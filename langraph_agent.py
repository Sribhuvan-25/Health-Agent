"""
LangGraph Agentic Workflow for FHIR Healthcare APIs
Uses the APIs from main.py as tools in a simple workflow:
1. Intent Classification Node - Understands user query
2. Tool Execution Node - Calls appropriate FHIR API
3. Response Formatting Node - Formats output for user
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dotenv import load_dotenv
from main import (
    create_patient, get_patient, search_patients,
    create_appointment, get_appointment, list_appointments,
    reschedule_appointment, cancel_appointment,
    create_diagnostic_report, get_diagnostic_report, get_diagnostic_reports
)

# Load environment variables
load_dotenv()

# State definition
class AgentState(TypedDict):
    messages: Annotated[List, "The messages in the conversation"]
    intent: Annotated[Optional[str], "The classified intent of the user query"]
    tool_name: Annotated[Optional[str], "The tool to be executed"]
    tool_args: Annotated[Optional[Dict], "Arguments for the tool"]
    tool_result: Annotated[Optional[Dict], "Result from tool execution"]

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define tools from main.py
@tool
def create_patient_tool(full_name: str) -> str:
    """Create a new patient with the given full name."""
    return create_patient(full_name)

@tool
def get_patient_tool(patient_id: str) -> dict:
    """Get patient details by patient ID."""
    return get_patient(patient_id)

@tool
def search_patients_tool(name_fragment: Optional[str] = None) -> List[Dict]:
    """Search for patients by name fragment."""
    return search_patients(name_fragment)

@tool
def create_appointment_tool(patient_id: str, start_date: str, end_date: str, description: str = "Exam Appointment") -> str:
    """Create an appointment for a patient. Dates should be in ISO format (YYYY-MM-DDTHH:MM:SS)."""
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    return create_appointment(patient_id, start_dt, end_dt, description)

@tool
def get_appointment_tool(appointment_id: str) -> dict:
    """Get appointment details by appointment ID."""
    return get_appointment(appointment_id)

@tool
def list_appointments_tool(patient_id: str) -> List[Dict]:
    """List all appointments for a patient."""
    return list_appointments(patient_id)

@tool
def reschedule_appointment_tool(appointment_id: str, new_start_date: str, new_end_date: str) -> None:
    """Reschedule an appointment. Dates should be in ISO format (YYYY-MM-DDTHH:MM:SS)."""
    new_start_dt = datetime.fromisoformat(new_start_date)
    new_end_dt = datetime.fromisoformat(new_end_date)
    return reschedule_appointment(appointment_id, new_start_dt, new_end_dt)

@tool
def cancel_appointment_tool(appointment_id: str) -> None:
    """Cancel an appointment."""
    return cancel_appointment(appointment_id)

@tool
def create_diagnostic_report_tool(patient_id: str, code: str, text: str, value: float, unit: str) -> str:
    """Create a diagnostic report for a patient."""
    return create_diagnostic_report(patient_id, code, text, value, unit)

@tool
def get_diagnostic_report_tool(report_id: str) -> dict:
    """Get diagnostic report details by report ID."""
    return get_diagnostic_report(report_id)

@tool
def get_diagnostic_reports_tool(patient_id: str) -> List[Dict]:
    """Get all diagnostic reports for a patient."""
    return get_diagnostic_reports(patient_id)

# Create tool list
tools = [
    create_patient_tool,
    get_patient_tool,
    search_patients_tool,
    create_appointment_tool,
    get_appointment_tool,
    list_appointments_tool,
    reschedule_appointment_tool,
    cancel_appointment_tool,
    create_diagnostic_report_tool,
    get_diagnostic_report_tool,
    get_diagnostic_reports_tool
]

# Node 1: Intent Classification
def classify_intent(state: AgentState) -> AgentState:
    """Classify the user's intent and determine which tool to use."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Simple intent classification prompt
    prompt = f"""
    Analyze the user's request and classify their intent. Choose from these categories:
    
    PATIENT OPERATIONS:
    - create_patient: User wants to create a new patient
    - get_patient: User wants to get patient details by ID
    - search_patients: User wants to search for patients by name
    
    APPOINTMENT OPERATIONS:
    - create_appointment: User wants to create an appointment
    - get_appointment: User wants to get appointment details by ID
    - list_appointments: User wants to list appointments for a patient
    - reschedule_appointment: User wants to reschedule an appointment
    - cancel_appointment: User wants to cancel an appointment
    
    DIAGNOSTIC REPORT OPERATIONS:
    - create_diagnostic_report: User wants to create a diagnostic report
    - get_diagnostic_report: User wants to get diagnostic report details by ID
    - get_diagnostic_reports: User wants to get all diagnostic reports for a patient
    
    SPECIAL CASES:
    - If user mentions they are new/first time AND want an appointment, classify as "create_appointment"
    - If user just wants to create a patient record, classify as "create_patient"
    - If user mentions scheduling, booking, or making an appointment, classify as "create_appointment"
    
    User request: {last_message}
    
    Respond with just the tool name (e.g., "create_patient") or "unknown" if unclear.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip()
    
    return {
        **state,
        "intent": intent
    }

# Node 2: Tool Execution
def execute_tool(state: AgentState) -> AgentState:
    """Execute the appropriate tool based on the classified intent."""
    intent = state["intent"]
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Map intent to tool and extract arguments
    tool_mapping = {
        "create_patient": create_patient_tool,
        "get_patient": get_patient_tool,
        "search_patients": search_patients_tool,
        "create_appointment": create_appointment_tool,
        "get_appointment": get_appointment_tool,
        "list_appointments": list_appointments_tool,
        "reschedule_appointment": reschedule_appointment_tool,
        "cancel_appointment": cancel_appointment_tool,
        "create_diagnostic_report": create_diagnostic_report_tool,
        "get_diagnostic_report": get_diagnostic_report_tool,
        "get_diagnostic_reports": get_diagnostic_reports_tool
    }
    
    if intent not in tool_mapping:
        return {
            **state,
            "tool_result": {"error": f"Unknown intent: {intent}"}
        }
    
    tool = tool_mapping[intent]
    
    # Extract arguments from user message
    prompt = f"""
    Extract the necessary arguments for the {intent} tool from this user message:
    "{last_message}"
    
    Return a JSON object with the arguments. For example:
    - For create_patient: {{"full_name": "John Doe"}}
    - For get_patient: {{"patient_id": "12345"}}
    - For create_appointment: {{"patient_id": "12345", "start_date": "2024-01-15T10:00:00", "end_date": "2024-01-15T11:00:00", "description": "Checkup"}}
    
    IMPORTANT RULES:
    1. If the user mentions their name (like "My name is X" or "I am X"), extract it as "patient_name"
    2. If the user says it's their first time or they're new, they need a patient record created first
    3. For appointments, if no patient_id is provided but a name is mentioned, use "patient_name" instead
    4. If the user mentions scheduling, booking, or making an appointment, extract any available details
    5. For dates/times, extract them as natural language strings (e.g., "tomorrow at 2pm", "next week monday 3pm")
    6. For duration, extract as natural language (e.g., "30 minutes", "1 hour", "21min")
    
    Look for patterns like:
    - "My name is [Name]"
    - "I am [Name]"
    - "This is my first time"
    - "I'm new"
    - "Schedule for [date/time]"
    - "Appointment at [time]"
    - "for [duration]"
    
    Only return the JSON object, nothing else.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        tool_args = json.loads(response.content)
        
        # Special handling for new patients wanting appointments
        if intent == "create_appointment":
            # Check if we have a patient name but no patient_id
            if "patient_name" in tool_args and "patient_id" not in tool_args:
                # First create the patient
                patient_name = tool_args.pop("patient_name")
                print(f"🆕 Creating new patient: {patient_name}")
                patient_id = create_patient_tool.invoke({"full_name": patient_name})
                tool_args["patient_id"] = patient_id
            
            # Convert natural language dates/times to ISO format
            if "start_date" in tool_args and isinstance(tool_args["start_date"], str):
                start_date = tool_args["start_date"]
                if not start_date.startswith("20"):  # Not already in ISO format
                    # Convert natural language to ISO format
                    start_iso = convert_natural_date_to_iso(start_date)
                    if start_iso:
                        tool_args["start_date"] = start_iso
            
            if "end_date" in tool_args and isinstance(tool_args["end_date"], str):
                end_date = tool_args["end_date"]
                if not end_date.startswith("20"):  # Not already in ISO format
                    # Convert natural language to ISO format
                    end_iso = convert_natural_date_to_iso(end_date)
                    if end_iso:
                        tool_args["end_date"] = end_iso
            
            # If we have start_date but no end_date, calculate it from duration
            if "start_date" in tool_args and "end_date" not in tool_args and "duration" in tool_args:
                start_dt = datetime.fromisoformat(tool_args["start_date"])
                duration_minutes = parse_duration(tool_args["duration"])
                end_dt = start_dt + timedelta(minutes=duration_minutes)
                tool_args["end_date"] = end_dt.isoformat()
                del tool_args["duration"]  # Remove duration as we now have end_date
            
            # Handle missing required parameters
            missing_params = []
            if "patient_id" not in tool_args:
                missing_params.append("patient_id")
            if "start_date" not in tool_args:
                missing_params.append("start_date")
            if "end_date" not in tool_args:
                missing_params.append("end_date")
            
            if missing_params:
                return {
                    **state,
                    "tool_result": {
                        "error": f"Missing required parameters for appointment: {', '.join(missing_params)}. Please provide: patient ID, start date/time, and end date/time."
                    }
                }
        
        result = tool.invoke(tool_args)
        return {
            **state,
            "tool_name": intent,
            "tool_args": tool_args,
            "tool_result": result
        }
    except Exception as e:
        return {
            **state,
            "tool_result": {"error": f"Tool execution failed: {str(e)}"}
        }

def convert_natural_date_to_iso(natural_date: str) -> str:
    """Convert natural language date/time to ISO format."""
    try:
        # Simple conversion for common patterns
        now = datetime.now()
        
        if "tomorrow" in natural_date.lower():
            target_date = now + timedelta(days=1)
        elif "today" in natural_date.lower():
            target_date = now
        else:
            # For now, assume tomorrow if no specific date mentioned
            target_date = now + timedelta(days=1)
        
        # Extract time
        time_str = natural_date.lower()
        if "am" in time_str or "pm" in time_str:
            # Handle 12-hour format
            if "am" in time_str:
                time_str = time_str.replace("am", "").strip()
                hour = int(time_str.split(":")[0]) if ":" in time_str else int(time_str)
                if hour == 12:
                    hour = 0
            else:
                time_str = time_str.replace("pm", "").strip()
                hour = int(time_str.split(":")[0]) if ":" in time_str else int(time_str)
                if hour != 12:
                    hour += 12
            
            minute = int(time_str.split(":")[1]) if ":" in time_str else 0
        else:
            # Assume 24-hour format or default to 9 AM
            if ":" in time_str:
                hour, minute = map(int, time_str.split(":"))
            else:
                hour, minute = 9, 0
        
        # Create datetime object
        target_datetime = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return target_datetime.isoformat()
    except:
        return None

def parse_duration(duration_str: str) -> int:
    """Parse duration string to minutes."""
    duration_str = duration_str.lower().strip()
    
    if "hour" in duration_str:
        hours = int(duration_str.split("hour")[0].strip())
        return hours * 60
    elif "minute" in duration_str:
        minutes = int(duration_str.split("minute")[0].strip())
        return minutes
    elif "min" in duration_str:
        minutes = int(duration_str.split("min")[0].strip())
        return minutes
    else:
        # Default to 30 minutes
        return 30

# Node 3: Response Formatting
def format_response(state: AgentState) -> AgentState:
    """Format the tool result into a user-friendly response."""
    tool_result = state["tool_result"]
    tool_name = state["tool_name"]
    
    if "error" in tool_result:
        error_msg = tool_result['error']
        if "Missing required parameters for appointment" in error_msg:
            # Check if we have a patient name in the original message
            original_message = state["messages"][0].content.lower()
            has_name = any(phrase in original_message for phrase in ["my name is", "i am", "i'm"])
            
            if has_name:
                response = f"❌ {error_msg}\n\n💡 I can see you mentioned your name! To complete your appointment booking, please provide:\n• Preferred date and time (e.g., 'tomorrow at 2 PM')\n• Duration (e.g., '30 minutes' or '1 hour')\n\nExample: 'Schedule for tomorrow at 2 PM for 30 minutes'"
            else:
                response = f"❌ {error_msg}\n\n💡 To schedule an appointment, please provide:\n• Your name (if you're a new patient)\n• Preferred date and time\n• Duration of the appointment\n\nExample: 'My name is John and I'd like an appointment tomorrow at 2 PM for 30 minutes'"
        else:
            response = f"❌ Error: {error_msg}"
    else:
        # Format based on tool type
        if tool_name == "create_patient":
            response = f"✅ Successfully created patient with ID: {tool_result}"
        elif tool_name == "get_patient":
            patient = tool_result
            name = patient.get("name", [{}])[0] if patient.get("name") else {}
            family = name.get("family", "Unknown")
            given = " ".join(name.get("given", []))
            response = f"📋 Patient Details:\nName: {given} {family}\nID: {patient.get('id')}\nStatus: {'Active' if patient.get('active') else 'Inactive'}"
        elif tool_name == "search_patients":
            patients = tool_result
            if patients:
                response = "🔍 Found patients:\n" + "\n".join([f"- {p.get('id')}: {p.get('name', [{}])[0].get('family', 'Unknown')}" for p in patients])
            else:
                response = "🔍 No patients found matching your search."
        elif tool_name == "create_appointment":
            response = f"📅 Successfully created appointment with ID: {tool_result}"
        elif tool_name == "get_appointment":
            appt = tool_result
            response = f"📅 Appointment Details:\nID: {appt.get('id')}\nDescription: {appt.get('description')}\nStatus: {appt.get('status')}"
        elif tool_name == "list_appointments":
            appts = tool_result
            if appts:
                response = "📅 Appointments:\n" + "\n".join([f"- {a.get('id')}: {a.get('description')} ({a.get('status')})" for a in appts])
            else:
                response = "📅 No appointments found for this patient."
        elif tool_name == "reschedule_appointment":
            response = "🔄 Appointment successfully rescheduled."
        elif tool_name == "cancel_appointment":
            response = "🚫 Appointment successfully cancelled."
        elif tool_name == "create_diagnostic_report":
            response = f"🧪 Successfully created diagnostic report with ID: {tool_result}"
        elif tool_name == "get_diagnostic_report":
            report = tool_result
            response = f"🧪 Diagnostic Report:\nID: {report.get('id')}\nCode: {report.get('code', {}).get('text', 'Unknown')}\nStatus: {report.get('status')}"
        elif tool_name == "get_diagnostic_reports":
            reports = tool_result
            if reports:
                response = "🧪 Diagnostic Reports:\n" + "\n".join([f"- {r.get('id')}: {r.get('code', {}).get('text', 'Unknown')}" for r in reports])
            else:
                response = "🧪 No diagnostic reports found for this patient."
        else:
            response = f"✅ Operation completed successfully: {tool_result}"
    
    # Add the response to messages
    messages = state["messages"] + [AIMessage(content=response)]
    
    return {
        **state,
        "messages": messages
    }

# Create the workflow
def create_workflow():
    """Create the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("format_response", format_response)
    
    # Add edges
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "execute_tool")
    workflow.add_edge("execute_tool", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# Main function to run the agent
def run_agent(user_input: str) -> str:
    """Run the agent with user input and return the response."""
    workflow = create_workflow()
    
    # Initialize state
    state = {
        "messages": [HumanMessage(content=user_input)],
        "intent": None,
        "tool_name": None,
        "tool_args": None,
        "tool_result": None
    }
    
    # Run the workflow
    result = workflow.invoke(state)
    
    # Return the last AI message
    return result["messages"][-1].content

# Example usage
if __name__ == "__main__":
    # Test the agent with some example queries
    test_queries = [
        "Create a patient named John Smith",
        "Search for patients named Alice",
        "Get patient details for patient ID 12345",
        "Create an appointment for patient 12345 tomorrow at 2 PM for 30 minutes",
        "List all appointments for patient 12345",
        "Create a diagnostic report for patient 12345 with HbA1c test, value 5.4%"
    ]
    
    print("🤖 FHIR Healthcare Agent")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = run_agent(query)
        print(f"Agent: {response}")
        print("-" * 50) 