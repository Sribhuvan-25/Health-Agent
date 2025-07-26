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
    missing_info: Annotated[Optional[Dict], "Information that needs to be collected from user"]
    ready_to_execute: Annotated[Optional[bool], "Whether we have all required information to execute the tool"]

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

# Node 1: Intent Classification and Information Extraction
def classify_intent_and_extract_info(state: AgentState) -> AgentState:
    """Classify the user's intent and extract available information in one step."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Build conversation context for the LLM
    conversation_context = ""
    if len(messages) > 1:
        # Include recent conversation history (last 3 messages)
        recent_messages = messages[-4:-1]  # Exclude the current message
        conversation_context = "Previous conversation:\n"
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                conversation_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_context += f"Agent: {msg.content}\n"
        conversation_context += "\n"
    
    # Comprehensive information extraction prompt
    prompt = f"""
    Analyze the user's request and extract both the intent and available information.
    
    {conversation_context}
    
    AVAILABLE OPERATIONS:
    - create_patient: Create a new patient record
    - get_patient: Get patient details by ID
    - search_patients: Search for patients by name
    - create_appointment: Create an appointment
    - get_appointment: Get appointment details by ID
    - list_appointments: List appointments for a patient
    - reschedule_appointment: Reschedule an appointment
    - cancel_appointment: Cancel an appointment
    - create_diagnostic_report: Create a diagnostic report
    - get_diagnostic_report: Get diagnostic report details by ID
    - get_diagnostic_reports: Get all diagnostic reports for a patient
    
    INFORMATION TO EXTRACT:
    - patient_name: Full name of the patient (e.g., "John Smith", "Dr. Alice Johnson")
    - patient_id: Existing patient ID (e.g., "12345", "Patient/abc123")
    - first_name: First name of new patient
    - last_name: Last name of new patient
    - appointment_id: Existing appointment ID
    - start_date: When the appointment should start (natural language like "tomorrow at 2pm", "next Monday 3pm")
    - end_date: When the appointment should end (natural language)
    - description: Description of the appointment or report
    - report_id: Existing diagnostic report ID
    - test_code: Medical test code (e.g., "HbA1c", "CBC")
    - test_value: Test result value (e.g., "5.4", "120")
    - test_unit: Unit of measurement (e.g., "%", "mg/dL")
    - is_new_patient: true/false - whether this appears to be a new patient
    
    SPECIAL RULES:
    1. If user mentions "my name is X" or "I am X", extract as patient_name and set is_new_patient to true
    2. If user says "first time", "new patient", "I'm new", set is_new_patient to true
    3. If user mentions an existing patient ID or "patient 12345", set is_new_patient to false
    4. For appointments, if patient_name is provided but no patient_id, we'll create the patient first
    5. For dates/times, extract natural language (e.g., "tomorrow at 2pm", "next week Monday 3pm")
    6. Don't expect duration from users - we'll use 30 minutes as default
    7. If user mentions scheduling, booking, or making an appointment, classify as create_appointment
    8. IMPORTANT: If user is scheduling an appointment but doesn't mention their name or patient ID, they are likely a new patient
    9. If user asks about "my appointments", "show my appointments", "view my appointments", classify as list_appointments
    10. If user mentions "patient ID" or "patient 12345" in context of appointments, extract the patient_id
    11. CRITICAL: If the user is continuing a previous conversation, combine information from both the current message and previous context
    12. If the agent previously asked for missing information, the user's response should be combined with the previous context
    13. IMPORTANT: Look at the conversation history above. If the agent asked for specific information (like name, date, etc.), and the user provides that information in their current message, combine it with what was mentioned before.
    14. For example: If agent asked for "name and date" and user says "James Goldmine and it is for a general health checkup", extract "James Goldmine" as patient_name and "general health checkup" as description, and keep the date from the previous message.
    
    User request: "{last_message}"
    
    Return a JSON object with this structure:
    {{
        "intent": "tool_name",
        "extracted_info": {{
            "patient_name": "extracted name if any",
            "patient_id": "extracted ID if any",
            "first_name": "extracted first name if any",
            "last_name": "extracted last name if any",
            "start_date": "extracted start date/time if any",
            "end_date": "extracted end date/time if any",
            "description": "extracted description if any",
            "appointment_id": "extracted appointment ID if any",
            "report_id": "extracted report ID if any",
            "test_code": "extracted test code if any",
            "test_value": "extracted test value if any",
            "test_unit": "extracted test unit if any",
            "is_new_patient": true/false
        }}
    }}
    
    Only return the JSON object, nothing else.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        result = json.loads(response.content)
        intent = result.get("intent", "unknown")
        extracted_info = result.get("extracted_info", {})
        
        # Merge with previous state if this is a continuation
        if len(messages) > 1 and state.get("tool_args"):
            previous_args = state["tool_args"]
            
            # Merge information, preferring new information over old
            merged_info = previous_args.copy()  # Start with all previous info
            for key in extracted_info:
                if extracted_info[key] and extracted_info[key] != "" and extracted_info[key] is not None:
                    merged_info[key] = extracted_info[key]
            
            extracted_info = merged_info
        
        return {
            **state,
            "intent": intent,
            "tool_args": extracted_info
        }
    except Exception as e:
        return {
            **state,
            "intent": "unknown",
            "tool_args": {}
        }

# Node 2: Check Information Completeness
def check_information_completeness(state: AgentState) -> AgentState:
    """Check if we have all required information for the intended operation."""
    intent = state["intent"]
    tool_args = state["tool_args"]
    
    # Define required parameters for each operation
    required_params = {
        "create_patient": ["patient_name"],
        "get_patient": ["patient_id"],
        "search_patients": [],  # Optional name_fragment
        "create_appointment": ["start_date"],  # patient_id OR patient_name is required
        "get_appointment": ["appointment_id"],
        "list_appointments": ["patient_id"],
        "reschedule_appointment": ["appointment_id", "start_date"],
        "cancel_appointment": ["appointment_id"],
        "create_diagnostic_report": ["patient_id", "test_code", "test_value", "test_unit"],
        "get_diagnostic_report": ["report_id"],
        "get_diagnostic_reports": ["patient_id"]
    }
    
    if intent not in required_params:
        return {
            **state,
            "missing_info": {"error": f"Unknown operation: {intent}"},
            "ready_to_execute": False
        }
    
    required = required_params[intent]
    missing = []
    
    for param in required:
        if param not in tool_args or not tool_args[param]:
            missing.append(param)
    
    # Special handling for appointments
    if intent == "create_appointment":
        # Check if we have either patient_id OR patient_name
        has_patient_id = "patient_id" in tool_args and tool_args["patient_id"]
        has_patient_name = "patient_name" in tool_args and tool_args["patient_name"]
        is_new_patient = tool_args.get("is_new_patient", True)  # Default to new patient if not specified
        
        if not has_patient_id and not has_patient_name:
            # Determine what to ask for based on context
            if is_new_patient:
                missing.append("patient_name")
            else:
                missing.append("patient_id")
    
    if missing:
        # Generate a helpful prompt for missing information
        prompt_messages = {
            "patient_id": "Please provide your patient ID",
            "patient_name": "Please provide your full name (first and last name)",
            "start_date": "Please provide when you'd like the appointment (e.g., 'tomorrow at 2pm', 'next Monday 3pm')",
            "appointment_id": "Please provide the appointment ID",
            "test_code": "Please provide the medical test code (e.g., 'HbA1c', 'CBC')",
            "test_value": "Please provide the test result value",
            "test_unit": "Please provide the unit of measurement (e.g., '%', 'mg/dL')",
            "report_id": "Please provide the diagnostic report ID"
        }
        
        missing_prompts = [prompt_messages.get(param, f"Please provide {param}") for param in missing]
        
        # Create a user-friendly message
        if intent == "create_appointment":
            if "patient_name" in missing and tool_args.get("is_new_patient", True):
                message = "I'd be happy to help you schedule an appointment! To get started, I need:\n"
                message += "• Your full name (first and last name)\n"
                if "start_date" not in missing:
                    message += "• When you'd like the appointment (e.g., 'tomorrow at 2pm')\n"
                message += "• Any specific details about the appointment"
            elif "patient_id" in missing and not tool_args.get("is_new_patient", True):
                message = "To schedule your appointment, I need:\n"
                message += "• Your patient ID\n"
                if "start_date" not in missing:
                    message += "• When you'd like the appointment (e.g., 'tomorrow at 2pm')\n"
                message += "• Any specific details about the appointment"
            else:
                message = "To schedule your appointment, I need:\n"
                message += "\n".join([f"• {prompt}" for prompt in missing_prompts])
        else:
            message = "To complete this request, I need:\n"
            message += "\n".join([f"• {prompt}" for prompt in missing_prompts])
        
        return {
            **state,
            "missing_info": {
                "missing_params": missing,
                "message": message
            },
            "ready_to_execute": False
        }
    
    return {
        **state,
        "missing_info": None,
        "ready_to_execute": True
    }

# Node 3: Tool Execution
def execute_tool(state: AgentState) -> AgentState:
    """Execute the appropriate tool based on the classified intent."""
    intent = state["intent"]
    tool_args = state["tool_args"]
    
    # Map intent to tool
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
    
    try:
        # Prepare arguments for the tool
        final_args = {}
        
        if intent == "create_patient":
            final_args["full_name"] = tool_args["patient_name"]
        
        elif intent == "get_patient":
            final_args["patient_id"] = tool_args["patient_id"]
        
        elif intent == "search_patients":
            if "patient_name" in tool_args and tool_args["patient_name"]:
                final_args["name_fragment"] = tool_args["patient_name"]
        
        elif intent == "create_appointment":
            # Handle new patient creation if needed
            # Check if we have a valid patient_id (not empty)
            has_valid_patient_id = "patient_id" in tool_args and tool_args["patient_id"] and tool_args["patient_id"].strip()
            
            if "patient_name" in tool_args and tool_args["patient_name"] and not has_valid_patient_id:
                patient_id = create_patient_tool.invoke({"full_name": tool_args["patient_name"]})
                final_args["patient_id"] = patient_id
            else:
                final_args["patient_id"] = tool_args["patient_id"]
            
            # Convert natural language dates to ISO format
            start_date = tool_args["start_date"]
            if not start_date.startswith("20"):  # Not already in ISO format
                start_iso = convert_natural_date_to_iso(start_date)
                if start_iso:
                    final_args["start_date"] = start_iso
                else:
                    return {
                        **state,
                        "tool_result": {"error": f"Could not parse start date: {start_date}"}
                    }
            else:
                final_args["start_date"] = start_date
            
            # Handle end date - use default 30 minutes if not provided
            if "end_date" in tool_args and tool_args["end_date"]:
                end_date = tool_args["end_date"]
                if not end_date.startswith("20"):  # Not already in ISO format
                    end_iso = convert_natural_date_to_iso(end_date)
                    if end_iso:
                        final_args["end_date"] = end_iso
                    else:
                        return {
                            **state,
                            "tool_result": {"error": f"Could not parse end date: {end_date}"}
                        }
                else:
                    final_args["end_date"] = end_date
            else:
                # Calculate end date from start date + 30 minutes default
                start_dt = datetime.fromisoformat(final_args["start_date"])
                end_dt = start_dt + timedelta(minutes=30)
                final_args["end_date"] = end_dt.isoformat()
            
            # Add description if provided
            if "description" in tool_args and tool_args["description"]:
                final_args["description"] = tool_args["description"]
            else:
                final_args["description"] = "Exam Appointment"
        
        elif intent == "get_appointment":
            final_args["appointment_id"] = tool_args["appointment_id"]
        
        elif intent == "list_appointments":
            final_args["patient_id"] = tool_args["patient_id"]
        
        elif intent == "reschedule_appointment":
            final_args["appointment_id"] = tool_args["appointment_id"]
            start_date = tool_args["start_date"]
            if not start_date.startswith("20"):
                start_iso = convert_natural_date_to_iso(start_date)
                if start_iso:
                    final_args["new_start_date"] = start_iso
                else:
                    return {
                        **state,
                        "tool_result": {"error": f"Could not parse start date: {start_date}"}
                    }
            else:
                final_args["new_start_date"] = start_date
            
            # Handle end date for reschedule
            if "end_date" in tool_args and tool_args["end_date"]:
                end_date = tool_args["end_date"]
                if not end_date.startswith("20"):
                    end_iso = convert_natural_date_to_iso(end_date)
                    if end_iso:
                        final_args["new_end_date"] = end_iso
                    else:
                        return {
                            **state,
                            "tool_result": {"error": f"Could not parse end date: {end_date}"}
                        }
                else:
                    final_args["new_end_date"] = end_date
            else:
                # Calculate end date from start date + 30 minutes default
                start_dt = datetime.fromisoformat(final_args["new_start_date"])
                end_dt = start_dt + timedelta(minutes=30)
                final_args["new_end_date"] = end_dt.isoformat()
        
        elif intent == "cancel_appointment":
            final_args["appointment_id"] = tool_args["appointment_id"]
        
        elif intent == "create_diagnostic_report":
            final_args["patient_id"] = tool_args["patient_id"]
            final_args["code"] = tool_args["test_code"]
            final_args["text"] = tool_args["test_code"]  # Use test_code as text
            final_args["value"] = float(tool_args["test_value"])
            final_args["unit"] = tool_args["test_unit"]
        
        elif intent == "get_diagnostic_report":
            final_args["report_id"] = tool_args["report_id"]
        
        elif intent == "get_diagnostic_reports":
            final_args["patient_id"] = tool_args["patient_id"]
        
        # Execute the tool
        result = tool.invoke(final_args)
        
        return {
            **state,
            "tool_name": intent,
            "tool_args": final_args,
            "tool_result": result
        }
        
    except Exception as e:
        return {
            **state,
            "tool_result": {"error": f"Tool execution failed: {str(e)}"}
        }

def convert_natural_date_to_iso(natural_date: str) -> str:
    """Convert natural language date/time to ISO format."""
    import re
    
    try:
        # Simple conversion for common patterns
        now = datetime.now()
        natural_date_lower = natural_date.lower().strip()
        
        # Determine the target date
        if "tomorrow" in natural_date_lower:
            target_date = now + timedelta(days=1)
        elif "today" in natural_date_lower:
            target_date = now
        elif "next week" in natural_date_lower:
            target_date = now + timedelta(days=7)
        else:
            # Try to parse specific dates like "28th July"
            date_pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)'
            match = re.search(date_pattern, natural_date_lower)
            
            if match:
                day = int(match.group(1))
                month_name = match.group(2)
                month_map = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                    'september': 9, 'october': 10, 'november': 11, 'december': 12
                }
                month = month_map.get(month_name, 1)
                year = now.year
                
                # If the date is in the past, assume next year
                target_date = datetime(year, month, day)
                if target_date < now:
                    target_date = datetime(year + 1, month, day)
            else:
                # For now, assume tomorrow if no specific date mentioned
                target_date = now + timedelta(days=1)
        
        # Extract time - handle various formats
        time_str = natural_date_lower
        
        # Remove date words to isolate time
        for date_word in ["tomorrow", "today", "next week", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            time_str = time_str.replace(date_word, "").strip()
        
        # Remove date patterns like "28th july", "july 28th"
        time_str = re.sub(r'\d{1,2}(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)', '', time_str)
        time_str = re.sub(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?', '', time_str)
        
        # Handle "at" and "on" keywords
        if "at" in time_str:
            time_str = time_str.split("at")[1].strip()
        elif "on" in time_str:
            # Split on "on" and take the part before it (which should be the time)
            parts = time_str.split("on")
            if len(parts) > 1:
                time_str = parts[0].strip()
        
        # Clean up any remaining text
        time_str = re.sub(r'[^\d:apm\s]', '', time_str).strip()
        
        # Handle 12-hour format with AM/PM
        if "am" in time_str or "pm" in time_str:
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
                # Try to parse as hour only
                try:
                    hour = int(time_str)
                    minute = 0
                except:
                    hour, minute = 9, 0
        
        # Create datetime object
        target_datetime = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return target_datetime.isoformat()
    except Exception as e:
        print(f"Error parsing date '{natural_date}': {e}")
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

# Node 4: Response Formatting
def format_response(state: AgentState) -> AgentState:
    """Format the tool result into a user-friendly response."""
    missing_info = state.get("missing_info")
    
    # If we have missing information, prompt the user
    if missing_info and "message" in missing_info:
        response = missing_info["message"]
    elif missing_info and "error" in missing_info:
        response = f"❌ Error: {missing_info['error']}"
    else:
        # Format tool result
        tool_result = state["tool_result"]
        tool_name = state["tool_name"]
        
        if "error" in tool_result:
            response = f"❌ Error: {tool_result['error']}"
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
                
                # Get patient details if available
                patient_name = "Unknown"
                if appt.get("participant") and len(appt["participant"]) > 0:
                    patient_ref = appt["participant"][0].get("actor", {}).get("reference", "")
                    if patient_ref.startswith("Patient/"):
                        try:
                            from main import get_patient
                            patient_id = patient_ref.split("/")[1]
                            patient = get_patient(patient_id)
                            if patient and patient.get("name"):
                                name = patient["name"][0]
                                family = name.get("family", "")
                                given = " ".join(name.get("given", []))
                                patient_name = f"{given} {family}".strip()
                        except:
                            patient_name = f"Patient {patient_id}"
                
                # Format date/time
                start_time = appt.get("start", "")
                end_time = appt.get("end", "")
                
                if start_time:
                    try:
                        from datetime import datetime
                        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00")) if end_time else start_dt
                        
                        # Format as readable date/time
                        start_str = start_dt.strftime("%A, %B %d, %Y at %I:%M %p")
                        end_str = end_dt.strftime("%I:%M %p")
                        
                        response = f"📅 Appointment Confirmation:\n"
                        response += f"Patient: {patient_name}\n"
                        response += f"Date & Time: {start_str}\n"
                        response += f"Duration: Until {end_str}\n"
                        response += f"Description: {appt.get('description', 'Exam Appointment')}\n"
                        response += f"Status: {appt.get('status', 'Unknown')}\n"
                        response += f"Appointment ID: {appt.get('id')} (save this for future reference)"
                    except:
                        # Fallback if date parsing fails
                        response = f"📅 Appointment Details:\n"
                        response += f"Patient: {patient_name}\n"
                        response += f"ID: {appt.get('id')}\n"
                        response += f"Description: {appt.get('description')}\n"
                        response += f"Status: {appt.get('status')}\n"
                        response += f"Start: {start_time}\n"
                        response += f"End: {end_time}"
                else:
                    response = f"📅 Appointment Details:\n"
                    response += f"Patient: {patient_name}\n"
                    response += f"ID: {appt.get('id')}\n"
                    response += f"Description: {appt.get('description')}\n"
                    response += f"Status: {appt.get('status')}"
            
            elif tool_name == "list_appointments":
                appts = tool_result
                if appts:
                    response = "📅 Your Appointments:\n"
                    for i, appt in enumerate(appts, 1):
                        # Get patient details
                        patient_name = "Unknown"
                        if appt.get("participant") and len(appt["participant"]) > 0:
                            patient_ref = appt["participant"][0].get("actor", {}).get("reference", "")
                            if patient_ref.startswith("Patient/"):
                                try:
                                    from main import get_patient
                                    patient_id = patient_ref.split("/")[1]
                                    patient = get_patient(patient_id)
                                    if patient and patient.get("name"):
                                        name = patient["name"][0]
                                        family = name.get("family", "")
                                        given = " ".join(name.get("given", []))
                                        patient_name = f"{given} {family}".strip()
                                except:
                                    patient_name = f"Patient {patient_id}"
                        
                        # Format date/time
                        start_time = appt.get("start", "")
                        end_time = appt.get("end", "")
                        date_str = "Unknown time"
                        
                        if start_time:
                            try:
                                from datetime import datetime
                                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                                date_str = start_dt.strftime("%A, %B %d at %I:%M %p")
                            except:
                                date_str = start_time
                        
                        response += f"{i}. {appt.get('description', 'Appointment')} - {date_str}\n"
                        response += f"   Patient: {patient_name} | Status: {appt.get('status', 'Unknown')} | ID: {appt.get('id')}\n\n"
                    
                    response += f"💡 Tip: You can use your patient ID ({appts[0].get('participant', [{}])[0].get('actor', {}).get('reference', '').split('/')[-1]}) to view your appointments anytime!"
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
    workflow.add_node("classify_intent", classify_intent_and_extract_info)
    workflow.add_node("check_info_completeness", check_information_completeness)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("format_response", format_response)
    
    # Add edges
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "check_info_completeness")
    
    # Conditional edge: only proceed to execute_tool if ready_to_execute is True
    def should_execute(state):
        return state.get("ready_to_execute", False)
    
    workflow.add_conditional_edges(
        "check_info_completeness",
        should_execute,
        {
            True: "execute_tool",
            False: "format_response"
        }
    )
    
    workflow.add_edge("execute_tool", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# Global conversation state
conversation_state = {
    "messages": [],
    "intent": None,
    "tool_name": None,
    "tool_args": None,
    "tool_result": None,
    "missing_info": None,
    "ready_to_execute": False
}

# Main function to run the agent with conversation memory
def run_agent(user_input: str) -> str:
    """Run the agent with user input and return the response. Maintains conversation context."""
    global conversation_state
    
    workflow = create_workflow()
    
    # Add the new user message to the conversation
    conversation_state["messages"].append(HumanMessage(content=user_input))
    
    # Run the workflow with the current conversation state
    result = workflow.invoke(conversation_state)
    
    # Update the global conversation state
    conversation_state.update(result)
    
    # Return the last AI message
    return result["messages"][-1].content

# Function to reset conversation (useful for testing)
def reset_conversation():
    """Reset the conversation state."""
    global conversation_state
    conversation_state = {
        "messages": [],
        "intent": None,
        "tool_name": None,
        "tool_args": None,
        "tool_result": None,
        "missing_info": None,
        "ready_to_execute": False
    }

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