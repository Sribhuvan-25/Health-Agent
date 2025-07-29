"""
ExamBuilder LangGraph Agent v2
Redesigned based on actual working API endpoints and real data structures.

Working endpoints discovered:
- get_instructor_id()
- list_exams(instructor_id)
- get_exam(instructor_id, exam_id)
- list_students(instructor_id)
- list_group_categories(instructor_id)

This agent focuses on what actually works with the API.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from dotenv import load_dotenv
from langsmith import Client
from exambuilder_tools import (
    get_instructor_id, list_exams, get_exam, list_students, get_student,
    list_group_categories, create_student, update_student, list_scheduled_exams,
    schedule_exam, get_exam_attempt, get_student_exam_statistics, search_student_by_student_id
)

# Load environment variables
load_dotenv()

# Configure LangSmith for telemetry
os.environ["LANGSMITH_TRACING_V2"] = "true"
# LangSmith API key and project will be loaded from .env file

# Initialize LangSmith client
langsmith_client = Client()

# State definition
class ExamBuilderAgentState(TypedDict):
    messages: Annotated[List, "The messages in the conversation"]
    intent: Annotated[Optional[str], "The classified intent of the user query"]
    instructor_id: Annotated[Optional[str], "The instructor ID for API calls"]
    tool_name: Annotated[Optional[str], "The tool to be executed"]
    tool_args: Annotated[Optional[Dict], "Arguments for the tool"]
    tool_result: Annotated[Optional[Dict], "Result from tool execution"]
    missing_info: Annotated[Optional[Dict], "Information that needs to be collected from user"]
    ready_to_execute: Annotated[Optional[bool], "Whether we have all required information to execute the tool"]
    cached_user_id: Annotated[Optional[str], "Cached user ID for a student to avoid repeated lookups"]
    cached_student_id: Annotated[Optional[str], "Cached student ID (email) corresponding to the user ID"]

# Initialize LLM with tracing
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0,
    tags=["exambuilder-agent", "gpt-3.5-turbo"]
)

# Define tools based on working endpoints
@tool
def get_instructor_id_tool() -> dict:
    """Get the instructor ID for the authenticated user. This is required for all other API operations."""
    return get_instructor_id()

@tool  
def list_exams_tool(instructor_id: str, exam_name: Optional[str] = None, exam_state: str = "all") -> dict:
    """List all exams available to the instructor. Use this to find exam IDs and names for scheduling. 
    Can filter by name and state. Active exams can be scheduled for students."""
    return list_exams(instructor_id, exam_name, exam_state)

@tool
def get_exam_tool(instructor_id: str, exam_id: str) -> dict:
    """Get detailed information about a specific exam using its exam ID. 
    Use this after list_exams_tool to get full details about an exam."""
    return get_exam(instructor_id, exam_id)

@tool
def list_students_tool(instructor_id: str, first_name: Optional[str] = None, 
                      last_name: Optional[str] = None, student_id: Optional[str] = None) -> dict:
    """List all students, optionally filtered by name or student ID. 
    Use this to find student User IDs which are needed for scheduling exams and getting results."""
    return list_students(instructor_id, first_name, last_name, student_id)

@tool
def get_student_tool(instructor_id: str, user_id: str) -> dict:
    """Get detailed information about a specific student using their User ID.
    Use this after list_students_tool or search_student_by_student_id_tool to get full student details."""
    return get_student(instructor_id, user_id)

@tool
def list_group_categories_tool(instructor_id: str) -> dict:
    """List all group categories available to the instructor. 
    This is useful for understanding the organizational structure of students."""
    return list_group_categories(instructor_id)

@tool
def create_student_tool(instructor_id: str, first_name: str, last_name: str, student_id: str, password: str) -> dict:
    """Create a new student account with required information. 
    The student_id should be an email address which will be used for login."""
    return create_student(instructor_id, first_name, last_name, student_id, password)

@tool
def update_student_tool(instructor_id: str, student_id: str, first_name: Optional[str] = None,
                       last_name: Optional[str] = None, new_student_id: Optional[str] = None,
                       password: Optional[str] = None, email: Optional[str] = None,
                       employee_number: Optional[str] = None) -> dict:
    """Update a student's information using their current student ID (email)."""
    return update_student(instructor_id, student_id, first_name, last_name, new_student_id, 
                         password, email, employee_number)

@tool
def list_scheduled_exams_tool(instructor_id: str, user_id: Optional[str] = None, 
                             exam_id: Optional[str] = None) -> dict:
    """List scheduled exams, optionally filtered by student User ID or exam ID. 
    This returns User Exam IDs which are needed for get_exam_attempt_tool and exam results.
    SEQUENCE: Use search_student_by_student_id_tool first to get User ID from email."""
    return list_scheduled_exams(instructor_id, user_id, exam_id)

@tool
def schedule_exam_tool(instructor_id: str, exam_id: str, user_id: str) -> dict:
    """Schedule an exam for a student using exam ID and student User ID.
    SEQUENCE: 1) Use list_exams_tool to get exam_id, 2) Use search_student_by_student_id_tool to get user_id."""
    return schedule_exam(instructor_id, exam_id, user_id)

@tool
def get_exam_attempt_tool(instructor_id: str, user_exam_id: str) -> dict:
    """Get details of a specific exam attempt using the User Exam ID.
    The User Exam ID comes from list_scheduled_exams_tool and represents a specific student's attempt at an exam.
    This returns the actual exam results, scores, and completion status."""
    return get_exam_attempt(instructor_id, user_exam_id)

@tool
def get_student_exam_statistics_tool(instructor_id: str, student_id: str, user_exam_id: str) -> dict:
    """Get detailed exam statistics for a specific student and exam attempt.
    Requires both student ID (email) and User Exam ID from list_scheduled_exams_tool."""
    return get_student_exam_statistics(instructor_id, student_id, user_exam_id)

@tool
def search_student_by_student_id_tool(instructor_id: str, student_id: str) -> dict:
    """Search for a student using their Student ID (email) and return their User ID.
    CRITICAL: Use this to convert email addresses to User IDs needed for scheduling and exam operations.
    This is often the first step when working with student emails."""
    return search_student_by_student_id(instructor_id, student_id)

# Create tool list
tools = [
    get_instructor_id_tool,
    list_exams_tool,
    get_exam_tool,
    list_students_tool,
    get_student_tool,
    list_group_categories_tool,
    create_student_tool,
    update_student_tool,
    list_scheduled_exams_tool,
    schedule_exam_tool,
    get_exam_attempt_tool,
    get_student_exam_statistics_tool,
    search_student_by_student_id_tool
]

# Node 1: Get Instructor ID
def get_instructor_id_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Get the instructor ID if not already available."""
    if state.get("instructor_id"):
        return state
    
    try:
        result = get_instructor_id()
        if result.get("status") and result.get("instructor_id"):
            return {
                **state,
                "instructor_id": result["instructor_id"]
            }
        else:
            return {
                **state,
                "tool_result": {"error": f"Failed to get instructor ID: {result.get('message', 'Unknown error')}"}
            }
    except Exception as e:
        return {
            **state,
            "tool_result": {"error": f"Failed to get instructor ID: {str(e)}"}
        }

# Node 2: Intent Classification and Information Extraction
def classify_intent_and_extract_info(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Classify the user's intent and extract available information."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Build conversation context
    conversation_context = ""
    if len(messages) > 1:
        recent_messages = messages[-4:-1]
        conversation_context = "Previous conversation:\n"
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                conversation_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_context += f"Agent: {msg.content}\n"
        conversation_context += "\n"
    
    # Check if this is a workflow continuation
    is_workflow_continuation = False
    workflow_context = {}
    
    if len(messages) > 1:
        # Initialize previous_ai_message
        previous_ai_message = None
        for msg in reversed(messages[:-1]):
            if isinstance(msg, AIMessage):
                previous_ai_message = msg.content
                break
        
        # Check if the previous state was awaiting exam selection
        if (state.get("tool_result") and 
            state["tool_result"].get("awaiting_exam_selection")):
            is_workflow_continuation = True
            workflow_context["type"] = "exam_selection_by_name"
        elif previous_ai_message:
            # Check for workflow indicators in the previous response
            if "Available Active Exams" in previous_ai_message and "Next Step" in previous_ai_message:
                is_workflow_continuation = True
                workflow_context["type"] = "exam_selection"
                # Extract available exams from the previous response
                if "Serengeti Practice Exam" in previous_ai_message:
                    workflow_context["available_exams"] = ["Serengeti Practice Exam", "Serengeti Certification"]
            elif "Please provide your email address" in previous_ai_message:
                is_workflow_continuation = True
                workflow_context["type"] = "email_provided"
            elif "what's your first name?" in previous_ai_message.lower():
                is_workflow_continuation = True
                workflow_context["type"] = "create_student_first_name"
            elif "what's your last name?" in previous_ai_message.lower():
                is_workflow_continuation = True
                workflow_context["type"] = "create_student_last_name"
            elif "your email address (this will be your student id" in previous_ai_message.lower():
                is_workflow_continuation = True
                workflow_context["type"] = "create_student_email"
            elif "create a password for your account" in previous_ai_message.lower():
                is_workflow_continuation = True
                workflow_context["type"] = "create_student_password"
            elif "failed to create student account" in previous_ai_message.lower() and "try a different email" in previous_ai_message.lower():
                is_workflow_continuation = True
                workflow_context["type"] = "create_student_retry_email"
            elif "please provide the user exam id" in previous_ai_message.lower():
                is_workflow_continuation = True
                workflow_context["type"] = "exam_results_followup"
            elif "which exam results would you like to see? please provide the reference id" in previous_ai_message.lower():
                is_workflow_continuation = True
                workflow_context["type"] = "exam_results_reference_id"
            elif "which exam results would you like to see? you can say the exam name or number" in previous_ai_message.lower():
                is_workflow_continuation = True
                workflow_context["type"] = "exam_selection_by_name"
    
    prompt = f"""
    Analyze the user's request for an ExamBuilder system and classify their intent.
    
    {conversation_context}
    
    Current user request: "{last_message}"
    
    WORKFLOW CONTEXT: {"This is a continuation of: " + workflow_context.get("type", "") if is_workflow_continuation else "New conversation"}
    
    {f"IMPORTANT: This is an exam selection workflow continuation. The user is choosing from a list of exams. ALWAYS classify as 'get_exam_attempt_by_student'." if workflow_context.get("type") == "exam_selection_by_name" else ""}
    
    AVAILABLE INTENTS:
    - list_exams: User wants to see available exams
    - get_exam: User wants details about a specific exam
    - list_students: User wants to see students
    - get_student: User wants details about a specific student
    - create_student: User wants to create a new student account/register
    - schedule_exam: User wants to schedule an exam (needs student ID and exam ID)
    - list_scheduled_exams: User wants to see scheduled exams
    - get_exam_attempt: User wants to see exam attempt details/results
    - get_exam_attempt_by_student: User wants exam results by providing student ID
    - get_student_exam_statistics: User wants to see exam statistics/results
    - help: User wants to know system capabilities
    - status: User wants system status
    - unsupported: Request not supported
    
    EXTRACTION RULES:
    - If this is a create_student workflow continuation, ALWAYS classify as "create_student"
    - If this is exam_selection_by_name workflow continuation, ALWAYS classify as "get_exam_attempt_by_student"
    - For create_student_first_name: extract the user's response as first_name
    - For create_student_last_name: extract the user's response as last_name  
    - For create_student_email: extract the user's response as student_id
    - For create_student_password: extract the user's response as password
    - For create_student_retry_email: extract the user's response as student_id (new email attempt)
    - For exam_results_followup: classify as "get_exam_attempt" and extract user_exam_id
    - For exam_results_reference_id: classify as "get_exam_attempt" and extract user_exam_id from the response
    - For exam_selection_by_name: extract exam_name or exam_number from user response (any text like "1", "2", "practice", "certification", etc.)
    - Extract exam names (like "Serengeti Practice Exam", "Certification")
    - Extract student emails/IDs from phrases like "for TEST123", "for student TEST123", "TEST123 scheduled"
    - Extract 32-character hex strings as user IDs or user_exam_ids
    - Extract user_exam_id (reference IDs) from phrases like "results for abc123", "show results abc123"
    - If user mentions "register", "create account", "don't have account", "new account", classify as "create_student"
    - If user mentions scheduling + exam name, classify as "schedule_exam"
    - If user asks about available/list of exams, classify as "list_exams"
    - If user asks about scheduled exams + any ID/email, classify as "list_scheduled_exams" and extract the ID
    - If user asks about "results", "show results", "exam results" + student ID + exam name, classify as "get_exam_attempt_by_student" and extract both student_id and exam_name
    - If user asks about "results", "show results", "exam results" + student ID, classify as "get_exam_attempt_by_student" and extract student_id
    - If user asks about "statistics", "stats", "scores" + reference ID, classify as "get_student_exam_statistics"
    
    WORKFLOW-SPECIFIC EXTRACTION:
    {f"- This is step: {workflow_context.get('type', '')}" if is_workflow_continuation else ""}
    {f"- Extract the user's response as the requested information for this step" if is_workflow_continuation else ""}
    
    Return only a JSON object:
    {{
        "intent": "classified_intent",
        "extracted_info": {{
            "exam_name": "extracted exam name or null",
            "student_id": "extracted email/ID or null",
            "user_exam_id": "extracted reference/user_exam_id or null",
            "first_name": "extracted first name or null",
            "last_name": "extracted last name or null", 
            "password": "extracted password or null",
            "exam_number": "extracted exam number (1, 2, etc.) or null"
        }}
    }}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        result = json.loads(response.content)
        intent = result.get("intent", "unknown")
        extracted_info = result.get("extracted_info", {})
        
        # Merge with previous state if this is a continuation
        if len(messages) > 1 and state.get("tool_args"):
            previous_args = state["tool_args"]
            merged_info = previous_args.copy()
            for key in extracted_info:
                if extracted_info[key] and extracted_info[key] != "" and extracted_info[key] is not None:
                    merged_info[key] = extracted_info[key]
            extracted_info = merged_info
        
        # Handle preserved context from failed operations (e.g., account creation retry)
        if (len(messages) > 1 and state.get("tool_result") and 
            state["tool_result"].get("preserve_context") and 
            state["tool_result"].get("partial_data")):
            
            preserved_data = state["tool_result"]["partial_data"]
            for key, value in preserved_data.items():
                if key not in extracted_info or not extracted_info[key]:
                    extracted_info[key] = value
        
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

# Node 3: Check Information Completeness
def check_information_completeness(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Check if we have all required information for the intended operation."""
    intent = state["intent"]
    tool_args = state["tool_args"]
    
    # Define required parameters for each operation
    required_params = {
        "list_exams": [],  # All parameters are optional
        "get_exam": ["exam_id"],
        "list_students": [],  # All parameters are optional
        "get_student": ["user_id"],
        "help": [],
        "status": [],
        "unsupported": [],  # No parameters needed for unsupported operations
        "schedule_exam": [],  # Will be handled dynamically
        "list_scheduled_exams": [],  # Optional parameters
        "get_exam_attempt": ["user_exam_id"],
        "get_exam_attempt_by_student": ["student_id"],
        "get_student_exam_statistics": ["student_id", "user_exam_id"],
        "create_student": [],  # Will be handled step-by-step in execute_tool
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
    
    if missing:
        # Generate helpful prompts for missing information
        prompt_messages = {
            "exam_id": "Please provide the exam ID (32-character hex string)",
            "user_id": "Please provide the user ID for the student",
            "first_name": "Please provide the student's first name",
            "last_name": "Please provide the student's last name",
            "student_id": "Please provide the student's email address (this will be used as the Student ID)",
            "password": "Please provide a password for the student account",
            "user_exam_id": "Please provide the user exam ID",
            "employee_number": "Please provide the employee number (optional)"
        }
        
        missing_prompts = [prompt_messages.get(param, f"Please provide {param}") for param in missing]
        
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

# Node 4: Tool Execution
def execute_tool(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Execute the appropriate tool based on the classified intent."""
    intent = state["intent"]
    tool_args = state["tool_args"]
    instructor_id = state["instructor_id"]
    
    if not instructor_id:
        return {
            **state,
            "tool_result": {"error": "Instructor ID not available"}
        }
    
    # Handle each intent
    if intent == "help":
        return {
            **state,
            "tool_name": "help",
            "tool_result": {
                "status": True,
                "capabilities": [
                    "List all available exams",
                    "Get detailed exam information",
                    "List all students", 
                    "Get detailed student information",
                    "Create new student accounts",
                    "Schedule exams for students",
                    "List scheduled exams",
                    "Show system status"
                ]
            }
        }
    
    elif intent == "status":
        # Get system status and available exams
        try:
            exams_result = list_exams_tool.invoke({"instructor_id": instructor_id})
            if exams_result.get("status"):
                exams = exams_result.get("exams", [])
                active_exams = [exam for exam in exams if exam.get("EXAMSTATE") == "Active"]
                return {
                    **state,
                    "tool_name": "status",
                    "tool_result": {
                        "status": True,
                        "system_status": "Connected to ExamBuilder API",
                        "total_exams": len(exams),
                        "active_exams": len(active_exams),
                        "instructor_id": instructor_id
                    }
                }
            else:
                return {
                    **state,
                    "tool_name": "status", 
                    "tool_result": {
                        "status": False,
                        "message": "Unable to retrieve system status"
                    }
                }
        except Exception as e:
            return {
                **state,
                "tool_name": "status",
                "tool_result": {
                    "status": False,
                    "message": f"Error getting system status: {str(e)}"
                }
            }
    
    elif intent == "list_exams":
        try:
            exam_name = tool_args.get("exam_name")
            result = list_exams_tool.invoke({
                "instructor_id": instructor_id,
                "exam_name": exam_name
            })
            return {
                **state,
                "tool_name": "list_exams",
                "tool_result": result
            }
        except Exception as e:
            return {
                **state,
                "tool_name": "list_exams",
                "tool_result": {
                    "status": False,
                    "message": f"Error listing exams: {str(e)}"
                }
            }
    
    elif intent == "list_students":
        try:
            result = list_students_tool.invoke({"instructor_id": instructor_id})
            return {
                **state,
                "tool_name": "list_students",
                "tool_result": result
            }
        except Exception as e:
            return {
                **state,
                "tool_name": "list_students",
                "tool_result": {
                    "status": False,
                    "message": f"Error listing students: {str(e)}"
                }
            }
    
    elif intent == "schedule_exam":
        exam_name = tool_args.get("exam_name")
        student_id = tool_args.get("student_id")
        
        if not exam_name:
            # User wants to schedule but didn't specify exam - show available exams
            try:
                exams_result = list_exams_tool.invoke({"instructor_id": instructor_id})
                if exams_result.get("status"):
                    exams = exams_result.get("exams", [])
                    active_exams = [exam for exam in exams if exam.get("EXAMSTATE") == "Active"]
                    return {
                        **state,
                        "tool_name": "schedule_exam",
                        "tool_result": {
                            "status": True,
                            "message": "Please select an exam to schedule:",
                            "available_exams": active_exams,
                            "next_step": "Please specify which exam you'd like to schedule"
                        }
                    }
                else:
                    return {
                        **state,
                        "tool_name": "schedule_exam",
                        "tool_result": {
                            "status": False,
                            "message": "Unable to retrieve available exams"
                        }
                    }
            except Exception as e:
                return {
                    **state,
                    "tool_name": "schedule_exam", 
                    "tool_result": {
                        "status": False,
                        "message": f"Error retrieving exams: {str(e)}"
                    }
                }
        
        if not student_id:
            return {
                **state,
                "tool_name": "schedule_exam",
                "tool_result": {
                    "status": False,
                    "message": "Please provide a student email or ID to schedule the exam",
                    "next_step": "Please provide the student's email address"
                }
            }
        
        # Both exam and student provided - attempt scheduling
        try:
            # First, find the exam_id from exam_name
            exams_result = list_exams_tool.invoke({"instructor_id": instructor_id})
            exam_id = None
            
            if exams_result.get("status") and exams_result.get("exams"):
                for exam in exams_result["exams"]:
                    if exam.get("EXAMNAME", "").lower() == exam_name.lower():
                        exam_id = exam.get("EXAMID")
                        break
            
            if not exam_id:
                return {
                    **state,
                    "tool_name": "schedule_exam",
                    "tool_result": {
                        "status": False,
                        "message": f"Could not find exam '{exam_name}'. Please check the exam name and try again.",
                        "suggestion": "Use 'Show me available exams' to see exact exam names"
                    }
                }
            
            # Check if student_id is actually a user_id (32-character hex string)
            import re
            if re.match(r'^[a-f0-9]{32}$', student_id.lower()):
                # This looks like a user_id, use it directly
                user_id = student_id
                student_name = "Student"
            else:
                # This is likely an email/student_id, need to look up the user_id
                search_result = search_student_by_student_id_tool.invoke({
                    "instructor_id": instructor_id,
                    "student_id": student_id
                })
                
                if search_result.get("found"):
                    user_id = search_result.get("user_id")
                    student_info = search_result.get("student", {})
                    student_name = f"{student_info.get('FIRSTNAME', '')} {student_info.get('LASTNAME', '')}".strip()
                else:
                    return {
                        **state,
                        "tool_name": "schedule_exam",
                        "tool_result": {
                            "status": False,
                            "message": f"No student found with ID/email: {student_id}",
                            "suggestion": "Please check the student ID or create a new student account"
                        }
                    }
            
            # Now attempt to schedule the exam
            schedule_result = schedule_exam_tool.invoke({
                "instructor_id": instructor_id,
                "exam_id": exam_id,
                "user_id": user_id
            })
            
            if schedule_result.get("status"):
                return {
                    **state,
                    "tool_name": "schedule_exam",
                    "tool_result": {
                        "status": True,
                        "message": f"✅ Exam '{exam_name}' scheduled successfully for {student_name}!",
                        "student_name": student_name,
                        "exam_name": exam_name,
                        "user_id": user_id,
                        "exam_id": exam_id
                    }
                }
            else:
                # Handle scheduling errors
                error_result = {
                    **state,
                    "tool_name": "schedule_exam",
                    "tool_result": {
                        "status": False,
                        "message": f"❌ Failed to schedule exam: {schedule_result.get('message', 'Unknown error')}",
                        "student_name": student_name,
                        "exam_name": exam_name,
                        "suggestion": "Please try again or contact support"
                    }
                }
                
                # Check if it's an "already scheduled" error
                if schedule_result.get("returnCode") == "STUDENT_ALREADY_SCHEDULED" or schedule_result.get("already_scheduled"):
                    error_result["tool_result"]["already_scheduled"] = True
                    error_result["tool_result"]["message"] = f"ℹ️ Student {student_name} is already scheduled for '{exam_name}'"
                    error_result["tool_result"]["suggestion"] = "You can check scheduled exams or choose a different exam"
                
                return error_result
                
        except Exception as e:
            return {
                **state,
                "tool_name": "schedule_exam",
                "tool_result": {
                    "status": False,
                    "message": f"❌ Error during scheduling: {str(e)}",
                    "suggestion": "Please try again or contact support"
                }
            }
    
    elif intent == "list_scheduled_exams":
        try:
            student_id = tool_args.get("student_id")
            user_id = None
            
            # If student_id is provided, look up the user_id
            if student_id:
                import re
                if re.match(r'^[a-f0-9]{32}$', student_id.lower()):
                    # This looks like a user_id, use it directly
                    user_id = student_id
                else:
                    # This is likely an email/student_id, need to look up the user_id
                    search_result = search_student_by_student_id_tool.invoke({
                        "instructor_id": instructor_id,
                        "student_id": student_id
                    })
                    
                    if search_result.get("found"):
                        user_id = search_result.get("user_id")
                    else:
                        return {
                            **state,
                            "tool_name": "list_scheduled_exams",
                            "tool_result": {
                                "status": False,
                                "message": f"No student found with ID/email: {student_id}",
                                "suggestion": "Please check the student ID"
                            }
                        }
            
            # Call list_scheduled_exams with the user_id if available
            if user_id:
                # API limitation: need to check each exam individually
                try:
                    # Get all available exams
                    exams_result = list_exams_tool.invoke({"instructor_id": instructor_id})
                    user_scheduled_exams = []
                    
                    if exams_result.get("status") and exams_result.get("exams"):
                        for exam in exams_result["exams"]:
                            exam_id = exam.get("EXAMID")
                            if exam_id:
                                # Check if user is scheduled for this specific exam
                                check_result = list_scheduled_exams_tool.invoke({
                                    "instructor_id": instructor_id,
                                    "user_id": user_id,
                                    "exam_id": exam_id
                                })
                                
                                if (check_result.get("status") and 
                                    check_result.get("students") and 
                                    len(check_result["students"]) > 0 and
                                    check_result["students"][0].get("EXAMID")):
                                    user_scheduled_exams.extend(check_result["students"])
                    
                    # Create a result in the expected format
                    result = {
                        "status": True,
                        "returnCode": "S",
                        "message": "Data returned successfully.",
                        "students": user_scheduled_exams if user_scheduled_exams else [{'NULL': None}]
                    }
                    
                except Exception as e:
                    result = {
                        "status": False,
                        "message": f"Error checking scheduled exams: {str(e)}"
                    }
            else:
                # Without user_id, we can't check scheduled exams due to API limitations
                result = {
                    "status": False,
                    "message": "Cannot list all scheduled exams. Please specify a student ID."
                }
            
            return {
                **state,
                "tool_name": "list_scheduled_exams",
                "tool_result": result
            }
        except Exception as e:
            return {
                **state,
                "tool_name": "list_scheduled_exams",
                "tool_result": {
                    "status": False,
                    "message": f"Error listing scheduled exams: {str(e)}"
                }
            }
    
    elif intent == "get_exam_attempt":
        try:
            user_exam_id = tool_args.get("user_exam_id")
            if not user_exam_id:
                return {
                    **state,
                    "tool_name": "get_exam_attempt",
                    "tool_result": {
                        "status": False,
                        "message": "Please provide a User Exam ID to get exam attempt details.",
                        "next_step": "Please provide the User Exam ID"
                    }
                }
            
            result = get_exam_attempt_tool.invoke({
                "instructor_id": instructor_id,
                "user_exam_id": user_exam_id
            })
            
            if result.get("status"):
                return {
                    **state,
                    "tool_name": "get_exam_attempt",
                    "tool_result": result
                }
            else:
                return {
                    **state,
                    "tool_name": "get_exam_attempt",
                    "tool_result": {
                        "status": False,
                        "message": f"❌ Failed to get exam attempt: {result.get('message', 'Unknown error')}",
                        "user_exam_id": user_exam_id
                    }
                }
        except Exception as e:
            return {
                **state,
                "tool_name": "get_exam_attempt",
                "tool_result": {
                    "status": False,
                    "message": f"❌ Error getting exam attempt: {str(e)}",
                    "user_exam_id": tool_args.get("user_exam_id")
                }
            }
    
    elif intent == "get_student_exam_statistics":
        try:
            student_id = tool_args.get("student_id")
            user_exam_id = tool_args.get("user_exam_id")
            
            if not student_id or not user_exam_id:
                return {
                    **state,
                    "tool_name": "get_student_exam_statistics",
                    "tool_result": {
                        "status": False,
                        "message": "Please provide both Student ID and User Exam ID to get statistics.",
                        "next_step": "Please provide the Student ID and User Exam ID"
                    }
                }
            
            result = get_student_exam_statistics_tool.invoke({
                "instructor_id": instructor_id,
                "student_id": student_id,
                "user_exam_id": user_exam_id
            })
            
            if result.get("status"):
                return {
                    **state,
                    "tool_name": "get_student_exam_statistics",
                    "tool_result": result
                }
            else:
                return {
                    **state,
                    "tool_name": "get_student_exam_statistics",
                    "tool_result": {
                        "status": False,
                        "message": f"❌ Failed to get exam statistics: {result.get('message', 'Unknown error')}",
                        "student_id": student_id,
                        "user_exam_id": user_exam_id
                    }
                }
        except Exception as e:
            return {
                **state,
                "tool_name": "get_student_exam_statistics",
                "tool_result": {
                    "status": False,
                    "message": f"❌ Error getting exam statistics: {str(e)}",
                    "student_id": tool_args.get("student_id"),
                    "user_exam_id": tool_args.get("user_exam_id")
                }
            }
    
    elif intent == "create_student":
        try:
            first_name = tool_args.get("first_name")
            last_name = tool_args.get("last_name")
            student_id = tool_args.get("student_id")
            password = tool_args.get("password")
            
            # Ask for information one by one
            if not first_name:
                return {
                    **state,
                    "tool_name": "create_student",
                    "tool_result": {
                        "status": True,
                        "message": "Let's create your student account! First, what's your first name?",
                        "next_step": "waiting_for_first_name",
                        "partial_data": {
                            "step": "first_name"
                        }
                    }
                }
            elif not last_name:
                return {
                    **state,
                    "tool_name": "create_student",
                    "tool_result": {
                        "status": True,
                        "message": f"Great! Hi {first_name}! What's your last name?",
                        "next_step": "waiting_for_last_name",
                        "partial_data": {
                            "step": "last_name",
                            "first_name": first_name
                        }
                    }
                }
            elif not student_id:
                return {
                    **state,
                    "tool_name": "create_student",
                    "tool_result": {
                        "status": True,
                        "message": f"Perfect, {first_name} {last_name}! Now I need your email address (this will be your Student ID for logging in):",
                        "next_step": "waiting_for_email",
                        "partial_data": {
                            "step": "student_id",
                            "first_name": first_name,
                            "last_name": last_name
                        }
                    }
                }
            elif not password:
                return {
                    **state,
                    "tool_name": "create_student",
                    "tool_result": {
                        "status": True,
                        "message": f"Almost done! Please create a password for your account (email: {student_id}):",
                        "next_step": "waiting_for_password",
                        "partial_data": {
                            "step": "password",
                            "first_name": first_name,
                            "last_name": last_name,
                            "student_id": student_id
                        }
                    }
                }
            
            # All information collected - create the account
            result = create_student_tool.invoke({
                "instructor_id": instructor_id,
                "first_name": first_name,
                "last_name": last_name,
                "student_id": student_id,
                "password": password
            })
            
            if result.get("status"):
                return {
                    **state,
                    "tool_name": "create_student",
                    "tool_result": {
                        "status": True,
                        "message": f"🎉 Congratulations {first_name}! Your student account has been created successfully!\n\n📧 **Email/Student ID**: {student_id}\n🔑 **Password**: {password}\n\n✅ You can now register for exams. Would you like me to help you register for the Serengeti Practice Exam?",
                        "student_name": f"{first_name} {last_name}",
                        "student_id": student_id,
                        "account_created": True
                    }
                }
            else:
                # Get the actual error message from the API response
                error_message = result.get('message') or result.get('error', 'Unknown error')
                return {
                    **state,
                    "tool_name": "create_student",
                    "tool_result": {
                        "status": False,
                        "message": f"❌ Failed to create student account: {error_message}\n\n💡 This might happen if an account with email '{student_id}' already exists. Please try a different email address.",
                        "suggestion": "Please try again with a different email address",
                        "preserve_context": True,
                        "partial_data": {
                            "first_name": first_name,
                            "last_name": last_name,
                            "password": password,
                            "failed_student_id": student_id
                        }
                    }
                }
        except Exception as e:
            return {
                **state,
                "tool_name": "create_student",
                "tool_result": {
                    "status": False,
                    "message": f"❌ Error creating student account: {str(e)}",
                    "suggestion": "Please try again or contact support"
                }
            }
    
    elif intent == "get_exam_attempt_by_student":
        try:
            # Check if this is an exam selection from a previous list
            if (state.get("tool_result") and 
                state["tool_result"].get("awaiting_exam_selection") and 
                state["tool_result"].get("exams")):
                
                # User is selecting from a previous exam list
                stored_exams = state["tool_result"]["exams"]
                student_name = state["tool_result"]["student_name"]
                student_id_from_state = state["tool_result"]["student_id"]
                
                # Get user's exam choice
                exam_choice = tool_args.get("exam_name") or tool_args.get("exam_number")
                if not exam_choice:
                    # Try to extract from the raw user input
                    messages = state.get("messages", [])
                    if messages:
                        last_user_message = messages[-1].content.strip()
                        exam_choice = last_user_message
                
                selected_exam = None
                
                # Try to match by number first (1, 2, 3, etc.)
                try:
                    if exam_choice.isdigit():
                        exam_index = int(exam_choice) - 1
                        if 0 <= exam_index < len(stored_exams):
                            selected_exam = stored_exams[exam_index]
                except:
                    pass
                
                # If not found by number, try to match by exam name
                if not selected_exam:
                    for exam in stored_exams:
                        exam_name = exam.get('EXAMNAME', '').lower()
                        if exam_choice.lower() in exam_name or exam_name in exam_choice.lower():
                            selected_exam = exam
                            break
                
                if selected_exam:
                    # Found the exam, get results
                    user_exam_id = selected_exam.get('USEREXAMID')
                    if user_exam_id:
                        attempt_result = get_exam_attempt_tool.invoke({
                            "instructor_id": instructor_id,
                            "user_exam_id": user_exam_id
                        })
                        
                        if attempt_result.get("status") and attempt_result.get("exam_attempt"):
                            return {
                                **state,
                                "tool_name": "get_exam_attempt_by_student",
                                "tool_result": {
                                    "status": True,
                                    "exam_attempt": attempt_result["exam_attempt"],
                                    "message": f"Exam results for {student_name}"
                                }
                            }
                        else:
                            return {
                                **state,
                                "tool_name": "get_exam_attempt_by_student",
                                "tool_result": {
                                    "status": False,
                                    "message": f"❌ Could not retrieve exam results: {attempt_result.get('message', 'Unknown error')}",
                                    "suggestion": "The exam may not have been started yet"
                                }
                            }
                    else:
                        return {
                            **state,
                            "tool_name": "get_exam_attempt_by_student",
                            "tool_result": {
                                "status": False,
                                "message": f"❌ Could not find exam attempt details for {student_name}",
                                "suggestion": "The exam may not have been started yet"
                            }
                        }
                else:
                    # Exam choice not found, show the list again
                    exam_list = ""
                    for i, exam in enumerate(stored_exams, 1):
                        exam_name = exam.get('EXAMNAME', 'Unknown')
                        started = exam.get('DATETIMESTARTED', 'Not started')
                        status_indicator = "🟡 In Progress" if started != 'Not Yet' and started != 'Not started' else "⚫ Not Started"
                        exam_list += f"{i}. **{exam_name}** ({status_indicator})\n"
                    
                    return {
                        **state,
                        "tool_name": "get_exam_attempt_by_student",
                        "tool_result": {
                            "status": False,
                            "message": f"❌ Sorry, I couldn't find an exam matching '{exam_choice}'.\n\nPlease choose from:\n{exam_list}\nYou can say the exam name or number (e.g., '1', '2', or 'Serengeti Certification').",
                            "multiple_exams": True,
                            "exams": stored_exams,
                            "student_name": student_name,
                            "student_id": student_id_from_state,
                            "awaiting_exam_selection": True
                        }
                    }
            
            # Original logic for new student ID requests
            student_id = tool_args.get("student_id")
            exam_name = tool_args.get("exam_name")  # Check if exam name was provided in initial request
            
            if not student_id:
                return {
                    **state,
                    "tool_name": "get_exam_attempt_by_student",
                    "tool_result": {
                        "status": False,
                        "message": "Please provide a student ID (email) to get exam results.",
                        "next_step": "Please provide the student's email address"
                    }
                }
            
            # First, find the user_id from student_id
            search_result = search_student_by_student_id_tool.invoke({
                "instructor_id": instructor_id,
                "student_id": student_id
            })
            
            if not search_result.get("found"):
                return {
                    **state,
                    "tool_name": "get_exam_attempt_by_student",
                    "tool_result": {
                        "status": False,
                        "message": f"❌ No student found with ID/email: {student_id}",
                        "suggestion": "Please check the student ID and try again"
                    }
                }
            
            user_id = search_result.get("user_id")
            student_info = search_result.get("student", {})
            student_name = f"{student_info.get('FIRSTNAME', '')} {student_info.get('LASTNAME', '')}".strip()
            
            # Get all scheduled exams for this student
            try:
                # Get all available exams
                exams_result = list_exams_tool.invoke({"instructor_id": instructor_id})
                user_scheduled_exams = []
                
                if exams_result.get("status") and exams_result.get("exams"):
                    for exam in exams_result["exams"]:
                        exam_id = exam.get("EXAMID")
                        if exam_id:
                            # Check if user is scheduled for this specific exam
                            check_result = list_scheduled_exams_tool.invoke({
                                "instructor_id": instructor_id,
                                "user_id": user_id,
                                "exam_id": exam_id
                            })
                            
                            if (check_result.get("status") and 
                                check_result.get("students") and 
                                len(check_result["students"]) > 0 and
                                check_result["students"][0].get("EXAMID")):
                                user_scheduled_exams.extend(check_result["students"])
                
                if not user_scheduled_exams:
                    return {
                        **state,
                        "tool_name": "get_exam_attempt_by_student",
                        "tool_result": {
                            "status": False,
                            "message": f"📅 No scheduled exams found for student {student_name} ({student_id})",
                            "suggestion": "This student may not have any scheduled exams yet"
                        }
                    }
                
                # If exam_name was provided in the initial request, try to find it directly
                if exam_name:
                    matching_exams = []
                    for exam in user_scheduled_exams:
                        exam_name_from_list = exam.get('EXAMNAME', '').lower()
                        if exam_name.lower() in exam_name_from_list or exam_name_from_list in exam_name.lower():
                            matching_exams.append(exam)
                    
                    if len(matching_exams) == 1:
                        # Found exactly one match - get results directly
                        selected_exam = matching_exams[0]
                        user_exam_id = selected_exam.get('USEREXAMID')
                        
                        if user_exam_id:
                            attempt_result = get_exam_attempt_tool.invoke({
                                "instructor_id": instructor_id,
                                "user_exam_id": user_exam_id
                            })
                            
                            if attempt_result.get("status") and attempt_result.get("exam_attempt"):
                                return {
                                    **state,
                                    "tool_name": "get_exam_attempt_by_student",
                                    "tool_result": {
                                        "status": True,
                                        "exam_attempt": attempt_result["exam_attempt"],
                                        "message": f"Exam results for {student_name}"
                                    }
                                }
                            else:
                                return {
                                    **state,
                                    "tool_name": "get_exam_attempt_by_student",
                                    "tool_result": {
                                        "status": False,
                                        "message": f"❌ Could not retrieve exam results: {attempt_result.get('message', 'Unknown error')}",
                                        "suggestion": "The exam may not have been started yet"
                                    }
                                }
                        else:
                            return {
                                **state,
                                "tool_name": "get_exam_attempt_by_student",
                                "tool_result": {
                                    "status": False,
                                    "message": f"❌ Could not find exam attempt details for {student_name}",
                                    "suggestion": "The exam may not have been started yet"
                                }
                            }
                    elif len(matching_exams) > 1:
                        # Multiple matches found - show the matching ones for selection
                        exam_list = ""
                        for i, exam in enumerate(matching_exams, 1):
                            exam_name_display = exam.get('EXAMNAME', 'Unknown')
                            started = exam.get('DATETIMESTARTED', 'Not started')
                            completed = exam.get('DATETIMECOMPLETED', 'Not completed')
                            
                            # Proper status logic: only "Not Started" or "Completed"
                            if completed and completed != 'Not completed':
                                status_indicator = "✅ Completed"
                            else:
                                status_indicator = "⚫ Not Started"
                            
                            exam_list += f"• **{exam_name_display}** ({status_indicator})\n"
                        
                        return {
                            **state,
                            "tool_name": "get_exam_attempt_by_student",
                            "tool_result": {
                                "status": True,
                                "message": f"📊 Found {len(matching_exams)} exams matching '{exam_name}' for {student_name} ({student_id}):\n\n{exam_list}\nWhich one would you like to see? You can say the number (e.g., '1', '2').",
                                "multiple_exams": True,
                                "exams": matching_exams,
                                "student_name": student_name,
                                "student_id": student_id,
                                "awaiting_exam_selection": True
                            }
                        }
                    else:
                        # No matches found - show all available exams
                        exam_list = ""
                        for i, exam in enumerate(user_scheduled_exams, 1):
                            exam_name_display = exam.get('EXAMNAME', 'Unknown')
                            started = exam.get('DATETIMESTARTED', 'Not started')
                            completed = exam.get('DATETIMECOMPLETED', 'Not completed')
                            
                            # Proper status logic: only "Not Started" or "Completed"
                            if completed and completed != 'Not completed':
                                status_indicator = "✅ Completed"
                            else:
                                status_indicator = "⚫ Not Started"
                            
                            exam_list += f"• **{exam_name_display}** ({status_indicator})\n"
                        
                        return {
                            **state,
                            "tool_name": "get_exam_attempt_by_student",
                            "tool_result": {
                                "status": False,
                                "message": f"❌ No exam found matching '{exam_name}' for {student_name} ({student_id}).\n\nAvailable exams:\n{exam_list}\nPlease choose from the list above.",
                                "multiple_exams": True,
                                "exams": user_scheduled_exams,
                                "student_name": student_name,
                                "student_id": student_id,
                                "awaiting_exam_selection": True
                            }
                        }
                
                # If no exam_name provided or no direct match, show all exams (original logic)
                elif len(user_scheduled_exams) == 1:
                    exam_attempt = user_scheduled_exams[0]
                    user_exam_id = exam_attempt.get('USEREXAMID')
                    
                    if user_exam_id:
                        attempt_result = get_exam_attempt_tool.invoke({
                            "instructor_id": instructor_id,
                            "user_exam_id": user_exam_id
                        })
                        
                        # Ensure the result has the proper format for display
                        if attempt_result.get("status") and attempt_result.get("exam_attempt"):
                            return {
                                **state,
                                "tool_name": "get_exam_attempt_by_student",
                                "tool_result": {
                                    "status": True,
                                    "exam_attempt": attempt_result["exam_attempt"],
                                    "message": f"Exam results for {student_name}"
                                }
                            }
                        else:
                            return {
                                **state,
                                "tool_name": "get_exam_attempt_by_student",
                                "tool_result": {
                                    "status": False,
                                    "message": f"❌ Could not retrieve exam results: {attempt_result.get('message', 'Unknown error')}",
                                    "suggestion": "The exam may not have been started yet"
                                }
                            }
                    else:
                        return {
                            **state,
                            "tool_name": "get_exam_attempt_by_student",
                            "tool_result": {
                                "status": False,
                                "message": f"❌ Could not find exam attempt details for {student_name}",
                                "suggestion": "The exam may not have been started yet"
                            }
                        }
                else:
                    # Multiple exams - ask user to choose with human-readable options
                    exam_list = ""
                    for i, exam in enumerate(user_scheduled_exams, 1):
                        exam_name = exam.get('EXAMNAME', 'Unknown')
                        started = exam.get('DATETIMESTARTED', 'Not started')
                        completed = exam.get('DATETIMECOMPLETED', 'Not completed')
                        
                        # Proper status logic: only "Not Started" or "Completed"
                        if completed and completed != 'Not completed':
                            status_indicator = "✅ Completed"
                        else:
                            status_indicator = "⚫ Not Started"
                        
                        exam_list += f"• **{exam_name}** ({status_indicator})\n"
                    
                    return {
                        **state,
                        "tool_name": "get_exam_attempt_by_student",
                        "tool_result": {
                            "status": True,
                            "message": f"📊 Found {len(user_scheduled_exams)} scheduled exams for {student_name} ({student_id}):\n\n{exam_list}\nWhich exam results would you like to see? You can say the exam name or number (e.g., '1', '2', or 'Serengeti Certification').",
                            "multiple_exams": True,
                            "exams": user_scheduled_exams,
                            "student_name": student_name,
                            "student_id": student_id,
                            "awaiting_exam_selection": True
                        }
                    }
                    
            except Exception as e:
                return {
                    **state,
                    "tool_name": "get_exam_attempt_by_student",
                    "tool_result": {
                        "status": False,
                        "message": f"❌ Error getting scheduled exams: {str(e)}",
                        "suggestion": "Please try again or contact support"
                    }
                }
                
        except Exception as e:
            return {
                **state,
                "tool_name": "get_exam_attempt_by_student",
                "tool_result": {
                    "status": False,
                    "message": f"❌ Error processing exam results request: {str(e)}",
                    "suggestion": "Please try again or contact support"
                }
            }
    
    else:
        return {
            **state,
            "tool_name": "unsupported",
            "tool_result": {
                "status": False,
                "message": "This operation is not currently supported.",
                "suggestion": "Try asking: 'What can you help me with?' to see available capabilities"
            }
        }

# Node 5: Response Formatting
def format_response(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
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
            error_msg = tool_result['error']
            response = f"❌ Error: {error_msg}\n\n"
            
            # Add specific guidance based on the error type and tool
            if "400 Client Error: Bad Request" in error_msg:
                if tool_name == "get_student":
                    response += f"💡 **Note**: The User ID you provided is not valid. You can find valid User IDs by listing all students first with: 'Show me all students'"
                elif tool_name == "get_exam_attempt":
                    response += f"💡 **Note**: The User Exam ID you provided is not valid. You can find valid User Exam IDs by listing scheduled exams first with: 'Show me my scheduled exams'"
                elif tool_name == "get_exam":
                    response += f"💡 **Note**: The Exam ID you provided is not valid. You can find valid Exam IDs by listing all exams first with: 'Show me all available exams'"
                else:
                    response += f"💡 **Note**: The ID you provided is not valid. Please check your input and try again."
            elif "500 Server Error: Internal Server Error" in error_msg:
                if tool_name == "list_scheduled_exams":
                    response += f"💡 **Note**: This might be due to server issues or no scheduled exams available. You can try scheduling an exam first."
                else:
                    response += f"💡 **Note**: This appears to be a server-side issue. Please try again later or contact support."
            else:
                response += f"💡 **Note**: Please check your input and try again. If the problem persists, contact support."
        else:
            # Format based on tool type
            if tool_name == "help":
                capabilities = tool_result.get("capabilities", [])
                response = "🎓 ExamBuilder Agent Capabilities:\n"
                for i, capability in enumerate(capabilities, 1):
                    response += f"{i}. {capability}\n"
                response += "\nJust ask me in natural language what you'd like to do!"
            
            elif tool_name == "status":
                response = f"📊 System Status:\n"
                response += f"• Connected to ExamBuilder API ✅\n"
                response += f"• Instructor ID: {tool_result.get('instructor_id', 'Unknown')}\n" 
                response += f"• Available Exams: {tool_result.get('active_exams', 0)}\n"
                response += f"• System Status: {tool_result.get('system_status', 'Unknown')}"
            
            elif tool_name == "list_exams":
                if tool_result.get("status") and tool_result.get("exams"):
                    exams = tool_result["exams"]
                    
                    # Separate active and inactive exams for better organization
                    active_exams = [exam for exam in exams if exam.get('EXAMSTATE') == 'Active']
                    inactive_exams = [exam for exam in exams if exam.get('EXAMSTATE') != 'Active']
                    
                    response = f"📝 **Available Exams** ({len(exams)} total)\n\n"
                    
                    # Show active exams first
                    if active_exams:
                        response += "🟢 **ACTIVE EXAMS** (Ready for scheduling)\n\n"
                        for i, exam in enumerate(active_exams, 1):
                            exam_name = exam.get('EXAMNAME', 'Untitled')
                            created_date = exam.get('DATETIMECREATED', 'Unknown')
                            response += f"• **{exam_name}**\n"
                            response += f"  - Status: Ready ✅\n"
                            response += f"  - Created: {created_date}\n\n"
                    
                    # Show inactive exams if any
                    if inactive_exams:
                        response += "🔴 **INACTIVE EXAMS** (Not available for scheduling)\n\n"
                        for i, exam in enumerate(inactive_exams, 1):
                            exam_name = exam.get('EXAMNAME', 'Untitled')
                            created_date = exam.get('DATETIMECREATED', 'Unknown')
                            response += f"• **{exam_name}**\n"
                            response += f"  - Status: Inactive ⏸️\n"
                            response += f"  - Created: {created_date}\n\n"
                    
                    # Add helpful footer
                    response += "💡 **Available Actions:**\n"
                    response += "• Schedule exam: *'Schedule [exam name] for [student email]'*\n"
                    response += "• View exam details: *'Show details for [exam name]'*\n"
                    response += "• Check scheduled exams: *'Show scheduled exams for [student email]'*"
                    
                else:
                    response = "📝 No exams found."
            
            elif tool_name == "get_exam":
                if tool_result.get("status") and tool_result.get("exam"):
                    exam = tool_result["exam"]
                    response = f"📋 Exam Details:\n\n"
                    response += f"**Name**: {exam.get('EXAMNAME', 'Unknown')}\n"
                    response += f"**ID**: `{exam.get('EXAMID', 'Unknown')}`\n"
                    response += f"**State**: {exam.get('EXAMSTATE', 'Unknown')}\n"
                    response += f"**Created**: {exam.get('DATETIMECREATED', 'Unknown')}\n"
                    response += f"**Last Modified**: {exam.get('DATETIMEEDITED', 'Unknown')}\n"
                    if exam.get('DATETIMEACTIVATED'):
                        response += f"**Activated**: {exam.get('DATETIMEACTIVATED')}\n"
                else:
                    response = f"❌ Exam not found: {tool_result.get('message', 'Unknown error')}"
            
            elif tool_name == "list_students":
                if tool_result.get("status"):
                    students = tool_result.get("students", []) or tool_result.get("student_list", [])
                    if students:
                        response = f"👥 Found {len(students)} student(s):\n\n"
                        for i, student in enumerate(students[:10], 1):  # Limit to 10
                            response += f"• **{student.get('FIRSTNAME', '')} {student.get('LASTNAME', '')}**\n"
                            response += f"  - User ID: `{student.get('USERID', 'Unknown')}`\n"
                            response += f"  - Student ID: `{student.get('STUDENTID', 'Unknown')}`\n"
                            response += f"  - Created: {student.get('DATETIMECREATED', 'Unknown')}\n\n"
                        if len(students) > 10:
                            response += f"... and {len(students) - 10} more students\n"
                    else:
                        response = "👥 No students found."
                else:
                    response = f"❌ Failed to list students: {tool_result.get('message', 'Unknown error')}"
            
            elif tool_name == "get_student":
                if tool_result.get("status") and tool_result.get("student"):
                    student = tool_result["student"]
                    response = f"👤 Student Details:\n\n"
                    response += f"**Name**: {student.get('FIRSTNAME', '')} {student.get('LASTNAME', '')}\n"
                    response += f"**User ID**: `{student.get('USERID', 'Unknown')}`\n"
                    response += f"**Student ID**: `{student.get('STUDENTID', 'Unknown')}`\n"
                    response += f"**Created**: {student.get('DATETIMECREATED', 'Unknown')}\n"
                    response += f"**Last Login**: {student.get('DATETIMELOGGED', 'Never')}\n"
                    
                    # Add group information if available
                    if tool_result.get("student_groups"):
                        groups = tool_result["student_groups"]
                        response += f"**Groups**: {', '.join([g.get('GROUPNAME', '') for g in groups])}\n"
                else:
                    response = f"❌ Student not found: {tool_result.get('message', 'Unknown error')}\n\n"
                    response += f"💡 **Note**: Make sure you're using a valid User ID. You can find User IDs by listing all students first."
            
            elif tool_name == "list_group_categories":
                if tool_result.get("status"):
                    categories = tool_result.get("categories", []) or tool_result.get("groupCategories", [])
                    if categories:
                        response = f"📂 Found {len(categories)} group categories:\n\n"
                        for i, cat in enumerate(categories, 1):
                            response += f"• **{cat.get('CATEGORYNAME', 'Unknown')}**\n"
                            response += f"  - ID: `{cat.get('CATEGORYID', 'Unknown')}`\n\n"
                    else:
                        response = "📂 No group categories found."
                else:
                    response = f"❌ Failed to list group categories: {tool_result.get('message', 'Unknown error')}"
            
            elif tool_name == "unsupported":
                response = f"⚠️ {tool_result.get('message', 'This operation is not available')}\n\n"
                if tool_result.get("suggestion"):
                    response += f"💡 **Suggestion**: {tool_result['suggestion']}"
            
            elif tool_name == "schedule_exam":
                if tool_result.get("status"):
                    if tool_result.get("message") and "Please select an exam to schedule:" in tool_result["message"]:
                        response = f"✅ {tool_result['message']}\n\n"
                        response += f"**Available Active Exams**:\n"
                        for exam in tool_result.get("available_exams", []):
                            response += f"• **{exam.get('EXAMNAME', 'Unknown')}**\n"
                        response += f"\n📋 **Next Step**: {tool_result.get('next_step', 'Please specify which exam you want to schedule')}"
                    elif tool_result.get("next_step"):
                        response = f"✅ {tool_result['message']}\n\n"
                        response += f"📋 **Next Step**: {tool_result['next_step']}"
                    elif tool_result.get("already_scheduled"):
                        response = f"ℹ️ {tool_result['message']}\n\n"
                        response += f"💡 **Suggestion**: {tool_result['suggestion']}"
                    else:
                        response = f"✅ {tool_result['message']}"
                else:
                    response = f"❌ {tool_result.get('message', 'Failed to process scheduling request')}\n\n"
                    if tool_result.get("suggestion"):
                        response += f"💡 **Suggestion**: {tool_result['suggestion']}"
            
            elif tool_name == "list_scheduled_exams":
                if tool_result.get("status"):
                    scheduled_exams = tool_result.get("students", [])
                    if (scheduled_exams and len(scheduled_exams) > 0 and 
                        scheduled_exams[0].get('EXAMID') and 
                        'NULL' not in scheduled_exams[0]):
                        
                        response = f"📅 **Scheduled Exams** ({len(scheduled_exams)} found)\n\n"
                        
                        for i, exam in enumerate(scheduled_exams, 1):
                            exam_name = exam.get('EXAMNAME', 'Unknown')
                            student_name = f"{exam.get('FIRSTNAME', '')} {exam.get('LASTNAME', '')}".strip()
                            student_id = exam.get('STUDENTID', 'Unknown')
                            user_exam_id = exam.get('USEREXAMID', 'Unknown')
                            started = exam.get('DATETIMESTARTED', 'Not started')
                            completed = exam.get('DATETIMECOMPLETED', 'Not completed')
                            
                            # Proper status logic
                            if completed and completed != 'Not completed':
                                status = "✅ Completed"
                            else:
                                status = "⚫ Not Started"
                            
                            response += f"• **{exam_name}**\n"
                            response += f"  - Student: {student_name} ({student_id})\n"
                            response += f"  - Status: {status}\n"
                            response += f"  - Reference: `{user_exam_id}`\n\n"
                        
                        response += "💡 **Available Actions:**\n"
                        response += "• Check results: *'Show results for [reference ID]'*\n"
                        response += "• Unschedule exam: *'Unschedule [reference ID]'*\n"
                        response += "• View exam details: *'Show details for [exam name]'*"
                        
                    else:
                        response = "📅 **No Scheduled Exams Found**\n\n"
                        response += "🔍 This student doesn't have any exams scheduled yet.\n\n"
                        response += "💡 **Next Steps:**\n"
                        response += "• Schedule an exam: *'Schedule [exam name] for [student email]'*\n"
                        response += "• View available exams: *'Show me available exams'*"
                else:
                    response = f"❌ **Error Getting Scheduled Exams**\n\n"
                    response += f"Details: {tool_result.get('message', 'Unknown error')}\n\n"
                    response += f"💡 **Troubleshooting:**\n"
                    response += f"• Check the student email/ID is correct\n"
                    response += f"• Try: *'Show me available exams'* first"
            
            elif tool_name == "schedule_exam":
                if tool_result.get("status"):
                    response = f"✅ Exam scheduled successfully!"
                    if tool_result.get("message"):
                        response += f" {tool_result['message']}"
                else:
                    response = f"❌ Failed to schedule exam: {tool_result.get('message', 'Unknown error')}"
            
            elif tool_name == "get_exam_attempt":
                if tool_result.get("status") and tool_result.get("exam_attempt"):
                    attempt = tool_result["exam_attempt"]
                    exam_name = attempt.get('EXAMNAME', 'Unknown')
                    student_name = f"{attempt.get('FIRSTNAME', '')} {attempt.get('LASTNAME', '')}".strip()
                    
                    response = f"📊 **Exam Results** - {exam_name}\n\n"
                    response += f"👤 **Student**: {student_name}\n"
                    response += f"📧 **Student ID**: {attempt.get('STUDENTID', 'Unknown')}\n"
                    response += f"🎯 **Passing Score**: {attempt.get('PASSINGSCORE', 'Unknown')}%\n"
                    response += f"📊 **Current Score**: {attempt.get('SCORE') or 'Not completed yet'}\n"
                    response += f"⏱️ **Started**: {attempt.get('DATETIMESTARTED', 'Not started')}\n"
                    response += f"✅ **Completed**: {attempt.get('DATETIMECOMPLETED') or 'Not completed'}\n"
                    response += f"📝 **Attempt Number**: {attempt.get('EXAMATTEMPT', 'Unknown')}\n\n"
                    
                    # Add status indicator
                    if attempt.get('DATETIMECOMPLETED'):
                        if attempt.get('SCORE'):
                            score = float(attempt.get('SCORE', 0))
                            passing = float(attempt.get('PASSINGSCORE', 70))
                            if score >= passing:
                                response += "🟢 **Status**: PASSED ✅\n"
                            else:
                                response += "🔴 **Status**: FAILED ❌\n"
                        else:
                            response += "⚪ **Status**: COMPLETED (Score pending)\n"
                    else:
                        response += "⚫ **Status**: NOT STARTED\n"
                        
                else:
                    response = f"❌ Failed to get exam attempt: {tool_result.get('message', 'Unknown error')}\n\n"
                    response += f"💡 **Note**: Make sure you're using a valid User Exam ID. You can find this by listing scheduled exams first."
            
            elif tool_name == "get_student_exam_statistics":
                if tool_result.get("status") and tool_result.get("statistics"):
                    stats = tool_result["statistics"]
                    response = f"📈 Exam Statistics:\n\n"
                    response += f"**Total Questions**: {stats.get('TOTALQUESTIONS', 'Unknown')}\n"
                    response += f"**Correct Answers**: {stats.get('CORRECTANSWERS', 'Unknown')}\n"
                    response += f"**Score Percentage**: {stats.get('SCOREPERCENTAGE', 'Unknown')}%\n"
                    response += f"**Time Taken**: {stats.get('TIMETAKEN', 'Unknown')}\n"
                else:
                    response = f"❌ Failed to get exam statistics: {tool_result.get('message', 'Unknown error')}\n\n"
                    response += f"💡 **Note**: Make sure you're using valid Student ID and User Exam ID. You can find these by listing students and scheduled exams first."
            
            elif tool_name == "search_student_by_student_id":
                if tool_result.get("found"):
                    student = tool_result.get("student", {})
                    response = f"✅ Student found!\n\n"
                    response += f"**Name**: {student.get('FIRSTNAME', '')} {student.get('LASTNAME', '')}\n"
                    response += f"**Student ID**: `{student.get('STUDENTID', 'Unknown')}`\n"
                    response += f"**User ID**: `{student.get('USERID', 'Unknown')}`\n"
                    response += f"**Email**: {student.get('STUDENTID', 'Not provided')}\n\n"
                    response += f"💡 You can now use the User ID for scheduling exams and getting results!"
                else:
                    response = f"❌ No student found with that email address. You may need to create a new account."
            
            elif tool_name == "create_student":
                if tool_result.get("status"):
                    response = f"✅ {tool_result['message']}"
                else:
                    response = f"❌ {tool_result.get('message', 'Failed to create student account')}\n\n"
                    if tool_result.get("suggestion"):
                        response += f"💡 **Suggestion**: {tool_result['suggestion']}"
            
            elif tool_name == "get_exam_attempt_by_student":
                if tool_result.get("status"):
                    if tool_result.get("multiple_exams"):
                        # Multiple exams found - show selection list
                        response = f"📊 {tool_result['message']}"
                    elif tool_result.get("exam_attempt"):
                        # Single exam result found - format like regular exam attempt
                        attempt = tool_result["exam_attempt"]
                        exam_name = attempt.get('EXAMNAME', 'Unknown')
                        student_name = f"{attempt.get('FIRSTNAME', '')} {attempt.get('LASTNAME', '')}".strip()
                        
                        response = f"📊 **Exam Results** - {exam_name}\n\n"
                        response += f"👤 **Student**: {student_name}\n"
                        response += f"📧 **Student ID**: {attempt.get('STUDENTID', 'Unknown')}\n"
                        response += f"🎯 **Passing Score**: {attempt.get('PASSINGSCORE', 'Unknown')}%\n"
                        response += f"📊 **Current Score**: {attempt.get('SCORE') or 'Not completed yet'}\n"
                        response += f"⏱️ **Started**: {attempt.get('DATETIMESTARTED', 'Not started')}\n"
                        response += f"✅ **Completed**: {attempt.get('DATETIMECOMPLETED') or 'Not completed'}\n"
                        response += f"📝 **Attempt Number**: {attempt.get('EXAMATTEMPT', 'Unknown')}\n\n"
                        
                        # Add status indicator
                        if attempt.get('DATETIMECOMPLETED'):
                            if attempt.get('SCORE'):
                                score = float(attempt.get('SCORE', 0))
                                passing = float(attempt.get('PASSINGSCORE', 70))
                                if score >= passing:
                                    response += "🟢 **Status**: PASSED ✅\n"
                                else:
                                    response += "🔴 **Status**: FAILED ❌\n"
                            else:
                                response += "⚪ **Status**: COMPLETED (Score pending)\n"
                        else:
                            response += "⚫ **Status**: NOT STARTED\n"
                    else:
                        response = f"✅ {tool_result.get('message', 'Exam results retrieved successfully')}"
                else:
                    response = f"❌ {tool_result.get('message', 'Failed to get exam results')}\n\n"
                    if tool_result.get("suggestion"):
                        response += f"💡 **Suggestion**: {tool_result['suggestion']}"
            
            else:
                # Generic response for other operations
                if tool_result.get("status"):
                    response = f"✅ Operation completed successfully."
                    if tool_result.get("message"):
                        response += f" {tool_result['message']}"
                else:
                    response = f"❌ Operation failed: {tool_result.get('message', 'Unknown error')}"
    
    # Add the response to messages
    messages = state["messages"] + [AIMessage(content=response)]
    
    return {
        **state,
        "messages": messages
    }

# Create the workflow
def create_workflow():
    """Create the LangGraph workflow."""
    workflow = StateGraph(ExamBuilderAgentState)
    
    # Add nodes
    workflow.add_node("get_instructor_id", get_instructor_id_node)
    workflow.add_node("classify_intent", classify_intent_and_extract_info)
    workflow.add_node("check_info_completeness", check_information_completeness)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("format_response", format_response)
    
    # Add edges
    workflow.set_entry_point("get_instructor_id")
    workflow.add_edge("get_instructor_id", "classify_intent")
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

# Global workflow instance
workflow_instance = None

def get_workflow():
    """Get or create the workflow instance."""
    global workflow_instance
    if workflow_instance is None:
        workflow_instance = create_workflow()
    return workflow_instance

# Session-based state management
session_states = {}

def get_session_state(session_id: str) -> dict:
    """Get or create session state for a specific session."""
    if session_id not in session_states:
        session_states[session_id] = {
            "messages": [],
            "intent": None,
            "instructor_id": None,
            "tool_name": None,
            "tool_args": None,
            "tool_result": None,
            "missing_info": None,
            "ready_to_execute": False,
            "cached_user_id": None,
            "cached_student_id": None
        }
    return session_states[session_id]

def run_exambuilder_agent_v2(user_input: str, session_id: str = "default") -> str:
    """Run the ExamBuilder agent v2 with user input and return the response."""
    
    workflow = get_workflow()
    
    # Get session-specific state
    conversation_state = get_session_state(session_id)
    
    # Add the new user message to the conversation
    conversation_state["messages"].append(HumanMessage(content=user_input))
    
    # Run the workflow with the current conversation state
    result = workflow.invoke(conversation_state)
    
    # Update the session-specific state
    session_states[session_id] = result
    
    # Return the last AI message
    return result["messages"][-1].content

def reset_conversation(session_id: str = "default"):
    """Reset the conversation state for a specific session."""
    if session_id in session_states:
        del session_states[session_id]

# Example usage
if __name__ == "__main__":
    # Test the agent with working queries
    test_queries = [
        "What can you help me with?",
        "Show me system status",
        "List all available exams",
        "Show me details for Serengeti Practice Exam",
        "List all students"
    ]
    
    print("🎓 ExamBuilder Agent v2 - Based on Working Endpoints")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = run_exambuilder_agent_v2(query)
        print(f"Agent: {response}")
        print("-" * 60) 