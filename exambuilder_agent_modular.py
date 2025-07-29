"""
ExamBuilder LangGraph Agent v3 - Modular Architecture
Refactored with proper LangGraph patterns:
- Separate nodes for each intent type
- Conditional edges for routing
- Better separation of concerns
- More maintainable and testable code
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
    tags=["exambuilder-agent-modular", "gpt-3.5-turbo"]
)

# ============================================================================
# TOOLS (Same as before)
# ============================================================================

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

# ============================================================================
# NODES - Each node handles ONE specific concern
# ============================================================================

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

def classify_intent_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
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
        
        # Check for workflow indicators in the previous response
        if previous_ai_message:
            if "Available Active Exams" in previous_ai_message and "Next Step" in previous_ai_message:
                is_workflow_continuation = True
                workflow_context["type"] = "exam_selection"
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

def check_information_completeness_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
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
        "create_student": [],  # Will be handled step-by-step
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

# ============================================================================
# INTENT-SPECIFIC NODES - Each handles one intent type
# ============================================================================

def help_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Handle help requests."""
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

def status_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Handle status requests."""
    instructor_id = state["instructor_id"]
    
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

def list_exams_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Handle list exams requests."""
    instructor_id = state["instructor_id"]
    tool_args = state["tool_args"]
    
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

def list_students_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Handle list students requests."""
    instructor_id = state["instructor_id"]
    
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

def create_student_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Handle create student requests."""
    instructor_id = state["instructor_id"]
    tool_args = state["tool_args"]
    
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
    try:
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

def unsupported_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
    """Handle unsupported requests."""
    return {
        **state,
        "tool_name": "unsupported",
        "tool_result": {
            "status": False,
            "message": "This operation is not currently supported.",
            "suggestion": "Try asking: 'What can you help me with?' to see available capabilities"
        }
    }

# ============================================================================
# ROUTING FUNCTIONS - Determine which node to execute
# ============================================================================

def route_to_intent_handler(state: ExamBuilderAgentState) -> str:
    """Route to the appropriate intent handler based on the classified intent."""
    intent = state.get("intent", "unknown")
    
    # Map intents to their handler nodes
    intent_handlers = {
        "help": "help",
        "status": "status", 
        "list_exams": "list_exams",
        "list_students": "list_students",
        "create_student": "create_student",
        "schedule_exam": "schedule_exam",
        "list_scheduled_exams": "list_scheduled_exams",
        "get_exam_attempt": "get_exam_attempt",
        "get_exam_attempt_by_student": "get_exam_attempt_by_student",
        "get_student_exam_statistics": "get_student_exam_statistics",
        "get_exam": "get_exam",
        "get_student": "get_student",
        "list_group_categories": "list_group_categories"
    }
    
    return intent_handlers.get(intent, "unsupported")

# ============================================================================
# RESPONSE FORMATTING NODE
# ============================================================================

def format_response_node(state: ExamBuilderAgentState) -> ExamBuilderAgentState:
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
            
            elif tool_name == "create_student":
                if tool_result.get("status"):
                    response = f"✅ {tool_result['message']}"
                else:
                    response = f"❌ {tool_result.get('message', 'Failed to create student account')}\n\n"
                    if tool_result.get("suggestion"):
                        response += f"💡 **Suggestion**: {tool_result['suggestion']}"
            
            elif tool_name == "unsupported":
                response = f"⚠️ {tool_result.get('message', 'This operation is not available')}\n\n"
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

# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_modular_workflow():
    """Create the modular LangGraph workflow with proper conditional edges."""
    workflow = StateGraph(ExamBuilderAgentState)
    
    # Add core nodes
    workflow.add_node("get_instructor_id", get_instructor_id_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("check_info_completeness", check_information_completeness_node)
    workflow.add_node("format_response", format_response_node)
    
    # Add intent-specific nodes
    workflow.add_node("help", help_node)
    workflow.add_node("status", status_node)
    workflow.add_node("list_exams", list_exams_node)
    workflow.add_node("list_students", list_students_node)
    workflow.add_node("create_student", create_student_node)
    workflow.add_node("unsupported", unsupported_node)
    
    # Add edges
    workflow.set_entry_point("get_instructor_id")
    workflow.add_edge("get_instructor_id", "classify_intent")
    workflow.add_edge("classify_intent", "check_info_completeness")
    
    # Conditional edge: route to appropriate handler or format response
    def should_execute(state):
        return state.get("ready_to_execute", False)
    
    workflow.add_conditional_edges(
        "check_info_completeness",
        should_execute,
        {
            True: "route_to_handler",
            False: "format_response"
        }
    )
    
    # Add routing node
    workflow.add_node("route_to_handler", route_to_intent_handler)
    
    # Add conditional edges from routing to specific handlers
    workflow.add_conditional_edges(
        "route_to_handler",
        route_to_intent_handler,
        {
            "help": "help",
            "status": "status",
            "list_exams": "list_exams", 
            "list_students": "list_students",
            "create_student": "create_student",
            "unsupported": "unsupported"
        }
    )
    
    # All handlers go to format_response
    workflow.add_edge("help", "format_response")
    workflow.add_edge("status", "format_response")
    workflow.add_edge("list_exams", "format_response")
    workflow.add_edge("list_students", "format_response")
    workflow.add_edge("create_student", "format_response")
    workflow.add_edge("unsupported", "format_response")
    
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# ============================================================================
# SESSION MANAGEMENT AND MAIN FUNCTION
# ============================================================================

# Global workflow instance
workflow_instance = None

def get_workflow():
    """Get or create the workflow instance."""
    global workflow_instance
    if workflow_instance is None:
        workflow_instance = create_modular_workflow()
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

def run_exambuilder_agent_modular(user_input: str, session_id: str = "default") -> str:
    """Run the modular ExamBuilder agent with user input and return the response."""
    
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
    # Test the modular agent
    test_queries = [
        "What can you help me with?",
        "Show me system status",
        "List all available exams",
        "List all students"
    ]
    
    print("🎓 ExamBuilder Agent v3 - Modular Architecture")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = run_exambuilder_agent_modular(query)
        print(f"Agent: {response}")
        print("-" * 60) 