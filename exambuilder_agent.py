"""
ExamBuilder LangGraph Agent v2
Redesigned based on actual working API endpoints and real data structures.

Working endpoints discovered:
- get_instructor_id() ✅
- list_exams(instructor_id) ✅ 
- get_exam(instructor_id, exam_id) ✅
- list_students(instructor_id) ✅
- list_group_categories(instructor_id) ✅

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
os.environ["LANGSMITH_API_KEY"] = "LANGSMITH_API_KEY_REMOVED"
os.environ["LANGSMITH_PROJECT"] = "exambuilder-agent"

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
    """Get the instructor ID for the authenticated user."""
    return get_instructor_id()

@tool  
def list_exams_tool(instructor_id: str, exam_name: Optional[str] = None, exam_state: str = "all") -> dict:
    """List all exams available to the instructor. Can filter by name and state."""
    return list_exams(instructor_id, exam_name, exam_state)

@tool
def get_exam_tool(instructor_id: str, exam_id: str) -> dict:
    """Get detailed information about a specific exam."""
    return get_exam(instructor_id, exam_id)

@tool
def list_students_tool(instructor_id: str, first_name: Optional[str] = None, 
                      last_name: Optional[str] = None, student_id: Optional[str] = None) -> dict:
    """List all students, optionally filtered by name or student ID."""
    return list_students(instructor_id, first_name, last_name, student_id)

@tool
def get_student_tool(instructor_id: str, user_id: str) -> dict:
    """Get detailed information about a specific student by user ID."""
    return get_student(instructor_id, user_id)

@tool
def list_group_categories_tool(instructor_id: str) -> dict:
    """List all group categories available to the instructor."""
    return list_group_categories(instructor_id)

@tool
def create_student_tool(instructor_id: str, first_name: str, last_name: str, student_id: str, 
                       password: str, email: Optional[str] = None, employee_number: Optional[str] = None) -> dict:
    """Create a new student account with required information."""
    return create_student(instructor_id, first_name, last_name, student_id, password, email, employee_number)

@tool
def update_student_tool(instructor_id: str, student_id: str, first_name: Optional[str] = None,
                       last_name: Optional[str] = None, new_student_id: Optional[str] = None,
                       password: Optional[str] = None, email: Optional[str] = None,
                       employee_number: Optional[str] = None) -> dict:
    """Update a student's information."""
    return update_student(instructor_id, student_id, first_name, last_name, new_student_id, 
                         password, email, employee_number)

@tool
def list_scheduled_exams_tool(instructor_id: str, user_id: Optional[str] = None, 
                             exam_id: Optional[str] = None) -> dict:
    """List scheduled exams, optionally filtered by student or exam."""
    return list_scheduled_exams(instructor_id, user_id, exam_id)

@tool
def schedule_exam_tool(instructor_id: str, exam_id: str, user_id: str) -> dict:
    """Schedule an exam for a student."""
    return schedule_exam(instructor_id, exam_id, user_id)

@tool
def get_exam_attempt_tool(instructor_id: str, user_exam_id: str) -> dict:
    """Get details of a specific exam attempt."""
    return get_exam_attempt(instructor_id, user_exam_id)

@tool
def get_student_exam_statistics_tool(instructor_id: str, student_id: str, user_exam_id: str) -> dict:
    """Get exam statistics for a specific student and exam."""
    return get_student_exam_statistics(instructor_id, student_id, user_exam_id)

@tool
def search_student_by_student_id_tool(instructor_id: str, student_id: str) -> dict:
    """Search for a student by their Student ID (email) and return their User ID."""
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
        # Check if the previous response was a workflow step
        previous_ai_message = None
        for msg in reversed(messages[:-1]):
            if isinstance(msg, AIMessage):
                previous_ai_message = msg.content
                break
        
        if previous_ai_message:
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
    
    prompt = f"""
    Analyze the user's request and extract both the intent and available information for an ExamBuilder system.
    
    {conversation_context}
    
    WORKFLOW CONTEXT: {workflow_context if is_workflow_continuation else "None"}
    IS WORKFLOW CONTINUATION: {is_workflow_continuation}
    
    AVAILABLE OPERATIONS:
    
    EXAM MANAGEMENT:
    - list_exams: List all available exams (can filter by name or state)
    - get_exam: Get detailed information about a specific exam
    
    STUDENT MANAGEMENT:
    - list_students: List all students (can filter by name or student ID)
    - get_student: Get detailed information about a specific student
    - create_student: Create a new student account
    - update_student: Update a student's information
    - search_student_by_student_id: Search for a student by email address
    
    SCHEDULING & RESULTS:
    - list_scheduled_exams: List scheduled exams (automatically converts student email to user ID)
    - schedule_exam: Schedule an exam for a student
    - schedule_exam_with_student_id: Schedule an exam using student email (automatic user ID lookup)
    - get_exam_attempt: Get details of a specific exam attempt
    - get_student_exam_statistics: Get exam statistics for a student
    
    GROUP MANAGEMENT:
    - list_group_categories: List all group categories
    
    GENERAL:
    - help: Provide information about what the system can do
    - status: Show system status and available exams
    - unsupported: For operations not currently available
    
    WORKFLOW INTENTS:
    - schedule_exam_workflow: Complete workflow for scheduling an exam (check student, create if needed, schedule)
    - create_student_workflow: Complete workflow for student registration
    
    INFORMATION TO EXTRACT:
    - exam_id: Exam ID (32-character hex string like "0344b2749fc8e3118d04269aa02d2675")
    - exam_name: Exam name to search for
    - exam_state: Exam state filter ("active", "inactive", "all")
    - user_id: User ID for student lookup
    - student_id: Student's email address (used as Student ID)
    - first_name: Student's first name for search or creation
    - last_name: Student's last name for search or creation
    - password: Student's password
    - user_exam_id: User exam ID for results
    
    IMPORTANT: When extracting names, look for patterns like "John Doe", "John, Doe", "John Doe email@example.com"
    IMPORTANT: When extracting email, look for patterns like "email@example.com", "john.doe@example.com"
    IMPORTANT: When extracting password, look for patterns like "password", "pass", "pwd" followed by a string
    
    IMPORTANT CONTEXT - AVAILABLE EXAMS:
    Note: Exam availability is determined dynamically via API calls. Examples of exam types:
    - Practice exams (typically for training purposes)
    - Certification exams (for official certification)
    - Assessment exams (for evaluation)
    
    The agent should use the list_exams API to get current available exams rather than relying on hardcoded values.
    
    SPECIAL RULES:
    1. If user mentions "schedule exam" AND provides email address → classify as schedule_exam_with_student_id
    2. If user mentions "book exam" AND provides email address → classify as schedule_exam_with_student_id
    3. If user mentions "schedule" AND provides email address → classify as schedule_exam_with_student_id
    4. If user mentions "schedule [exam name] for [email]" → classify as schedule_exam_with_student_id
    5. If user mentions "schedule [exam name] for email [email]" → classify as schedule_exam_with_student_id
    6. If this is a workflow continuation AND user mentions exam name → classify as schedule_exam_with_student_id
    7. If this is a workflow continuation AND user says "yes" or "schedule" → classify as schedule_exam_with_student_id
    8. If user provides 32-character hex string (User ID) AND mentions "schedule" → classify as schedule_exam_with_user_id (treat as student_id)
    9. If user provides 32-character hex string AND mentions exam name → classify as schedule_exam_with_student_id (treat hex as student_id)
    10. If user mentions "practice test" or "practice exam" → extract exam_name as "practice" (search for practice exams via API)
    11. If user mentions "certification" without "practice" → extract exam_name as "certification" (search for certification exams via API)
    12. If user asks about "available exams", "list exams", "show exams" → classify as list_exams
    13. If user mentions specific exam name like "Pearson" or "Serengeti" → classify as list_exams with exam_name filter
    14. If user provides exam ID → classify as get_exam
    15. If user asks about "students", "list students" → classify as list_students
    16. If user provides user ID or wants specific student info → classify as get_student
    17. If user asks about "groups", "categories" → classify as list_group_categories
    18. If user asks "what can you do", "help", "capabilities" → classify as help
    19. If user asks for "status", "overview" → classify as status
    20. If user asks about "scheduling", "schedule exam", "book exam" → classify as schedule_exam_workflow
    21. If user asks about "creating student", "register student", "new student account" → classify as create_student_workflow
    22. If user asks about "updating", "modifying" student info → classify as update_student
    23. If user asks about "exam results", "statistics", "attempts" → classify as get_student_exam_statistics
    24. If user asks about "scheduled exams", "my exams" → classify as list_scheduled_exams
    25. If user asks about "scheduled exams for [student ID]" or "exams for student [ID]" → classify as list_scheduled_exams
    26. If user asks to "find student", "search student", "look up student" and provides email → classify as search_student_by_student_id
    27. If user provides email address in any context → extract it as student_id
    28. If user asks for "student details", "student info", "get student" without providing ID → suggest listing students first
    29. Look at conversation history to combine information from previous messages
    30. If user says "I want to schedule [exam name] for [student ID]" → classify as schedule_exam_with_student_id
    31. If user says "Schedule [exam name] for student [student ID]" → classify as schedule_exam_with_student_id
    32. If user mentions both an exam name AND a student ID in the same request → classify as schedule_exam_with_student_id
    33. IMPORTANT: Always extract both exam_name AND exam_id when user mentions exam names
    34. IMPORTANT: Always extract student_id when user provides email or student ID or 32-character hex strings
    35. IMPORTANT: 32-character hex strings (like "a1b2c3d4e5f6...") should be treated as student_id
    
    EXAM NAME TO ID MAPPING RULES:
    - Use list_exams API to get current exam names and IDs dynamically
    - When user mentions exam names like "Practice Exam" or "Certification", use API to find matching exams
    - If user mentions general terms like "practice" without specifying, show available options
    - Avoid hardcoded exam mappings - always query the API for current exam data
    
    User request: "{last_message}"
    
    Return a JSON object with this structure:
    {{
        "intent": "tool_name",
        "extracted_info": {{
            "exam_id": "extracted exam ID if any",
            "exam_name": "extracted exam name if any",
            "exam_state": "extracted exam state if any",
            "user_id": "extracted user ID if any",
            "student_id": "extracted email address if any",
            "first_name": "extracted first name if any",
            "last_name": "extracted last name if any",
            "password": "extracted password if any",
            "user_exam_id": "extracted user exam ID if any"
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
        "list_group_categories": [],
        "help": [],
        "status": [],
        "unsupported": [],  # No parameters needed for unsupported operations
        "create_student": ["first_name", "last_name", "student_id", "password"],
        "update_student": ["student_id"],  # At least one field to update
        "list_scheduled_exams": [],  # user_id and exam_id are optional, student_id can be converted to user_id
        "schedule_exam": ["exam_id", "user_id"],
        "get_exam_attempt": ["user_exam_id"],
        "get_student_exam_statistics": ["student_id", "user_exam_id"],
        "search_student_by_student_id": ["student_id"],
        "schedule_exam_workflow": [],  # Will be handled specially
        "create_student_workflow": [],   # Will be handled specially
        "schedule_exam_with_student_id": ["student_id"]  # Student ID (email) is required
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
    
    # Handle special intents
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
                    "Create new student accounts (First Name, Last Name, Email, Password)",
                    "Update student information",
                    "Search for students by email address",
                    "List scheduled exams",
                    "Schedule exams for students",
                    "Get exam attempt details",
                    "Get exam statistics and results",
                    "List group categories",
                    "Show system status"
                ]
            }
        }
    
    elif intent == "status":
        # Get system status by listing exams
        try:
            exams_result = list_exams_tool.invoke({"instructor_id": instructor_id})
            return {
                **state,
                "tool_name": "status",
                "tool_result": {
                    "status": True,
                    "instructor_id": instructor_id,
                    "exams_available": len(exams_result.get("exams", [])) if exams_result.get("status") else 0,
                    "system_status": "Connected to ExamBuilder API"
                }
            }
        except Exception as e:
            return {
                **state,
                "tool_result": {"error": f"Failed to get status: {str(e)}"}
            }
    
    elif intent == "unsupported":
        # Handle unsupported operations with helpful messages
        last_message = state["messages"][-1].content.lower()
        
        if any(word in last_message for word in ["schedule", "booking", "book exam"]):
            return {
                **state,
                "tool_name": "unsupported",
                "tool_result": {
                    "status": False,
                    "message": "Exam scheduling is not currently available in this demo version. The system can only view and list exams. Please contact your administrator for scheduling capabilities.",
                    "suggestion": "You can list available exams with: 'Show me all available exams'"
                }
            }
        elif any(word in last_message for word in ["create student", "register student", "add student", "new student"]):
            return {
                **state,
                "tool_name": "unsupported", 
                "tool_result": {
                    "status": False,
                    "message": "Student registration is not currently available in this demo version. The system can only view existing students. Please contact your administrator for student management capabilities.",
                    "suggestion": "You can list existing students with: 'Show me all students'"
                }
            }
        elif any(word in last_message for word in ["update", "modify", "edit", "delete", "remove"]):
            return {
                **state,
                "tool_name": "unsupported",
                "tool_result": {
                    "status": False,
                    "message": "Data modification operations are not currently available in this demo version. The system can only view and list information. Please contact your administrator for modification capabilities.",
                    "suggestion": "You can view available data with: 'What can you help me with?'"
                }
            }
        else:
            return {
                **state,
                "tool_name": "unsupported",
                "tool_result": {
                    "status": False,
                    "message": "This operation is not currently available in this demo version. The system focuses on viewing and listing information.",
                    "suggestion": "Try asking: 'What can you help me with?' to see available capabilities"
                }
            }
    
    elif intent == "schedule_exam_workflow":
        # Handle the complete exam scheduling workflow
        return {
            **state,
            "tool_name": "schedule_exam_workflow",
            "tool_result": {
                "status": True,
                "message": "I'll help you schedule an exam. First, I need to check if you have a student account. Please provide your email address to search for your account, or let me know if you need to create a new account.",
                "workflow_step": "check_student_account",
                "next_action": "Please provide your email address or say 'I need to create a new account'"
            }
        }
    
    elif intent == "schedule_exam_with_student_id":
        # Handle scheduling when student ID (email) is provided
        student_id = tool_args.get("student_id")
        exam_id = tool_args.get("exam_id")
        exam_name = tool_args.get("exam_name")
        
        if not student_id:
            return {
                **state,
                "tool_name": "schedule_exam_workflow",
                "tool_result": {
                    "status": False,
                    "message": "I need your email address to schedule an exam. Please provide your email.",
                    "workflow_step": "need_student_id",
                    "next_action": "Please provide your email address"
                }
            }
        
        # If we have exam_name but no exam_id, find matching exam via API
        if exam_name and not exam_id:
            try:
                # Search for exams that match the exam name
                exam_search_result = list_exams_tool.invoke({
                    "instructor_id": instructor_id,
                    "exam_name": exam_name,
                    "exam_state": "active"
                })
                
                if exam_search_result.get("status") and exam_search_result.get("exams"):
                    # Find the best matching exam
                    matching_exams = exam_search_result["exams"]
                    if len(matching_exams) == 1:
                        exam_id = matching_exams[0].get("EXAMID")
                    elif len(matching_exams) > 1:
                        # Multiple matches - let user choose
                        return {
                            **state,
                            "tool_name": "schedule_exam_with_student_id",
                            "tool_result": {
                                "status": True,
                                "message": f"Found multiple exams matching '{exam_name}'. Please choose:",
                                "workflow_step": "select_exam",
                                "available_exams": matching_exams,
                                "next_action": "Please specify which exam you want to schedule"
                            }
                        }
            except Exception as e:
                # If search fails, continue without exam_id
                pass
        
        # Check if student_id is actually a user_id (32-character hex string)
        import re
        if re.match(r'^[a-f0-9]{32}$', student_id.lower()):
            # This looks like a user_id, use it directly
            user_id = student_id
            student_name = "Student"  # We'll get the real name if scheduling succeeds
            
            # If exam_id is provided, try to schedule directly
            if exam_id:
                try:
                    schedule_result = schedule_exam_tool.invoke({
                        "instructor_id": instructor_id,
                        "exam_id": exam_id,
                        "user_id": user_id
                    })
                    
                    if schedule_result.get("status"):
                        # Get exam name dynamically from API if possible
                        exam_display_name = exam_name or "Selected Exam"
                        try:
                            exam_details = get_exam_tool.invoke({
                                "instructor_id": instructor_id,
                                "exam_id": exam_id
                            })
                            if exam_details.get("status") and exam_details.get("exam"):
                                exam_display_name = exam_details["exam"].get("EXAMNAME", exam_display_name)
                        except:
                            pass  # Use fallback name
                        
                        return {
                            **state,
                            "tool_name": "schedule_exam_with_student_id",
                            "tool_result": {
                                "status": True,
                                "message": f"✅ Exam scheduled successfully!",
                                "student_name": student_name,
                                "user_id": user_id,
                                "exam_id": exam_id,
                                "exam_name": exam_display_name
                            }
                        }
                    else:
                        error_result = {
                            **state,
                            "tool_name": "schedule_exam_with_student_id",
                            "tool_result": {
                                "status": False,
                                "message": f"❌ Failed to schedule exam: {schedule_result.get('message', 'Unknown error')}",
                                "student_name": student_name,
                                "user_id": user_id,
                                "suggestion": "Please try again or contact support"
                            }
                        }
                        
                        # Check if it's an "already scheduled" error
                        if schedule_result.get("returnCode") == "STUDENT_ALREADY_SCHEDULED" or schedule_result.get("already_scheduled"):
                            error_result["tool_result"]["already_scheduled"] = True
                            error_result["tool_result"]["message"] = "This student is already scheduled for this exam."
                        
                        return error_result
                except Exception as e:
                    return {
                        **state,
                        "tool_name": "schedule_exam_with_student_id",
                        "tool_result": {
                            "status": False,
                            "message": f"❌ Error scheduling exam: {str(e)}",
                            "student_name": student_name,
                            "user_id": user_id,
                            "suggestion": "Please try again or contact support"
                        }
                    }
            else:
                # No exam specified, show available exams
                try:
                    exams_result = list_exams_tool.invoke({
                        "instructor_id": instructor_id,
                        "exam_state": "active"
                    })
                    
                    if exams_result.get("status") and exams_result.get("exams"):
                        return {
                            **state,
                            "tool_name": "schedule_exam_with_student_id", 
                            "tool_result": {
                                "status": True,
                                "message": f"✅ Student found! Please choose an exam to schedule:",
                                "workflow_step": "select_exam",
                                "available_exams": exams_result["exams"],
                                "student_name": student_name,
                                "user_id": user_id,
                                "next_action": "Please specify which exam you want to schedule"
                            }
                        }
                    else:
                        return {
                            **state,
                            "tool_name": "schedule_exam_with_student_id",
                            "tool_result": {
                                "status": False,
                                "message": "❌ No active exams available for scheduling",
                                "suggestion": "Please contact your administrator"
                            }
                        }
                except Exception as e:
                    return {
                        **state,
                        "tool_name": "schedule_exam_with_student_id",
                        "tool_result": {
                            "status": False,
                            "message": f"❌ Error retrieving exams: {str(e)}",
                            "suggestion": "Please try again or contact support"
                        }
                    }
        
        # Original email-based lookup logic
        # First, search for the student by email
        try:
            search_result = search_student_by_student_id_tool.invoke({
                "instructor_id": instructor_id,
                "student_id": student_id
            })
            
            if search_result.get("found"):
                user_id = search_result.get("user_id")
                student_name = f"{search_result.get('student', {}).get('FIRSTNAME', '')} {search_result.get('student', {}).get('LASTNAME', '')}"
                
                # If exam_id is provided, try to schedule directly
                if exam_id:
                    try:
                        schedule_result = schedule_exam_tool.invoke({
                            "instructor_id": instructor_id,
                            "exam_id": exam_id,
                            "user_id": user_id
                        })
                        
                        if schedule_result.get("status"):
                            # Get exam name dynamically from API if possible
                            exam_display_name = exam_name or "Selected Exam"
                            try:
                                exam_details = get_exam_tool.invoke({
                                    "instructor_id": instructor_id,
                                    "exam_id": exam_id
                                })
                                if exam_details.get("status") and exam_details.get("exam"):
                                    exam_display_name = exam_details["exam"].get("EXAMNAME", exam_display_name)
                            except:
                                pass  # Use fallback name
                            
                            return {
                                **state,
                                "tool_name": "schedule_exam_with_student_id",
                                "tool_result": {
                                    "status": True,
                                    "message": f"✅ Exam scheduled successfully for {student_name}!",
                                    "student_name": student_name,
                                    "user_id": user_id,
                                    "exam_id": exam_id,
                                    "exam_name": exam_display_name
                                }
                            }
                        else:
                            error_result = {
                                **state,
                                "tool_name": "schedule_exam_with_student_id",
                                "tool_result": {
                                    "status": False,
                                    "message": f"❌ Failed to schedule exam: {schedule_result.get('message', 'Unknown error')}",
                                    "student_name": student_name,
                                    "user_id": user_id,
                                    "suggestion": "Please try again or contact support"
                                }
                            }
                            
                            # Check if it's an "already scheduled" error
                            if schedule_result.get("returnCode") == "STUDENT_ALREADY_SCHEDULED" or schedule_result.get("already_scheduled"):
                                error_result["tool_result"]["already_scheduled"] = True
                                error_result["tool_result"]["message"] = "This student is already scheduled for this exam."
                            
                            return error_result
                    except Exception as e:
                        return {
                            **state,
                            "tool_name": "schedule_exam_with_student_id",
                            "tool_result": {
                                "status": False,
                                "message": f"❌ Error scheduling exam: {str(e)}",
                                "student_name": student_name,
                                "user_id": user_id
                            }
                        }
                else:
                    # No specific exam provided, show available exams
                    try:
                        exams_result = list_exams_tool.invoke({"instructor_id": instructor_id})
                        if exams_result.get("status") and exams_result.get("exams"):
                            active_exams = [exam for exam in exams_result["exams"] if exam.get("EXAMSTATE") == "Active"]
                            return {
                                **state,
                                "tool_name": "schedule_exam_with_student_id",
                                "tool_result": {
                                    "status": True,
                                    "message": f"✅ Found your account: {student_name} (User ID: {user_id})",
                                    "student_name": student_name,
                                    "user_id": user_id,
                                    "available_exams": active_exams,
                                    "workflow_step": "select_exam",
                                    "next_action": f"Please specify which exam you'd like to schedule. Available active exams: {', '.join([exam.get('EXAMNAME', 'Unknown') for exam in active_exams])}"
                                }
                            }
                        else:
                            return {
                                **state,
                                "tool_name": "schedule_exam_with_student_id",
                                "tool_result": {
                                    "status": False,
                                    "message": "❌ No active exams available for scheduling",
                                    "student_name": student_name,
                                    "user_id": user_id
                                }
                            }
                    except Exception as e:
                        return {
                            **state,
                            "tool_name": "schedule_exam_with_student_id",
                            "tool_result": {
                                "status": False,
                                "message": f"❌ Error getting available exams: {str(e)}",
                                "student_name": student_name,
                                "user_id": user_id
                            }
                        }
            else:
                return {
                    **state,
                    "tool_name": "schedule_exam_with_student_id",
                    "tool_result": {
                        "status": False,
                        "message": f"❌ No student account found with email: {student_id}",
                        "suggestion": "You may need to create a new account first. Say 'I need to create a new account' to get started."
                    }
                }
        except Exception as e:
            return {
                **state,
                "tool_name": "schedule_exam_with_student_id",
                "tool_result": {
                    "status": False,
                    "message": f"❌ Error looking up student account: {str(e)}"
                }
            }
    
    elif intent == "create_student_workflow":
        # Handle the complete student creation workflow
        return {
            **state,
            "tool_name": "create_student_workflow",
            "tool_result": {
                "status": True,
                "message": "I'll help you create a new student account. Please provide the following information:",
                "workflow_step": "collect_student_info",
                "required_fields": ["first_name", "last_name", "student_id", "password"],
                "next_action": "Please provide: First Name, Last Name, Email Address (Student ID), and Password"
            }
        }
    
    elif intent == "list_scheduled_exams":
        # Handle listing scheduled exams - convert student_id to user_id if needed
        user_id = tool_args.get("user_id")
        student_id = tool_args.get("student_id")
        exam_id = tool_args.get("exam_id")
        
        # If we have student_id but no user_id, convert it
        if student_id and not user_id:
            # First check if we have a cached user_id for this student_id
            if (state.get("cached_student_id") == student_id and 
                state.get("cached_user_id")):
                user_id = state["cached_user_id"]
            else:
                try:
                    # Check if student_id is actually a user_id (32-character hex string)
                    import re
                    if re.match(r'^[a-f0-9]{32}$', student_id.lower()):
                        # This looks like a user_id, use it directly
                        user_id = student_id
                    else:
                        # This is an email/student_id, convert to user_id
                        search_result = search_student_by_student_id_tool.invoke({
                            "instructor_id": instructor_id,
                            "student_id": student_id
                        })
                        
                        if search_result.get("found"):
                            user_id = search_result.get("user_id")
                            # Cache the user_id in state for future use
                            state["cached_user_id"] = user_id
                            state["cached_student_id"] = student_id
                        else:
                            return {
                                **state,
                                "tool_name": "list_scheduled_exams",
                                "tool_result": {
                                    "status": False,
                                    "message": f"❌ No student account found with email: {student_id}",
                                    "suggestion": "Please check the email address or create a new account first."
                                }
                            }
                except Exception as e:
                    return {
                        **state,
                        "tool_name": "list_scheduled_exams",
                        "tool_result": {
                            "status": False,
                            "message": f"❌ Error looking up student account: {str(e)}"
                        }
                    }
        
        # Now call list_scheduled_exams with the user_id
        try:
            final_args = {"instructor_id": instructor_id}
            if user_id:
                final_args["user_id"] = user_id
            if exam_id:
                final_args["exam_id"] = exam_id
                
            result = list_scheduled_exams_tool.invoke(final_args)
            
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
                    "message": f"❌ Error retrieving scheduled exams: {str(e)}"
                }
            }
    
    # Map intent to tool
    tool_mapping = {
        "list_exams": list_exams_tool,
        "get_exam": get_exam_tool,
        "list_students": list_students_tool,
        "get_student": get_student_tool,
        "list_group_categories": list_group_categories_tool,
        "create_student": create_student_tool,
        "update_student": update_student_tool,
        "schedule_exam": schedule_exam_tool,
        "get_exam_attempt": get_exam_attempt_tool,
        "get_student_exam_statistics": get_student_exam_statistics_tool,
        "search_student_by_student_id": search_student_by_student_id_tool
    }
    
    if intent not in tool_mapping:
        return {
            **state,
            "tool_result": {"error": f"Unknown intent: {intent}"}
        }
    
    tool = tool_mapping[intent]
    
    try:
        # Prepare arguments for the tool
        final_args = {"instructor_id": instructor_id}
        
        # Add specific arguments based on intent
        if intent == "list_exams":
            if "exam_name" in tool_args and tool_args["exam_name"]:
                final_args["exam_name"] = tool_args["exam_name"]
            if "exam_state" in tool_args and tool_args["exam_state"]:
                final_args["exam_state"] = tool_args["exam_state"]
        
        elif intent == "get_exam":
            final_args["exam_id"] = tool_args["exam_id"]
        
        elif intent == "list_students":
            if "first_name" in tool_args and tool_args["first_name"]:
                final_args["first_name"] = tool_args["first_name"]
            if "last_name" in tool_args and tool_args["last_name"]:
                final_args["last_name"] = tool_args["last_name"]
            if "student_id" in tool_args and tool_args["student_id"]:
                final_args["student_id"] = tool_args["student_id"]
        
        elif intent == "get_student":
            final_args["user_id"] = tool_args["user_id"]
        
        elif intent == "create_student":
            final_args["first_name"] = tool_args["first_name"]
            final_args["last_name"] = tool_args["last_name"]
            final_args["student_id"] = tool_args["student_id"]  # This is the email
            final_args["password"] = tool_args["password"]
        
        elif intent == "update_student":
            final_args["student_id"] = tool_args["student_id"]
            if "first_name" in tool_args and tool_args["first_name"]:
                final_args["first_name"] = tool_args["first_name"]
            if "last_name" in tool_args and tool_args["last_name"]:
                final_args["last_name"] = tool_args["last_name"]
            if "new_student_id" in tool_args and tool_args["new_student_id"]:
                final_args["new_student_id"] = tool_args["new_student_id"]
            if "password" in tool_args and tool_args["password"]:
                final_args["password"] = tool_args["password"]
            if "email" in tool_args and tool_args["email"]:
                final_args["email"] = tool_args["email"]
            if "employee_number" in tool_args and tool_args["employee_number"]:
                final_args["employee_number"] = tool_args["employee_number"]
        
        elif intent == "schedule_exam":
            final_args["exam_id"] = tool_args["exam_id"]
            final_args["user_id"] = tool_args["user_id"]
        
        elif intent == "get_exam_attempt":
            final_args["user_exam_id"] = tool_args["user_exam_id"]
        
        elif intent == "get_student_exam_statistics":
            final_args["student_id"] = tool_args["student_id"]
            final_args["user_exam_id"] = tool_args["user_exam_id"]
        
        elif intent == "search_student_by_student_id":
            final_args["student_id"] = tool_args["student_id"]
        
        # Execute the tool
        result = tool.invoke(final_args)
        
        return {
            **state,
            "tool_name": intent,
            "tool_result": result
        }
        
    except Exception as e:
        return {
            **state,
            "tool_result": {"error": f"Tool execution failed: {str(e)}"}
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
                response += f"• Available Exams: {tool_result.get('exams_available', 0)}\n"
                response += f"• System Status: {tool_result.get('system_status', 'Unknown')}"
            
            elif tool_name == "list_exams":
                if tool_result.get("status") and tool_result.get("exams"):
                    exams = tool_result["exams"]
                    response = f"📝 Found {len(exams)} exam(s):\n\n"
                    for i, exam in enumerate(exams, 1):
                        exam_id = exam.get('EXAMID', 'Unknown')
                        exam_name = exam.get('EXAMNAME', 'Untitled')
                        exam_state = exam.get('EXAMSTATE', 'Unknown')
                        response += f"{i}. **{exam_name}**\n"
                        response += f"   • ID: `{exam_id}`\n"
                        response += f"   • State: {exam_state}\n"
                        response += f"   • Created: {exam.get('DATETIMECREATED', 'Unknown')}\n\n"
                    response += "💡 To get detailed info about an exam, just ask: 'Show me details for [exam name]'"
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
                            response += f"{i}. **{student.get('FIRSTNAME', '')} {student.get('LASTNAME', '')}**\n"
                            response += f"   • User ID: `{student.get('USERID', 'Unknown')}`\n"
                            response += f"   • Student ID: `{student.get('STUDENTID', 'Unknown')}`\n"
                            response += f"   • Created: {student.get('DATETIMECREATED', 'Unknown')}\n\n"
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
                            response += f"{i}. **{cat.get('CATEGORYNAME', 'Unknown')}**\n"
                            response += f"   • ID: `{cat.get('CATEGORYID', 'Unknown')}`\n\n"
                    else:
                        response = "📂 No group categories found."
                else:
                    response = f"❌ Failed to list group categories: {tool_result.get('message', 'Unknown error')}"
            
            elif tool_name == "unsupported":
                response = f"⚠️ {tool_result.get('message', 'This operation is not available')}\n\n"
                if tool_result.get("suggestion"):
                    response += f"💡 **Suggestion**: {tool_result['suggestion']}"
            
            elif tool_name == "schedule_exam_workflow":
                response = f"🎯 {tool_result.get('message', 'Starting exam scheduling workflow')}\n\n"
                if tool_result.get("next_action"):
                    response += f"📋 **Next Step**: {tool_result['next_action']}"
            
            elif tool_name == "schedule_exam_with_student_id":
                if tool_result.get("status"):
                    if tool_result.get("message") and "scheduled successfully" in tool_result["message"]:
                        response = f"✅ **Exam Scheduled Successfully!**\n\n"
                        response += f"**Student**: {tool_result.get('student_name', 'Unknown')}\n"
                        response += f"**Exam**: {tool_result.get('exam_name', 'Practice Exam')}\n\n"
                        response += f"🎯 Your exam is now ready! You can proceed to take it whenever you're ready."
                    elif tool_result.get("workflow_step") == "select_exam":
                        response = f"✅ {tool_result['message']}\n\n"
                        response += f"**Available Active Exams**:\n"
                        for exam in tool_result.get("available_exams", []):
                            response += f"• **{exam.get('EXAMNAME', 'Unknown')}**\n"
                        response += f"\n📋 **Next Step**: {tool_result.get('next_action', 'Please specify which exam you want to schedule')}"
                    else:
                        response = f"✅ {tool_result['message']}"
                else:
                    if tool_result.get("already_scheduled"):
                        response = f"ℹ️ **Already Scheduled**: This student is already scheduled for this exam.\n\n"
                        response += f"📋 You can check your scheduled exams or choose a different exam to schedule."
                    else:
                        response = f"❌ {tool_result.get('message', 'Failed to process scheduling request')}\n\n"
                        if tool_result.get("suggestion"):
                            response += f"💡 **Suggestion**: {tool_result['suggestion']}"
            
            elif tool_name == "create_student_workflow":
                response = f"👤 {tool_result.get('message', 'Starting student creation workflow')}\n\n"
                if tool_result.get("next_action"):
                    response += f"📋 **Next Step**: {tool_result['next_action']}"
            
            elif tool_name == "create_student":
                if tool_result.get("status"):
                    student = tool_result.get("student", {})
                    response = f"✅ Student account created successfully!\n\n"
                    response += f"**Name**: {student.get('FIRSTNAME', '')} {student.get('LASTNAME', '')}\n"
                    response += f"**Student ID**: `{student.get('STUDENTID', 'Unknown')}`\n"
                    response += f"**User ID**: `{student.get('USERID', 'Unknown')}`\n"
                    response += f"**Email**: {student.get('STUDENTID', 'Not provided')}\n\n"
                    response += f"💡 You can now schedule exams with this account!"
                else:
                    response = f"❌ Failed to create student account: {tool_result.get('message', 'Unknown error')}"
            
            elif tool_name == "update_student":
                if tool_result.get("status"):
                    response = f"✅ Student information updated successfully!"
                    if tool_result.get("message"):
                        response += f" {tool_result['message']}"
                else:
                    response = f"❌ Failed to update student: {tool_result.get('message', 'Unknown error')}"
            
            elif tool_name == "list_scheduled_exams":
                if tool_result.get("status"):
                    scheduled_exams = tool_result.get("scheduledExams", []) or tool_result.get("students", [])
                    if scheduled_exams and len(scheduled_exams) > 0 and scheduled_exams[0] != {'NULL': None}:
                        response = f"📅 Found {len(scheduled_exams)} scheduled exam(s):\n\n"
                        for i, exam in enumerate(scheduled_exams, 1):
                            response += f"{i}. **{exam.get('EXAMNAME', 'Unknown')}**\n"
                            response += f"   • Exam ID: `{exam.get('EXAMID', 'Unknown')}`\n"
                            response += f"   • User Exam ID: `{exam.get('USEREXAMID', 'Unknown')}`\n"
                            response += f"   • Scheduled: {exam.get('DATETIMESCHEDULED', 'Unknown')}\n\n"
                    else:
                        response = "📅 No scheduled exams found.\n\n"
                        response += f"💡 **Note**: Students need to be scheduled for exams first. You can schedule exams using the exam scheduling workflow."
                else:
                    response = f"❌ Failed to list scheduled exams: {tool_result.get('message', 'Unknown error')}\n\n"
                    response += f"💡 **Note**: This might be due to server issues or no scheduled exams available."
            
            elif tool_name == "schedule_exam":
                if tool_result.get("status"):
                    response = f"✅ Exam scheduled successfully!"
                    if tool_result.get("message"):
                        response += f" {tool_result['message']}"
                else:
                    response = f"❌ Failed to schedule exam: {tool_result.get('message', 'Unknown error')}"
            
            elif tool_name == "get_exam_attempt":
                if tool_result.get("status") and tool_result.get("attempt"):
                    attempt = tool_result["attempt"]
                    response = f"📊 Exam Attempt Details:\n\n"
                    response += f"**Status**: {attempt.get('STATUS', 'Unknown')}\n"
                    response += f"**Score**: {attempt.get('SCORE', 'Not available')}\n"
                    response += f"**Started**: {attempt.get('DATETIMESTARTED', 'Unknown')}\n"
                    response += f"**Completed**: {attempt.get('DATETIMECOMPLETED', 'Not completed')}\n"
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

# Global conversation state
conversation_state = {
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

# Global workflow instance
workflow_instance = None

def get_workflow():
    """Get or create the workflow instance."""
    global workflow_instance
    if workflow_instance is None:
        workflow_instance = create_workflow()
    return workflow_instance

# Main function to run the agent with conversation memory
def run_exambuilder_agent_v2(user_input: str) -> str:
    """Run the ExamBuilder agent v2 with user input and return the response."""
    global conversation_state
    
    workflow = get_workflow()
    
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
        "instructor_id": None,
        "tool_name": None,
        "tool_args": None,
        "tool_result": None,
        "missing_info": None,
        "ready_to_execute": False,
        "cached_user_id": None,
        "cached_student_id": None
    }

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