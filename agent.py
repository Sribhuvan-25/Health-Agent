"""
ExamBuilder LangGraph Agent - Proper Implementation
A properly structured LangGraph agent for exam management system
"""

import json
import os
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from tool_registry import get_tool_registry
from config import get_config
import langsmith
from langsmith import trace

# Load environment variables
load_dotenv()

# Get configuration
config = get_config()

# Setup LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "exambuilder-langgraph-agent"

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """The state of our LangGraph agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    instructor_id: Optional[str]
    user_id: Optional[str]
    exam_data: Optional[List[Dict]]
    user_exam_id: Optional[str]
    extracted_entities: Optional[Dict]
    current_intent: Optional[str]
    missing_info: Optional[List[str]]
    context: Optional[Dict]

# ============================================================================
# LANGGRAPH TOOLS INTEGRATION
# ============================================================================

def create_langgraph_tools():
    """Create LangGraph-compatible tools from our tool registry"""
    from langchain_core.tools import tool
    
    tool_registry = get_tool_registry()
    tools = []
    
    # Create a LangGraph tool for each function in our registry
    for tool_name in tool_registry.list_tools():
        tool_func = tool_registry.get_tool(tool_name)
        metadata = tool_registry.get_metadata(tool_name)
        
        if tool_func and metadata:
            # Create a wrapper that handles the tool registry execution
            def create_tool_wrapper(name, description, required_params):
                @tool(name=name, description=description)
                def tool_wrapper(**kwargs):
                    """Dynamically created tool wrapper"""
                    registry = get_tool_registry()
                    result = registry.execute_tool(name, **kwargs)
                    if result.get("status"):
                        return result.get("data", {})
                    else:
                        return {"error": result.get("error", "Tool execution failed")}
                
                return tool_wrapper
            
            # Create the tool with proper metadata
            langgraph_tool = create_tool_wrapper(
                tool_name,
                metadata.description,
                metadata.required_parameters
            )
            tools.append(langgraph_tool)
    
    return tools

# ============================================================================
# LANGGRAPH NODES
# ============================================================================

def intent_classifier_node(state: AgentState) -> AgentState:
    """Classify user intent from the latest message"""
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Get the latest human message
    latest_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break
    
    if not latest_message:
        return state
    
    # Check if we have a previous intent and missing info (context continuation)
    previous_intent = state.get("current_intent")
    missing_info = state.get("missing_info", [])
    
    # If we have missing info and user provides simple input, maintain context
    if previous_intent and missing_info:
        # Simple inputs that are likely continuation of previous intent
        simple_patterns = [
            len(latest_message.split()) <= 3,  # 3 words or less
            latest_message.lower().startswith(("my ", "i am ", "john", "doe", "password")),
            any(word in latest_message.lower() for word in ["john", "doe", "password", "email", "@"]),
            latest_message.replace(" ", "").replace(",", "").replace(".", "").isalnum()  # Simple alphanumeric
        ]
        
        if any(simple_patterns):
            print(f"🔄 Maintaining previous intent: {previous_intent}")
            state["current_intent"] = previous_intent
            return state
    
    # Get conversation context for better classification
    recent_messages = []
    for msg in reversed(state["messages"][-4:]):  # Last 4 messages for context
        if isinstance(msg, HumanMessage):
            recent_messages.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            recent_messages.append(f"Agent: {msg.content[:100]}...")
    
    context = "\n".join(reversed(recent_messages))
    
    prompt = f"""
        You are an intent classifier for an exam management system.

        User input: "{latest_message}"
        Previous intent: {previous_intent}

        Recent conversation context:
        {context}

        Available intents:
        - list_exams: User wants to see available exams
        - get_exam: User wants details about a specific exam  
        - list_students: User wants to see students
        - get_student: User wants details about a specific student
        - create_student: User wants to create a new student account
        - schedule_exam: User wants to schedule an exam for a student
        - list_scheduled_exams: User wants to see their scheduled/registered exams
        - get_results: User wants to see exam results
        - help: User needs help
        - status: User wants system status

        IMPORTANT RULES:
        1. If user is providing missing information for previous intent, keep the same intent
        2. Look for keywords: 
        - "register", "schedule" = schedule_exam
        - "results" = get_results
        - "create", "new account" = create_student
        - "show", "my exams", "scheduled", "registered" = list_scheduled_exams
        3. If user says single words/names after create_student context, maintain create_student intent
        4. If user provides student ID after asking for registration, maintain schedule_exam intent

        Respond with ONLY the intent name, nothing else.
    """
    
    try:
        with trace("intent_classification"):
            response = llm.invoke(prompt)
            intent = response.content.strip().lower()
            
        state["current_intent"] = intent
        print(f"🎯 Classified intent: {intent}")
        
    except Exception as e:
        print(f"Intent classification error: {e}")
        state["current_intent"] = "help"
    
    return state

def entity_extractor_node(state: AgentState) -> AgentState:
    """Extract entities from user input"""
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Get the latest human message
    latest_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break
    
    if not latest_message:
        return state
    
    intent = state.get("current_intent", "")
    
    # Get previous entities to maintain context
    previous_entities = state.get("extracted_entities", {})
    missing_info = state.get("missing_info", [])
    
    # Get conversation context (last few messages)
    recent_messages = []
    for msg in reversed(state["messages"][-6:]):  # Last 6 messages for context
        if isinstance(msg, HumanMessage):
            recent_messages.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            recent_messages.append(f"Agent: {msg.content[:100]}...")
    
    context = "\n".join(reversed(recent_messages))
    
    prompt = f"""
        Extract entities from this user input: "{latest_message}"

        Intent: {intent}
        Previous entities found: {previous_entities}
        Missing information: {missing_info}

        Recent conversation context:
        {context}

        CONTEXT ANALYSIS:
        - If intent is "create_student" and missing first_name: extract single word/name as first_name (e.g., "Tim" → first_name:"Tim")
        - If intent is "create_student" and missing last_name: extract single word/name as last_name (e.g., "David" → last_name:"David")  
        - If intent is "create_student" and missing student_id: extract any string as student_id (e.g., "Tim1212" → student_id:"Tim1212")
        - If intent is "create_student" and missing password: extract any input as password
        - For simple single-word inputs, map to the FIRST missing field in this order: first_name, last_name, student_id, password
        - Simple inputs in create_student context should be mapped to missing fields

        Extract ONLY the following entities if present:
        - student_id: Any student identifier including email addresses, usernames, or IDs (like "SAMPLE+2523350510825", "john@example.com", "john123")
        - exam_id: Exam IDs (usually alphanumeric strings)
        - exam_name: Exam names (like "Serengeti Certification", "Pearson Test 1", "Serengeti Practice Exam")
        - first_name: First names
        - last_name: Last names
        - password: Passwords

        IMPORTANT RULES:
        1. If the user mentions "Serengetic" they likely mean "Serengeti"
        2. Extract email addresses as student_id (emails are valid student IDs)
        3. For exam names, check for partial matches (e.g., "Serengetic" → "Serengeti Certification")
        4. Preserve previously extracted entities if they're still relevant
        5. If user says "my student ID is X" or "my email is X", extract X as student_id
        6. If user mentions an exam name, extract it even if spelled slightly wrong
        7. Parse comma-separated values: "John, Doe, password123" = first_name:"John", last_name:"Doe", password:"password123"
        8. Look for patterns like "John Doe" for first and last names
        9. For create_student intent: if user gives simple input, map to the missing field (single word usually goes to the currently missing field)
        10. Extract "Tim" as first_name, "David" as last_name, "Tim1212" as student_id, "MyPass123" as password based on context
        11. When user provides an email address, always extract it as student_id, not as a separate email field
        12. Pattern matching: "my [field] is X" should extract X as that field

        Respond with a JSON object containing only the found entities.
        Examples:
        - Input "Tim" (when expecting first_name) → {{"first_name": "Tim"}}
        - Input "David" (when expecting last_name) → {{"last_name": "David"}}
        - Input "My last name is David" → {{"last_name": "David"}}
        - Input "Tim1212" (when expecting student_id) → {{"student_id": "Tim1212"}}
        - Input "JohnDoe" (when expecting password) → {{"password": "JohnDoe"}}
    """
    
    try:
        with trace("entity_extraction"):
            response = llm.invoke(prompt)
            new_entities = json.loads(response.content)
            
        # Merge with previous entities, giving priority to new ones
        merged_entities = previous_entities.copy()
        merged_entities.update(new_entities)
        
        state["extracted_entities"] = merged_entities
        print(f"🔍 Extracted entities: {merged_entities}")
        
    except Exception as e:
        print(f"Entity extraction error: {e}")
        # Keep previous entities if extraction fails
        state["extracted_entities"] = previous_entities
    
    return state

def validation_node(state: AgentState) -> AgentState:
    """Validate if we have required information for the intent"""
    
    intent = state.get("current_intent", "")
    entities = state.get("extracted_entities", {})
    missing_info = []
    
    # Define required fields for each intent
    required_fields = {
        "schedule_exam": ["student_id", "exam_name"],
        "get_results": ["student_id", "exam_name"],
        "create_student": ["first_name", "last_name", "student_id", "password"],
        "list_scheduled_exams": ["student_id"]
    }
    
    if intent in required_fields:
        for field in required_fields[intent]:
            if not entities.get(field):
                missing_info.append(field)
    
    state["missing_info"] = missing_info
    print(f"✅ Validation - Missing info: {missing_info}")
    
    return state

def tool_execution_node(state: AgentState) -> AgentState:
    """Execute tools based on intent and entities"""
    
    intent = state.get("current_intent", "")
    entities = state.get("extracted_entities", {})
    missing_info = state.get("missing_info", [])
    
    # If we have missing info, skip tool execution
    if missing_info:
        return state
    
    # Ensure we have instructor_id
    if not state.get("instructor_id"):
        # Get instructor ID first
        tool_registry = get_tool_registry()
        result = tool_registry.execute_tool("get_instructor_id")
        if result.get("status"):
            instructor_data = result.get("data", {})
            state["instructor_id"] = instructor_data.get("instructor_id")
            print(f"🔑 Got instructor_id: {state['instructor_id']}")
    
    instructor_id = state.get("instructor_id")
    if not instructor_id:
        state["context"] = {"error": "Failed to get instructor ID"}
        return state
    
    # Execute tools based on intent
    tool_registry = get_tool_registry()
    results = {}
    
    try:
        if intent == "list_exams":
            result = tool_registry.execute_tool("list_exams", instructor_id=instructor_id)
            if result.get("status"):
                results["exams"] = result.get("data", {})
                
        elif intent == "schedule_exam":
            student_id = entities.get("student_id")
            exam_name = entities.get("exam_name")
            
            # Step 1: Get exam data
            exams_result = tool_registry.execute_tool("list_exams", instructor_id=instructor_id)
            if exams_result.get("status"):
                exam_data = exams_result.get("data", {}).get("exams", [])
                state["exam_data"] = exam_data
                
                # Find exam ID by name
                exam_id = None
                for exam in exam_data:
                    if exam.get("EXAMNAME") == exam_name:
                        exam_id = exam.get("EXAMID")
                        break
                
                if exam_id:
                    # Step 2: Get student user_id
                    student_result = tool_registry.execute_tool(
                        "search_student_by_student_id",
                        instructor_id=instructor_id,
                        student_id=student_id
                    )
                    
                    if student_result.get("status") and student_result.get("data", {}).get("found"):
                        user_id = student_result.get("data", {}).get("user_id")
                        state["user_id"] = user_id
                        
                        # Step 3: Schedule the exam
                        schedule_result = tool_registry.execute_tool(
                            "schedule_exam",
                            instructor_id=instructor_id,
                            exam_id=exam_id,
                            user_id=user_id
                        )
                        results["schedule"] = schedule_result.get("data", schedule_result)
                    else:
                        results["error"] = "Student not found"
                else:
                    results["error"] = f"Exam '{exam_name}' not found"
                        
        elif intent == "get_results":
            student_id = entities.get("student_id")
            exam_name = entities.get("exam_name")
            
            # Step 1: Get student user_id
            student_result = tool_registry.execute_tool(
                "search_student_by_student_id",
                instructor_id=instructor_id,
                student_id=student_id
            )
            
            if student_result.get("status") and student_result.get("data", {}).get("found"):
                user_id = student_result.get("data", {}).get("user_id") 
                state["user_id"] = user_id
                
                # Step 2: Get exam ID
                exams_result = tool_registry.execute_tool("list_exams", instructor_id=instructor_id)
                if exams_result.get("status"):
                    exam_data = exams_result.get("data", {}).get("exams", [])
                    exam_id = None
                    for exam in exam_data:
                        if exam.get("EXAMNAME") == exam_name:
                            exam_id = exam.get("EXAMID")
                            break
                    
                    if exam_id:
                        # Step 3: Get scheduled exams
                        scheduled_result = tool_registry.execute_tool(
                            "list_scheduled_exams",
                            instructor_id=instructor_id,
                            user_id=user_id,
                            exam_id=exam_id
                        )
                        
                        if scheduled_result.get("status"):
                            scheduled_exams = scheduled_result.get("data", {}).get("students", [])
                            
                            # Find ALL attempts for this student and exam
                            matching_attempts = []
                            for exam in scheduled_exams:
                                if (exam.get("STUDENTID", "").lower() == student_id.lower() and 
                                    exam.get("EXAMNAME", "").lower() == exam_name.lower()):
                                    matching_attempts.append(exam)
                            
                            if matching_attempts:
                                print(f"🔧 Found {len(matching_attempts)} attempts for {student_id}")
                                
                                # Get detailed info for all attempts
                                all_attempts = []
                                for attempt in matching_attempts:
                                    user_exam_id = attempt.get("USEREXAMID")
                                    
                                    # Get basic attempt info
                                    attempt_result = tool_registry.execute_tool(
                                        "get_exam_attempt",
                                        instructor_id=instructor_id,
                                        user_exam_id=user_exam_id
                                    )
                                    
                                    # Try to get statistics (may fail for some attempts)
                                    stats_result = tool_registry.execute_tool(
                                        "get_student_exam_statistics",
                                        instructor_id=instructor_id,
                                        student_id=student_id,
                                        user_exam_id=user_exam_id
                                    )
                                    
                                    all_attempts.append({
                                        "user_exam_id": user_exam_id,
                                        "attempt_info": attempt_result.get("data", attempt_result),
                                        "statistics": stats_result.get("data", stats_result),
                                        "scheduled_data": attempt  # Original scheduled exam data
                                    })
                                
                                results["results"] = {
                                    "all_attempts": all_attempts,
                                    "student_id": student_id,
                                    "exam_name": exam_name,
                                    "total_attempts": len(all_attempts)
                                }
                            else:
                                results["error"] = "No exam attempt found for this student"
                        else:
                            results["error"] = "Failed to get scheduled exams"
                    else:
                        results["error"] = f"Exam '{exam_name}' not found"
            else:
                results["error"] = "Student not found"
        
        elif intent == "create_student":
            first_name = entities.get("first_name")
            last_name = entities.get("last_name")
            student_id = entities.get("student_id")
            password = entities.get("password")
            
            result = tool_registry.execute_tool(
                "create_student",
                instructor_id=instructor_id,
                first_name=first_name,
                last_name=last_name,
                student_id=student_id,
                password=password
            )
            results["create_student"] = result.get("data", result)
        
        elif intent == "list_students":
            result = tool_registry.execute_tool("list_students", instructor_id=instructor_id)
            if result.get("status"):
                results["students"] = result.get("data", {})
        
        elif intent == "list_scheduled_exams":
            student_id = entities.get("student_id")
            
            # First get the user_id from student_id
            student_result = tool_registry.execute_tool(
                "search_student_by_student_id",
                instructor_id=instructor_id,
                student_id=student_id
            )
            
            if student_result.get("status") and student_result.get("data", {}).get("found"):
                user_id = student_result.get("data", {}).get("user_id")
                
                # Get all available exams first
                exams_result = tool_registry.execute_tool("list_exams", instructor_id=instructor_id)
                
                if exams_result.get("status"):
                    all_exams = exams_result.get("data", {}).get("exams", [])
                    all_scheduled_exams = []
                    
                    # Check each exam individually for scheduling
                    for exam in all_exams:
                        exam_id = exam.get("EXAMID")
                        if exam_id:
                            # Get scheduled exams for this specific exam
                            scheduled_result = tool_registry.execute_tool(
                                "list_scheduled_exams",
                                instructor_id=instructor_id,
                                user_id=user_id,
                                exam_id=exam_id
                            )
                            
                            if scheduled_result.get("status"):
                                scheduled_exams = scheduled_result.get("data", {}).get("students", [])
                                # Filter out NULL entries and add valid scheduled exams
                                for scheduled_exam in scheduled_exams:
                                    if (scheduled_exam and 
                                        scheduled_exam != {"NULL": None} and 
                                        scheduled_exam.get("EXAMID")):
                                        all_scheduled_exams.append(scheduled_exam)
                    
                    results["scheduled_exams"] = {"students": all_scheduled_exams}
                    results["student_info"] = {"student_id": student_id, "user_id": user_id}
                else:
                    results["error"] = "Failed to retrieve exams list"
            else:
                results["error"] = f"Student '{student_id}' not found"
                
        state["context"] = results
        print(f"🔧 Tool execution results: {len(results)} results")
        
    except Exception as e:
        print(f"Tool execution error: {e}")
        state["context"] = {"error": str(e)}
    
    return state

def format_contextual_missing_info_response(intent: str, missing_info: List[str], entities: Dict) -> str:
    """Format contextual missing information responses"""
    
    if intent == "schedule_exam":
        if "student_id" in missing_info and "exam_name" not in missing_info:
            return """🤖 **Student ID Required**

I see you want to register for an exam. I need your student ID or email address.

**Please provide your student ID** like:
• "My student ID is john@example.com"
• "I am John1212"
• "Student ID: SAMPLE+123456"
"""
        elif "exam_name" in missing_info and "student_id" not in missing_info:
            return """🤖 **Exam Name Required**

I have your student ID. Which exam would you like to register for?

**Available exams:**
• Serengeti Practice Exam
• Pearson Test 1  
• Serengeti Certification

**Just say:** "I want to register for [exam name]"
"""
        else:
            return """🤖 **Registration Information Needed**

To register you for an exam, I need:

1. **Your student ID** (email or ID number)
2. **Exam name** you want to register for

**Example:** "I am john@example.com and want to register for Serengeti Practice Exam"
"""
    
    elif intent == "get_results":
        if "student_id" in missing_info and "exam_name" not in missing_info:
            return """🤖 **Student ID Required**

I see you want exam results. I need your student ID or email address.

**Please provide your student ID** like:
• "My student ID is john@example.com"
• "I am SAMPLE+123456"
"""
        elif "exam_name" in missing_info and "student_id" not in missing_info:
            return """🤖 **Exam Name Required**

I have your student ID. Which exam results do you want to see?

**Just say:** "Results for [exam name]"
**Example:** "Results for Serengeti Certification"
"""
        else:
            return """🤖 **Results Information Needed**

To get your exam results, I need:

1. **Your student ID** (email or ID number)
2. **Exam name** you want results for

**Example:** "My ID is john@example.com, results for Serengeti Certification"
"""
    
    elif intent == "create_student":
        return format_student_creation_response(missing_info, entities)
    
    elif intent == "list_scheduled_exams":
        return """🤖 **Student ID Required**

To show your scheduled exams, I need your student ID or email address.

**Please provide your student ID** like:
• "My student ID is john@example.com"
• "I am John1212"
• "Student ID: SAMPLE+123456"
"""
    
    else:
        missing_fields = ", ".join(missing_info)
        return f"""🤖 **Information Required**

To {intent.replace('_', ' ')}, I need: {missing_fields}

Please provide the missing information and I'll help you!
"""

def format_student_creation_response(missing_info: List[str], entities: Dict) -> str:
    """Format progressive student creation response"""
    
    if len(missing_info) == 4:  # All fields missing
        return """🤖 **Let's Create Your Student Account**

I'll help you create a new student account. I need a few details:

**Please provide your first name to get started.**
**Example:** "My first name is John"
"""
    elif "first_name" in missing_info:
        return """🤖 **First Name Needed**

**Please provide your first name.**
**Example:** "My first name is John" or just "John"
"""
    elif "last_name" in missing_info:
        return """🤖 **Last Name Needed**

**Please provide your last name.**
**Example:** "My last name is Doe" or just "Doe"
"""
    elif "student_id" in missing_info:
        first_name = entities.get("first_name", "")
        return f"""🤖 **Student ID Needed**

Great! Hi {first_name}. **Please provide a student ID for your account.**
**Example:** "My student ID is Tim1212" or just "Tim1212"
"""
    elif "password" in missing_info:
        first_name = entities.get("first_name", "")
        return f"""🤖 **Password Needed**

Almost done {first_name}! **Please create a password for your account.**
**Example:** "My password is SecurePass123"
"""
    else:
        return """🤖 **Account Creation**

I have all the information needed. Let me create your student account!
"""

def response_formatter_node(state: AgentState) -> AgentState:
    """Format the final response"""
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    intent = state.get("current_intent", "")
    missing_info = state.get("missing_info", [])
    context = state.get("context", {})
    entities = state.get("extracted_entities", {})
    
    # Handle missing information
    if missing_info:
        response_text = format_contextual_missing_info_response(intent, missing_info, entities)
        response = AIMessage(content=response_text)
        state["messages"].append(response)
        return state
    
    # Handle errors
    if "error" in context:
        error_msg = context["error"]
        response_text = f"❌ Error: {error_msg}"
        response = AIMessage(content=response_text)
        state["messages"].append(response)
        return state
    
    # Format successful responses
    if intent == "list_exams" and "exams" in context:
        exams = context["exams"].get("exams", [])
        response_text = f"""
### 📚 Available Exams

Found **{len(exams)}** exams:

"""
        for exam in exams[:10]:  # Limit to first 10
            response_text += f"• **{exam.get('EXAMNAME', 'Unknown')}**\n"
            response_text += f"  ID: {exam.get('EXAMID', 'N/A')}\n\n"
    
    elif intent == "schedule_exam" and "schedule" in context:
        student_id = entities.get("student_id", "")
        exam_name = entities.get("exam_name", "")
        response_text = f"""
### ✅ Exam Scheduled Successfully!

**Student:** {student_id}
**Exam:** {exam_name}

The exam has been scheduled and the student can now take it.
"""
    
    elif intent == "get_results" and "results" in context:
        results = context["results"]
        student_id = results.get("student_id", entities.get("student_id", ""))
        exam_name = results.get("exam_name", entities.get("exam_name", ""))
        
        response_text = f"""
### 📊 Exam Results

**Student:** {student_id}
**Exam:** {exam_name}

"""
        
        # Handle multiple attempts
        if "all_attempts" in results:
            all_attempts = results["all_attempts"]
            total_attempts = results.get("total_attempts", len(all_attempts))
            
            response_text += f"**Total Attempts:** {total_attempts}\n\n"
            
            for i, attempt_data in enumerate(all_attempts, 1):
                attempt_info = attempt_data.get("attempt_info", {})
                scheduled_data = attempt_data.get("scheduled_data", {})
                
                response_text += f"### 📝 Attempt #{i}\n\n"
                
                if attempt_info and "exam_attempt" in attempt_info:
                    exam_data = attempt_info["exam_attempt"]
                    
                    # Basic exam info
                    attempt_num = exam_data.get("EXAMATTEMPT", "N/A")
                    passing_score = exam_data.get("PASSINGSCORE", "N/A")
                    signup_date = exam_data.get("DATETIMESIGNEDUP", "N/A")
                    started_date = exam_data.get("DATETIMESTARTED", "Not Started")
                    completed_date = exam_data.get("DATETIMECOMPLETED", "Not Completed")
                    score = exam_data.get("SCORE")
                    
                    response_text += f"**Attempt Number:** {attempt_num}\n"
                    response_text += f"**Signed Up:** {signup_date}\n"
                    response_text += f"**Started:** {started_date}\n"
                    response_text += f"**Completed:** {completed_date}\n"
                    
                    if score is not None and score != "":
                        response_text += f"**Score:** {score}%\n"
                        if passing_score != "N/A":
                            try:
                                if float(score) >= float(passing_score):
                                    response_text += f"**Result:** ✅ **PASSED** (Score: {score}% ≥ Required: {passing_score}%)\n"
                                else:
                                    response_text += f"**Result:** ❌ **FAILED** (Score: {score}% < Required: {passing_score}%)\n"
                            except:
                                response_text += f"**Result:** Score: {score}%\n"
                    else:
                        if completed_date and completed_date != "Not Completed" and completed_date != "None":
                            response_text += f"**Status:** Completed but score not available\n"
                        elif started_date and started_date != "Not Yet" and started_date != "Not Started":
                            response_text += f"**Status:** In progress\n"
                        else:
                            response_text += f"**Status:** Not started\n"
                    
                    response_text += f"\n"
                
                elif scheduled_data:
                    # Fallback to scheduled data if attempt_info is not available
                    signup_date = scheduled_data.get("DATETIMESIGNEDUP", "N/A")
                    started_date = scheduled_data.get("DATETIMESTARTED", "Not Started") 
                    completed_date = scheduled_data.get("DATETIMECOMPLETED", "Not Completed")
                    attempt_num = scheduled_data.get("EXAMATTEMPT", "N/A")
                    score = scheduled_data.get("SCORE")
                    
                    response_text += f"**Attempt Number:** {attempt_num}\n"
                    response_text += f"**Signed Up:** {signup_date}\n"
                    response_text += f"**Started:** {started_date}\n"
                    response_text += f"**Completed:** {completed_date}\n"
                    
                    if score is not None and score != "":
                        response_text += f"**Score:** {score}%\n"
                    else:
                        response_text += f"**Status:** No score available\n"
                    
                    response_text += f"\n"
            
            # Show passing score info at the end
            if all_attempts and all_attempts[0].get("attempt_info", {}).get("exam_attempt", {}).get("PASSINGSCORE"):
                passing_score = all_attempts[0]["attempt_info"]["exam_attempt"]["PASSINGSCORE"]
                response_text += f"**Passing Score Required:** {passing_score}%\n"
        
        else:
            response_text += "**Status:** No exam attempt data found.\n"
            response_text += "This student may not have started the exam yet.\n"
    
    elif intent == "create_student" and "create_student" in context:
        student_result = context["create_student"]
        entities = state.get("extracted_entities", {})
        first_name = entities.get("first_name", "")
        student_id = entities.get("student_id", "")
        
        if student_result.get("status"):
            response_text = f"""
### ✅ Student Account Created Successfully!

**Name:** {first_name}
**Student ID:** {student_id}

Your account has been created and you can now register for exams!
"""
        else:
            error_msg = student_result.get("error", "Unknown error occurred")
            response_text = f"""
### ❌ Account Creation Failed

**Error:** {error_msg}

Please try again or contact support if the problem persists.
"""
    
    elif intent == "list_students" and "students" in context:
        students = context["students"].get("students", [])
        response_text = f"""
### 👥 Students List

Found **{len(students)}** students:

"""
        for student in students[:10]:  # Limit to first 10
            response_text += f"• **{student.get('FIRSTNAME', '')} {student.get('LASTNAME', '')}**\n"
            response_text += f"  Email: {student.get('STUDENTID', 'N/A')}\n\n"
    
    elif intent == "list_scheduled_exams" and "scheduled_exams" in context:
        scheduled_data = context["scheduled_exams"]
        student_info = context.get("student_info", {})
        student_id = student_info.get("student_id", "Unknown")
        
        scheduled_exams = scheduled_data.get("students", [])
        
        response_text = f"""
### 📅 Your Scheduled Exams

**Student:** {student_id}

"""
        
        if scheduled_exams and len(scheduled_exams) > 0 and scheduled_exams[0] != {"NULL": None}:
            response_text += f"Found **{len(scheduled_exams)}** scheduled exam(s):\n\n"
            
            for exam in scheduled_exams:
                exam_name = exam.get("EXAMNAME", "Unknown Exam")
                exam_id = exam.get("EXAMID", "N/A")
                user_exam_id = exam.get("USEREXAMID", "N/A")
                signup_date = exam.get("DATETIMESIGNEDUP", "N/A")
                started_date = exam.get("DATETIMESTARTED", "Not Started")
                completed_date = exam.get("DATETIMECOMPLETED", "Not Completed")
                attempt_num = exam.get("EXAMATTEMPT", "1")
                score = exam.get("SCORE", "No score yet")
                
                response_text += f"• **{exam_name}**\n"
                response_text += f"  Exam ID: {exam_id}\n"
                response_text += f"  Attempt #{attempt_num}\n"
                response_text += f"  Signed up: {signup_date}\n"
                response_text += f"  Started: {started_date}\n"
                response_text += f"  Completed: {completed_date}\n"
                response_text += f"  Score: {score}\n\n"
        else:
            response_text += "**No scheduled exams found.**\n\nYou can register for available exams by saying:\n\"I want to register for [exam name]\""
    
    else:
        # Default response
        response_text = f"""
### 🤖 ExamBuilder Assistant

I can help you with:
• List available exams
• Schedule exams for students  
• Get exam results
• Manage student accounts
• View system status

**Example commands:**
- "List all exams"
- "I am john@example.com and want to schedule Serengeti Practice Exam"
- "Get results for john@example.com for Serengeti Practice Exam"

How can I help you today?
"""
    
    response = AIMessage(content=response_text)
    state["messages"].append(response)
    print(f"📝 Generated response ({len(response_text)} chars)")
    
    return state

# ============================================================================
# ROUTING LOGIC  
# ============================================================================

def should_continue(state: AgentState) -> str:
    """Determine the next node based on state"""
    
    missing_info = state.get("missing_info", [])
    if missing_info:
        return "response_formatter"
    
    intent = state.get("current_intent", "")
    if intent in ["help", "status"]:
        return "response_formatter"
    
    return "tool_execution"

# ============================================================================
# LANGGRAPH SETUP
# ============================================================================

def create_langgraph_agent():
    """Create the LangGraph agent"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("entity_extractor", entity_extractor_node) 
    workflow.add_node("validation", validation_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("response_formatter", response_formatter_node)
    
    # Add edges
    workflow.add_edge(START, "intent_classifier")
    workflow.add_edge("intent_classifier", "entity_extractor")
    workflow.add_edge("entity_extractor", "validation")
    
    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validation",
        should_continue,
        {
            "tool_execution": "tool_execution",
            "response_formatter": "response_formatter"
        }
    )
    
    workflow.add_edge("tool_execution", "response_formatter")
    workflow.add_edge("response_formatter", END)
    
    # Create memory for persistence
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Global agent instance
langgraph_agent = create_langgraph_agent()

def run_langgraph_agent(user_input: str, session_id: str = "default") -> str:
    """Main interface for the LangGraph agent"""
    
    try:
        with trace("langgraph_agent_execution"):
            # Create the input message
            input_message = HumanMessage(content=user_input)
            
            # Run the agent
            config_dict = {"configurable": {"thread_id": session_id}}
            
            result = langgraph_agent.invoke(
                {"messages": [input_message]},
                config=config_dict
            )
            
            # Get the latest AI message
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    return msg.content
            
            return "I'm sorry, I couldn't process that request."
            
    except Exception as e:
        print(f"LangGraph agent error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ System error: {str(e)}"

def reset_langgraph_session(session_id: str = "default"):
    """Reset a session in the LangGraph agent"""
    # LangGraph with MemorySaver handles this automatically
    # We could clear specific thread data if needed
    print(f"🔄 Session {session_id} reset (handled by LangGraph)")

if __name__ == "__main__":
    # Test the LangGraph agent
    test_inputs = [
        "help",
        "list all exams", 
        "I am john@example.com and want to register for Serengeti Practice Exam"
    ]
    
    print("🤖 ExamBuilder LangGraph Agent")
    print("=" * 50)
    
    for test_input in test_inputs:
        print(f"\n🧪 Testing: {test_input}")
        response = run_langgraph_agent(test_input)
        print(f"📝 Response: {response}")
        print("-" * 50)