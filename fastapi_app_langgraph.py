"""
FastAPI App for ExamBuilder LangGraph Agent
Proper LangGraph implementation with clean architecture
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from typing import Optional
from agent import run_langgraph_agent, reset_langgraph_session
from tool_registry import get_tool_registry
from config import get_config

# Initialize FastAPI app
app = FastAPI(
    title="ExamBuilder LangGraph Agent API",
    description="AI Agent with Proper LangGraph Implementation",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get configuration
config = get_config()

# Configure environment variables for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "exambuilder-langgraph-agent"

# Session management
sessions = {}

def get_session_id(request: Request) -> str:
    """Get or create a session ID for the current user."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = {"created": True}
    
    return session_id

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    status: str
    response: str
    session_id: str

class StatusResponse(BaseModel):
    status: str
    message: str
    instructor_id: Optional[str] = None
    available_tools: Optional[list] = None
    config_valid: bool = False

class ResetResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class ToolInfoResponse(BaseModel):
    tools: list
    categories: dict

# Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>ExamBuilder LangGraph Agent</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    height: 100vh; 
                    display: flex; 
                    flex-direction: column;
                    background: #f5f5f5;
                }
                .header { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 15px 20px; 
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .header h1 { font-size: 24px; margin-bottom: 5px; }
                .header p { font-size: 14px; opacity: 0.9; }
                
                .chat-container { 
                    flex: 1; 
                    display: flex; 
                    flex-direction: column; 
                    max-width: 900px; 
                    margin: 0 auto;
                    width: 100%;
                    background: white;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }
                
                .messages-container { 
                    flex: 1; 
                    overflow-y: auto; 
                    padding: 20px; 
                    background: #fafafa;
                }
                
                .message { 
                    margin-bottom: 15px; 
                    animation: slideIn 0.3s ease-out;
                }
                
                @keyframes slideIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                .user-message { 
                    text-align: right; 
                }
                
                .user-message .message-content { 
                    background: #007bff; 
                    color: white; 
                    display: inline-block; 
                    padding: 12px 16px; 
                    border-radius: 18px 18px 4px 18px; 
                    max-width: 70%;
                    word-wrap: break-word;
                }
                
                .agent-message .message-content { 
                    background: white; 
                    border: 1px solid #e0e0e0;
                    display: inline-block; 
                    padding: 12px 16px; 
                    border-radius: 18px 18px 18px 4px; 
                    max-width: 80%;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                
                .error .message-content { 
                    background: #f8d7da; 
                    border-color: #f5c6cb; 
                    color: #721c24;
                }
                
                .input-container { 
                    padding: 20px; 
                    border-top: 1px solid #e0e0e0; 
                    background: white;
                }
                
                .input-group { 
                    display: flex; 
                    gap: 10px; 
                    align-items: center;
                }
                
                #messageInput { 
                    flex: 1; 
                    padding: 12px 16px; 
                    border: 2px solid #e0e0e0; 
                    border-radius: 25px; 
                    font-size: 16px;
                    outline: none;
                    transition: border-color 0.2s;
                }
                
                #messageInput:focus { 
                    border-color: #007bff; 
                }
                
                button { 
                    padding: 12px 20px; 
                    background: #007bff; 
                    color: white; 
                    border: none; 
                    border-radius: 25px; 
                    cursor: pointer; 
                    font-weight: 600;
                    transition: background 0.2s;
                }
                
                button:hover { 
                    background: #0056b3; 
                }
                
                button:disabled { 
                    background: #ccc; 
                    cursor: not-allowed; 
                }
                
                .status-bar {
                    padding: 10px 20px;
                    background: #e8f5e8;
                    border-bottom: 1px solid #c3e6c3;
                    font-size: 14px;
                    color: #2d5a2d;
                }
                
                .links { 
                    padding: 15px 20px; 
                    background: #f8f9fa; 
                    border-top: 1px solid #e0e0e0;
                    text-align: center;
                }
                
                .links a { 
                    margin: 0 10px; 
                    color: #007bff; 
                    text-decoration: none; 
                    font-size: 14px;
                }
                
                .links a:hover { 
                    text-decoration: underline; 
                }
                
                .typing-indicator {
                    padding: 10px 16px;
                    color: #666;
                    font-style: italic;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎓 ExamBuilder LangGraph Agent</h1>
            </div>
            
            <div class="status-bar" id="statusBar">
                🔄 Connecting...
            </div>
            
            <div class="chat-container">
                <div class="messages-container" id="messagesContainer">
                    <div class="message agent-message">
                        <div class="message-content">
                            👋 Welcome! I can help you with:
                            • List available exams
                            • Schedule exams for students  
                            • Get exam results
                            • Manage student accounts
                            
                            Try saying: "I need the list of exams" or "My student ID is [your-id] and I need results for [exam-name]"
                        </div>
                    </div>
                </div>
                
                <div class="input-container">
                    <div class="input-group">
                        <input type="text" id="messageInput" placeholder="Type your message here..." onkeypress="handleKeypress(event)">
                        <button id="sendButton" onclick="sendMessage()">Send</button>
                        <button onclick="resetChat()">Reset</button>
                    </div>
                </div>
            </div>
            
            <div class="links">
                <a href="/api/status">System Status</a>
                <a href="/api/tools">Available Tools</a>
                <a href="/docs">API Documentation</a>
                <a href="/api/health">Health Check</a>
            </div>

            <script>
                let isProcessing = false;

                function handleKeypress(event) {
                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        sendMessage();
                    }
                }

                async function sendMessage() {
                    if (isProcessing) return;
                    
                    const input = document.getElementById('messageInput');
                    const sendButton = document.getElementById('sendButton');
                    const message = input.value.trim();
                    
                    if (!message) return;

                    // Disable input while processing
                    isProcessing = true;
                    input.disabled = true;
                    sendButton.disabled = true;
                    sendButton.textContent = 'Sending...';

                    // Add user message
                    addUserMessage(message);
                    input.value = '';

                    // Show typing indicator
                    const typingId = showTypingIndicator();

                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message: message })
                        });

                        const data = await response.json();
                        removeTypingIndicator(typingId);

                        if (data.status === 'success') {
                            addAgentMessage(data.response);
                        } else {
                            addErrorMessage(data.detail || 'Unknown error');
                        }
                    } catch (error) {
                        removeTypingIndicator(typingId);
                        addErrorMessage('Network error: ' + error.message);
                    } finally {
                        // Re-enable input
                        isProcessing = false;
                        input.disabled = false;
                        sendButton.disabled = false;
                        sendButton.textContent = 'Send';
                        input.focus();
                    }
                }

                function addUserMessage(message) {
                    const container = document.getElementById('messagesContainer');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message user-message';
                    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(message)}</div>`;
                    container.appendChild(messageDiv);
                    scrollToBottom();
                }

                function addAgentMessage(message) {
                    const container = document.getElementById('messagesContainer');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message agent-message';
                    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(message)}</div>`;
                    container.appendChild(messageDiv);
                    scrollToBottom();
                }

                function addErrorMessage(message) {
                    const container = document.getElementById('messagesContainer');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message agent-message error';
                    messageDiv.innerHTML = `<div class="message-content">❌ ${escapeHtml(message)}</div>`;
                    container.appendChild(messageDiv);
                    scrollToBottom();
                }

                function showTypingIndicator() {
                    const container = document.getElementById('messagesContainer');
                    const typingDiv = document.createElement('div');
                    const id = 'typing-' + Date.now();
                    typingDiv.id = id;
                    typingDiv.className = 'message agent-message';
                    typingDiv.innerHTML = '<div class="message-content typing-indicator">🤖 Agent is thinking...</div>';
                    container.appendChild(typingDiv);
                    scrollToBottom();
                    return id;
                }

                function removeTypingIndicator(id) {
                    const element = document.getElementById(id);
                    if (element) {
                        element.remove();
                    }
                }

                function scrollToBottom() {
                    const container = document.getElementById('messagesContainer');
                    setTimeout(() => {
                        container.scrollTop = container.scrollHeight;
                    }, 50);
                }

                function escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }

                async function resetChat() {
                    try {
                        const response = await fetch('/api/reset', { method: 'POST' });
                        const data = await response.json();
                        
                        // Clear messages except welcome message
                        const container = document.getElementById('messagesContainer');
                        container.innerHTML = `
                            <div class="message agent-message">
                                <div class="message-content">
                                    🔄 Chat reset successfully!
                                    
                                    👋 I can help you with:
                                    • List available exams
                                    • Schedule exams for students  
                                    • Get exam results
                                    • Manage student accounts
                                    
                                    Try saying: "I need the list of exams" or "My student ID is [your-id] and I need results for [exam-name]"
                                </div>
                            </div>
                        `;
                        scrollToBottom();
                    } catch (error) {
                        addErrorMessage('Failed to reset chat: ' + error.message);
                    }
                }

                // Test connection on page load
                window.onload = async function() {
                    const statusBar = document.getElementById('statusBar');
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        if (data.status === 'success') {
                            statusBar.innerHTML = '✅ Connected | Instructor ID: ' + data.instructor_id + ' | Tools: ' + data.available_tools.length;
                            statusBar.style.background = '#d4edda';
                            statusBar.style.color = '#155724';
                        } else {
                            statusBar.innerHTML = '⚠️ Connection issue: ' + data.message;
                            statusBar.style.background = '#f8d7da';
                            statusBar.style.color = '#721c24';
                        }
                    } catch (error) {
                        statusBar.innerHTML = '❌ Failed to connect: ' + error.message;
                        statusBar.style.background = '#f8d7da';
                        statusBar.style.color = '#721c24';
                    }
                    
                    // Focus input
                    document.getElementById('messageInput').focus();
                };
            </script>
        </body>
    </html>
    """)

@app.get("/api/status", response_model=StatusResponse)
async def status():
    """Check if the LangGraph agent system is ready."""
    try:
        # Validate config
        config_valid = config.validate()
        
        # Get available tools
        tool_registry = get_tool_registry()
        available_tools = tool_registry.list_tools()
        
        # Test getting instructor ID
        result = tool_registry.execute_tool("get_instructor_id")
        instructor_id = None
        
        if result.get("status"):
            instructor_data = result.get("data", {})
            instructor_id = instructor_data.get("instructor_id")
        
        if instructor_id and config_valid:
            return StatusResponse(
                status="success",
                message="LangGraph Agent Connected Successfully",
                instructor_id=instructor_id,
                available_tools=available_tools,
                config_valid=config_valid
            )
        else:
            error_msg = "Failed to connect"
            if not config_valid:
                error_msg += " - Invalid configuration"
            if not instructor_id:
                error_msg += " - No instructor ID"
                
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
            
    except Exception as e:
        print(f"Status endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Connection error: {str(e)}"
        )

@app.get("/api/tools", response_model=ToolInfoResponse)
async def list_tools():
    """List all available tools and their categories."""
    try:
        tool_registry = get_tool_registry()
        
        # Get tools by category
        categories = {}
        for category in tool_registry.categories:
            categories[category] = tool_registry.list_tools(category)
        
        # Get all tools with metadata
        tools_info = []
        for tool_name in tool_registry.list_tools():
            metadata = tool_registry.get_metadata(tool_name)
            if metadata:
                tools_info.append({
                    "name": tool_name,
                    "description": metadata.description,
                    "category": metadata.category,
                    "tags": metadata.tags,
                    "required_parameters": metadata.required_parameters,
                    "optional_parameters": metadata.optional_parameters
                })
        
        return ToolInfoResponse(
            tools=tools_info,
            categories=categories
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing tools: {str(e)}"
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_msg: ChatMessage, request: Request, response: Response):
    """Handle chat messages and return LangGraph agent responses."""
    try:
        user_message = chat_msg.message.strip()
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="Empty message"
            )
        
        # Get session ID for this user
        session_id = get_session_id(request)
        
        # Set session ID cookie for future requests
        response.set_cookie(
            key="session_id", 
            value=session_id, 
            httponly=True, 
            samesite="lax",
            max_age=86400  # 24 hours
        )
        
        # Run the LangGraph agent
        agent_response = run_langgraph_agent(user_message, session_id)
        
        return ChatResponse(
            status="success",
            response=agent_response,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )

@app.post("/api/reset", response_model=ResetResponse)
async def reset_conversation_endpoint(request: Request):
    """Reset the conversation state."""
    try:
        session_id = get_session_id(request)
        reset_langgraph_session(session_id)
        return ResetResponse(
            status="success",
            message="LangGraph conversation reset successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset conversation: {str(e)}"
        )

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="ExamBuilder LangGraph Agent API",
        version="2.0.0"
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🎓 Starting ExamBuilder LangGraph Agent Server...")
    print("📱 UI available at: http://localhost:8002")
    print("📚 API Documentation: http://localhost:8002/docs")
    print("🔗 API endpoints:")
    print("   - GET  /api/status  - Check connection")
    print("   - GET  /api/tools   - List available tools")
    print("   - POST /api/chat    - Send message")
    print("   - POST /api/reset   - Reset conversation")
    print("   - GET  /api/health  - Health check")
    print("=" * 50)
    
    uvicorn.run(
        "fastapi_app_langgraph:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )