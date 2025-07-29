from fastapi import FastAPI, HTTPException, Request, Response, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import traceback
import uuid
from typing import Optional
from exambuilder_agent import run_exambuilder_agent_v2, reset_conversation

# Initialize FastAPI app
app = FastAPI(
    title="ExamBuilder Agent API",
    description="AI Agent for ExamBuilder exam management",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure environment variables
os.environ["LANGSMITH_TRACING_V2"] = "true"
# LangSmith API key and project will be loaded from .env file

# In-memory session store (in production, use Redis or database)
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

class ResetResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    service: str

# Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <head><title>ExamBuilder Agent</title></head>
            <body>
                <h1>🎓 ExamBuilder Agent API</h1>
                <p>The API is running! Use the endpoints to interact with the agent.</p>
                <h3>Available Endpoints:</h3>
                <ul>
                    <li><a href="/docs">/docs</a> - API Documentation</li>
                    <li><a href="/api/status">/api/status</a> - Check connection</li>
                    <li><a href="/api/health">/api/health</a> - Health check</li>
                </ul>
            </body>
        </html>
        """)

@app.get("/api/status", response_model=StatusResponse)
async def status():
    """Check if the agent is ready."""
    try:
        # Simple test to see if the agent can be initialized
        from exambuilder_tools import get_instructor_id
        result = get_instructor_id()
        
        if result.get("status"):
            return StatusResponse(
                status="success",
                message="Connected to ExamBuilder API",
                instructor_id=result.get("instructor_id")
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to connect to ExamBuilder API"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Connection error: {str(e)}"
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_msg: ChatMessage, request: Request, response: Response):
    """Handle chat messages and return agent responses."""
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
        response.set_cookie(key="session_id", value=session_id, httponly=True, samesite="lax")
        
        # Run the agent with the user message and session ID
        agent_response = run_exambuilder_agent_v2(user_message, session_id)
        
        return ChatResponse(
            status="success",
            response=agent_response,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/reset", response_model=ResetResponse)
async def reset_conversation_endpoint(request: Request):
    """Reset the conversation state."""
    try:
        session_id = get_session_id(request)
        reset_conversation(session_id)
        return ResetResponse(
            status="success",
            message="Conversation reset successfully"
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
        service="ExamBuilder Agent API"
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🎓 Starting ExamBuilder Agent Server (FastAPI)...")
    print("📱 UI available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔗 API endpoints:")
    print("   - GET  /api/status  - Check connection")
    print("   - POST /api/chat    - Send message")
    print("   - POST /api/reset   - Reset conversation")
    print("   - GET  /api/health  - Health check")
    print("=" * 50)
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 