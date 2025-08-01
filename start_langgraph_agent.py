#!/usr/bin/env python3
"""
Simple startup script for ExamBuilder LangGraph Agent
"""

import uvicorn
from fastapi_app_langgraph import app

if __name__ == "__main__":
    print("🎓 Starting ExamBuilder LangGraph Agent...")
    print("📱 Web UI: http://localhost:8003")
    print("📚 API Docs: http://localhost:8003/docs")
    print("=" * 50)
    
    uvicorn.run(
        "fastapi_app_langgraph:app",
        host="0.0.0.0",
        port=8003,
        reload=True
    )