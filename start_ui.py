#!/usr/bin/env python3
"""
Startup script for ExamBuilder Agent UI
Sets up environment variables and starts the Flask server
"""

import os
import sys

def setup_environment():
    """Set up environment variables for the application."""
    # LangSmith configuration
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "LANGSMITH_API_KEY_REMOVED"
    os.environ["LANGSMITH_PROJECT"] = "exambuilder-agent"
    
    # OpenAI API key (you'll need to set this)
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_openai_api_key_here'")
        sys.exit(1)

def main():
    """Main function to start the application."""
    print("🎓 ExamBuilder Agent UI")
    print("=" * 40)
    
    # Setup environment
    setup_environment()
    
    print("✅ Environment configured")
    print("🔗 LangSmith telemetry enabled")
    print("🌐 Starting web server...")
    print()
    
    # Import and run the Flask app
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == "__main__":
    main() 