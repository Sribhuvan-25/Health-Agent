from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import traceback
from exambuilder_agent import run_exambuilder_agent_v2, reset_conversation

app = Flask(__name__)
CORS(app)

# Configure environment variables
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "LANGSMITH_API_KEY_REMOVED"
os.environ["LANGSMITH_PROJECT"] = "exambuilder-agent"

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/api/status')
def status():
    """Check if the agent is ready."""
    try:
        # Simple test to see if the agent can be initialized
        from exambuilder_tools import get_instructor_id
        result = get_instructor_id()
        
        if result.get("status"):
            return jsonify({
                "status": "success",
                "message": "Connected to ExamBuilder API",
                "instructor_id": result.get("instructor_id")
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to connect to ExamBuilder API"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Connection error: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and return agent responses."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "status": "error",
                "message": "No message provided"
            }), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                "status": "error", 
                "message": "Empty message"
            }), 400
        
        # Run the agent with the user message
        response = run_exambuilder_agent_v2(user_message)
        
        return jsonify({
            "status": "success",
            "response": response
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the conversation state."""
    try:
        reset_conversation()
        return jsonify({
            "status": "success",
            "message": "Conversation reset successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to reset conversation: {str(e)}"
        }), 500

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "ExamBuilder Agent API"
    })

if __name__ == '__main__':
    print("🎓 Starting ExamBuilder Agent Server...")
    print("📱 UI available at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - GET  /api/status  - Check connection")
    print("   - POST /api/chat    - Send message")
    print("   - POST /api/reset   - Reset conversation")
    print("   - GET  /api/health  - Health check")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001) 