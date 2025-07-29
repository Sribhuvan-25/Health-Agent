"""
ExamBuilder Agent - Interactive Demo
Simple interactive interface for the ExamBuilder agent.
"""

import os
from exambuilder_agent import run_exambuilder_agent_v2, reset_conversation

def main():
    print("🎓 ExamBuilder Agent - Interactive Demo")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        return
    
    print("✅ Connected to ExamBuilder API")
    print("✅ Agent ready for exam scheduling and student management")
    print("\n🔧 **Interactive Mode**")
    print("Ask me anything about exam scheduling, student management, or results!")
    print("(Type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                response = run_exambuilder_agent_v2(user_input)
                print(f"🤖 Agent: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n👋 Thank you for using the ExamBuilder Agent!")

if __name__ == "__main__":
    main() 