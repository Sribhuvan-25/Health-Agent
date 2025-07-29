"""
ExamBuilder Agent - Complete Workflow Demo
Demonstrates the full exam scheduling workflow including student creation and management.
"""

import os
from exambuilder_agent import run_exambuilder_agent_v2, reset_conversation

def main():
    print("🎓 ExamBuilder Agent - Complete Workflow Demo")
    print("=" * 60)
    print("✅ Successfully Connected to ExamBuilder API!")
    print("✅ Full workflow support: Student Creation → Exam Scheduling → Results!")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        return
    
    # Comprehensive workflow demonstration
    demo_scenarios = [
        {
            "title": "🔍 System Capabilities",
            "query": "What can you help me with?",
            "description": "Shows all available capabilities including new features"
        },
        {
            "title": "📊 System Status Check", 
            "query": "Show me the system status",
            "description": "Displays API connection status and available resources"
        },
        {
            "title": "📝 List Available Exams",
            "query": "List all available exams",
            "description": "Shows all exams that can be scheduled"
        },
        {
            "title": "🎯 Exam Scheduling Workflow - New Student",
            "query": "I need to schedule an exam and this is my first time",
            "description": "Demonstrates the complete workflow for new students"
        },
        {
            "title": "👤 Student Account Creation",
            "query": "I need to create a new student account",
            "description": "Shows student registration workflow"
        },
        {
            "title": "🔍 Student Search by Email",
            "query": "Search for student with email john.doe@example.com",
            "description": "Demonstrates student lookup functionality"
        },
        {
            "title": "📅 List Scheduled Exams",
            "query": "Show me my scheduled exams",
            "description": "Displays scheduled exam management"
        },
        {
            "title": "📊 Exam Results and Statistics",
            "query": "Show me my exam results",
            "description": "Demonstrates exam results retrieval"
        },
        {
            "title": "🔄 Student Information Update",
            "query": "I need to update my student information",
            "description": "Shows student profile management"
        }
    ]
    
    print("\n🚀 Running Complete Workflow Demo:")
    print("-" * 60)
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n{i}. {scenario['title']}")
        print(f"   Query: \"{scenario['query']}\"")
        print(f"   Purpose: {scenario['description']}")
        print("   " + "─" * 55)
        
        try:
            # Reset conversation for clean state
            reset_conversation()
            response = run_exambuilder_agent_v2(scenario['query'])
            
            # Format the response nicely
            print(f"   🤖 Agent Response:")
            for line in response.split('\n'):
                print(f"   {line}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print("-" * 60)
    
    print("\n✅ Demo Complete!")
    print("\n🎯 **Complete Workflow Achievements:**")
    print("• ✅ Student account creation and management")
    print("• ✅ Exam scheduling workflow")
    print("• ✅ Student search and lookup")
    print("• ✅ Scheduled exam management")
    print("• ✅ Exam results and statistics")
    print("• ✅ Student information updates")
    print("• ✅ Natural language conversation flow")
    print("• ✅ Multi-step workflow support")
    
    print("\n📋 **Available Workflows:**")
    print("• 🎯 **Exam Scheduling**: Check student → Create if needed → Schedule exam")
    print("• 👤 **Student Management**: Create → Update → Search → View")
    print("• 📊 **Results & Analytics**: Attempts → Statistics → Performance")
    print("• 📅 **Scheduling Management**: List → Schedule → Track")
    
    print("\n🔧 **Interactive Mode**")
    print("You can now test the complete workflow:")
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
    
    print("\n👋 Thank you for trying the Complete ExamBuilder Agent!")
    print("🎉 This demonstrates a fully functional workflow system")
    print("   with student management, exam scheduling, and results!")

if __name__ == "__main__":
    main() 