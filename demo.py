"""
Demo script for the FHIR Healthcare LangGraph Agent
Shows how to use the agent with interactive examples
"""

import os
from langraph_agent import run_agent

def main():
    print("🏥 FHIR Healthcare Agent Demo")
    print("=" * 50)
    print("This agent can help you with:")
    print("• Patient management (create, search, get details)")
    print("• Appointment scheduling (create, list, reschedule, cancel)")
    print("• Diagnostic reports (create, view)")
    print("=" * 50)
    
    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("You can do this by creating a .env file with:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    print("✅ OpenAI API key configured successfully!")
    
    # Example queries to demonstrate the agent
    examples = [
        {
            "query": "Create a patient named Sarah Johnson",
            "description": "Creates a new patient in the FHIR system"
        },
        {
            "query": "Search for patients named John",
            "description": "Searches for patients with 'John' in their name"
        },
        {
            "query": "Create an appointment for patient 12345 tomorrow at 3 PM for 1 hour with description 'Annual Checkup'",
            "description": "Creates an appointment with specific details"
        },
        {
            "query": "List all appointments for patient 12345",
            "description": "Shows all appointments for a specific patient"
        },
        {
            "query": "Create a diagnostic report for patient 12345 with code '4548-4', text 'Hemoglobin A1c', value 5.4, unit '%'",
            "description": "Creates a diagnostic report with lab results"
        }
    ]
    
    print("\n📋 Example Queries:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['query']}")
        print(f"   {example['description']}")
        print()
    
    print("=" * 50)
    print("Try your own queries or press Ctrl+C to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🤖 Enter your query: ").strip()
            if not user_input:
                continue
                
            print("\n🔄 Processing...")
            response = run_agent(user_input)
            print(f"\n✅ Response: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 