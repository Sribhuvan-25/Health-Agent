#!/usr/bin/env python3
"""
Test script for the modular ExamBuilder agent
"""

from exambuilder_agent_modular import run_exambuilder_agent_modular, reset_conversation

def test_modular_agent():
    """Test the modular agent with various queries."""
    
    print("🧪 Testing Modular ExamBuilder Agent")
    print("=" * 50)
    
    # Test queries
    test_cases = [
        {
            "query": "What can you help me with?",
            "expected_intent": "help"
        },
        {
            "query": "Show me system status",
            "expected_intent": "status"
        },
        {
            "query": "List all available exams",
            "expected_intent": "list_exams"
        },
        {
            "query": "List all students",
            "expected_intent": "list_students"
        },
        {
            "query": "I want to create a new account",
            "expected_intent": "create_student"
        },
        {
            "query": "This is not a valid request",
            "expected_intent": "unsupported"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['query']}")
        print(f"Expected Intent: {test_case['expected_intent']}")
        
        try:
            # Reset conversation for clean state
            reset_conversation()
            
            # Run the query
            response = run_exambuilder_agent_modular(test_case['query'])
            
            print(f"✅ Response: {response[:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 50)
    
    print("\n🎉 Modular agent testing completed!")

if __name__ == "__main__":
    test_modular_agent() 