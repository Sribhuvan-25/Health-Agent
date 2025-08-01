#!/usr/bin/env python3
"""
Test script for ExamBuilder LangGraph Agent
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_config():
    """Test configuration"""
    print("🔧 Testing configuration...")
    
    try:
        from config import get_config
        config = get_config()
        
        if config.validate():
            print("✅ Configuration is valid")
            return True
        else:
            print("❌ Configuration validation failed")
            return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_tool_registry():
    """Test tool registry"""
    print("🔧 Testing tool registry...")
    
    try:
        from tool_registry import get_tool_registry
        registry = get_tool_registry()
        
        tools = registry.list_tools()
        print(f"✅ Found {len(tools)} tools: {', '.join(tools[:5])}...")
        
        # Test a simple tool execution
        result = registry.execute_tool("get_instructor_id")
        if result.get("status"):
            instructor_id = result.get("data", {}).get("instructor_id")
            print(f"✅ Got instructor ID: {instructor_id}")
            return True
        else:
            print(f"❌ Failed to get instructor ID: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Tool registry error: {e}")
        return False

def test_langgraph_agent():
    """Test LangGraph agent"""
    print("🔧 Testing LangGraph agent...")
    
    try:
        from agent import run_langgraph_agent
        
        # Test simple help query
        response = run_langgraph_agent("help", "test_session")
        
        if response and len(response) > 50:
            print(f"✅ Agent responded with {len(response)} characters")
            print(f"Response preview: {response[:100]}...")
            return True
        else:
            print(f"❌ Agent returned short/empty response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ LangGraph agent error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exam_listing():
    """Test exam listing functionality"""
    print("🔧 Testing exam listing...")
    
    try:
        from agent import run_langgraph_agent
        
        response = run_langgraph_agent("list all exams", "test_session")
        
        if "exams" in response.lower() or "exam" in response.lower():
            print("✅ Exam listing appears to work")
            print(f"Response preview: {response[:200]}...")
            return True
        else:
            print(f"❌ Unexpected exam listing response: {response[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Exam listing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ExamBuilder LangGraph Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Tool Registry", test_tool_registry),
        ("LangGraph Agent", test_langgraph_agent),
        ("Exam Listing", test_exam_listing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
        print("-" * 30)
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)