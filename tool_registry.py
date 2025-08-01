"""
Dynamic Tool Registry for ExamBuilder Multi-Agent System
Automatically discovers and registers tools without hard-coding
"""

import inspect
import importlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
from langsmith import trace

@dataclass
class ToolMetadata:
    """Metadata for a tool"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_parameters: List[str]
    optional_parameters: List[str]
    return_type: str
    category: str
    tags: List[str]

class ToolCategory(Enum):
    """Tool categories for organization"""
    AUTHENTICATION = "authentication"
    EXAM_MANAGEMENT = "exam_management"
    STUDENT_MANAGEMENT = "student_management"
    SCHEDULING = "scheduling"
    RESULTS = "results"
    UTILITY = "utility"

class DynamicToolRegistry:
    """Dynamic tool registry with automatic discovery"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.categories: Dict[str, List[str]] = {}
        
        # Auto-discover tools from exambuilder_tools module
        self._discover_tools()
    
    def _discover_tools(self):
        """Automatically discover tools from the exambuilder_tools module"""
        try:
            import exambuilder_tools
            
            # Get all functions from the module
            for name, obj in inspect.getmembers(exambuilder_tools):
                if inspect.isfunction(obj) and not name.startswith('_'):
                    # Register the tool
                    self.register_tool(name, obj)
                    
        except ImportError as e:
            print(f"⚠️  Could not import exambuilder_tools: {e}")
    
    def register_tool(self, name: str, func: Callable, metadata: Optional[ToolMetadata] = None):
        """Register a tool with optional metadata"""
        self.tools[name] = func
        
        # Generate metadata if not provided
        if metadata is None:
            metadata = self._generate_metadata(name, func)
        
        self.metadata[name] = metadata
        
        # Add to category
        category = metadata.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)
        
        print(f"🔧 Registered tool: {name} ({category})")
    
    def _generate_metadata(self, name: str, func: Callable) -> ToolMetadata:
        """Generate metadata from function signature and docstring"""
        sig = inspect.signature(func)
        doc = func.__doc__ or ""
        
        # Parse parameters
        parameters = {}
        required_params = []
        optional_params = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            parameters[param_name] = {
                "type": param_type,
                "default": param.default if param.default != inspect.Parameter.empty else None,
                "required": param.default == inspect.Parameter.empty
            }
            
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                optional_params.append(param_name)
        
        # Determine category based on function name
        category = self._determine_category(name)
        
        # Extract description from docstring
        description = doc.split('\n')[0] if doc else f"Tool for {name}"
        
        return ToolMetadata(
            name=name,
            description=description,
            parameters=parameters,
            required_parameters=required_params,
            optional_parameters=optional_params,
            return_type=str(sig.return_annotation) if sig.return_annotation != inspect.Parameter.empty else "Any",
            category=category,
            tags=self._extract_tags(name, doc)
        )
    
    def _determine_category(self, name: str) -> str:
        """Determine tool category based on name"""
        name_lower = name.lower()
        
        if 'auth' in name_lower or 'instructor' in name_lower:
            return ToolCategory.AUTHENTICATION.value
        elif 'exam' in name_lower:
            return ToolCategory.EXAM_MANAGEMENT.value
        elif 'student' in name_lower:
            return ToolCategory.STUDENT_MANAGEMENT.value
        elif 'schedule' in name_lower:
            return ToolCategory.SCHEDULING.value
        elif 'result' in name_lower or 'stat' in name_lower:
            return ToolCategory.RESULTS.value
        else:
            return ToolCategory.UTILITY.value
    
    def _extract_tags(self, name: str, doc: str) -> List[str]:
        """Extract tags from function name and docstring"""
        tags = []
        name_lower = name.lower()
        doc_lower = doc.lower()
        
        # Common patterns
        if 'list' in name_lower:
            tags.append('list')
        if 'get' in name_lower:
            tags.append('get')
        if 'create' in name_lower:
            tags.append('create')
        if 'update' in name_lower:
            tags.append('update')
        if 'delete' in name_lower:
            tags.append('delete')
        if 'search' in name_lower:
            tags.append('search')
        
        return tags
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name"""
        return self.metadata.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools, optionally filtered by category"""
        if category:
            return self.categories.get(category, [])
        return list(self.tools.keys())
    
    def get_tools_by_category(self, category: str) -> Dict[str, Callable]:
        """Get all tools in a category"""
        tool_names = self.categories.get(category, [])
        return {name: self.tools[name] for name in tool_names}
    
    def search_tools(self, query: str) -> List[str]:
        """Search tools by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for name, metadata in self.metadata.items():
            # Search in name
            if query_lower in name.lower():
                results.append(name)
                continue
            
            # Search in description
            if query_lower in metadata.description.lower():
                results.append(name)
                continue
            
            # Search in tags
            for tag in metadata.tags:
                if query_lower in tag.lower():
                    results.append(name)
                    break
        
        return results
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        if name not in self.tools:
            return {"status": False, "error": f"Tool '{name}' not found"}
        
        try:
            with trace(f"tool_execution_{name}"):
                tool = self.tools[name]
                metadata = self.metadata[name]
                
                # Validate required parameters
                missing_params = []
                for param in metadata.required_parameters:
                    if param not in kwargs:
                        missing_params.append(param)
                
                if missing_params:
                    return {
                        "status": False, 
                        "error": f"Missing required parameters: {', '.join(missing_params)}"
                    }
                
                # Execute the tool
                result = tool(**kwargs)
                return {"status": True, "data": result}
            
        except Exception as e:
            return {"status": False, "error": f"Tool execution error: {str(e)}"}
    
    def get_tool_suggestions(self, intent: str, entities: Dict[str, Any]) -> List[str]:
        """Get tool suggestions based on intent and entities"""
        suggestions = []
        
        # Map intents to likely tools
        intent_mapping = {
            "list_exams": ["list_exams"],
            "get_exam": ["get_exam"],
            "list_students": ["list_students"],
            "get_student": ["get_student", "search_student_by_student_id"],
            "create_student": ["create_student"],
            "schedule_exam": ["schedule_exam"],
            "get_results": ["get_exam_attempt", "get_student_exam_statistics"],
            "authentication": ["get_instructor_id"]
        }
        
        # Get suggestions based on intent
        if intent in intent_mapping:
            suggestions.extend(intent_mapping[intent])
        
        # Add tools based on entities
        if "student_id" in entities:
            suggestions.extend(["get_student", "search_student_by_student_id"])
        if "exam_id" in entities:
            suggestions.extend(["get_exam", "schedule_exam"])
        
        # Remove duplicates and return
        return list(set(suggestions))
    
    def get_tool_dependencies(self, tool_name: str) -> List[str]:
        """Get dependencies for a tool (e.g., authentication)"""
        dependencies = []
        
        # Most tools require authentication
        if tool_name != "get_instructor_id":
            dependencies.append("get_instructor_id")
        
        # Specific dependencies
        if tool_name == "schedule_exam":
            dependencies.extend(["get_student", "get_exam"])
        elif tool_name == "get_student_exam_statistics":
            dependencies.extend(["get_student", "get_exam_attempt"])
        
        return dependencies

# Global tool registry instance
tool_registry = DynamicToolRegistry()

def get_tool_registry() -> DynamicToolRegistry:
    """Get the global tool registry instance"""
    return tool_registry 