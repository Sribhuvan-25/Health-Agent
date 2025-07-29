"""
ExamBuilder API Tools
Contains all the API functions for ExamBuilder that will be used as tools in the LangGraph agent.
Based on the ExamBuilder API documentation at https://api.exambuilder.com/

This file contains only the VERIFIED WORKING endpoints from testing.
"""

import requests
import json
import base64
from typing import Dict, Optional

# ExamBuilder API Configuration
BASE_URL = "https://instructor.exambuilder.com/v2"
API_KEY = "FE0F8C82239FF183"
API_SECRET = "A227A6838F3D180A15E6D8ED"

# Create base64 encoded credentials for Basic Auth
credentials = f"{API_KEY}:{API_SECRET}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()

# Authentication headers
AUTH_HEADERS = {
    "Authorization": f"Basic {encoded_credentials}",
    "Content-Type": "application/json"
}

def _make_request(method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
    """Make an authenticated request to the ExamBuilder API."""
    url = f"{BASE_URL}/{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=AUTH_HEADERS, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, headers=AUTH_HEADERS, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=AUTH_HEADERS, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=AUTH_HEADERS)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Try to get more specific error information
        error_details = f"API request failed: {str(e)}"
        if hasattr(e.response, 'text') and e.response is not None:
            try:
                # Try to parse JSON error response
                error_json = e.response.json()
                if 'error' in error_json:
                    error_details = f"API request failed: {str(e)} - Server response: {error_json['error']}"
                elif 'message' in error_json:
                    error_details = f"API request failed: {str(e)} - Server response: {error_json['message']}"
            except:
                # If not JSON, include raw response text
                error_details = f"API request failed: {str(e)} - Server response: {e.response.text[:200]}"
        
        return {
            "error": error_details,
            "status": False,
            "returnCode": "API_ERROR"
        }

# ============================================================================
# VERIFIED WORKING ENDPOINTS
# ============================================================================

def get_instructor_id() -> Dict:
    """
    Get the instructor ID for the authenticated user.
    This is required for most other API calls.
    
    ✅ VERIFIED WORKING
    """
    # The validate.json endpoint is at the root level, not under /v2/
    url = "https://instructor.exambuilder.com/v2/validate.json"
    
    try:
        response = requests.get(url, headers=AUTH_HEADERS)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {
            "error": f"API request failed: {str(e)}",
            "status": False,
            "returnCode": "API_ERROR"
        }

def list_exams(instructor_id: str, exam_name: Optional[str] = None, exam_state: str = "all") -> Dict:
    """
    List all exams available to the instructor.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        exam_name: Optional exam name to search for
        exam_state: Filter by exam state ("active", "notactive", "archived", "all")
    
    ✅ VERIFIED WORKING
    """
    endpoint = f"instructor/{instructor_id}/exam/list.json"
    params = {"examstate": exam_state}
    if exam_name:
        params["examname"] = exam_name
    
    return _make_request("GET", endpoint, params=params)

def get_exam(instructor_id: str, exam_id: str) -> Dict:
    """
    Get details of a specific exam.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        exam_id: The ID of the exam to retrieve
    
    ✅ VERIFIED WORKING
    """
    endpoint = f"instructor/{instructor_id}/exam/{exam_id}.json"
    return _make_request("GET", endpoint)

def list_students(instructor_id: str, first_name: Optional[str] = None, last_name: Optional[str] = None,
                 student_id: Optional[str] = None, sort: Optional[str] = None, sort_direction: Optional[str] = None) -> Dict:
    """
    List all students, optionally filtered by search criteria.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        first_name: Optional first name to search for
        last_name: Optional last name to search for
        student_id: Optional student ID to search for
        sort: Optional sort field ("firstname", "lastname", "datetimecreated")
        sort_direction: Optional sort direction ("asc", "desc")
    
    ✅ VERIFIED WORKING
    """
    endpoint = f"instructor/{instructor_id}/student/list.json"
    params = {}
    if first_name:
        params["firstname"] = first_name
    if last_name:
        params["lastname"] = last_name
    if student_id:
        params["studentid"] = student_id
    if sort:
        params["sort"] = sort
    if sort_direction:
        params["sortdirection"] = sort_direction
    
    return _make_request("GET", endpoint, params=params)

def get_student(instructor_id: str, student_id: str) -> Dict:
    """
    Get details of a specific student.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        student_id: The ID of the student to retrieve
    
    ✅ VERIFIED WORKING
    """
    endpoint = f"instructor/{instructor_id}/student/{student_id}.json"
    return _make_request("GET", endpoint)

def list_group_categories(instructor_id: str) -> Dict:
    """
    List all group categories available to the instructor.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
    
    ✅ VERIFIED WORKING
    """
    endpoint = f"instructor/{instructor_id}/category/list.json"
    return _make_request("GET", endpoint)

# ============================================================================
# STUDENT MANAGEMENT FUNCTIONS
# ============================================================================

def create_student(instructor_id: str, first_name: str, last_name: str, student_id: str, password: str) -> Dict:
    """
    Create a new student account.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        first_name: Student's first name
        last_name: Student's last name
        student_id: Student's email address (used as student ID)
        password: Student's password
    
    🔧 READY FOR TESTING
    """
    endpoint = f"instructor/{instructor_id}/student.json"
    data = {
        "firstName": first_name,
        "lastName": last_name,
        "studentId": student_id,  # This should be the email address
        "password": password
    }
    
    return _make_request("POST", endpoint, data=data)

def update_student(instructor_id: str, student_id: str, first_name: Optional[str] = None, 
                  last_name: Optional[str] = None, new_student_id: Optional[str] = None,
                  password: Optional[str] = None, email: Optional[str] = None, 
                  employee_number: Optional[str] = None) -> Dict:
    """
    Update a student's information.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        student_id: The current student ID
        first_name: New first name (optional)
        last_name: New last name (optional)
        new_student_id: New student ID (optional)
        password: New password (optional)
        email: New email (optional)
        employee_number: New employee number (optional)
    
    🔧 READY FOR TESTING
    """
    endpoint = f"instructor/{instructor_id}/student/{student_id}.json"
    data = {}
    if first_name:
        data["firstName"] = first_name
    if last_name:
        data["lastName"] = last_name
    if new_student_id:
        data["studentId"] = new_student_id
    if password:
        data["password"] = password
    if email:
        data["email"] = email
    if employee_number:
        data["employee_number"] = employee_number
    
    return _make_request("PUT", endpoint, data=data)

# ============================================================================
# SCHEDULING FUNCTIONS
# ============================================================================

def list_scheduled_exams(instructor_id: str, user_id: Optional[str] = None, exam_id: Optional[str] = None) -> Dict:
    """
    List scheduled exams, optionally filtered by student or exam.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        user_id: Optional user ID to filter by student
        exam_id: Optional exam ID to filter by exam
    
    🔧 READY FOR TESTING
    """
    endpoint = f"instructor/{instructor_id}/student/scheduled.json"
    params = {}
    if user_id:
        params["userid"] = user_id
    if exam_id:
        params["examid"] = exam_id
    
    return _make_request("GET", endpoint, params=params)

def schedule_exam(instructor_id: str, exam_id: str, user_id: str) -> Dict:
    """
    Schedule an exam for a student.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        exam_id: The ID of the exam to schedule
        user_id: The user ID of the student
    
    🔧 READY FOR TESTING
    """
    # First check if student is already scheduled for this exam
    scheduled_result = list_scheduled_exams(instructor_id, user_id=user_id, exam_id=exam_id)
    if scheduled_result.get("status") and scheduled_result.get("scheduled_exams"):
        # Student is already scheduled
        return {
            "status": False,
            "message": "This student is already scheduled to take this exam.",
            "returnCode": "STUDENT_ALREADY_SCHEDULED",
            "already_scheduled": True
        }
    
    endpoint = f"instructor/{instructor_id}/student/exam/{exam_id}/schedule.json"
    
    # Try the official API documentation format first: "userId" (capital I)
    data = {"userId": user_id}
    result = _make_request("POST", endpoint, data=data)
    
    # If that fails with a specific error about the parameter, try the lowercase version
    if "error" in result and ("userId" in str(result.get("error", "")).lower() or "parameter" in str(result.get("error", "")).lower()):
        print(f"⚠️  Trying alternative parameter format: userid (lowercase)")
        data = {"userid": user_id}  # Try lowercase version
        result = _make_request("POST", endpoint, data=data)
    
    # Handle specific error cases based on API documentation
    if "error" in result:
        error_msg = result["error"]
        if "STUDENT_ALREADY_SCHEDULED" in error_msg or "already scheduled" in error_msg.lower():
            return {
                "status": False,
                "message": "This student is already scheduled to take this exam.",
                "returnCode": "STUDENT_ALREADY_SCHEDULED",
                "already_scheduled": True
            }
        elif "INVALID_INSTRUCTOR" in error_msg:
            return {
                "status": False,
                "message": "The Instructor ID was invalid",
                "returnCode": "INVALID_INSTRUCTOR"
            }
        elif "ROUTE_PERMISSION_ERROR" in error_msg:
            return {
                "status": False,
                "message": "The Super User of this account did not grant permission to access this resource",
                "returnCode": "ROUTE_PERMISSION_ERROR"
            }
        elif "API_AUTHENTICATION_FAILED" in error_msg:
            return {
                "status": False,
                "message": "The API Key and API Secret combination was invalid",
                "returnCode": "API_AUTHENTICATION_FAILED"
            }
        else:
            return {
                "status": False,
                "message": f"Failed to schedule exam: {error_msg}",
                "returnCode": "UNKNOWN_ERROR"
            }
    
    # If no error, the request was successful
    return {
        "status": True,
        "message": "Exam scheduled successfully",
        "data": result
    }

def get_exam_attempt(instructor_id: str, user_exam_id: str) -> Dict:
    """
    Get details of a specific exam attempt.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        user_exam_id: The user exam ID from list_scheduled_exams
    
    🔧 READY FOR TESTING
    """
    endpoint = f"instructor/{instructor_id}/student/userexam/{user_exam_id}/attempt.json"
    return _make_request("GET", endpoint)

def get_student_exam_statistics(instructor_id: str, student_id: str, user_exam_id: str) -> Dict:
    """
    Get exam statistics for a specific student and exam.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        student_id: The ID of the student
        user_exam_id: The user exam ID (from list_scheduled_exams)
    
    🔧 READY FOR TESTING
    """
    endpoint = f"instructor/{instructor_id}/student/{student_id}/userexam/{user_exam_id}/stats.json"
    return _make_request("GET", endpoint)

def unschedule_exam(instructor_id: str, user_exam_id: str) -> Dict:
    """
    Unschedule an exam (delete exam attempt) for a specific student.
    
    This request can delete:
    - An exam that has not been started yet
    - An exam that has been started but not completed
    - A completed exam
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        user_exam_id: The user exam ID (can be obtained by using the List Scheduled Exams request)
    
    🔧 READY FOR TESTING
    """
    endpoint = f"instructor/{instructor_id}/student/userexam/{user_exam_id}/unschedule.json"
    return _make_request("DELETE", endpoint)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def search_student_by_student_id(instructor_id: str, student_id: str) -> Dict:
    """
    Search for a student by their Student ID (email) and return their User ID.
    
    Args:
        instructor_id: The instructor ID from get_instructor_id()
        student_id: The student's email address (Student ID)
    
    Returns:
        Dict with student info and User ID if found
    
    🔧 READY FOR TESTING
    """
    # Use list_students to search for the student
    result = list_students(instructor_id, student_id=student_id)
    
    if result.get("status"):
        students = result.get("students", []) or result.get("student_list", [])
        for student in students:
            if student.get("STUDENTID", "").lower() == student_id.lower():
                return {
                    "status": True,
                    "student": student,
                    "found": True,
                    "user_id": student.get("USERID"),
                    "student_id": student.get("STUDENTID")
                }
    
    return {
        "status": True,
        "student": None,
        "found": False,
        "user_id": None,
        "student_id": None
    } 