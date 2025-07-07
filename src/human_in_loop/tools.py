"""
Human-in-the-Loop Tools for LangGraph Integration.

This module provides tools that integrate with LangGraph's interrupt mechanism
to enable human-in-the-loop functionality in the Moveworks AI system.
"""

import logging
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

from langchain_core.tools import tool
from langgraph.types import interrupt


logger = logging.getLogger(__name__)


@tool
def request_user_confirmation(
    action_description: str,
    importance: Literal["low", "medium", "high"] = "medium",
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 300
) -> str:
    """
    Request user confirmation for an action.
    
    This tool pauses the agent execution and waits for user confirmation
    through the AG-UI interface.
    
    Args:
        action_description: Description of the action requiring confirmation
        importance: Importance level of the confirmation
        context: Additional context for the confirmation
        timeout_seconds: Timeout for user response
    
    Returns:
        User's confirmation response
    """
    confirmation_data = {
        "type": "confirmation_request",
        "action": action_description,
        "importance": importance,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
        "timeout_seconds": timeout_seconds
    }
    
    logger.info(f"Requesting user confirmation: {action_description}")
    
    # Use LangGraph's interrupt to pause execution
    user_response = interrupt(value=confirmation_data)
    
    # Process the response
    if isinstance(user_response, dict):
        confirmed = user_response.get("confirmed", False)
        response_text = user_response.get("response", "")
    else:
        # Handle string responses
        response_lower = str(user_response).lower().strip()
        confirmed = any(word in response_lower for word in ["yes", "y", "confirm", "approve", "ok", "proceed"])
        response_text = str(user_response)
    
    result = "confirmed" if confirmed else "rejected"
    logger.info(f"User confirmation result: {result}")
    
    return f"User {result} the action: {action_description}. Response: {response_text}"


@tool
def request_slot_clarification(
    slot_name: str,
    clarification_message: str,
    options: Optional[List[str]] = None,
    slot_type: str = "text",
    required: bool = True
) -> str:
    """
    Request clarification for a slot value from the user.
    
    This tool pauses execution to get clarification on slot values
    that are missing or ambiguous.
    
    Args:
        slot_name: Name of the slot requiring clarification
        clarification_message: Message to display to the user
        options: Optional list of predefined options
        slot_type: Type of slot (text, number, date, etc.)
        required: Whether the slot is required
    
    Returns:
        Clarified slot value from user
    """
    clarification_data = {
        "type": "slot_clarification",
        "slot_name": slot_name,
        "message": clarification_message,
        "options": options or [],
        "slot_type": slot_type,
        "required": required,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Requesting slot clarification for: {slot_name}")
    
    # Use LangGraph's interrupt to pause execution
    user_response = interrupt(value=clarification_data)
    
    # Extract the clarified value
    if isinstance(user_response, dict):
        clarified_value = user_response.get("value", user_response.get("response", ""))
    else:
        clarified_value = str(user_response)
    
    logger.info(f"Slot {slot_name} clarified with value: {clarified_value}")
    
    return f"Slot '{slot_name}' clarified with value: {clarified_value}"


@tool
def request_user_input(
    prompt: str,
    input_type: Literal["text", "number", "email", "phone", "date"] = "text",
    validation_rules: Optional[Dict[str, Any]] = None,
    placeholder: Optional[str] = None
) -> str:
    """
    Request general input from the user.
    
    This tool can be used to collect any type of input from the user
    with optional validation rules.
    
    Args:
        prompt: Prompt to display to the user
        input_type: Type of input expected
        validation_rules: Optional validation rules
        placeholder: Placeholder text for the input
    
    Returns:
        User's input
    """
    input_data = {
        "type": "user_input_request",
        "prompt": prompt,
        "input_type": input_type,
        "validation_rules": validation_rules or {},
        "placeholder": placeholder,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Requesting user input: {prompt}")
    
    # Use LangGraph's interrupt to pause execution
    user_response = interrupt(value=input_data)
    
    # Extract the input value
    if isinstance(user_response, dict):
        input_value = user_response.get("value", user_response.get("response", ""))
    else:
        input_value = str(user_response)
    
    logger.info(f"User provided input: {input_value}")
    
    return f"User input received: {input_value}"


@tool
def request_choice_selection(
    question: str,
    choices: List[str],
    allow_multiple: bool = False,
    required: bool = True
) -> str:
    """
    Request the user to select from a list of choices.
    
    Args:
        question: Question to ask the user
        choices: List of available choices
        allow_multiple: Whether multiple selections are allowed
        required: Whether a selection is required
    
    Returns:
        User's selected choice(s)
    """
    choice_data = {
        "type": "choice_selection",
        "question": question,
        "choices": choices,
        "allow_multiple": allow_multiple,
        "required": required,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Requesting choice selection: {question}")
    
    # Use LangGraph's interrupt to pause execution
    user_response = interrupt(value=choice_data)
    
    # Extract the selected choice(s)
    if isinstance(user_response, dict):
        selected = user_response.get("selected", user_response.get("response", ""))
    else:
        selected = str(user_response)
    
    logger.info(f"User selected: {selected}")
    
    return f"User selected: {selected}"


@tool
def show_information_and_wait(
    title: str,
    message: str,
    information_type: Literal["info", "warning", "error", "success"] = "info",
    auto_dismiss_seconds: Optional[int] = None
) -> str:
    """
    Show information to the user and optionally wait for acknowledgment.
    
    Args:
        title: Title of the information message
        message: Information message to display
        information_type: Type of information (affects styling)
        auto_dismiss_seconds: Auto-dismiss after seconds (None = wait for user)
    
    Returns:
        Acknowledgment from user
    """
    info_data = {
        "type": "information_display",
        "title": title,
        "message": message,
        "information_type": information_type,
        "auto_dismiss_seconds": auto_dismiss_seconds,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Showing information to user: {title}")
    
    if auto_dismiss_seconds:
        # Don't wait for user response, just show the message
        return f"Information displayed: {title}"
    else:
        # Wait for user acknowledgment
        user_response = interrupt(value=info_data)
        return f"User acknowledged: {title}"


@tool
def request_file_upload(
    prompt: str,
    accepted_file_types: Optional[List[str]] = None,
    max_file_size_mb: int = 10,
    required: bool = True
) -> str:
    """
    Request the user to upload a file.
    
    Args:
        prompt: Prompt for file upload
        accepted_file_types: List of accepted file extensions
        max_file_size_mb: Maximum file size in MB
        required: Whether file upload is required
    
    Returns:
        Information about uploaded file
    """
    upload_data = {
        "type": "file_upload_request",
        "prompt": prompt,
        "accepted_file_types": accepted_file_types or [],
        "max_file_size_mb": max_file_size_mb,
        "required": required,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Requesting file upload: {prompt}")
    
    # Use LangGraph's interrupt to pause execution
    user_response = interrupt(value=upload_data)
    
    # Extract file information
    if isinstance(user_response, dict):
        file_info = user_response.get("file_info", {})
        filename = file_info.get("filename", "unknown")
        file_size = file_info.get("size", 0)
    else:
        filename = "uploaded_file"
        file_size = 0
    
    logger.info(f"File uploaded: {filename}")
    
    return f"File uploaded: {filename} ({file_size} bytes)"


# Collection of all human-in-the-loop tools
HUMAN_IN_LOOP_TOOLS = [
    request_user_confirmation,
    request_slot_clarification,
    request_user_input,
    request_choice_selection,
    show_information_and_wait,
    request_file_upload
]


def get_human_in_loop_tools() -> List:
    """Get all human-in-the-loop tools for LangGraph integration."""
    return HUMAN_IN_LOOP_TOOLS
