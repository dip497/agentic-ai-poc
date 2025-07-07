"""
Built-in actions for Moveworks-style conversational processes.

This module implements Moveworks-style built-in actions using LangChain:
- generate_text_action (equivalent to mw.generate_text_action)
- generate_structured_value_action (equivalent to mw.generate_structured_value_action)
- get_user_by_email (equivalent to mw.get_user_by_email)
- send_chat_notification (equivalent to mw.send_plaintext_chat_notification)
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from ..models.moveworks import (
    GenerateTextActionInput, GenerateStructuredValueActionInput,
    GetUserByEmailInput, SendChatNotificationInput, ActivityResult
)
from ..llm.llm_factory import LLMFactory


logger = logging.getLogger(__name__)


class MoveworksBuiltinActions:
    """
    Built-in actions that replicate Moveworks functionality using LangChain.
    
    These actions provide the core LLM-powered capabilities that Moveworks
    offers as built-in actions in their platform.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize built-in actions."""
        if llm_config is None:
            llm_config = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000
            }

        self.llm = LLMFactory.create_llm(**llm_config)
        
        # Mock user database (in production, this would be a real user service)
        self.user_database = {
            "john.doe@company.com": {
                "id": "user_001",
                "full_name": "John Doe",
                "email_addr": "john.doe@company.com",
                "department": "Engineering",
                "role": "Senior Developer",
                "manager": "jane.smith@company.com",
                "location": "San Francisco",
                "employee_id": "EMP001"
            },
            "jane.smith@company.com": {
                "id": "user_002",
                "full_name": "Jane Smith",
                "email_addr": "jane.smith@company.com",
                "department": "Engineering",
                "role": "Engineering Manager",
                "manager": "bob.wilson@company.com",
                "location": "San Francisco",
                "employee_id": "EMP002"
            }
        }
    
    async def generate_text_action(self, input_data: GenerateTextActionInput) -> ActivityResult:
        """
        Generate text using LLM (equivalent to mw.generate_text_action).
        
        Args:
            input_data: Input parameters for text generation
            
        Returns:
            ActivityResult with generated text
        """
        try:
            # Create prompt template
            if input_data.system_prompt:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", input_data.system_prompt),
                    ("human", input_data.user_input)
                ])
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("human", input_data.user_input)
                ])
            
            # Create chain with specified model
            llm = ChatOpenAI(
                model=input_data.model,
                temperature=0.1,
                max_tokens=2000
            )
            
            chain = prompt | llm
            
            # Execute generation
            result = await chain.ainvoke({})
            
            # Format response to match Moveworks structure
            response_data = {
                "openai_chat_completions_response": {
                    "choices": [
                        {
                            "message": {
                                "content": result.content
                            }
                        }
                    ]
                }
            }
            
            return ActivityResult(
                success=True,
                output_data=response_data,
                requires_user_input=False
            )
        
        except Exception as e:
            logger.error(f"Error in generate_text_action: {e}")
            return ActivityResult(
                success=False,
                error_message=f"Text generation failed: {str(e)}",
                requires_user_input=False
            )
    
    async def generate_structured_value_action(self, input_data: GenerateStructuredValueActionInput) -> ActivityResult:
        """
        Generate structured output using LLM (equivalent to mw.generate_structured_value_action).
        
        Args:
            input_data: Input parameters for structured generation
            
        Returns:
            ActivityResult with structured output
        """
        try:
            # Create JSON output parser with the provided schema
            json_parser = JsonOutputParser()
            
            # Create prompt template
            if input_data.system_prompt:
                system_message = input_data.system_prompt
            else:
                system_message = f"""Extract structured information from the provided payload according to the JSON schema.

Output Schema: {json.dumps(input_data.output_schema, indent=2)}

Return only valid JSON that matches the schema exactly."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", f"Payload to analyze: {json.dumps(input_data.payload, indent=2)}")
            ])
            
            # Create chain
            llm = ChatOpenAI(
                model=input_data.model,
                temperature=0.1,
                max_tokens=2000
            )
            
            chain = prompt | llm | json_parser
            
            # Execute generation
            result = await chain.ainvoke({})
            
            # Format response to match Moveworks structure
            response_data = {
                "openai_chat_completions_response": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(result)
                            }
                        }
                    ]
                }
            }
            
            return ActivityResult(
                success=True,
                output_data=response_data,
                requires_user_input=False
            )
        
        except Exception as e:
            logger.error(f"Error in generate_structured_value_action: {e}")
            return ActivityResult(
                success=False,
                error_message=f"Structured generation failed: {str(e)}",
                requires_user_input=False
            )
    
    async def get_user_by_email(self, input_data: GetUserByEmailInput) -> ActivityResult:
        """
        Get user information by email (equivalent to mw.get_user_by_email).
        
        Args:
            input_data: Input parameters with user email
            
        Returns:
            ActivityResult with user information
        """
        try:
            user_email = input_data.user_email.lower()
            
            if user_email in self.user_database:
                user_data = self.user_database[user_email]
                
                response_data = {
                    "user": user_data
                }
                
                return ActivityResult(
                    success=True,
                    output_data=response_data,
                    requires_user_input=False
                )
            else:
                return ActivityResult(
                    success=False,
                    error_message=f"User not found: {input_data.user_email}",
                    requires_user_input=False
                )
        
        except Exception as e:
            logger.error(f"Error in get_user_by_email: {e}")
            return ActivityResult(
                success=False,
                error_message=f"User lookup failed: {str(e)}",
                requires_user_input=False
            )
    
    async def send_chat_notification(self, input_data: SendChatNotificationInput) -> ActivityResult:
        """
        Send chat notification (equivalent to mw.send_plaintext_chat_notification).
        
        Args:
            input_data: Input parameters for notification
            
        Returns:
            ActivityResult with notification status
        """
        try:
            # In a real implementation, this would integrate with chat platforms
            # For demo purposes, we'll simulate sending a notification
            
            logger.info(f"Sending notification to user {input_data.user_record_id}: {input_data.message}")
            
            response_data = {
                "notification_id": f"notif_{input_data.user_record_id}_{asyncio.get_event_loop().time()}",
                "status": "sent",
                "recipient": input_data.user_record_id,
                "message": input_data.message
            }
            
            return ActivityResult(
                success=True,
                output_data=response_data,
                requires_user_input=False
            )
        
        except Exception as e:
            logger.error(f"Error in send_chat_notification: {e}")
            return ActivityResult(
                success=False,
                error_message=f"Notification failed: {str(e)}",
                requires_user_input=False
            )
    
    async def classify_text(self, text: str, categories: List[str], system_prompt: Optional[str] = None) -> ActivityResult:
        """
        Classify text into predefined categories using LLM.
        
        Args:
            text: Text to classify
            categories: Available categories
            system_prompt: Optional system prompt
            
        Returns:
            ActivityResult with classification
        """
        try:
            if not system_prompt:
                system_prompt = f"""Classify the following text into one of these categories: {', '.join(categories)}

Return only the category name, nothing else."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", text)
            ])
            
            chain = prompt | self.llm
            result = await chain.ainvoke({})
            
            classification = result.content.strip()
            
            # Validate classification
            if classification not in categories:
                # Try to find closest match
                classification = self._find_closest_category(classification, categories)
            
            response_data = {
                "classification": classification,
                "confidence": 0.8,  # In a real implementation, this would be calculated
                "available_categories": categories
            }
            
            return ActivityResult(
                success=True,
                output_data=response_data,
                requires_user_input=False
            )
        
        except Exception as e:
            logger.error(f"Error in classify_text: {e}")
            return ActivityResult(
                success=False,
                error_message=f"Text classification failed: {str(e)}",
                requires_user_input=False
            )
    
    def _find_closest_category(self, classification: str, categories: List[str]) -> str:
        """Find the closest matching category."""
        classification_lower = classification.lower()
        
        for category in categories:
            if category.lower() in classification_lower or classification_lower in category.lower():
                return category
        
        # Default to first category if no match
        return categories[0] if categories else "unknown"
    
    async def execute_builtin_action(self, action_name: str, input_data: Dict[str, Any]) -> ActivityResult:
        """
        Execute a built-in action by name.
        
        Args:
            action_name: Name of the action to execute
            input_data: Input data for the action
            
        Returns:
            ActivityResult with execution results
        """
        try:
            if action_name == "generate_text_action":
                input_obj = GenerateTextActionInput(**input_data)
                return await self.generate_text_action(input_obj)
            
            elif action_name == "generate_structured_value_action":
                input_obj = GenerateStructuredValueActionInput(**input_data)
                return await self.generate_structured_value_action(input_obj)
            
            elif action_name == "get_user_by_email":
                input_obj = GetUserByEmailInput(**input_data)
                return await self.get_user_by_email(input_obj)
            
            elif action_name == "send_chat_notification":
                input_obj = SendChatNotificationInput(**input_data)
                return await self.send_chat_notification(input_obj)
            
            elif action_name == "classify_text":
                return await self.classify_text(
                    text=input_data.get("text", ""),
                    categories=input_data.get("categories", []),
                    system_prompt=input_data.get("system_prompt")
                )
            
            else:
                return ActivityResult(
                    success=False,
                    error_message=f"Unknown built-in action: {action_name}",
                    requires_user_input=False
                )
        
        except Exception as e:
            logger.error(f"Error executing built-in action {action_name}: {e}")
            return ActivityResult(
                success=False,
                error_message=f"Action execution failed: {str(e)}",
                requires_user_input=False
            )


# Example usage and testing
async def test_builtin_actions():
    """Test the built-in actions."""
    actions = MoveworksBuiltinActions()
    
    print("Testing Built-in Actions:")
    print("=" * 50)
    
    # Test generate_text_action
    print("1. Testing generate_text_action:")
    text_input = GenerateTextActionInput(
        system_prompt="You are a helpful assistant that classifies purchase requests as Opex or Capex.",
        user_input="I need to buy 100 pens for the office",
        model="gpt-4o-mini"
    )
    
    result = await actions.generate_text_action(text_input)
    print(f"Success: {result.success}")
    if result.success:
        content = result.output_data["openai_chat_completions_response"]["choices"][0]["message"]["content"]
        print(f"Generated text: {content}")
    print("-" * 30)
    
    # Test get_user_by_email
    print("2. Testing get_user_by_email:")
    user_input = GetUserByEmailInput(user_email="john.doe@company.com")
    
    result = await actions.get_user_by_email(user_input)
    print(f"Success: {result.success}")
    if result.success:
        user_data = result.output_data["user"]
        print(f"User found: {user_data['full_name']} ({user_data['role']})")
    print("-" * 30)
    
    # Test generate_structured_value_action
    print("3. Testing generate_structured_value_action:")
    structured_input = GenerateStructuredValueActionInput(
        payload={"abstract": "This paper discusses machine learning techniques for natural language processing."},
        output_schema={
            "type": "object",
            "properties": {
                "topic_tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["topic_tags"]
        },
        system_prompt="Extract topic tags from the research paper abstract."
    )
    
    result = await actions.generate_structured_value_action(structured_input)
    print(f"Success: {result.success}")
    if result.success:
        content = result.output_data["openai_chat_completions_response"]["choices"][0]["message"]["content"]
        print(f"Structured output: {content}")
    print("-" * 30)


if __name__ == "__main__":
    asyncio.run(test_builtin_actions())
