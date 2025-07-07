"""
LangChain-based process matching for Moveworks-style conversational processes.

This module uses LLM to semantically match user utterances to conversational processes
based on trigger utterances, similar to how Moveworks operates.
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.moveworks import ConversationalProcess, ProcessMatchResult
from .llm_factory import LLMFactory


logger = logging.getLogger(__name__)


class ProcessMatchOutput(BaseModel):
    """Structured output for process matching."""
    matched_process: str = Field(..., description="Name of the best matching process")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Explanation of why this process was selected")
    trigger_utterance: str = Field(..., description="Most similar trigger utterance")


class MoveworksProcessMatcher:
    """
    LangChain-based process matcher that semantically matches user utterances
    to conversational processes using LLM understanding.
    """

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the process matcher."""
        if llm_config is None:
            llm_config = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1000
            }

        self.llm = LLMFactory.create_llm(**llm_config)
        
        self.output_parser = PydanticOutputParser(pydantic_object=ProcessMatchOutput)
        
        # Process matching prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at matching user utterances to conversational processes in a Moveworks-style AI assistant.

Your task is to analyze a user's utterance and determine which conversational process best matches their intent based on the trigger utterances for each process.

Consider:
1. Semantic similarity between the user utterance and trigger utterances
2. Intent and purpose behind the user's request
3. Context and domain of the request
4. Exact phrase matching as well as conceptual similarity

Be precise in your matching and provide a confidence score. Only match if there's a clear semantic relationship.

{format_instructions}"""),
            ("human", """User utterance: "{user_utterance}"

Available conversational processes:
{processes_info}

Analyze the user utterance and determine the best matching process. If no process is a good match, set confidence to 0.0 and matched_process to "no_match".""")
        ])
    
    async def match_process(
        self, 
        user_utterance: str, 
        processes: List[ConversationalProcess],
        min_confidence: float = 0.6
    ) -> Optional[ProcessMatchResult]:
        """
        Match user utterance to the best conversational process.
        
        Args:
            user_utterance: User's input utterance
            processes: Available conversational processes
            min_confidence: Minimum confidence threshold
            
        Returns:
            ProcessMatchResult if match found, None otherwise
        """
        if not processes:
            return None
        
        try:
            # Prepare process information for the LLM
            processes_info = self._format_processes_for_llm(processes)
            
            # Create the chain
            chain = self.prompt_template | self.llm | self.output_parser
            
            # Execute the chain
            result = await chain.ainvoke({
                "user_utterance": user_utterance,
                "processes_info": processes_info,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            # Check confidence threshold
            if result.confidence < min_confidence or result.matched_process == "no_match":
                logger.info(f"No process matched with sufficient confidence. Best match: {result.matched_process} (confidence: {result.confidence})")
                return None
            
            # Return the match result
            return ProcessMatchResult(
                process_name=result.matched_process,
                confidence=result.confidence,
                trigger_utterance=result.trigger_utterance
            )
        
        except Exception as e:
            logger.error(f"Error in process matching: {e}")
            return None
    
    def _format_processes_for_llm(self, processes: List[ConversationalProcess]) -> str:
        """Format processes information for LLM consumption."""
        formatted_processes = []
        
        for process in processes:
            process_info = f"""
Process: {process.title}
Description: {process.description}
Trigger Utterances:
{chr(10).join(f"  - {utterance}" for utterance in process.trigger_utterances)}
"""
            formatted_processes.append(process_info.strip())
        
        return "\n\n".join(formatted_processes)
    
    async def batch_match_processes(
        self,
        utterances: List[str],
        processes: List[ConversationalProcess],
        min_confidence: float = 0.6
    ) -> List[Optional[ProcessMatchResult]]:
        """
        Match multiple utterances to processes in batch.
        
        Args:
            utterances: List of user utterances
            processes: Available conversational processes
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of ProcessMatchResult (None for no matches)
        """
        tasks = [
            self.match_process(utterance, processes, min_confidence)
            for utterance in utterances
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error matching utterance {i}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def explain_match(
        self,
        user_utterance: str,
        matched_process: ConversationalProcess,
        confidence: float
    ) -> str:
        """
        Generate an explanation for why a process was matched.
        
        Args:
            user_utterance: Original user utterance
            matched_process: The matched process
            confidence: Matching confidence
            
        Returns:
            Human-readable explanation
        """
        explanation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are explaining why a user utterance matched a specific conversational process. Be clear and concise."),
            ("human", """User said: "{user_utterance}"

Matched Process: {process_title}
Process Description: {process_description}
Trigger Utterances: {trigger_utterances}
Confidence: {confidence}

Explain in 1-2 sentences why this utterance matched this process.""")
        ])
        
        chain = explanation_prompt | self.llm
        
        try:
            result = await chain.ainvoke({
                "user_utterance": user_utterance,
                "process_title": matched_process.title,
                "process_description": matched_process.description,
                "trigger_utterances": ", ".join(matched_process.trigger_utterances),
                "confidence": confidence
            })
            
            return result.content
        
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Matched based on semantic similarity to process '{matched_process.title}' with {confidence:.1%} confidence."


class ProcessMatchingCache:
    """Simple cache for process matching results."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache."""
        self.cache: Dict[str, ProcessMatchResult] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def get(self, utterance: str) -> Optional[ProcessMatchResult]:
        """Get cached result."""
        if utterance in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(utterance)
            self.access_order.append(utterance)
            return self.cache[utterance]
        return None
    
    def put(self, utterance: str, result: ProcessMatchResult) -> None:
        """Cache a result."""
        if utterance in self.cache:
            # Update existing
            self.access_order.remove(utterance)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[utterance] = result
        self.access_order.append(utterance)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


# Example usage and testing
async def test_process_matcher():
    """Test the process matcher with sample data."""
    from ..models.moveworks import ConversationalProcess, Slot, DataType, SlotInferencePolicy
    
    # Sample processes
    pto_process = ConversationalProcess(
        title="get_pto_balance_process",
        description="Fetches the PTO balance for a given user",
        trigger_utterances=[
            "How can I check my current paid time off balance?",
            "Can I take PTO next week?",
            "I want to take PTO the first week of May",
            "Check PTO balance",
            "What is my remaining paid time off balance"
        ],
        slots=[
            Slot(
                name="pto_type",
                data_type=DataType.STRING,
                slot_description="Type of PTO balance to fetch",
                slot_inference_policy=SlotInferencePolicy.INFER_IF_AVAILABLE
            )
        ]
    )
    
    feature_request_process = ConversationalProcess(
        title="update_feature_request_process",
        description="This will help users update the status of an existing feature request",
        trigger_utterances=[
            "How do I update a feature request?",
            "Can you update a feature request?",
            "Change status of FR-12345 to Limited Preview",
            "update feature request",
            "Change feature request status"
        ]
    )
    
    processes = [pto_process, feature_request_process]
    
    # Test utterances
    test_utterances = [
        "What's my vacation balance?",
        "Update FR-123 to completed",
        "How do I reset my password?",  # Should not match
        "I need to check my time off"
    ]
    
    matcher = MoveworksProcessMatcher()
    
    print("Testing Process Matcher:")
    print("=" * 50)
    
    for utterance in test_utterances:
        result = await matcher.match_process(utterance, processes)
        
        if result:
            explanation = await matcher.explain_match(
                utterance, 
                next(p for p in processes if p.title == result.process_name),
                result.confidence
            )
            print(f"Utterance: '{utterance}'")
            print(f"Matched: {result.process_name} (confidence: {result.confidence:.2f})")
            print(f"Explanation: {explanation}")
        else:
            print(f"Utterance: '{utterance}'")
            print("No match found")
        
        print("-" * 30)


if __name__ == "__main__":
    asyncio.run(test_process_matcher())
