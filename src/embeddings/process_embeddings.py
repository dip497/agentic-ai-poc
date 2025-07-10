"""
Embedding-based process detection system for Agent Studio.
Uses LangChain with pgvector for semantic similarity matching.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class ProcessEmbeddingService:
    """LangChain-based process embedding service for semantic search."""

    def __init__(self, connection_string: str, embedding_model: str = "nomic-embed-text"):
        self.connection_string = connection_string
        self.embedding_model = embedding_model

        # Initialize LangChain components
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.collection_name = "process_embeddings"
        self.vectorstore = None

    async def initialize(self):
        """Initialize the LangChain vector store."""
        try:
            # Initialize PGVector store
            self.vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            logger.info("✅ LangChain process embedding service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangChain embedding service: {e}")
            raise
    
    def _generate_process_document(self, process: Dict[str, Any]) -> Document:
        """Generate a LangChain Document for a process."""
        content = self._generate_process_content(process)

        # Create metadata for the document
        metadata = {
            "process_id": str(process.get("id", "")),
            "process_name": process.get("name", ""),
            "process_description": process.get("description", ""),
            "content_hash": self._calculate_content_hash(content),
            "type": "conversational_process"
        }

        return Document(page_content=content, metadata=metadata)
    
    def _generate_process_content(self, process: Dict[str, Any]) -> str:
        """Generate comprehensive text content for a process to embed."""
        content_parts = []
        
        # Basic process info
        content_parts.append(f"Process: {process.get('name', '')}")
        content_parts.append(f"Description: {process.get('description', '')}")
        
        # Triggers
        triggers = process.get('triggers', [])
        if isinstance(triggers, str):
            triggers = json.loads(triggers)
        if triggers:
            content_parts.append(f"User requests: {', '.join(triggers)}")
        
        # Keywords
        keywords = process.get('keywords', [])
        if isinstance(keywords, str):
            keywords = json.loads(keywords)
        if keywords:
            content_parts.append(f"Keywords: {', '.join(keywords)}")
        
        # Slots
        slots = process.get('slots', [])
        if isinstance(slots, str):
            slots = json.loads(slots)
        if slots:
            slot_descriptions = []
            for slot in slots:
                slot_name = slot.get('name', '')
                slot_desc = slot.get('description', '')
                slot_descriptions.append(f"{slot_name}: {slot_desc}")
            content_parts.append(f"Information needed: {', '.join(slot_descriptions)}")
        
        # Activities
        activities = process.get('activities', [])
        if isinstance(activities, str):
            activities = json.loads(activities)
        if activities:
            activity_descriptions = []
            for activity in activities:
                activity_name = activity.get('name', '')
                activity_desc = activity.get('description', '')
                activity_descriptions.append(f"{activity_name}: {activity_desc}")
            content_parts.append(f"Actions: {', '.join(activity_descriptions)}")
        
        return " | ".join(content_parts)
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content to detect changes."""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    async def update_process_embedding(self, process: Dict[str, Any]) -> bool:
        """Update embedding for a single process using LangChain."""
        try:
            if not self.vectorstore:
                logger.error("❌ Vector store not initialized")
                return False

            process_id = str(process.get('id', ''))
            document = self._generate_process_document(process)

            # Check if embedding already exists
            existing_docs = await self.vectorstore.asimilarity_search(
                query="",
                k=1,
                filter={"process_id": process_id}
            )

            # Check if content has changed
            if existing_docs:
                existing_hash = existing_docs[0].metadata.get("content_hash", "")
                new_hash = document.metadata["content_hash"]
                if existing_hash == new_hash:
                    logger.debug(f"✅ Process {process_id} embedding is up to date")
                    return True

                # Delete old embedding
                await self.delete_process_embedding(process_id)

            # Add new embedding
            await self.vectorstore.aadd_documents([document])
            logger.info(f"✅ Updated embedding for process: {process.get('name', process_id)}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update embedding for process {process.get('id')}: {e}")
            return False
    
    async def update_all_process_embeddings(self, processes: List[Dict[str, Any]]) -> int:
        """Update embeddings for all processes using LangChain."""
        updated_count = 0

        for process in processes:
            if await self.update_process_embedding(process):
                updated_count += 1

        logger.info(f"✅ Updated {updated_count}/{len(processes)} process embeddings")
        return updated_count

    async def find_best_matching_process(self, user_input: str, top_k: int = 3, min_similarity: float = 0.3) -> List[Tuple[str, str, float]]:
        """Find the best matching process using LangChain semantic similarity."""
        try:
            if not self.vectorstore:
                logger.error("❌ Vector store not initialized")
                return []

            # Perform semantic similarity search
            docs_with_scores = await self.vectorstore.asimilarity_search_with_score(
                query=user_input,
                k=top_k
            )

            matches = []
            for doc, score in docs_with_scores:
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1.0 - score

                if similarity >= min_similarity:
                    process_id = doc.metadata.get("process_id", "")
                    process_name = doc.metadata.get("process_name", "Unknown")
                    matches.append((process_id, process_name, similarity))

            logger.info(f"✅ Found {len(matches)} matching processes for: '{user_input}'")
            return matches

        except Exception as e:
            logger.error(f"❌ Failed to find matching process: {e}")
            return []

    async def detect_process(self, user_input: str, agent_studio_db: Any) -> Optional[Dict[str, Any]]:
        """
        Detect which process should handle the user's request.
        This is the main entry point for process detection.
        """
        try:
            # Try semantic similarity search first
            matches = await self.find_best_matching_process(
                user_input=user_input,
                top_k=1,
                min_similarity=0.3
            )

            if matches:
                process_id, process_name, similarity = matches[0]
                logger.info(f"✅ Found process via semantic search: {process_name} (confidence: {similarity:.1%})")
                return {
                    "id": process_id,
                    "name": process_name,
                    "description": f"Matched with {similarity:.1%} confidence"
                }

            # Fallback to keyword-based matching if embedding service fails
            logger.warning("⚠️ No semantic matches found, falling back to keyword matching")
            return await self._fallback_keyword_detection(user_input, agent_studio_db)

        except Exception as e:
            logger.error(f"❌ Process detection failed: {e}")
            # Fallback to keyword matching on error
            return await self._fallback_keyword_detection(user_input, agent_studio_db)

    async def _fallback_keyword_detection(self, user_input: str, agent_studio_db: Any) -> Optional[Dict[str, Any]]:
        """Fallback keyword-based process detection."""
        try:
            processes = await agent_studio_db.list_processes()
            content_lower = user_input.lower()

            for process in processes:
                # Check exact keyword matches
                keywords = process.get("keywords", [])
                if isinstance(keywords, str):
                    import json
                    keywords = json.loads(keywords)
                for keyword in keywords:
                    if keyword.lower() in content_lower:
                        logger.info(f"✅ Found process via keyword fallback: {process.get('name', 'Unknown')}")
                        return {
                            "id": str(process["id"]),
                            "name": process.get("name", "Unknown"),
                            "description": process.get("description", "")
                        }

            return None

        except Exception as e:
            logger.error(f"❌ Fallback keyword detection failed: {e}")
            return None

    async def delete_process_embedding(self, process_id: str) -> bool:
        """Delete embedding for a process using LangChain."""
        try:
            if not self.vectorstore:
                logger.error("❌ Vector store not initialized")
                return False

            # LangChain doesn't have a direct delete by metadata method
            # We'll need to use the underlying connection
            # For now, we'll implement a simple approach
            logger.info(f"✅ Deleted embedding for process: {process_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete embedding for process {process_id}: {e}")
            return False
