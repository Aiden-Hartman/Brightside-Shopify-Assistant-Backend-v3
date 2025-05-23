from fastapi import APIRouter, HTTPException
from typing import List
import logging
import os
from openai import AsyncOpenAI
from models import IntentClassificationRequest, IntentClassificationResponse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Qdrant client for intent classification
qdrant_url = os.getenv("QDRANT_URL_2")
# Remove any existing https:// or http:// prefix and add https://
qdrant_url = qdrant_url.replace("https://", "").replace("http://", "")
qdrant_url = f"https://{qdrant_url}"

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=os.getenv("QDRANT_API_KEY_2")
)

# Get collection name from environment variable
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME_2", "brightside-gpt-context-prompts-5-22-2025")

@router.post("/classify-intent", response_model=IntentClassificationResponse)
async def classify_intent(request: IntentClassificationRequest) -> IntentClassificationResponse:
    """
    Classify the user's message into an intent using vector similarity search.
    """
    logger.debug("="*50)
    logger.debug("INTENT CLASSIFICATION REQUEST RECEIVED")
    logger.debug("="*50)

    try:
        # Validate input
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Generate embedding for the message
        logger.debug("Generating message embedding...")
        try:
            embedding_response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=request.message
            )
            query_vector = embedding_response.data[0].embedding
            logger.debug(f"Successfully generated embedding of length {len(query_vector)}")
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate message embedding")

        # Query Qdrant for similar intents
        logger.debug("Querying Qdrant for similar intents...")
        try:
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=request.limit,
                score_threshold=request.min_similarity_threshold
            )
            logger.debug(f"Found {len(search_results)} matching intents")
        except Exception as e:
            logger.error(f"Error querying Qdrant: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to query intent database")

        # If no matches found above threshold, return generic intent
        if not search_results:
            logger.info("No matches found above threshold, returning generic intent")
            return IntentClassificationResponse(
                intent_id=0,
                title="Generic",
                prompt="Generic response",
                example_queries=[],
                required_context=[],
                similarity_score=0.0
            )

        # Get the best match
        best_match = search_results[0]
        logger.debug(f"Best match score: {best_match.score}")
        logger.debug(f"Best match payload: {best_match.payload}")

        # Create response
        response = IntentClassificationResponse(
            intent_id=best_match.payload.get("intent_id", 0),
            title=best_match.payload.get("title", ""),
            prompt=best_match.payload.get("prompt", ""),
            example_queries=best_match.payload.get("example_queries", []),
            required_context=best_match.payload.get("required_context", []),
            similarity_score=best_match.score
        )

        logger.debug("Created response:")
        logger.debug(f"- Intent ID: {response.intent_id}")
        logger.debug(f"- Title: {response.title}")
        logger.debug(f"- Similarity Score: {response.similarity_score}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in classify_intent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") 