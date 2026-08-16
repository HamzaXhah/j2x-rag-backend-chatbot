"""
AI Integration endpoints that match the j2x-backend/frontend API contract.

These endpoints allow the existing j2x system to use this RAG microservice
without changing their frontend or backend code — just point AI_BASE_URL here.

Endpoints:
  POST /ai/chat                                         — Streaming chat (frontend calls this)
  POST /ai/companies/{id}/text-based-file/embeddings    — Create embeddings from file URL
  DELETE /ai/companies/{id}/embeddings/{file_name}      — Delete embeddings for a file
"""

import logging
import httpx
import os
import tempfile
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.vector_store import vector_store
from app.services.retrieval.retriever import rag_retriever
from app.services.retrieval.generator import rag_generator
from app.services.document_processor import document_processor
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str
    company_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class EmbeddingsRequest(BaseModel):
    file_src: str
    file_name: str


@router.post("/chat")
async def ai_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Streaming chat endpoint that matches the frontend contract.
    Frontend calls: POST /ai/chat with {query, company_id}
    Returns a streaming text response.
    """
    company_id = request.company_id
    collection_name = f"company_{company_id}" if company_id else "documents"

    async def generate_stream():
        try:
            collection_names = [collection_name]
            existing_collections = list(settings.COLLECTIONS.keys())
            if collection_name not in existing_collections:
                try:
                    vector_store.get_collection(collection_name)
                except Exception:
                    collection_names = existing_collections

            result = rag_generator.generate_response(
                query=request.query,
                collection_names=collection_names,
                conversation_history=request.conversation_history,
                db=db,
            )

            answer = result.get("answer", "I couldn't find relevant information to answer your question.")
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                yield answer[i:i + chunk_size]

        except Exception as e:
            logger.error(f"Error in ai_chat stream: {e}")
            yield f"Error processing your question: {str(e)}"

    return StreamingResponse(generate_stream(), media_type="text/plain")


@router.post("/companies/{company_id}/text-based-file/embeddings")
async def create_embeddings(
    company_id: int,
    request: EmbeddingsRequest,
    db: Session = Depends(get_db),
):
    """
    Create embeddings from a file URL.
    j2x-backend calls this when files are uploaded to the Data Room.

    It downloads the file from the provided URL (DigitalOcean Spaces),
    processes it, and stores embeddings in a company-scoped collection.
    """
    file_src = request.file_src
    file_name = request.file_name
    collection_name = f"company_{company_id}"

    logger.info(
        "Creating embeddings for company %s, file: %s, source: %s",
        company_id, file_name, file_src,
    )

    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(file_src)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not download file from {file_src}: HTTP {response.status_code}",
                )
            file_content = response.content

        extension = os.path.splitext(file_name)[1].lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        mime_type = mime_map.get(extension, "application/octet-stream")

        from app.services.parsers.factory import parser_factory
        from langchain.schema import Document
        from io import BytesIO
        import uuid

        file_obj = BytesIO(file_content)
        parser = parser_factory.get_parser(mime_type)
        if not parser:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {extension}",
            )

        metadata = {
            "filename": file_name,
            "mime_type": mime_type,
            "company_id": str(company_id),
            "source_url": file_src,
        }
        parsed_documents = parser.parse(file_obj, metadata)

        if not parsed_documents:
            raise HTTPException(
                status_code=400,
                detail=f"No content could be extracted from {file_name}",
            )

        doc_id = str(uuid.uuid4())
        for doc in parsed_documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["document_id"] = doc_id
            doc.metadata["company_id"] = str(company_id)
            doc.metadata["file_name"] = file_name
            doc.metadata["source_url"] = file_src

        vector_ids = vector_store.add_documents(
            documents=parsed_documents,
            collection_name=collection_name,
        )

        logger.info(
            "Successfully created %d embeddings for company %s, file: %s",
            len(vector_ids), company_id, file_name,
        )

        return {
            "status": "success",
            "document_id": doc_id,
            "collection": collection_name,
            "chunks_created": len(vector_ids),
            "file_name": file_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating embeddings: {str(e)}",
        )


@router.delete("/companies/{company_id}/embeddings/{file_name}")
async def delete_embeddings(
    company_id: int,
    file_name: str,
    db: Session = Depends(get_db),
):
    """
    Delete embeddings for a specific file from a company's collection.
    j2x-backend calls this when files are deleted from the Data Room.
    """
    collection_name = f"company_{company_id}"

    logger.info(
        "Deleting embeddings for company %s, file: %s",
        company_id, file_name,
    )

    try:
        collection = vector_store.get_collection(collection_name)
        results = collection.similarity_search(
            query="",
            k=1000,
            filter={"file_name": file_name},
        )

        if results:
            ids_to_delete = [
                doc.metadata.get("id") for doc in results
                if doc.metadata.get("id")
            ]
            if ids_to_delete:
                vector_store.delete(ids=ids_to_delete, collection_name=collection_name)
                logger.info("Deleted %d vectors for file %s", len(ids_to_delete), file_name)

        return {
            "status": "success",
            "message": f"Embeddings deleted for {file_name}",
            "company_id": company_id,
        }

    except Exception as e:
        logger.error(f"Error deleting embeddings: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting embeddings: {str(e)}",
        )
