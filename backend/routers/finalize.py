"""
Finalize Router — Case Completion & PDF Payload Generation

POST /api/v1/action/finalize

Receives the user-submitted form data (extracted entities confirmed/edited
by the user), persists the case to MongoDB, and returns the structured
PDF payload for frontend document rendering.
"""

from uuid import uuid4
from fastapi import APIRouter, HTTPException

from models.requests import FinalizeRequest
from models.responses import FinalizeResponse
from db.mongodb import SessionStore
from services.output_formatter import build_pdf_payload
from models.domain import LegalDomain, IntentResult, RAGResult

router = APIRouter(prefix="/api/v1/action", tags=["AI Pipeline"])


@router.post("/finalize", response_model=FinalizeResponse)
async def finalize_case(data: FinalizeRequest):
    """
    Finalize a case with user-confirmed entity data.

    Requires a valid session_id from a prior /analyze call.
    Returns the complete PDF payload for document rendering.
    """
    # Load the session to get intent and RAG results
    session = await SessionStore.get(data.session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{data.session_id}' not found. Run /analyze first.",
        )

    # Reconstruct intent from stored session data
    intent_data = session.get("intent", {})
    try:
        domain = LegalDomain(intent_data.get("domain", data.case_type))
    except ValueError:
        domain = LegalDomain.GENERAL_LEGAL_QUERY

    intent = IntentResult(
        domain=domain,
        confidence=intent_data.get("confidence", 1.0),
        summary=intent_data.get("summary", ""),
    )

    # Reconstruct RAG results from stored session data
    rag_data = session.get("rag_results", [])
    rag_results = [
        RAGResult(
            text=r.get("text", ""),
            act_name=r.get("act_name", ""),
            section=r.get("section", ""),
            relevance_score=0.0,
        )
        for r in rag_data
    ]

    # Build the PDF payload with user-confirmed data
    pdf_payload = build_pdf_payload(intent, rag_results, data.extracted_data)

    # Generate a permanent case ID
    case_id = f"case_{uuid4().hex[:10]}"

    # Persist finalized data
    await SessionStore.update(
        data.session_id,
        case_id=case_id,
        case_type=data.case_type,
        extracted_entities=data.extracted_data,
        pdf_payload=pdf_payload,
        status="completed",
    )

    return FinalizeResponse(
        status="success",
        case_id=case_id,
        session_id=data.session_id,
        pdf_payload=pdf_payload,
    )
