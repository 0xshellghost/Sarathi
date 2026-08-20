"""
Sessions Router — History & Case Retrieval

GET /api/v1/sessions/history   — List all sessions (most recent first)
GET /api/v1/sessions/cases/{case_id} — Retrieve a specific finalized case
"""

from fastapi import APIRouter, HTTPException

from models.responses import CaseHistoryResponse, CaseSummary
from db.mongodb import SessionStore

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@router.get("/history", response_model=CaseHistoryResponse)
async def get_history():
    """List all sessions, most recent first."""
    sessions = await SessionStore.list_all(limit=50)

    cases = [
        CaseSummary(
            case_id=s.get("case_id", s["session_id"]),
            session_id=s["session_id"],
            type=s.get("intent", {}).get("domain", "unknown"),
            summary=s.get("intent", {}).get("summary", s.get("user_input", "")[:100]),
            created_at=s["created_at"],
            status=s.get("status", "unknown"),
        )
        for s in sessions
    ]

    return CaseHistoryResponse(cases=cases)


@router.get("/cases/{case_id}")
async def get_case(case_id: str):
    """
    Retrieve a specific finalized case by its case_id.

    Returns the full session data including the PDF payload.
    """
    session = await SessionStore.get_by_case_id(case_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Case '{case_id}' not found.",
        )

    return {
        "case_id": session.get("case_id", case_id),
        "session_id": session["session_id"],
        "case_type": session.get("case_type"),
        "user_input": session.get("user_input"),
        "intent": session.get("intent"),
        "extracted_entities": session.get("extracted_entities"),
        "pdf_payload": session.get("pdf_payload"),
        "status": session.get("status"),
        "created_at": session.get("created_at"),
    }
