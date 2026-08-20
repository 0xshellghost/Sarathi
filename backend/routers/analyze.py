"""
Analyze Router — SSE Streaming Endpoint

POST /api/v1/action/analyze

Accepts the user's problem description, runs it through the deterministic
AI pipeline, and streams results back as Server-Sent Events.

SSE event types (matching frontend contract in lib/stream.ts):
  • {type: "token", text: "..."}       — streamed explanation word/token
  • {type: "form_request", schema: {}} — entity form for user to fill
  • {type: "complete", session_id: ""} — pipeline finished
  • {type: "error", message: "..."}    — recoverable error
"""

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import StreamingResponse

from models.requests import AnalyzeRequest
from services.ai_pipeline import run_pipeline

router = APIRouter(prefix="/api/v1/action", tags=["AI Pipeline"])


@router.post("/analyze")
async def analyze_problem(request: Request):
    """
    Analyze a legal problem and stream the AI's response.

    The request body must contain `user_input` (the plain-text problem).
    Optionally include `session_id` to resume a prior session.
    """
    body = await request.json()
    data = AnalyzeRequest(**body)

    return StreamingResponse(
        run_pipeline(data.user_input, data.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
