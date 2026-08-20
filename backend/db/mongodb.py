"""
MongoDB Async Client — Session Persistence

Uses Motor (async pymongo driver) for non-blocking MongoDB operations.
Each user interaction creates or updates a "session" document that tracks
the full lifecycle: user input → intent → RAG results → entities → PDF payload.
"""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure

from config import get_settings

logger = logging.getLogger("sarathi.db.mongo")

# ── Module-level client (initialized in lifespan) ────────────────
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect() -> None:
    """Open the MongoDB connection pool. Called once at app startup."""
    global _client, _db
    settings = get_settings()

    _client = AsyncIOMotorClient(
        settings.MONGODB_URI,
        maxPoolSize=20,
        minPoolSize=2,
        serverSelectionTimeoutMS=5000,
    )

    # Verify connectivity
    try:
        await _client.admin.command("ping")
        logger.info("Connected to MongoDB at %s", settings.MONGODB_URI)
    except ConnectionFailure:
        logger.error("MongoDB unreachable at %s", settings.MONGODB_URI)
        raise

    _db = _client[settings.MONGODB_DB_NAME]

    # Ensure indexes
    await _db.sessions.create_index("session_id", unique=True)
    await _db.sessions.create_index("created_at")


async def disconnect() -> None:
    """Close the MongoDB connection pool. Called at app shutdown."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed.")


def _get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB not initialized. Call connect() first.")
    return _db


# ── Session CRUD ─────────────────────────────────────────────────


class SessionStore:
    """Async CRUD operations for user sessions."""

    @staticmethod
    async def create(user_input: str, session_id: str | None = None) -> dict:
        """Create a new session and return its document."""
        db = _get_db()
        doc = {
            "session_id": session_id or f"sess_{uuid4().hex[:12]}",
            "user_input": user_input,
            "intent": None,
            "rag_results": [],
            "extracted_entities": {},
            "pdf_payload": None,
            "status": "analyzing",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        await db.sessions.insert_one(doc)
        logger.info("Session created: %s", doc["session_id"])
        return doc

    @staticmethod
    async def get(session_id: str) -> dict | None:
        """Retrieve a session by its ID."""
        db = _get_db()
        doc = await db.sessions.find_one(
            {"session_id": session_id}, {"_id": 0}
        )
        return doc

    @staticmethod
    async def update(session_id: str, **fields) -> bool:
        """Partial update — only set the provided fields."""
        db = _get_db()
        fields["updated_at"] = datetime.now(timezone.utc)
        result = await db.sessions.update_one(
            {"session_id": session_id},
            {"$set": fields},
        )
        return result.modified_count > 0

    @staticmethod
    async def list_all(limit: int = 50) -> list[dict]:
        """List sessions, most recent first."""
        db = _get_db()
        cursor = (
            db.sessions.find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    @staticmethod
    async def get_by_case_id(case_id: str) -> dict | None:
        """Retrieve a session by its case_id (set after finalization)."""
        db = _get_db()
        doc = await db.sessions.find_one(
            {"case_id": case_id}, {"_id": 0}
        )
        return doc
