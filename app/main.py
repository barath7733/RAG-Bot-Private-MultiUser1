"""
FastAPI application entry point.

Exposes endpoints for auth (register/login/logout), chat (general +
RAG + web), document upload/list/delete, chat history, image
generation, vision (image analysis), and a health check. Serves the
frontend from /static and /templates.

AUTHENTICATION MODEL
---------------------
Login issues an HttpOnly, SameSite=Lax session cookie containing a
signed JWT (see app/auth.py). Every protected endpoint depends on
`get_current_user`, which resolves the *authenticated* user id from
that cookie — a user id is never accepted from the request body,
query string, or a header. All document, RAG, and chat-history
operations are then scoped to that resolved user id end-to-end
(app/rag.py, app/pinecone_db.py), so one user's data is structurally
unreachable by another, even if they tamper with ids in the request.
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import groq_client, image_gen, rag
from app.auth import (
    COOKIE_NAME,
    JWT_ALGORITHM,
    create_session_and_token,
    get_current_user,
    get_current_user_optional,
    hash_password,
    revoke_session,
    verify_password,
)
from app.config import configure_logging, get_settings, validate_required_settings
from app.database import get_db, init_db
from app.db_models import User
from app.image_gen import ImageGenerationError
from app.models import (
    AuthResponse,
    ChatHistoryMessage,
    ChatRequest,
    ChatResponse,
    ChatSessionDetail,
    ChatSessionSummary,
    DeleteResponse,
    DocumentInfo,
    ErrorResponse,
    HealthResponse,
    ImageGenerateRequest,
    ImageGenerateResponse,
    LoginRequest,
    RegisterRequest,
    UploadResponse,
    UserOut,
    VisionAnalyzeResponse,
)
from app.pinecone_db import is_index_ready
from app.rag import RAGError
from app.web_search import WebSearchError

configure_logging()
logger = logging.getLogger("rag_chatbot.main")

settings = get_settings()

app = FastAPI(
    title="General-Purpose AI Assistant + RAG Document Intelligence (Multi-User)",
    description="A multi-user chatbot that answers general questions via Groq and grounds document-related "
                "questions in retrieved context using per-user Pinecone vector search.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024


@app.on_event("startup")
async def on_startup() -> None:
    warnings = validate_required_settings()
    for warning in warnings:
        logger.warning(warning)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path("data/documents").mkdir(parents=True, exist_ok=True)
    init_db()
    logger.info(
        "Application startup complete. Groq model=%s, Embedding dimension configured, DB=%s",
        settings.groq_model, settings.database_url,
    )


# --------------------------------------------------------------------------
# Error handling — never leak stack traces or secrets to the client.
# --------------------------------------------------------------------------

@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
    return JSONResponse(status_code=422, content=ErrorResponse(error=str(exc)).model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="An internal error occurred. Please try again.",
        ).model_dump(),
    )


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )


# --------------------------------------------------------------------------
# Frontend pages
# --------------------------------------------------------------------------

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index(request: Request, user: User | None = Depends(get_current_user_optional)) -> Response:
    """Protected page: unauthenticated visitors are redirected to /login server-side."""
    if user is None:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("index.html", {"request": request, "user_email": user.email})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User | None = Depends(get_current_user_optional)) -> Response:
    if user is not None:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user: User | None = Depends(get_current_user_optional)) -> Response:
    if user is not None:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("register.html", {"request": request})


# --------------------------------------------------------------------------
# Health / status (public — no user data revealed)
# --------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    warnings = validate_required_settings()
    embedding_dimension: int | None = None
    try:
        from app.embeddings import get_embedding_dimension
        embedding_dimension = get_embedding_dimension()
    except Exception:  # noqa: BLE001
        embedding_dimension = None

    pinecone_ready = False
    if settings.pinecone_api_key:
        pinecone_ready = is_index_ready()

    return HealthResponse(
        status="ok" if not warnings else "degraded",
        groq_configured=bool(settings.groq_api_key),
        pinecone_configured=bool(settings.pinecone_api_key),
        embedding_model=settings.embedding_model,
        embedding_dimension=embedding_dimension,
        pinecone_index_ready=pinecone_ready,
        warnings=warnings,
    )


@app.get("/api/features")
async def features() -> dict:
    """Lightweight capability flags the frontend uses to enable/disable UI features."""
    return {
        "web_search_enabled": bool(settings.tavily_api_key),
        "image_generation_enabled": True,  # Pollinations.ai requires no key
    }


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=AuthResponse, status_code=201)
async def register(payload: RegisterRequest, response: Response, db: Session = Depends(get_db)) -> AuthResponse:
    email_normalized = payload.email.strip().lower()

    existing = db.query(User).filter(User.email == email_normalized).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user = User(email=email_normalized, password_hash=hash_password(payload.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    db.refresh(user)

    token, _ = create_session_and_token(db, user)
    _set_auth_cookie(response, token)

    return AuthResponse(
        user=UserOut(id=user.id, email=user.email, created_at=user.created_at.isoformat()),
        message="Account created successfully.",
    )


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> AuthResponse:
    email_normalized = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email_normalized).first()

    # Same generic error whether the email doesn't exist or the password is
    # wrong, so a login attempt can't be used to enumerate registered emails.
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token, _ = create_session_and_token(db, user)
    _set_auth_cookie(response, token)

    return AuthResponse(
        user=UserOut(id=user.id, email=user.email, created_at=user.created_at.isoformat()),
        message="Logged in successfully.",
    )


@app.post("/api/auth/logout")
async def logout(
    response: Response,
    request: Request,
    user: User = Depends(get_current_user),  # ensures only an already-valid session can be logged out
    db: Session = Depends(get_db),
) -> dict:
    """
    Revoke the current session server-side (so the token stops working
    immediately, not just once it naturally expires) and clear the cookie.
    """
    raw_token = request.cookies.get(COOKIE_NAME)
    if raw_token:
        # get_current_user already proved this token is valid; decode again
        # here purely to recover the session id (`jti`) to delete it.
        try:
            decoded = jwt.decode(raw_token, settings.secret_key, algorithms=[JWT_ALGORITHM])
            session_id = decoded.get("jti")
            if session_id:
                revoke_session(db, session_id)
        except jwt.InvalidTokenError:
            pass

    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"message": "Logged out successfully."}


@app.get("/api/auth/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut(id=user.id, email=user.email, created_at=user.created_at.isoformat())


# --------------------------------------------------------------------------
# Chat (protected)
# --------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    session = rag.get_or_create_session(db, user.id, payload.session_id, payload.question)
    rag.record_message(db, session, role="user", content=payload.question)

    try:
        answer, mode_used, sources, web_sources, found = rag.answer_question(
            db=db,
            user_id=user.id,
            question=payload.question,
            mode=payload.mode,
            history=payload.history,
            document_id=payload.document_id,
        )
    except (RuntimeError, WebSearchError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    rag.record_message(db, session, role="assistant", content=answer, mode_used=mode_used.value)

    return ChatResponse(
        answer=answer,
        mode_used=mode_used,
        sources=sources,
        web_sources=web_sources,
        found_in_documents=found,
        session_id=session.id,
    )


# --------------------------------------------------------------------------
# Vision (protected) — user uploads a photo and asks about it, integrated
# into chat history the same way a normal text turn is.
# --------------------------------------------------------------------------

ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/api/vision/analyze", response_model=VisionAnalyzeResponse)
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    question: str = Form(default=""),
    session_id: str | None = Form(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VisionAnalyzeResponse:
    if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PNG, JPEG, WEBP, or GIF images are supported.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds the maximum allowed size of 10 MB.")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{file.content_type};base64,{encoded}"

    user_message = question.strip() or f"[Uploaded image: {file.filename or 'photo'}] Describe this image."
    session = rag.get_or_create_session(db, user.id, session_id, user_message)
    rag.record_message(db, session, role="user", content=user_message)

    try:
        answer = groq_client.analyze_image(data_url, question or None)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    rag.record_message(db, session, role="assistant", content=answer, mode_used="vision")

    return VisionAnalyzeResponse(answer=answer, session_id=session.id)


# --------------------------------------------------------------------------
# Chat history (protected, owner-checked)
# --------------------------------------------------------------------------

@app.get("/api/chat-history", response_model=list[ChatSessionSummary])
async def list_chat_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[ChatSessionSummary]:
    return rag.list_chat_sessions(db, user.id)


@app.get("/api/chat-history/{session_id}", response_model=ChatSessionDetail)
async def get_chat_history(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatSessionDetail:
    detail = rag.get_chat_session_detail(db, user.id, session_id)
    if detail is None:
        # Identical response whether the session doesn't exist or belongs
        # to another user — never confirm another user's session id exists.
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return detail


@app.delete("/api/chat-history/{session_id}", response_model=DeleteResponse)
async def delete_chat_history(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    deleted = rag.delete_chat_session(db, user.id, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return DeleteResponse(document_id=session_id, deleted=True, message="Chat session deleted successfully.")


# --------------------------------------------------------------------------
# Image generation (protected)
# --------------------------------------------------------------------------

@app.post("/api/image/generate", response_model=ImageGenerateResponse)
async def generate_image(
    payload: ImageGenerateRequest,
    user: User = Depends(get_current_user),
) -> ImageGenerateResponse:
    try:
        url, final_prompt, note = image_gen.generate_image(
            payload.prompt, width=payload.width, height=payload.height, enhance=payload.enhance,
        )
    except ImageGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ImageGenerateResponse(image_url=url, prompt=payload.prompt, final_prompt=final_prompt, note=note)


# --------------------------------------------------------------------------
# Documents (protected, owner-scoped)
# --------------------------------------------------------------------------

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    replace_existing: bool = Form(default=False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UploadResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Sanitize filename to avoid path traversal / unsafe characters.
    safe_name = Path(file.filename or f"document-{uuid.uuid4().hex[:8]}.pdf").name

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the maximum allowed size of {settings.max_upload_size_mb} MB.",
        )

    try:
        doc_info = rag.ingest_pdf(db, user.id, file_bytes, safe_name, replace_existing=replace_existing)
    except RAGError:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return UploadResponse(document=doc_info, message=f"'{safe_name}' processed and indexed successfully.")


@app.get("/api/documents", response_model=list[DocumentInfo])
async def get_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[DocumentInfo]:
    return rag.list_documents(db, user.id)


@app.delete("/api/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    try:
        deleted = rag.delete_document(db, user.id, document_id)
    except RAGError:
        raise
    if not deleted:
        # Also returned when the document belongs to a different user —
        # never distinguish "not found" from "not yours".
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(document_id=document_id, deleted=True, message="Document deleted successfully.")