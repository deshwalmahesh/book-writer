"""Authenticated HTTP interface for the book-writing service."""

from __future__ import annotations

import os
import logging
import secrets
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Path as ApiPath, Query, Request
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pydantic import BaseModel, Field, model_validator

from book_writer import DEFAULT_BRIEF, load_config
from jobs import ARTIFACT_NAMES, JobConflict, JobNotFound, JobService


PASSWORD_HASH = PasswordHash.recommended()
DUMMY_HASH = PASSWORD_HASH.hash("missing-user-password")
BEARER = OAuth2PasswordBearer(tokenUrl="/auth/token")


class Registration(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=12, max_length=128)


class UserPublic(BaseModel):
    id: int
    username: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ReviewRequest(BaseModel):
    approved: bool
    feedback: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def check_feedback(self) -> "ReviewRequest":
        self.feedback = self.feedback.strip()
        if not self.approved and not self.feedback:
            raise ValueError("A rejected chapter needs feedback")
        if self.approved and self.feedback:
            raise ValueError("Feedback is only used when rejecting a chapter")
        return self


class JobSummary(BaseModel):
    id: str
    status: str
    chapter_index: int
    created_at: str
    updated_at: str


class JobDetail(JobSummary):
    review_chapter: int | None
    review_preview: str | None
    error: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config(Path(os.environ.get("BOOK_ENV_FILE", ".env")))
    jwt_secret = config.get("JWT_SECRET", "")
    registration_key = config.get("REGISTRATION_KEY", "")
    if len(jwt_secret) < 32 or len(registration_key) < 32:
        raise RuntimeError("JWT_SECRET and REGISTRATION_KEY must each contain at least 32 characters")
    service = JobService(Path(config.get("DATA_DIR", "data")), config)
    app.state.service = service
    app.state.jwt_secret = jwt_secret
    app.state.registration_key = registration_key
    service.start()
    try:
        yield
    finally:
        service.stop()


app = FastAPI(title="Book Writer", version="1.0", lifespan=lifespan)


@app.exception_handler(JobNotFound)
async def not_found_handler(request: Request, exc: JobNotFound):
    raise HTTPException(status_code=404, detail="Job not found")


@app.exception_handler(JobConflict)
async def conflict_handler(request: Request, exc: JobConflict):
    raise HTTPException(status_code=409, detail=str(exc))


def service_for(request: Request) -> JobService:
    return request.app.state.service


def current_user(
    request: Request,
    token: Annotated[str, Depends(BEARER)],
    service: Annotated[JobService, Depends(service_for)],
) -> dict:
    unauthorized = HTTPException(status_code=401, detail="Invalid or expired credentials",
                                 headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, request.app.state.jwt_secret, algorithms=["HS256"],
                             options={"require": ["sub", "exp"]})
        user_id = int(payload["sub"])
    except (InvalidTokenError, ValueError, TypeError, KeyError) as exc:
        raise unauthorized from exc
    user = service.store.user_by_id(user_id)
    if user is None:
        raise unauthorized
    return user


@app.get("/health")
def health(service: Annotated[JobService, Depends(service_for)]) -> dict:
    with service.store.db() as conn:
        conn.execute("SELECT 1").fetchone()
    if not service.healthy():
        raise HTTPException(status_code=503, detail="Job workers are unavailable")
    return {"status": "ok"}


@app.post("/auth/register", response_model=UserPublic, status_code=201)
def register(
    payload: Registration,
    request: Request,
    service: Annotated[JobService, Depends(service_for)],
    registration_key: Annotated[str | None, Header(alias="X-Registration-Key")] = None,
) -> dict:
    if not registration_key or not secrets.compare_digest(
        registration_key.encode(), request.app.state.registration_key.encode()
    ):
        raise HTTPException(status_code=403, detail="Registration is not allowed")
    username = payload.username.lower()
    password_hash = PASSWORD_HASH.hash(payload.password)
    try:
        user_id = service.store.create_user(username, password_hash)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists") from exc
    return {"id": user_id, "username": username}


@app.post("/auth/token", response_model=TokenResponse)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    service: Annotated[JobService, Depends(service_for)],
) -> dict:
    user = service.store.user_by_name(form.username.lower())
    if user is None:
        PASSWORD_HASH.verify(form.password, DUMMY_HASH)
    if user is None or not PASSWORD_HASH.verify(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    token = jwt.encode({"sub": str(user["id"]), "exp": expires}, request.app.state.jwt_secret,
                       algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserPublic)
def me(user: Annotated[dict, Depends(current_user)]) -> dict:
    return user


@app.post("/jobs", response_model=JobDetail, status_code=202)
def create_job(
    user: Annotated[dict, Depends(current_user)],
    service: Annotated[JobService, Depends(service_for)],
) -> dict:
    job_id = service.store.create_job(user["id"], DEFAULT_BRIEF)
    return service.store.get_job(user["id"], job_id)


@app.get("/jobs", response_model=list[JobSummary])
def list_jobs(
    user: Annotated[dict, Depends(current_user)],
    service: Annotated[JobService, Depends(service_for)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[dict]:
    return service.store.list_jobs(user["id"], limit)


@app.get("/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str, user: Annotated[dict, Depends(current_user)],
            service: Annotated[JobService, Depends(service_for)]) -> dict:
    return service.store.get_job(user["id"], job_id)


@app.post("/jobs/{job_id}/pause", response_model=JobDetail, status_code=202)
def pause_job(job_id: str, user: Annotated[dict, Depends(current_user)],
              service: Annotated[JobService, Depends(service_for)]) -> dict:
    service.store.request_pause(user["id"], job_id)
    return service.store.get_job(user["id"], job_id)


@app.post("/jobs/{job_id}/resume", response_model=JobDetail, status_code=202)
def resume_job(job_id: str, user: Annotated[dict, Depends(current_user)],
               service: Annotated[JobService, Depends(service_for)]) -> dict:
    service.store.resume(user["id"], job_id)
    return service.store.get_job(user["id"], job_id)


@app.post("/jobs/{job_id}/review", response_model=JobDetail, status_code=202)
def review_job(job_id: str, payload: ReviewRequest,
               user: Annotated[dict, Depends(current_user)],
               service: Annotated[JobService, Depends(service_for)]) -> dict:
    service.store.submit_review(user["id"], job_id, payload.approved, payload.feedback)
    return service.store.get_job(user["id"], job_id)


@app.get("/jobs/{job_id}/artifacts/{name}")
def download_artifact(
    job_id: str,
    name: Annotated[str, ApiPath(pattern=r"^[A-Za-z0-9.-]+$")],
    user: Annotated[dict, Depends(current_user)],
    service: Annotated[JobService, Depends(service_for)],
) -> FileResponse:
    job = service.store.get_job(user["id"], job_id)
    if name not in ARTIFACT_NAMES or (name.startswith("book.") and job["status"] != "completed"):
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = service.store.artifacts_dir / job_id / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type = "application/pdf" if name.endswith(".pdf") else "text/markdown; charset=utf-8"
    return FileResponse(path, media_type=media_type, filename=name)
