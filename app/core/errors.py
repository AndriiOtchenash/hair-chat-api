"""One error body for REST and SSE: {"code": "...", "message": "..."} (no `detail` wrapper)."""

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

MAX_VALIDATION_ERRORS_IN_MESSAGE = 3


class ErrorBody(BaseModel):
    code: str
    message: str


def api_error(
    status_code: int, code: str, message: str, headers: dict[str, str] | None = None
) -> HTTPException:
    """HTTPException whose response body is exactly ErrorBody(code, message)."""
    return HTTPException(status_code, detail=ErrorBody(code=code, message=message), headers=headers)


def _default_code(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase.lower().replace(" ", "_").replace("-", "_")
    except ValueError:
        return "error"


async def _http_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    # Registered for Starlette's base class so routing 404/405 get the same shape.
    assert isinstance(exc, StarletteHTTPException)
    if isinstance(exc.detail, ErrorBody):
        body = exc.detail
    else:
        body = ErrorBody(code=_default_code(exc.status_code), message=str(exc.detail))
    return JSONResponse(body.model_dump(), status_code=exc.status_code, headers=exc.headers)


async def _validation_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    # Location + reason only; input values are not echoed back.
    assert isinstance(exc, RequestValidationError)
    errors = exc.errors()
    parts = [
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
        for err in errors[:MAX_VALIDATION_ERRORS_IN_MESSAGE]
    ]
    if len(errors) > MAX_VALIDATION_ERRORS_IN_MESSAGE:
        parts.append(f"and {len(errors) - MAX_VALIDATION_ERRORS_IN_MESSAGE} more")
    body = ErrorBody(code="validation_error", message="; ".join(parts) or "Invalid request")
    return JSONResponse(body.model_dump(), status_code=422)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
