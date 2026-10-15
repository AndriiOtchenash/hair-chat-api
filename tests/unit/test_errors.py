from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from app.core.errors import api_error, install_error_handlers


class Payload(BaseModel):
    name: str
    age: int


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/api-error")
    async def raises_api_error() -> None:
        raise api_error(409, "already_exists", "Already exists", {"X-Hint": "1"})

    @app.get("/plain-http-error")
    async def raises_plain() -> None:
        raise HTTPException(404, "Conversation not found")

    @app.post("/validated")
    async def validated(payload: Payload) -> None: ...

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_api_error_has_flat_code_and_message(client: AsyncClient) -> None:
    response = await client.get("/api-error")

    assert response.status_code == 409
    assert response.json() == {"code": "already_exists", "message": "Already exists"}
    assert response.headers["X-Hint"] == "1"


async def test_plain_http_exception_gets_code_from_status(client: AsyncClient) -> None:
    response = await client.get("/plain-http-error")

    assert response.json() == {"code": "not_found", "message": "Conversation not found"}


async def test_unknown_route_and_wrong_method_use_same_shape(client: AsyncClient) -> None:
    assert (await client.get("/nope")).json() == {"code": "not_found", "message": "Not Found"}
    response = await client.get("/validated")
    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"


async def test_validation_error_is_summarised_without_echoing_input(client: AsyncClient) -> None:
    response = await client.post("/validated", json={"age": "secret-value"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert "body.name: Field required" in body["message"]
    assert "body.age" in body["message"]
    assert "secret-value" not in body["message"]
