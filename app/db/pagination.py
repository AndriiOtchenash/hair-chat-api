"""Keyset (cursor) pagination: the cursor is the sort key of the last row of the previous page.

Unlike OFFSET it stays fast on deep pages and does not skip or repeat rows when new ones arrive.
The cursor is opaque to clients: base64url of `<iso timestamp>|<uuid>`.
"""

import base64
import binascii
import uuid
from dataclasses import dataclass
from datetime import datetime


class InvalidCursor(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Cursor:
    at: datetime
    id: uuid.UUID

    def encode(self) -> str:
        raw = f"{self.at.isoformat()}|{self.id}".encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, value: str) -> "Cursor":
        try:
            raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
            at, _, id_ = raw.partition("|")
            parsed = datetime.fromisoformat(at)
            if parsed.tzinfo is None:
                raise ValueError("naive timestamp")
            return cls(parsed, uuid.UUID(id_))
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            raise InvalidCursor("invalid cursor") from exc


@dataclass(frozen=True, slots=True)
class Page[T]:
    items: list[T]
    next_cursor: str | None
