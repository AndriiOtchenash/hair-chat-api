from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from app.db.models import Consent, ConsentType


@dataclass(frozen=True)
class ConsentStatus:
    type: ConsentType
    granted: bool  # an active consent for the required version exists
    # The required-version consent if granted, else the latest active one of any version,
    # so the client can say "you accepted an older text, please re-accept".
    version: str | None
    granted_at: datetime | None


def consent_statuses(active: Sequence[Consent], required_version: str) -> list[ConsentStatus]:
    """One status per ConsentType, from the user's non-revoked consents."""
    statuses = []
    for consent_type in ConsentType:
        of_type = [c for c in active if c.type == consent_type and c.revoked_at is None]
        current = next((c for c in of_type if c.version == required_version), None)
        shown = current or max(of_type, key=lambda c: c.granted_at, default=None)
        statuses.append(
            ConsentStatus(
                type=consent_type,
                granted=current is not None,
                version=shown.version if shown else None,
                granted_at=shown.granted_at if shown else None,
            )
        )
    return statuses


def missing_consents(active: Sequence[Consent], required_version: str) -> list[ConsentType]:
    return [s.type for s in consent_statuses(active, required_version) if not s.granted]
