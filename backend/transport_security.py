"""Resolve optional HTTPS without making TLS a startup requirement."""

from __future__ import annotations

import ssl
from dataclasses import dataclass, field
from pathlib import Path

from config_env import RuntimeConfig


@dataclass(frozen=True)
class TransportSecurity:
    """The transport that can actually be used by this process."""

    requested: bool
    active: bool
    fallback_reason: str | None = None
    cert_file: Path | None = field(default=None, repr=False)
    key_file: Path | None = field(default=None, repr=False)

    @property
    def scheme(self) -> str:
        return "https" if self.active else "http"

    def public_summary(self) -> dict[str, object]:
        return {
            "https_requested": self.requested,
            "https_active": self.active,
            "scheme": self.scheme,
            "http_fallback": self.requested and not self.active,
            "fallback_reason": self.fallback_reason,
        }


def _certificate_pair_is_valid(cert_file: Path, key_file: Path) -> bool:
    """Ask OpenSSL to parse the pair without retaining secret material."""

    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    except (OSError, ssl.SSLError, TypeError, ValueError):
        return False
    return True


def resolve_transport_security(config: RuntimeConfig) -> TransportSecurity:
    """Validate the PEM certificate pair and otherwise select HTTP safely."""

    if not config.https_enabled:
        return TransportSecurity(requested=False, active=False)

    cert_file = config.tls_cert_file
    key_file = config.tls_key_file
    if cert_file is None or key_file is None:
        return TransportSecurity(
            requested=True,
            active=False,
            fallback_reason="TLS_CERT_FILE and TLS_KEY_FILE must both be configured",
        )
    if not cert_file.is_file():
        return TransportSecurity(
            requested=True,
            active=False,
            fallback_reason="TLS certificate file is unavailable",
        )
    if not key_file.is_file():
        return TransportSecurity(
            requested=True,
            active=False,
            fallback_reason="TLS private-key file is unavailable",
        )

    if not _certificate_pair_is_valid(cert_file, key_file):
        return TransportSecurity(
            requested=True,
            active=False,
            fallback_reason="TLS certificate or private key is invalid",
        )

    return TransportSecurity(
        requested=True,
        active=True,
        cert_file=cert_file,
        key_file=key_file,
    )

