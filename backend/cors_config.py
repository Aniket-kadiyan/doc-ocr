"""Credential-compatible unrestricted CORS for unknown client LAN layouts."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def install_unrestricted_cors(
    app: FastAPI,
    *,
    expose_headers: Sequence[str] = (),
) -> None:
    """Allow every browser origin while echoing it for credentialed requests.

    A literal ``*`` origin cannot be used reliably with cookies. A regular
    expression makes Starlette return the request's exact Origin while still
    accepting any client hostname, IP address, port, or scheme.
    """

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=r".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=list(expose_headers),
    )
