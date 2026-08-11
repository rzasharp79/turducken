"""FastAPI application factory.

The app binds to localhost by default and has no authentication, because it is
a single-user tool that reads and writes local files on the user's own machine.
That is a deliberate scope decision, and :func:`create_app` warns loudly if it
is asked to bind to a public interface — at which point the missing auth stops
being a reasonable simplification and becomes a way to hand a stranger your
filesystem.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .routes import jobs, learn, recipes, system

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("turducken")


def create_app(*, host: str = "127.0.0.1") -> FastAPI:
    app = FastAPI(
        title="turducken",
        version=__version__,
        description="User friendly, low overhead, custom LoRA training.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    if host not in {"127.0.0.1", "localhost", "::1"}:
        log.warning(
            "Binding to %s, which may be reachable from other machines. turducken has no "
            "authentication and exposes local filesystem paths through its API — only do "
            "this on a network you trust.",
            host,
        )

    for module in (system, recipes, jobs, learn):
        app.include_router(module.router, prefix="/api")

    @app.exception_handler(FileNotFoundError)
    async def _missing_file(_request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
