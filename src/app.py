from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.router import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(title="RightShip Risk Classifier API", version="0.2.0")

    allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/v1")
    return app


app = create_app()


