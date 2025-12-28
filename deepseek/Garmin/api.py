# [cite_start]Fonte: [cite: 464-520]
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import datetime
import logging

# Importações dos nossos módulos
from models import DailySummary
from llm_service import get_daily_summary
from garmin_utils import start_garmin

# Configuração simples de logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Garmin Health Summary API")


class Model(str, Enum):
    chat = "deepseek-chat"
    reasoner = "deepseek-reasoner"


class HealthSummaryRequest(BaseModel):
    date: str = Field(
        default_factory=lambda: datetime.date.today().isoformat(),
        description="Date in YYYY-MM-DD format, defaults to today",
        example="2025-08-19"
    )
    model: Model = Field(
        default=Model.chat,
        description="Model to use for the summary"
    )


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/health-summary", response_model=DailySummary)
async def get_health_summary_endpoint(
        request: HealthSummaryRequest,
        garmin_email: str = Header(..., description="Garmin email address"),
        garmin_password: str = Header(..., description="Garmin password"),
) -> DailySummary:
    # Nota: Em produção, injetar credenciais via Header não é o ideal para segurança,
    # mas segue o exemplo didático do livro.

    # Definindo variáveis de ambiente temporariamente para a função start_garmin
    # (Ou refatorar start_garmin para aceitar argumentos diretos)
    os.environ["GARMIN_EMAIL"] = garmin_email
    os.environ["GARMIN_PASSWORD"] = garmin_password

    try:
        garmin_client = start_garmin()
        summary = get_daily_summary(garmin_client, request.date, request.model)

        logger.info(f"Daily Health Summary API request completed successfully for {request.date}")
        return summary

    except ValueError as e:
        logger.error(f"Invalid date format provided: {request.date}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Failed to generate Daily Health Summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")

# Para rodar: uv run fastapi run api.py