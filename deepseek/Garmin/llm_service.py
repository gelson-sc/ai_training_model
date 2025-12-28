import os
import json
import datetime
from typing import Literal, Tuple, Dict, Optional, Any
from openai import OpenAI  # Biblioteca padrão compatível com DeepSeek
from models import DailySummary, DayType  # Importando do arquivo models.py
from garmin_utils import get_daily_health_summary, build_llm_context_md  # Importando utils


# [cite_start]Fonte: [cite: 346-362]
def llm(messages: list[dict], model: str, response_format: dict | None = None) -> tuple[dict, str | None]:
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0.0
    )
    message = response.choices[0].message

    reasoning_content = getattr(message, "reasoning_content", None)

    return json.loads(message.content), reasoning_content


# [cite_start]Fonte: [cite: 387-400] - Definição do System Prompt
def get_system_prompt():
    # Nota: Em um código real, os exemplos seriam injetados aqui conforme o texto
    return f"""
    Instruções:
    * Você receberá um resumo dos dados de saúde e condicionamento físico do usuário para hoje.
    * Seu objetivo é gerar um resumo que será exibido no smartwatch do usuário.
    * Mantenha as coisas curtas, mas também interessantes.
    * O resumo deve estar no formato JSON.
    ---ESQUEMA JSON---
    {DailySummary.model_json_schema()}
    ---FIM DO ESQUEMA JSON---
    """


# [cite_start]Fonte: [cite: 409-436]
def get_daily_summary(
        garmin: Any,
        date: str,
        model: Literal["deepseek-chat", "deepseek-reasoner"],
) -> DailySummary:
    # 1. Converter datas
    date_for_summary = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    past_period_start = date_for_summary - datetime.timedelta(days=7)
    past_period_end = date_for_summary - datetime.timedelta(days=1)

    # 2. Obter dados da Garmin
    summary_in_date = get_daily_health_summary(garmin, date_for_summary, date_for_summary)
    summary_in_past_period = get_daily_health_summary(garmin, past_period_start, past_period_end)

    # 3. Criar Contexto
    prompt = build_llm_context_md(summary_in_date, summary_in_past_period)

    # 4. Preparar mensagens
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt},
    ]

    # 5. Chamar LLM
    response_dict, _ = llm(messages, model, {"type": "json_object"})

    # 6. Validar e retornar
    health_summary = DailySummary.model_validate(response_dict)
    return health_summary