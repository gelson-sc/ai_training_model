import os
import datetime
import numpy as np
from typing import Any, List, Dict
from garminconnect import Garmin  # Requer: pip install garminconnect


# --- Autenticação ---

def start_garmin() -> Garmin:
    """Inicializa a conexão com a Garmin."""
    GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
    GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")

    try:
        api = Garmin(email=GARMIN_EMAIL, password=GARMIN_PASSWORD, is_cn=False)
        api.login()
        print("Você está logado.")
        return api
    except Exception as e:
        print(f"Não foi possível fazer login com e-mail e senha: {e}")
        raise


# --- Extração de Dados ---
# [cite_start]Fonte: [cite: 112-146]
def get_daily_health_summary(api: Any, start: datetime.date, end: datetime.date) -> list[dict[str, Any]]:
    def dstr(d: datetime.date) -> str:
        return d.isoformat()

    def daterange(a: datetime.date, b: datetime.date):
        for i in range((b - a).days + 1):
            yield a + datetime.timedelta(days=i)

    out = []
    for day in daterange(start, end):
        s = dstr(day)
        day_of_week = day.strftime("%A")

        # Obtém o resumo do usuário da API
        summary = api.get_user_summary(s) or {}

        rhr = summary.get("restingHeartRate")
        steps = summary.get("totalSteps")
        stress_level = summary.get("averageStressLevel")  # Corrigido typo do texto original

        # Tenta pegar body battery final
        body_battery_final = summary.get("bodyBatteryMostRecentValue") or summary.get("mostRecentBodyBattery")

        exercise_minutes = (summary.get("moderateIntensityMinutes") or 0) + (
                    summary.get("vigorousIntensityMinutes") or 0)
        sleep_seconds = summary.get("sleepingSeconds")
        sleep_hours = round(sleep_seconds / 3600, 2) if sleep_seconds else 0

        body_battery_start = summary.get("bodyBatteryAtWakeTime")
        total_distance_meters = summary.get("totalDistanceMeters")

        out.append({
            "date": s,
            "day_of_week": day_of_week,
            "resting_heart_rate": rhr,
            "exercise_minutes": exercise_minutes,
            "stress_level": stress_level,
            "sleep_hours": sleep_hours,
            "steps": steps,
            "total_distance_meters": total_distance_meters,
            "body_battery_start_day": body_battery_start,
            "body_battery_end_day": body_battery_final,
        })
    return out


# --- Construção do Contexto para IA ---
def detect_trend(values, pct_threshold=5):
    if not values: return "flat"
    recent, earlier = np.mean(values[-3:]), np.mean(values[:3])
    if recent > earlier * (1 + pct_threshold / 100):
        return "up"
    elif recent < earlier * (1 - pct_threshold / 100):
        return "down"
    else:
        return "flat"


def build_llm_context_md(summary_for_today: list[dict], summary_for_past_7_days: list[dict]) -> str:
    assert len(summary_for_today) == 1, "Expected 1 day of summary"
    today = summary_for_today[0]

    # Filtra chaves que não são métricas
    metrics = [key for key in today.keys() if key not in ["date", "day_of_week"]]

    lines = [
        f"# Daily Metrics Summary for {today['date']} ({today['day_of_week']})",
        "_Note: All comparisons use the **previous 7 days only**, excluding today._",
    ]

    better_is_lower = ["resting_heart_rate", "stress_level"]

    for metric in metrics:
        today_val = today[metric]
        if today_val is None:  # Tratamento se valor for nulo
            continue

        past_vals = [day[metric] for day in summary_for_past_7_days if day.get(metric) is not None]

        avg_7d = sum(past_vals) / len(past_vals) if past_vals else 0
        delta_pct = ((today_val - avg_7d) / avg_7d * 100) if avg_7d else 0

        trend_dir = detect_trend(past_vals)
        arrow = "↑" if trend_dir == "up" else ("↓" if trend_dir == "down" else "→")

        lines.append(
            f"## {metric.replace('_', ' ').title()}\n"
            f"- Today's value ({today['date']}): {today_val}\n"
            f"- 7-day baseline average (excluding today): {avg_7d:.2f}\n"
            f"- Percent change vs. baseline: {delta_pct:+.1f}%\n"
            f"- Trend over previous 7 days: {trend_dir} {arrow}\n"
            f"- Better is lower: {metric in better_is_lower}\n"
        )

    return "\n".join(lines)