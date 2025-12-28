from pydantic import BaseModel, Field
from enum import Enum

class DayType(str, Enum):
    TRAINING = "training"
    ACTIVE_RECOVERY = "active_recovery"
    REST = "rest"
    HIGH_STRESS = "high_stress"
    BALANCED = "balanced"

class DailySummary(BaseModel):
    day_type: DayType = Field(
        description="Classification of the day based on activity and recovery metrics"
    )
    title: str = Field(..., description="One sentence summary of the day")
    emoji: str = Field(..., description="Emoji to represent the day type")
    observation: str = Field(
        description="Two sentence observation about key metrics and patterns"
    )
    recommendation: str = Field(
        description="Two sentence actionable recommendation for tomorrow"
    )