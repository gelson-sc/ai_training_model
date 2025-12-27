import os

import instructor
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="../../.env")

client = instructor.from_provider(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    mode=instructor.Mode.TOOLS
    # or Mode.MD_JSON for reasoning models
)


class ProductReview(BaseModel):
    rating: int = Field(ge=1, le=5, description="1-5 star rating")
    summary: str = Field(max_length=100)

    pros: list[str]
    cons: list[str]


# Magic happens here - guaranteed valid ProductReview or exception
review = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Review of iPhone 15: Great camera, battery life could be better...",
        }
    ],
    response_model=ProductReview
)
print(review.model_dump_json(indent=2))
