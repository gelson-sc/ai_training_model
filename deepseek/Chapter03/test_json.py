from dotenv import load_dotenv
from openai import OpenAI
import json
import os

load_dotenv(dotenv_path="../../.env")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": "Extract user information as JSON. Example: {\"name\": \"Alice\", \"age\": 30}"
        },
        {
            "role": "user",
            "content": "My name is Bob and I'm 25 years old"
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.1
)
data = json.loads(response.choices[0].message.content)
