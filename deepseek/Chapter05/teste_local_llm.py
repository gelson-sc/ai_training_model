import os
from ollama import chat

messages = [
    {"role": "user", "content": "Responda em português: o que é uma LLM?"}
]

response = chat(
    model="deepseek-r1:1.5b",
    messages=messages,
    options={
        "temperature": 0.0,
    },
)

print(response["message"]["content"])


messages = [
    {"role": "system", "content": "Você é um assistente objetivo."},
    {"role": "user", "content": "Me dê 3 usos práticos de LLMs."},
    {"role": "user", "content": "Agora detalhe o segundo."},
]

resp = chat(model="deepseek-r1:1.5b", messages=messages)
print(resp["message"]["content"])