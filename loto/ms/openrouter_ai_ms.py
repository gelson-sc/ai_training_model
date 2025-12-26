import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from openai import OpenAI

# Carrega as variáveis de ambiente do arquivo .env
# na raiz do projeto
load_dotenv(dotenv_path="../../.env")

df = pd.read_excel(
    'Mega-Sena.xlsx',
    usecols=[0, 2, 3, 4, 5, 6, 7, 8],
    names=['Concurso', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6', 'Ganhadores_6_acertos'],
    header=0
)

# Removido prints de debug para manter a saída limpa
# print(df.head())
# print(f"\nShape: {df.shape}")
# print(df.tail())
# print(f"\nShape: {df.shape}")

# Extrair os resultados para uma lista de listas para facilitar o envio
resultados_anteriores = df[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].values.tolist()

# Preparar o prompt
prompt_text = f"""
Tenho um histórico de {len(resultados_anteriores)} sorteios da Mega-Sena. 
Os números variam de 1 a 60.

Regras e padrões observados:
1. Nunca repete exatamente o mesmo conjunto de 6 números de sorteios anteriores.
2. Raramente ocorrem sequências longas (ex: 1, 2, 3, 4, 5, 6).
3. Evitar que todos os números sejam pares ou todos sejam ímpares (equilíbrio é comum).
4. Analise as frequências e tendências baseadas nos dados históricos 
(fornecerei os últimos 1000 resultados para contexto, mas considere que 
existem {len(resultados_anteriores)} sorteios no total).
5. mostre os padroes que foram identificados

Últimos 2000 resultados (do mais antigo para o mais recente):
{resultados_anteriores[-2000:]}

Tarefa:
Gere exatamente 20 bilhetes diferentes de 6 números cada (de 1 a 60).
Os bilhetes devem ser plausíveis, respeitando as regras de não repetição de sorteios anteriores e o equilíbrio entre pares e ímpares.
Apresente o resultado em uma lista numerada de 1 a 20. Cada linha deve conter apenas os 6 números do bilhete.
Exemplo de formato:
01. 05, 12, 23, 34, 45, 56
02. ...
"""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
    model="deepseek/deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": "Você é um especialista em estatística de loterias. "
                       "Sua tarefa é gerar números plausíveis para a Mega-Sena com base em regras fornecidas. "
                       "Responda APENAS com a lista de bilhetes, sem explicações ou texto adicional."
        },
        {
            "role": "user",
            "content": prompt_text,
        }
    ],
)

print("\n--- Sugestões de Bilhetes (DeepSeek) ---\n")
content = completion.choices[0].message.content
reasoning = getattr(completion.choices[0].message, 'reasoning', None)

if content:
    print(content)
elif reasoning:
    print("O modelo forneceu apenas o raciocínio. Extraindo bilhetes do raciocínio...")
    print(reasoning)
else:
    print("O modelo retornou uma resposta vazia.")
    print(f"Resposta bruta: {completion}")

'''
01. 07, 14, 25, 36, 42, 53
02. 03, 18, 22, 37, 44, 55
03. 09, 16, 27, 38, 45, 52
04. 04, 13, 24, 35, 46, 57
05. 08, 15, 26, 33, 47, 54
06. 06, 17, 28, 39, 43, 56
07. 02, 19, 30, 41, 48, 58
08. 10, 21, 29, 40, 49, 59
09. 01, 20, 31, 34, 50, 60
10. 11, 23, 32, 36, 51, 57
11. 05, 12, 24, 37, 45, 58
12. 07, 16, 25, 38, 46, 59
13. 03, 14, 26, 39, 47, 60
14. 09, 18, 27, 40, 48, 53
15. 04, 15, 28, 41, 49, 54
16. 08, 17, 29, 42, 50, 55
17. 06, 19, 30, 43, 51, 56
18. 02, 20, 31, 44, 52, 57
19. 10, 22, 32, 45, 53, 58
20. 01, 13, 23, 35, 46, 59
-----------------------------------
01. 07, 14, 25, 36, 42, 53
02. 03, 18, 22, 37, 44, 59
03. 09, 16, 27, 38, 45, 57
04. 04, 13, 26, 35, 47, 58
05. 08, 17, 24, 39, 46, 55
06. 02, 15, 28, 33, 48, 56
07. 10, 19, 30, 41, 49, 54
08. 06, 20, 29, 34, 43, 52
09. 11, 21, 32, 40, 50, 60
10. 01, 12, 23, 31, 42, 51
11. 05, 13, 24, 36, 47, 58
12. 07, 16, 25, 39, 48, 57
13. 03, 14, 26, 35, 44, 59
14. 09, 18, 27, 38, 46, 55
15. 04, 15, 28, 37, 45, 56
16. 08, 17, 29, 40, 49, 54
17. 02, 19, 30, 41, 50, 60
18. 10, 20, 31, 42, 51, 53
19. 06, 21, 32, 43, 52, 57
20. 11, 22, 33, 44, 53, 58
'''
