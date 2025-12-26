import pandas as pd
import ollama
import sys

# 1. Carregar os dados com tratamento de erro
try:
    df = pd.read_excel(
        'Mega-Sena.xlsx',
        usecols=[0, 2, 3, 4, 5, 6, 7, 8],
        names=['Concurso', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6', 'Ganhadores'],
        header=0
    )
except Exception as e:
    print(f"Erro ao carregar o arquivo: {e}")
    sys.exit()

# 2. Preparar insights estatísticos para a IA
# Pegamos os últimos 15 concursos para dar mais contexto de "atraso"
ultimos_resultados = df.tail(15)[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].values.tolist()

# Frequência histórica (Top 15 números que mais saíram)
frequencia_total = df[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].stack().value_counts()
top_15_frequentes = frequencia_total.head(15).to_dict()
top_15_frequentes_python = {int(k): int(v) for k, v in top_15_frequentes.items()}

# 3. Engenharia de Prompt (Melhorada para o DeepSeek-R1)
prompt = f"""
Você é um especialista em análise combinatória e estatística de loterias.
Analise os seguintes dados da Mega-Sena para gerar uma estratégia de jogo:

HISTÓRICO RECENTE (Últimos 15 concursos):
{ultimos_resultados}

NÚMEROS MAIS FREQUENTES (Número: Vezes sorteado):
{top_15_frequentes_python}

SUA TAREFA:
1. Analise tendências: Veja se há números repetidos, equilíbrios entre pares/ímpares e números que não saem há muito tempo (atrasados).
2. Gere 20 cartões únicos de 6 números (entre 01 e 60).
3. Aplique filtros de qualidade: Evite sequências (ex: 01, 02, 03) e garanta que os jogos não fiquem todos no mesmo quadrante do volante.

Responda no final com uma lista clara dos 20 jogos formatados.
"""

# 4. Execução no Ollama (Aproveitando sua GPU de 12GB) deepseek-r1:8b llama3.2:latest
try:
    print(f"Enviando dados para o DeepSeek-R1 (8b) via GPU...")
    response = ollama.chat(model='llama3.2:latest', messages=[
        {'role': 'user', 'content': prompt},
    ])

    print("\n" + "="*50)
    print("ANÁLISE E SUGESTÕES DO DEEPSEEK")
    print("="*50)

    # Exibe a resposta completa (incluindo o raciocínio se o modelo gerar)
    print(response['message']['content'])

except Exception as e:
    print(f"Erro ao conectar com o Ollama: {e}")
    print("Certifique-se de que o comando 'ollama serve' está rodando.")