import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from openai import OpenAI
from collections import Counter

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path="../../.env")

# Carregar dados
df = pd.read_excel(
    'Mega-Sena.xlsx',
    usecols=['Concurso', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6'],
    header=0
)

# Converter para lista de listas
resultados = df[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].values.tolist()
total_sorteios = len(resultados)

# --- Análise Estatística Local ---

# 1. Frequência de números
todos_numeros = [n for sorteio in resultados for n in sorteio]
freq = Counter(todos_numeros)
mais_comuns = freq.most_common(10)
menos_comuns = freq.most_common()[:-11:-1]

# 2. Pares vs Ímpares
pares_count = []
for s in resultados:
    p = len([n for n in s if n % 2 == 0])
    pares_count.append(p)
dist_pares = Counter(pares_count)

# 3. Somas
somas = [sum(s) for s in resultados]
media_somas = np.mean(somas)
std_somas = np.std(somas)

# 4. Quadrantes (1-30, 31-60 ou mais detalhado 1-15, 16-30, 31-45, 46-60)
def get_quadrante(n):
    if n <= 10: return 1
    if n <= 20: return 2
    if n <= 30: return 3
    if n <= 40: return 4
    if n <= 50: return 5
    return 6

quadrantes_dist = []
for s in resultados:
    q = [get_quadrante(n) for n in s]
    quadrantes_dist.append(tuple(sorted(Counter(q).items())))

# 5. Números Consecutivos
def tem_consecutivo(s):
    s_sorted = sorted(s)
    for i in range(len(s_sorted)-1):
        if s_sorted[i+1] == s_sorted[i] + 1:
            return True
    return False

consecutivos_freq = len([s for s in resultados if tem_consecutivo(s)]) / total_sorteios

# Preparar o prompt
prompt_text = f"""
Você é um especialista em análise estatística avançada e ciência de dados aplicada a loterias.
Analise os padrões identificados em TODOS os {total_sorteios} sorteios da Mega-Sena realizados até hoje.

Dados Estatísticos Calculados:
1. Total de Sorteios: {total_sorteios}
2. Números mais frequentes: {mais_comuns}
3. Números menos frequentes: {menos_comuns}
4. Distribuição de Pares (Pares: Qtd Sorteios): {dict(sorted(dist_pares.items()))}
5. Média das somas dos números sorteados: {media_somas:.2f} (Desvio Padrão: {std_somas:.2f})
6. Frequência de sorteios com ao menos dois números consecutivos: {consecutivos_freq*100:.2f}%

Com base nesses dados e em técnicas avançadas de estatística (Distribuição de Poisson, Lei dos Grandes Números, Análise Combinatória e Teoria das Probabilidades), realize as seguintes tarefas:

1. Identifique e enumere os padrões estatísticos mais significativos observados em todo o histórico.
2. Explique como o equilíbrio entre quadrantes, paridade e somas se comporta ao longo de quase 3000 sorteios.
3. Identifique anomalias ou "clusters" de números que tendem a aparecer juntos.
4. Apresente uma análise técnica sobre a "memória" do sorteio (ou falta dela) e como os padrões de atraso (delay) influenciam os resultados futuros.
5. Liste os 10 padrões mais recorrentes identificados em pesquisas acadêmicas sobre a Mega-Sena.

Sua resposta deve ser técnica, detalhada e estruturada em tópicos claros.

Ao final, gere 20 sugestões de bilhetes (6 números) que melhor se alinham a esses padrões identificados.
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
            "content": "Você é um PhD em Estatística com especialização em jogos de azar e análise de grandes volumes de dados lotéricos."
        },
        {
            "role": "user",
            "content": prompt_text,
        }
    ],
)

print("\n--- ANÁLISE AVANÇADA DE PADRÕES DA MEGA-SENA ---\n")
content = completion.choices[0].message.content

if content:
    print(content)
else:
    print("O modelo retornou uma resposta vazia.")


'''
#### **5 Sugestões de Bilhetes (Alinhados aos Padrões)**  
1. **10, 34, 37, 41, 53, 59** (2 pares/4 ímpares; soma = 234; cluster 30-40).  
2. **5, 17, 27, 33, 38, 44** (3 pares/3 ímpares; soma = 164; números frequentes).  
3. **14, 19, 30, 35, 42, 55** (2 pares/4 ímpares; soma = 195; inclui atrasado 55).  
4. **8, 15, 26, 37, 49, 52** (3 pares/3 ímpares; soma = 187; quadrante balanceado).  
5. **11, 23, 34, 38, 45, 60** (2 pares/4 ímpares; soma = 211; sem consecutivos).  
'''