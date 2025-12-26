import pandas as pd
import numpy as np

# Carregar o arquivo Excel com as colunas especificas
df = pd.read_excel(
    'Mega-Sena.xlsx',
    usecols=[0, 2, 3, 4, 5, 6, 7, 8],
    names=['Concurso', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6', 'Ganhadores_6_acertos'],
    header=0
)

print(df.head())
print(f"\nShape: {df.shape}")

print(df.tail())
print(f"\nShape: {df.shape}")

# 1. Extrair todos os números sorteados em uma única lista
todos_numeros = df[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].values.flatten()

# 2. Calcular a frequência de cada número (1 a 60)
frequencia = pd.Series(todos_numeros).value_counts().sort_index()

# Garantir que todos os números de 1 a 60 estejam presentes na contagem
for i in range(1, 61):
    if i not in frequencia:
        frequencia[i] = 0

# 3. Criar pesos baseados na frequência
# Aqui, números que saem mais têm mais chance.
# Dica: Você pode inverter isso se preferir apostar nos que 'faltam' sair.
pesos = frequencia.values / frequencia.sum()
numeros_disponiveis = frequencia.index.values


def gerar_cartoes(n_cartoes, n_de_bolas=6):
    cartoes = []
    for _ in range(n_cartoes):
        # O numpy escolhe 6 números baseados nos pesos de probabilidade calculados do histórico
        escolha = np.random.choice(numeros_disponiveis, size=n_de_bolas, replace=False, p=pesos)
        escolha.sort()
        cartoes.append(escolha)
    return cartoes


# 4. Gerar 20 cartões
meus_jogos = gerar_cartoes(20)

print("\n--- Meus 20 Cartões Sugeridos (Baseados em Frequência) ---")
for i, jogo in enumerate(meus_jogos, 1):
    print(f"Jogo {i:02d}: {jogo}")

print("\n--- Fim da Execução ---")
