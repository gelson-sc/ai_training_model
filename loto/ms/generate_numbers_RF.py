import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ... (seu código anterior de carregamento do df permanece o mesmo) ...
# Carregar o arquivo Excel com as colunas especificas
df = pd.read_excel(
    'Mega-Sena.xlsx',
    usecols=[0, 2, 3, 4, 5, 6, 7, 8],
    names=['Concurso', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6', 'Ganhadores_6_acertos'],
    header=0
)
# 1. Preparação dos dados para o Modelo
# Vamos tentar prever os números do concurso N baseado no concurso N-1
df_train = df.copy()

# Criar colunas de 'lag' (resultados do concurso anterior)
for i in range(1, 7):
    df_train[f'Prev_Bola{i}'] = df_train[f'Bola{i}'].shift(1)

# Remover a primeira linha que ficará com NaN devido ao shift
df_train = df_train.dropna()

# Features: Número do concurso + resultados anteriores
X = df_train[['Concurso', 'Prev_Bola1', 'Prev_Bola2', 'Prev_Bola3', 'Prev_Bola4', 'Prev_Bola5', 'Prev_Bola6']]
# Targets: As 6 bolas do concurso atual
y = df_train[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']]

# 2. Configurar o modelo (limitando n_estimators para poupar RAM)
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

print("Treinando o modelo RandomForest (isso pode levar alguns segundos)...")
model.fit(X, y)

# 3. Preparar entrada para o próximo concurso (2955)
ultimo_concurso = df.iloc[-1]
proximo_id = ultimo_concurso['Concurso'] + 1

# Criamos um DataFrame em vez de um np.array para manter os nomes das colunas
entrada_previsao = pd.DataFrame([[
    proximo_id,
    ultimo_concurso['Bola1'], ultimo_concurso['Bola2'], ultimo_concurso['Bola3'],
    ultimo_concurso['Bola4'], ultimo_concurso['Bola5'], ultimo_concurso['Bola6']
]], columns=['Concurso', 'Prev_Bola1', 'Prev_Bola2', 'Prev_Bola3', 'Prev_Bola4', 'Prev_Bola5', 'Prev_Bola6'])

# 4. Gerar 20 cartões baseados na sugestão do modelo
# O modelo dá uma "média", adicionamos uma variação para diversificar os jogos
print(f"Gerando 20 cartões para o concurso {proximo_id}...")

# Agora o predict recebe o DataFrame e o warning desaparece
previsao_base = model.predict(entrada_previsao)[0]
meus_jogos_ia = []

for _ in range(20):
    # Adicionamos um pequeno desvio padrão para diversificar
    variacao = np.random.normal(0, 5, size=6) # Aumentei um pouco a variação para 5
    jogo = np.round(previsao_base + variacao).astype(int)

    # Validações: números únicos e no range 1-60
    jogo = np.clip(jogo, 1, 60)
    jogo = np.unique(jogo)

    # Se faltar números por serem repetidos, completa com aleatórios
    while len(jogo) < 6:
        novo_num = np.random.randint(1, 61)
        if novo_num not in jogo:
            jogo = np.append(jogo, novo_num)

    jogo.sort()
    meus_jogos_ia.append(jogo)

# EXIBIÇÃO: O bloco abaixo deve estar FORA do loop 'for _ in range(20)'
print("\n" + "="*45)
print(f"   20 CARTÕES PARA O CONCURSO {proximo_id}")
print("="*45)

for i, jogo in enumerate(meus_jogos_ia, 1):
    # Formatação elegante
    jogo_formatado = " - ".join([f"{int(n):02d}" for n in jogo])
    print(f"Jogo {i:02d}: [ {jogo_formatado} ]")

print("="*45)
print("Boa sorte! Lembre-se: IA ajuda, mas loteria é sorte.")