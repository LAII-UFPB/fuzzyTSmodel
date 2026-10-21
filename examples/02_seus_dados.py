"""Ponto de partida para aplicar o modelo nos seus próprios dados.

Troque o bloco "SEUS DADOS" e rode:  python examples/02_seus_dados.py
"""

import numpy as np
import matplotlib.pyplot as plt

from fuzzytsmodel import FuzzyTSModel

# =====================================================================
# SEUS DADOS
# =====================================================================
# Há dois formatos possíveis.
#
# (a) Série temporal: você tem uma única sequência e quer prever o próximo
#     valor a partir dos anteriores. Informe a série em `serie` e quantos
#     valores passados usar em N_LAGS.
#
# (b) Entradas e saída já separadas (ex.: temperatura e vento -> chuva):
#     monte X com shape (n_amostras, n_variaveis) e y com (n_amostras,),
#     e pule a construção de lags abaixo.
#
# Exemplo (a), com dados sintéticos — substitua por np.loadtxt("seus_dados.csv")
# ou pandas.read_csv(...).values:
N_LAGS = 3
serie = np.sin(np.arange(0, 6 * np.pi, 0.05)) + 0.1 * np.cos(np.arange(0, 6 * np.pi, 0.05) * 3)

nomes_entradas = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
nome_saida = "valor"

X = np.vstack([np.roll(serie, lag) for lag in range(1, N_LAGS + 1)]).T
X, y = X[N_LAGS:], serie[N_LAGS:]

# =====================================================================
# Treino / validação — sem embaralhar: em série temporal a ordem importa.
# =====================================================================
corte = int(len(y) * 0.7)
X_treino, y_treino = X[:corte], y[:corte]
X_val, y_val = X[corte:], y[corte:]

# =====================================================================
# Modelo
# =====================================================================
# N controla a granularidade: cada variável é dividida em 2N+1 regiões.
#   N baixo  -> poucas regras, modelo mais suave, pode não capturar detalhes
#   N alto   -> mais regras, mais detalhe, risco de decorar o ruído
# Comece em 5-7 e ajuste olhando o erro de validação.
#
# input_range="auto" deriva a faixa de cada variável dos próprios dados.
# Use faixas explícitas se os dados novos puderem extrapolar o treino.
model = FuzzyTSModel(
    input_names=nomes_entradas,
    output_name=nome_saida,
    N=6,
    input_range="auto",
    output_range="auto",
)

model.fit(X_treino, y_treino)

print(f"{len(X_treino)} amostras -> {len(model.explain())} regras\n")
for regra in model.explain()[:10]:
    print("  ", regra)

# =====================================================================
# Avaliação
# =====================================================================
y_pred = model.predict(X_val, prune=False)
metricas = model.score(y_pred=y_pred, y_true=y_val)
print("\nMétricas na validação:")
for nome, valor in metricas.items():
    print(f"  {nome}: {valor:.4f}")

plt.plot(y_val, label="Real")
plt.plot(y_pred, label="Previsto")
plt.title("Validação: real vs. previsto")
plt.legend()
plt.grid()
plt.show()

# model.save("models/meu_modelo")                        # salvar
# model = FuzzyTSModel.from_file("models/meu_modelo.npz")  # recarregar
