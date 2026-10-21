"""Previsão de uma onda senoidal com regras fuzzy aprendidas dos dados.

Corresponde ao exemplo dos slides "Previsão Usando Fuzzy - Onda Seno".
Execute com:  python examples/01_sine_wang_mendel.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from fuzzytsmodel import FuzzyTSModel

N_LAGS = 3

# ---------------------------------------------------------------- dados
# Série com três regimes: seno, seno invertido e seno amortecido. Os três
# trechos existem para mostrar que o modelo precisa de regras diferentes
# em cada região do espaço de entrada.
t = np.arange(0, 8 * np.pi, 0.02)
meio, dois_tercos = len(t) // 2, len(t) // 2 + len(t) // 3
y1 = np.sin(t[:meio])
y2 = -np.sin(t[meio:dois_tercos])
y3 = -np.exp(-(t[dois_tercos:] - t[dois_tercos])) * np.sin(t[dois_tercos:])
y = np.concatenate((y1, y2, y3))

# Entradas = valores passados da própria série (y[k-1], y[k-2], y[k-3]).
X = np.vstack([np.roll(y, lag) for lag in range(1, N_LAGS + 1)]).T
X, y, t = X[N_LAGS:], y[N_LAGS:], t[N_LAGS:]

corte = len(t) // 2
X_treino, y_treino = X[:corte], y[:corte]
X_val, y_val = X[corte:], y[corte:]

plt.plot(t[:corte], y_treino, label="Treino")
plt.plot(t[corte:], y_val, label="Validação")
plt.title("Dados de treino e validação")
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------------------------------- modelo
model = FuzzyTSModel(
    input_names=[f"y_k_menos_{i}" for i in range(1, N_LAGS + 1)],
    output_name="y_k",
    N=6,                       # 2*6+1 = 13 regiões por variável
    input_range=[-1.5, 1.5],
    output_range=[-1.5, 1.5],
    update_rule_window=50,
)

# Passo 1: as regiões fuzzy de uma das entradas.
primeira = model.var_manager.get("y_k_menos_1")
print("Graus de pertinência de y_k_menos_1 = 0.5:")
for termo, grau in primeira.get_values(0.5).items():
    if grau > 0:
        print(f"  {termo}: {grau:.3f}")
primeira.plot()

# Passos 2, 3 e 4: uma regra por amostra, pesada e com conflitos resolvidos.
model.fit(X_treino, y_treino)
print(f"\n{len(X_treino)} amostras -> {len(model.explain())} regras")
print("Primeiras regras aprendidas:")
for regra in model.explain()[:5]:
    print("  ", regra)

# --------------------------------------------------- salvar e recarregar
os.makedirs("models", exist_ok=True)
caminho = os.path.join("models", "fuzzyTS_sine")
model.save(caminho)
# from_file reconstrói o modelo inteiro, sem precisar repetir os parâmetros.
model = FuzzyTSModel.from_file(caminho + ".npz")

# ---------------------------------------------------------------- previsão
# prune=False mantém a base intacta: a mesma entrada dá sempre a mesma saída.
y_pred = model.predict(X_val, prune=False)

plt.plot(y_val, label="Real")
plt.plot(y_pred, label="Previsto")
plt.title("Validação: real vs. previsto")
plt.legend()
plt.grid()
plt.show()

metricas = model.score(y_pred=y_pred, y_true=y_val)
print("\nMétricas:")
for nome, valor in metricas.items():
    print(f"  {nome}: {valor:.4f}")
