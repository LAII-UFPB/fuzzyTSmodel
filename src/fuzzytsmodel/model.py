"""Modelo fuzzy para previsão de séries temporais com regras aprendidas dos dados."""

import json

import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from simpful import FuzzySystem

from .rules import FuzzyRuleManager
from .variables import FuzzyVariableManager

AUTO = "auto"


class FuzzyTSModel:
    """Sistema de inferência fuzzy cuja base de regras vem dos dados (Wang-Mendel).

    Uso típico:
        model = FuzzyTSModel(["x1", "x2"], "y", N=5,
                             input_range=[-1, 1], output_range=[-1, 1])
        model.fit(X, y)
        y_pred = model.predict(X_novo)
        print(model.explain())
    """

    def __init__(self, input_names: list, output_name: str, N, input_range, output_range,
                 update_rule_window: int = 15, max_rules: int = None,
                 aggregation_fun="product", prune_weight_threshold: float = 0.1,
                 prune_use_threshold: int = 0, prune_window: int = 15,
                 prune_error_threshold: float = 0.1):
        """
        Args:
            input_names: nomes das variáveis de entrada; definem a ordem das colunas de X.
            output_name: nome da variável de saída.
            N: 2N+1 regiões por variável. int, ou dict {nome: N} para granularidade
                diferente por variável.
            input_range: [min, max] das entradas, dict {nome: [min, max]}, ou "auto"
                para derivar de X em fit(). Com "auto", as variáveis só passam a
                existir depois de fit().
            output_range: [min, max] da saída, ou "auto".
            update_rule_window: tamanho do bloco de predição em predict_and_update().
            max_rules: limite de regras na base. None = ilimitado.
            aggregation_fun: forma de combinar os graus de pertinência no peso da regra.
            prune_*: parâmetros da poda (extensão, fora de Wang-Mendel).
        """
        self.input_names = list(input_names)
        self.output_name = output_name
        self.update_rule_window = update_rule_window
        self._N = N
        self._input_range = input_range
        self._output_range = output_range

        self.rule_manager = FuzzyRuleManager(
            max_rules=max_rules,
            aggregation_fun=aggregation_fun,
            prune_weight_threshold=prune_weight_threshold,
            prune_use_threshold=prune_use_threshold,
            prune_window=prune_window,
            prune_error_threshold=prune_error_threshold,
        )
        self.fs = FuzzySystem(show_banner=False)
        self.var_manager = None
        self.input_vars = None
        self.output_var = None

        if not self._needs_data_for_ranges():
            self._build_variables(self._input_range, self._output_range)

    @property
    def n_features(self) -> int:
        return len(self.input_names)

    def _needs_data_for_ranges(self) -> bool:
        return AUTO in (self._input_range, self._output_range)

    def _build_variables(self, input_range, output_range) -> None:
        """Passo 1: cria as variáveis linguísticas e as registra no sistema fuzzy."""
        self.var_manager = FuzzyVariableManager(self._N, input_range, output_range)
        self.input_vars = [self.var_manager.create_variable(name) for name in self.input_names]
        self.output_var = self.var_manager.create_variable(self.output_name, is_output=True)

        self.fs.add_linguistic_variable(self.output_name, self.output_var)
        for name, var in zip(self.input_names, self.input_vars):
            self.fs.add_linguistic_variable(name, var)

    def _ensure_variables(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.var_manager is not None:
            return
        input_range = self._input_range
        if input_range == AUTO:
            input_range = {
                name: [float(X[:, i].min()), float(X[:, i].max())]
                for i, name in enumerate(self.input_names)
            }
        output_range = self._output_range
        if output_range == AUTO:
            output_range = [float(np.min(y)), float(np.max(y))]
        self._build_variables(input_range, output_range)

    def _sync_rules(self) -> None:
        """Espelha a base de regras no sistema simpful."""
        # simpful não expõe um clear público; _rules é a única via de reset.
        self.fs._rules.clear()
        if self.rule_manager.rules:
            self.fs.add_rules(self.rule_manager.rules)

    def _check_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X deve ser 2D (n_amostras, {self.n_features}), recebido shape {X.shape}")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X tem {X.shape[1]} colunas, mas o modelo espera {self.n_features} "
                f"({', '.join(self.input_names)})"
            )
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None,
            progress: bool = True) -> "FuzzyTSModel":
        """Aprende a base de regras a partir do conjunto de dados.

        Args:
            sample_weight: grau de importância m(i) de cada amostra. Opcional.
            progress: exibe a barra de progresso.
        """
        X, y = np.asarray(X), np.asarray(y)
        assert X.shape[0] == y.shape[0], "X e y devem ter o mesmo número de amostras"
        self._ensure_variables(X, y)
        X = self._check_input(X)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        var_names = self.input_names + [self.output_name]
        amostras = zip(X, y, sample_weight)
        if progress:
            amostras = tqdm(amostras, total=X.shape[0], desc="Aprendendo regras")
        for xi, yi, wi in amostras:
            self.rule_manager.update_rules(
                self.input_vars, self.output_var, list(xi) + [yi], var_names, sample_weight=wi
            )
        self._sync_rules()
        return self

    def partial_fit(self, xi: np.ndarray, yi: float) -> "FuzzyTSModel":
        """Extensão: atualiza a base com uma única amostra (aprendizado incremental)."""
        if self.var_manager is None:
            raise RuntimeError("Chame fit() antes de partial_fit() quando usar input_range='auto'.")
        self.rule_manager.update_rules(
            self.input_vars, self.output_var, list(xi) + [yi],
            self.input_names + [self.output_name],
        )
        self._sync_rules()
        return self

    def predict(self, X: np.ndarray, y_true: np.ndarray = None, prune: bool = True,
                progress: bool = True) -> np.ndarray:
        """Infere a saída para cada linha de X.

        Args:
            y_true: valores reais, usados só para atribuir erro às regras (poda).
            prune: quando False, a predição não altera a base de regras — use assim
                para obter uma predição reproduzível. A poda é uma extensão, não
                faz parte de Wang-Mendel.
        """
        if not self.rule_manager.rules:
            raise RuntimeError("O modelo não tem regras. Chame fit() ou load() antes de predict().")
        X = self._check_input(X)

        predictions = []
        iterator = tqdm(X, total=X.shape[0], desc="Prevendo") if progress else X
        for idx, xi in enumerate(iterator):
            for name, val in zip(self.input_names, xi):
                self.fs.set_variable(name, val)
            y_pred = self.fs.inference()[self.output_name]
            predictions.append(y_pred)

            if prune:
                error = None if y_true is None else y_true[idx] - y_pred
                self.rule_manager.register_rule_usage(prediction_error=error)
                if self.rule_manager.prune_unused_rules():
                    self._sync_rules()

        return np.array(predictions)

    def predict_and_update(self, X: np.ndarray, y_true: np.ndarray = None,
                           abs_error_threshold: float = 0.05,
                           rel_error_threshold: float = None,
                           verbose: bool = True) -> np.ndarray:
        """Extensão: prevê em blocos e reaprende quando o erro passa do limiar.

        Sem `y_true` não há erro a medir, e o método equivale a predict().
        """
        X = self._check_input(X)
        if y_true is None:
            return self.predict(X)

        y_true = np.asarray(y_true)
        window = self.update_rule_window
        y_pred = np.zeros(X.shape[0])

        for start in range(0, X.shape[0], window):
            end = min(start + window, X.shape[0])
            X_batch, y_batch_true = X[start:end], y_true[start:end]
            y_batch_pred = self.predict(X_batch, y_true=y_batch_true)

            abs_errors = np.abs(y_batch_true - y_batch_pred)
            rel_errors = np.full_like(abs_errors, np.inf)
            nonzero = np.abs(y_batch_true) > 1e-8
            rel_errors[nonzero] = abs_errors[nonzero] / np.abs(y_batch_true[nonzero])

            for i, (xi, yi, yp) in enumerate(zip(X_batch, y_batch_true, y_batch_pred)):
                update = abs_error_threshold is not None and abs_errors[i] > abs_error_threshold
                if rel_error_threshold is not None and rel_errors[i] > rel_error_threshold:
                    update = True
                if update:
                    if verbose:
                        tqdm.write(
                            f"Atualizando: y_true={yi:.4f}, y_pred={yp:.4f}, "
                            f"erro_abs={abs_errors[i]:.4f}"
                        )
                    self.partial_fit(xi, yi)

            y_pred[start:end] = y_batch_pred

        return y_pred

    def explain(self) -> list[str]:
        """Base de regras aprendida, como lista de strings legíveis."""
        return list(self.rule_manager.rules)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """Métricas de erro da predição.

        MAPE é calculado só onde |y_true| > 1e-8: perto de zero ele é indefinido,
        e numa senoide isso distorceria o valor.
        """
        y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
        nonzero = np.abs(y_true) > 1e-8
        mape = (
            float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
            if np.any(nonzero) else float("nan")
        )
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MAPE": mape,
            "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "R2": float(r2_score(y_true, y_pred)),
        }

    def _meta(self) -> dict:
        return {
            "input_names": self.input_names,
            "output_name": self.output_name,
            "N": self._N,
            "input_range": self.var_manager.input_range if self.var_manager else self._input_range,
            "output_range": self.var_manager.output_range if self.var_manager else self._output_range,
            "update_rule_window": self.update_rule_window,
            "aggregation_fun": (
                self.rule_manager.aggregation_fun
                if isinstance(self.rule_manager.aggregation_fun, str) else "product"
            ),
            "max_rules": self.rule_manager.max_rules,
        }

    def save(self, path: str) -> None:
        """Salva regras, pesos e hiperparâmetros num .npz."""
        np.savez(
            path,
            rules=np.array(self.rule_manager.rules, dtype=object),
            weights=np.array(self.rule_manager.weights, dtype=float),
            usage=np.array(self.rule_manager.usage_count, dtype=int),
            meta=json.dumps(self._meta()),
        )

    def load(self, path: str) -> "FuzzyTSModel":
        """Carrega regras e pesos de um .npz para este modelo."""
        data = np.load(path, allow_pickle=True)
        self.rule_manager.rules = [str(r) for r in data["rules"].tolist()]
        self.rule_manager.weights = data["weights"].tolist()
        self.rule_manager.usage_count = data["usage"].tolist()
        self.rule_manager.error_contribution = [0.0 for _ in self.rule_manager.rules]
        self.rule_manager.rebuild_index()
        self._sync_rules()
        return self

    @classmethod
    def from_file(cls, path: str) -> "FuzzyTSModel":
        """Reconstrói o modelo inteiro a partir de um .npz, sem precisar de fit()."""
        data = np.load(path, allow_pickle=True)
        if "meta" not in data:
            raise ValueError(
                f"{path} foi salvo por uma versão antiga, sem hiperparâmetros. "
                "Recrie o modelo manualmente e use load()."
            )
        meta = json.loads(str(data["meta"]))
        model = cls(
            input_names=meta["input_names"],
            output_name=meta["output_name"],
            N=meta["N"],
            input_range=meta["input_range"],
            output_range=meta["output_range"],
            update_rule_window=meta["update_rule_window"],
            max_rules=meta["max_rules"],
            aggregation_fun=meta["aggregation_fun"],
        )
        return model.load(path)
