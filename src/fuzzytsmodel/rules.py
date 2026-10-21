"""Passos 2, 3 e 4 de Wang-Mendel: gerar regras, pesá-las e resolver conflitos."""

import numpy as np
from tqdm import tqdm
from simpful import LinguisticVariable

Antecedent = tuple[str, ...]


class FuzzyRuleManager:
    """Base de regras fuzzy aprendida a partir dos dados.

    Cada antecedente aparece no máximo uma vez: quando dois dados geram o mesmo
    antecedente com consequentes diferentes (conflito), vence o de maior peso
    — o Passo 4 do método. O índice `_index` mapeia antecedente -> posição em
    `rules`, de modo que a resolução do conflito atinja sempre a regra correta.
    """

    def __init__(self, prune_weight_threshold: float = 0.1, prune_use_threshold: int = 0,
                 prune_window: int = 15, prune_error_threshold: float = 0.1,
                 max_rules: int = None, aggregation_fun="product"):
        """
        Args:
            prune_weight_threshold: pertinência mínima para a regra contar como usada.
            prune_use_threshold: número mínimo de usos para a regra ser mantida.
            prune_error_threshold: erro médio abaixo do qual uma regra pouco usada
                é considerada descartável.
            prune_window: número de predições entre avaliações de uso.
            max_rules: limite de regras na base. None = ilimitado.
            aggregation_fun: como combinar os graus de pertinência para obter o peso
                ("product", "min", "max", "arit_mean" ou um callable).
        """
        assert prune_window > prune_use_threshold, \
            "prune_window deve ser maior que prune_use_threshold"

        self.rules: list[str] = []
        self.weights: list[float] = []
        self.usage_count: list[int] = []
        self.error_contribution: list[float] = []
        self._index: dict[Antecedent, int] = {}
        self.prune_count = 0

        self.prune_weight_threshold = prune_weight_threshold
        self.prune_use_threshold = prune_use_threshold
        self.prune_error_threshold = prune_error_threshold
        self.prune_window = prune_window
        self.max_rules = max_rules
        self.aggregation_fun = aggregation_fun

    @staticmethod
    def strong_pertinence(var: LinguisticVariable, value: float) -> tuple[str, float]:
        """Passo 2: região de maior grau de pertinência para um valor."""
        values = var.get_values(value)
        terms = list(values.keys())
        degrees = list(values.values())
        best = int(np.argmax(degrees))
        return terms[best], degrees[best]

    @staticmethod
    def build_rule(input_names: list[str], output_name: str, terms) -> str:
        """Monta a regra no formato aceito pelo simpful."""
        antecedent = " AND ".join(
            f"({name} IS {term})" for name, term in zip(input_names, terms)
        )
        return f"IF {antecedent} THEN ({output_name} IS {terms[-1]})"

    def _aggregate_weights(self, degrees: list[float]) -> float:
        """Passo 3: peso da regra a partir dos graus de pertinência."""
        if callable(self.aggregation_fun):
            return self.aggregation_fun(degrees)
        if self.aggregation_fun == "product":
            return float(np.prod(degrees))
        if self.aggregation_fun == "min":
            return float(min(degrees))
        if self.aggregation_fun == "max":
            return float(max(degrees))
        if self.aggregation_fun == "arit_mean":
            return float(np.mean(degrees))
        raise ValueError(f"Função de agregação desconhecida: {self.aggregation_fun!r}")

    def rebuild_index(self) -> None:
        """Recria o índice antecedente -> posição a partir de `rules`."""
        self._index = {
            self.antecedent_of(rule): i for i, rule in enumerate(self.rules)
        }

    @staticmethod
    def antecedent_of(rule: str) -> Antecedent:
        """Termos do antecedente de uma regra já montada."""
        head = rule.split(" THEN ")[0]
        return tuple(part.split(" IS ")[1].rstrip(")") for part in head.split(" AND "))

    def update_rules(self, input_vars: list[LinguisticVariable], output_var: LinguisticVariable,
                     values_io: list[float], var_names: list[str],
                     sample_weight: float = 1.0) -> None:
        """Gera a regra de uma amostra e a insere na base, resolvendo conflitos.

        `sample_weight` é o grau de importância m(i) do dado: quando há confiança
        diferente entre as medições, ele multiplica o peso da regra.
        """
        terms, degrees = [], []
        for var, value in zip(input_vars + [output_var], values_io):
            term, degree = self.strong_pertinence(var, value)
            terms.append(term)
            degrees.append(degree)

        new_weight = self._aggregate_weights(degrees) * sample_weight
        new_rule = self.build_rule(var_names[:-1], var_names[-1], terms)
        antecedent = tuple(terms[:-1])

        idx = self._index.get(antecedent)

        if idx is not None:
            # Passo 4: mesmo antecedente já existe -> fica a regra de maior peso.
            if new_weight > self.weights[idx]:
                self.rules[idx] = new_rule
                self.weights[idx] = new_weight
            return

        if self.max_rules is not None and len(self.rules) >= self.max_rules:
            weakest = int(np.argmin(self.weights))
            if new_weight <= self.weights[weakest]:
                return
            tqdm.write(
                f"Substituindo regra {weakest}: peso {self.weights[weakest]:.4f}"
                f" -> {new_weight:.4f}."
            )
            del self._index[self.antecedent_of(self.rules[weakest])]
            self.rules[weakest] = new_rule
            self.weights[weakest] = new_weight
            self.usage_count[weakest] = 0
            self.error_contribution[weakest] = 0.0
            self._index[antecedent] = weakest
            return

        self.rules.append(new_rule)
        self.weights.append(new_weight)
        self.usage_count.append(0)
        self.error_contribution.append(0.0)
        self._index[antecedent] = len(self.rules) - 1

    def register_rule_usage(self, prediction_error: float = None) -> None:
        """Conta uso das regras ativas e acumula o erro a elas associado."""
        for idx, weight in enumerate(self.weights):
            if weight > self.prune_weight_threshold:
                self.usage_count[idx] += 1
                if prediction_error is not None:
                    self.error_contribution[idx] += abs(prediction_error)

    def prune_unused_rules(self) -> bool:
        """Extensão (fora de Wang-Mendel): remove regras pouco usadas e de baixo impacto.

        Retorna True quando a base foi reavaliada (tenha havido remoção ou não).
        """
        if self.prune_count < self.prune_window:
            self.prune_count += 1
            return False

        to_remove = [
            i for i, count in enumerate(self.usage_count)
            if count <= self.prune_use_threshold
            and self.error_contribution[i] / max(1, count) < self.prune_error_threshold
        ]
        # Nunca esvaziar a base: sem regras não há inferência possível.
        if len(to_remove) == len(self.rules):
            to_remove = []

        if to_remove:
            tqdm.write(
                f"Podando {len(to_remove)} de {len(self.rules)} regras "
                f"-> {len(self.rules) - len(to_remove)} restantes."
            )
            for idx in sorted(to_remove, reverse=True):
                del self.rules[idx]
                del self.weights[idx]
                del self.usage_count[idx]
                del self.error_contribution[idx]
            self.rebuild_index()

        self.usage_count = [0 for _ in self.rules]
        self.error_contribution = [0.0 for _ in self.rules]
        self.prune_count = 0
        return True
