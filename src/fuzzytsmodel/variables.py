"""Passo 1 de Wang-Mendel: divisão dos domínios em 2N+1 regiões fuzzy."""

import numpy as np
from simpful import AutoTriangle, LinguisticVariable

CENTER_TERM = "CE"


def region_terms(N: int) -> list[str]:
    """Rótulos das 2N+1 regiões, na ordem do universo: S_N ... S1, CE, B1 ... B_N.

    Nomenclatura dos slides da disciplina (S = Small, B = Big, CE = Centro).
    """
    assert N > 0, "N deve ser maior que 0"
    return (
        [f"S{N - i}" for i in range(N)]
        + [CENTER_TERM]
        + [f"B{i + 1}" for i in range(N)]
    )


class FuzzyVariableManager:
    """Cria e guarda as variáveis linguísticas (entradas e saída).

    N e os domínios podem ser únicos (um valor para todas as variáveis) ou
    definidos por variável, via dict — o método permite granularidades
    diferentes para cada variável.
    """

    def __init__(self, N: int | dict[str, int], input_range, output_range):
        """
        Args:
            N: número de conjuntos por lado; cada variável recebe 2N+1 regiões.
                int aplica o mesmo N a todas, ou dict {nome_da_variavel: N}.
            input_range: [min, max] usado por todas as entradas, ou
                dict {nome_da_variavel: [min, max]}.
            output_range: [min, max] da variável de saída.
        """
        self._validate_N(N)
        self.N = N
        self.input_range = input_range
        self.output_range = output_range
        self.variables: dict[str, LinguisticVariable] = {}

        if not isinstance(input_range, dict):
            self._validate_range(input_range, "input_range")
        else:
            for name, rng in input_range.items():
                self._validate_range(rng, f"input_range[{name!r}]")
        self._validate_range(output_range, "output_range")

    @staticmethod
    def _validate_N(N) -> None:
        values = N.values() if isinstance(N, dict) else [N]
        for value in values:
            assert isinstance(value, (int, np.integer)) and value > 0, \
                f"N deve ser um inteiro maior que 0, recebido {value!r}"

    @staticmethod
    def _validate_range(rng, label: str) -> None:
        assert len(rng) == 2 and rng[0] < rng[1], \
            f"{label} deve ser [min, max] com min < max, recebido {rng!r}"

    def n_for(self, name: str) -> int:
        """N da variável `name` (cai no valor global quando não há um específico)."""
        if isinstance(self.N, dict):
            if name not in self.N:
                raise KeyError(
                    f"N não foi definido para a variável {name!r}. "
                    f"Variáveis com N definido: {sorted(self.N)}"
                )
            return self.N[name]
        return self.N

    def universe_for(self, name: str, is_output: bool):
        if is_output:
            return self.output_range
        if isinstance(self.input_range, dict):
            if name not in self.input_range:
                raise KeyError(
                    f"input_range não foi definido para a variável {name!r}. "
                    f"Variáveis com faixa definida: {sorted(self.input_range)}"
                )
            return self.input_range[name]
        return self.input_range

    def create_variable(self, name: str, is_output: bool = False) -> LinguisticVariable:
        """Divide o domínio de `name` em 2N+1 triângulos e registra a variável."""
        universe = self.universe_for(name, is_output)
        terms = region_terms(self.n_for(name))

        auto = AutoTriangle(
            n_sets=len(terms),
            terms=terms,
            universe_of_discourse=universe,
        )
        lv = LinguisticVariable(auto._FSlist, concept=name, universe_of_discourse=universe)
        self.variables[name] = lv
        return lv

    def get(self, name: str) -> LinguisticVariable:
        if name not in self.variables:
            raise KeyError(
                f"Variável {name!r} ainda não foi criada. "
                f"Disponíveis: {sorted(self.variables)}. "
                "Com input_range='auto' as variáveis só existem depois de fit()."
            )
        return self.variables[name]
