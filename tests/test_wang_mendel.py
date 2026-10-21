"""Testes do método de Wang-Mendel implementado em fuzzytsmodel."""

import numpy as np
import pytest

from fuzzytsmodel import FuzzyTSModel, FuzzyRuleManager, FuzzyVariableManager, region_terms


@pytest.fixture
def manager():
    return FuzzyVariableManager(N=2, input_range=[-1, 1], output_range=[-1, 1])


def sine_dataset(n_lags=3, step=0.05):
    t = np.arange(0, 8 * np.pi, step)
    y = np.sin(t)
    X = np.vstack([np.roll(y, lag) for lag in range(1, n_lags + 1)]).T
    return X[n_lags:], y[n_lags:]


# --- Passo 1: partição em 2N+1 regiões ------------------------------------

def test_region_terms_segue_nomenclatura_dos_slides():
    assert region_terms(2) == ["S2", "S1", "CE", "B1", "B2"]
    assert region_terms(3) == ["S3", "S2", "S1", "CE", "B1", "B2", "B3"]


def test_variavel_tem_2N_mais_1_regioes(manager):
    var = manager.create_variable("x")
    assert len(var.get_values(0.0)) == 2 * 2 + 1


def test_N_por_variavel():
    mgr = FuzzyVariableManager(N={"x": 2, "y": 4}, input_range=[-1, 1], output_range=[-1, 1])
    assert len(mgr.create_variable("x").get_values(0.0)) == 5
    assert len(mgr.create_variable("y", is_output=True).get_values(0.0)) == 9


def test_faixa_por_variavel():
    mgr = FuzzyVariableManager(
        N=2, input_range={"a": [0, 10], "b": [-5, 5]}, output_range=[0, 1]
    )
    assert mgr.create_variable("a")._universe_of_discourse == [0, 10]
    assert mgr.create_variable("b")._universe_of_discourse == [-5, 5]


# --- Passo 2: uma regra por amostra ---------------------------------------

def test_strong_pertinence_escolhe_a_regiao_de_maior_grau(manager):
    var = manager.create_variable("x")
    term, degree = FuzzyRuleManager.strong_pertinence(var, 1.0)
    assert term == "B2"
    assert degree == pytest.approx(1.0)
    assert FuzzyRuleManager.strong_pertinence(var, 0.0)[0] == "CE"


def test_build_rule_e_antecedent_of_sao_consistentes():
    rule = FuzzyRuleManager.build_rule(["a", "b"], "y", ["S1", "CE", "B2"])
    assert rule == "IF (a IS S1) AND (b IS CE) THEN (y IS B2)"
    assert FuzzyRuleManager.antecedent_of(rule) == ("S1", "CE")


# --- Passo 3: peso da regra ------------------------------------------------

def test_peso_e_o_produto_dos_graus():
    rm = FuzzyRuleManager()
    assert rm._aggregate_weights([0.8, 0.7, 0.9]) == pytest.approx(0.504)


def test_sample_weight_multiplica_o_peso(manager):
    var = manager.create_variable("x")
    out = manager.create_variable("y", is_output=True)
    rm = FuzzyRuleManager()
    rm.update_rules([var], out, [1.0, 1.0], ["x", "y"], sample_weight=0.5)
    assert rm.weights[0] == pytest.approx(0.5)


# --- Passo 4: conflito resolvido pelo maior peso ---------------------------

def test_conflito_substitui_a_regra_certa_no_meio_da_base():
    """Regressão: a resolução de conflito atingia a última regra, não a conflitante."""
    rm = FuzzyRuleManager()
    rm.rules = [
        "IF (v1 IS CE) AND (v2 IS B1) THEN (out IS S1)",
        "IF (v1 IS S1) AND (v2 IS S2) THEN (out IS CE)",   # a conflitante, no meio
        "IF (v1 IS B2) AND (v2 IS CE) THEN (out IS B1)",
    ]
    rm.weights = [0.9, 0.1, 0.9]
    rm.usage_count = [0, 0, 0]
    rm.error_contribution = [0.0, 0.0, 0.0]
    rm.rebuild_index()

    assert rm._index[("S1", "S2")] == 1

    mgr = FuzzyVariableManager(N=2, input_range=[-1, 1], output_range=[-1, 1])
    v1, v2 = mgr.create_variable("v1"), mgr.create_variable("v2")
    out = mgr.create_variable("out", is_output=True)
    # valores que caem em (S1, S2) com pertinência alta e consequente diferente
    rm.update_rules([v1, v2], out, [-0.5, -1.0, 1.0], ["v1", "v2", "out"])

    assert len(rm.rules) == 3, "conflito não deve criar regra nova"
    assert rm.rules[0] == "IF (v1 IS CE) AND (v2 IS B1) THEN (out IS S1)", "regra 0 intacta"
    assert rm.rules[2] == "IF (v1 IS B2) AND (v2 IS CE) THEN (out IS B1)", "regra 2 intacta"
    assert rm.rules[1].endswith("THEN (out IS B2)"), "a regra 1 deveria ter sido substituída"


def test_regra_de_menor_peso_nao_sobrescreve_a_de_maior():
    mgr = FuzzyVariableManager(N=2, input_range=[-1, 1], output_range=[-1, 1])
    v1 = mgr.create_variable("v1")
    out = mgr.create_variable("out", is_output=True)
    rm = FuzzyRuleManager()

    rm.update_rules([v1], out, [1.0, 1.0], ["v1", "out"])      # peso 1.0
    forte = rm.rules[0]
    rm.update_rules([v1], out, [0.9, -1.0], ["v1", "out"])     # mesmo antecedente, peso menor

    assert len(rm.rules) == 1
    assert rm.rules[0] == forte


def test_antecedentes_sao_unicos_apos_fit():
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=4,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5]).fit(X, y)
    antecedentes = [FuzzyRuleManager.antecedent_of(r) for r in model.explain()]
    assert len(antecedentes) == len(set(antecedentes))


def test_indice_fica_consistente_com_as_regras():
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=4,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5]).fit(X, y)
    rm = model.rule_manager
    for antecedente, idx in rm._index.items():
        assert FuzzyRuleManager.antecedent_of(rm.rules[idx]) == antecedente


# --- Modelo ----------------------------------------------------------------

def test_fit_duas_vezes_nao_duplica_regras():
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=4,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5])
    model.fit(X, y)
    n_regras = len(model.explain())
    model.fit(X, y)
    assert len(model.explain()) == n_regras
    assert len(model.fs.get_rules()) == n_regras


def test_predict_sem_prune_e_reproduzivel():
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=4,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5]).fit(X, y)
    primeira = model.predict(X[:40], prune=False, progress=False)
    segunda = model.predict(X[:40], prune=False, progress=False)
    np.testing.assert_allclose(primeira, segunda)


def test_predict_rejeita_numero_errado_de_colunas():
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=4,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5]).fit(X, y)
    with pytest.raises(ValueError, match="colunas"):
        model.predict(X[:, :2], progress=False)


def test_predict_antes_de_fit_da_erro_claro():
    model = FuzzyTSModel(["a"], "out", N=2, input_range=[-1, 1], output_range=[-1, 1])
    with pytest.raises(RuntimeError, match="não tem regras"):
        model.predict(np.zeros((3, 1)), progress=False)


def test_from_file_reconstroi_o_modelo_sem_fit(tmp_path):
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=4,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5]).fit(X, y)
    esperado = model.predict(X[:30], prune=False, progress=False)

    destino = tmp_path / "modelo"
    model.save(str(destino))

    recarregado = FuzzyTSModel.from_file(str(destino) + ".npz")
    assert recarregado.explain() == model.explain()
    np.testing.assert_allclose(
        recarregado.predict(X[:30], prune=False, progress=False), esperado
    )


def test_score_ignora_y_true_proximo_de_zero():
    model = FuzzyTSModel(["a"], "out", N=2, input_range=[-1, 1], output_range=[-1, 1])
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.5, 1.0, 2.0])
    metricas = model.score(y_pred=y_pred, y_true=y_true)
    assert np.isfinite(metricas["MAPE"])
    assert metricas["MAPE"] == pytest.approx(0.0)


def test_input_range_auto_deriva_dos_dados():
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=4,
                         input_range="auto", output_range="auto").fit(X, y)
    assert model.var_manager.output_range[0] == pytest.approx(y.min())
    assert model.var_manager.input_range["a"][1] == pytest.approx(X[:, 0].max())


def test_max_rules_limita_a_base():
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=6, max_rules=10,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5]).fit(X, y)
    assert len(model.explain()) <= 10


def test_predict_and_update_sem_y_true_equivale_a_predict():
    X, y = sine_dataset()

    def novo_modelo():
        return FuzzyTSModel(["a", "b", "c"], "out", N=4,
                            input_range=[-1.5, 1.5], output_range=[-1.5, 1.5]).fit(X, y)

    esperado = novo_modelo().predict(X[:30], progress=False)
    obtido = novo_modelo().predict_and_update(X[:30])
    np.testing.assert_allclose(obtido, esperado)


def test_poda_nunca_esvazia_a_base():
    """Com limiares extremos toda regra vira candidata; a base não pode zerar."""
    X, y = sine_dataset()
    model = FuzzyTSModel(["a", "b", "c"], "out", N=4,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5],
                         prune_weight_threshold=1.0, prune_use_threshold=0,
                         prune_window=5, prune_error_threshold=10 ** 6).fit(X, y)
    model.predict(X[:60], progress=False)
    assert len(model.explain()) > 0


def test_modelo_aprende_a_senoide():
    X, y = sine_dataset()
    corte = len(y) // 2
    model = FuzzyTSModel(["a", "b", "c"], "out", N=6,
                         input_range=[-1.5, 1.5], output_range=[-1.5, 1.5])
    model.fit(X[:corte], y[:corte])
    y_pred = model.predict(X[corte:], prune=False, progress=False)
    assert model.score(y_pred=y_pred, y_true=y[corte:])["R2"] > 0.95
