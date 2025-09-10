import numpy as np
from functools import reduce
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from simpful import FuzzySystem, AutoTriangle, LinguisticVariable


class FuzzyVariableManager:
    """Manages fuzzy variables (inputs and outputs)."""

    def __init__(self, N:int, input_range:list, output_range:list):
        """
        Args:
            N (int): Number of fuzzy sets per variable (total sets = 2*N + 1)
            input_range (list): [min, max] range for input variables
            output_range (list): [min, max] range for output variable
        """
        assert N > 0, "N must be greater than 0"
        assert len(input_range) == 2 and input_range[0] < input_range[1], \
            "input_range must be [min, max] with min < max"
        assert len(output_range) == 2 and output_range[0] < output_range[1], \
            "output_range must be [min, max] with min < max"
        
        self.N = N
        self.input_range = input_range
        self.output_range = output_range
        self.variables = {}

    def create_variable(self, name: str, is_output: bool =False) -> LinguisticVariable:
        """
        Create fuzzy variable based on Auto Triangle and convert to Linguistic Variable.
        Args:
            name (str): variable name
            is_output (bool): whether variable is output
        """
        universe = self.output_range if is_output else self.input_range
        regions = 2 * self.N + 1

        # region names (S=small, B=big, Z=zero)
        terms = []
        for i in range(regions):
            if i < self.N:
                terms.append(f"S{abs(i - self.N)}")
            elif i > self.N:
                terms.append(f"B{abs(i - self.N)}")
            else:
                terms.append("Z")

        auto = AutoTriangle(
            n_sets=regions,
            terms=terms,
            universe_of_discourse=universe
        )

        lv = LinguisticVariable(auto._FSlist, concept=name, universe_of_discourse=universe)
        self.variables[name] = lv
        return lv

    def get(self, name:str) -> LinguisticVariable:
        return self.variables[name]


class FuzzyRuleManager:
    """Manages a fuzzy rule base with adaptive learning and forgetting."""

    def __init__(self, prune_weight_threshold:float=0.1, prune_use_threshold:int=0,
                 prune_window:int=15, max_rules:int=None, aggregation_fun="product"):
        """
        Args:
            prune_weight_threshold (float): Minimum weight for a rule to be considered used.
            prune_use_threshold (int): Minimum number of uses for a rule to be retained.
            prune_window (int): Number of predictions after which to evaluate rule usage.
            max_rules (int): Maximum number of rules to store. None = unlimited.
            aggregation_fun (str or callable): Aggregation function for pertinence values
                                               ("product", "min", "max", "arit_mean" or callable).
        """

        self.rules = []
        self.weights = []
        self.usage_count = []
        self.error_contribution = []  # track per-rule contribution to error
        self.prune_count = 0

        self.prune_weight_threshold = prune_weight_threshold
        self.prune_use_threshold = prune_use_threshold
        self.prune_window = prune_window
        self.max_rules = max_rules
        self.aggregation_fun = aggregation_fun

        assert self.prune_window > self.prune_use_threshold, \
            "prune_window must be greater than prune_use_threshold"
        
    @staticmethod
    def strong_pertinence(var:LinguisticVariable, value:float) -> tuple[str,float]:
        """Selects the term with the highest membership degree for a value."""
        values = var.get_values(value)
        terms = list(values.keys())
        vals = list(values.values())
        strong_term = terms[np.argmax(vals)]
        value_term = vals[np.argmax(vals)]
        return strong_term, value_term

    @staticmethod
    def build_rule(input_names: list[str], output_name: str, terms: str) -> str:
        """Creates rule string in simpful-compatible format."""
        rule_string = "IF "
        for i, name in enumerate(input_names):
            rule_string += f"({name} IS {terms[i]}) AND "
        rule_string = rule_string[:-4] + f"THEN ({output_name} IS {terms[-1]})"
        return rule_string

    def _aggregate_weights(self, weights:list[float]) -> float:
        """Aggregate membership degrees according to selected aggregation function."""
        if callable(self.aggregation_fun):
            return self.aggregation_fun(weights)
        elif self.aggregation_fun == "product":
            return np.prod(weights)
        elif self.aggregation_fun == "min":
            return min(weights)
        elif self.aggregation_fun == "max":
            return max(weights)
        elif self.aggregation_fun == "arit_mean":
            return np.mean(weights)
        else:
            raise Exception(f"Unknown aggregation function {self.aggregation_fun}")

    def update_rules(self, input_vars: list[LinguisticVariable], output_var: LinguisticVariable,
                      values_io:list[float], var_names:list[str]) -> None:
        """Updates rule base (learns new rules or updates existing ones)."""
        terms_list, weight_list = [], []

        fuzzy_vars = input_vars + [output_var]
        for i in range(len(fuzzy_vars)):
            term, weight = self.strong_pertinence(fuzzy_vars[i], values_io[i])
            terms_list.append(term)
            weight_list.append(weight)

        new_weight = self._aggregate_weights(weight_list)
        new_rule = self.build_rule(var_names[:-1], var_names[-1], terms_list)

        if not any(new_rule[:new_rule.find('THEN')] in item for item in self.rules) or len(self.rules) == 0:
            # Check maximum rules limit
            if self.max_rules is not None and len(self.rules) >= self.max_rules:
                tqdm.write(f"Replacing rule {idx}: old_weight = {self.weights[idx]} -> new_weight = {new_weight}.")
                # Replace weakest rule
                idx = np.argmin(self.weights)
                self.rules[idx] = new_rule
                self.weights[idx] = new_weight
                self.usage_count[idx] = 0
                self.error_contribution[idx] = 0.0
            else:
                tqdm.write(f"Add new rule: now the fuzzy uses {len(self.rules)+1} rules.", end='\r')
                self.rules.append(new_rule)
                self.weights.append(new_weight)
                self.usage_count.append(0)
                self.error_contribution.append(0.0)
        else:
            arr = np.array(self.rules)
            mask = np.core.defchararray.find(arr.astype(str), new_rule[:new_rule.find('THEN')])
            old_weight = self.weights[mask[0]]
            if new_weight > old_weight:
                self.rules[mask[0]] = new_rule
                self.weights[mask[0]] = new_weight
    
    def register_rule_usage(self, prediction_error:float=None) -> None:
        """
        Increment usage count for rules with weights above threshold.
        Optionally register contribution to global error.new_rule
        """
        for idx, weight in enumerate(self.weights):
            if weight > self.prune_weight_threshold:
                self.usage_count[idx] += 1
                if prediction_error is not None:
                    self.error_contribution[idx] += abs(prediction_error)

    def prune_unused_rules(self) -> bool:
        """
        Remove unused or low-impact rules based on sliding window.
        Returns True if pruning occurred.
        """
        if self.prune_count >= self.prune_window:
            to_remove = []
            for i, count in enumerate(self.usage_count):
                avg_error = self.error_contribution[i] / max(1, count)
                if count <= self.prune_use_threshold and avg_error < self.prune_weight_threshold:
                    to_remove.append(i)

            if len(to_remove) > 0:
                tqdm.write(f"Pruning {len(to_remove)} rules out of {len(self.rules)} "
                            f"-> {len(self.rules) - len(to_remove)} remaining.")
                for idx in sorted(to_remove, reverse=True):
                    del self.rules[idx]
                    del self.weights[idx]
                    del self.usage_count[idx]
                    del self.error_contribution[idx]

            # Reset usage counters after pruning
            self.usage_count = [0 for _ in self.rules]
            self.error_contribution = [0.0 for _ in self.rules]
            self.prune_count = 0
            return True
        else:
            self.prune_count += 1
            return False

class FuzzyTSModel:
    """Fuzzy model for time series forecasting."""

    def __init__(self, input_names:list, output_name:str, N:int, input_range:list, output_range:list):
        self.input_names = input_names
        self.output_name = output_name
        self.var_manager = FuzzyVariableManager(N, input_range, output_range)
        self.rule_manager = FuzzyRuleManager()
        self.fs = FuzzySystem(show_banner=False)

        # Create variables
        self.input_vars = [self.var_manager.create_variable(name) for name in input_names]
        self.output_var = self.var_manager.create_variable(output_name, is_output=True)

        # Add variables to fuzzy system
        self.fs.add_linguistic_variable(self.output_name, self.output_var)
        for name, var in zip(self.input_names, self.input_vars):
            self.fs.add_linguistic_variable(name, var)
        
        # Useful variables
        self.X_train_dim = None

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learn fuzzy rules from data."""
        
        assert X.shape[0] == y.shape[0], f"The first dimension of X should be equals to the first dimension of y,\
              instead X dimension is {X.shape} and y dimension is {y.shape} "
        

        self.X_train_dim = X.shape

        for xi, yi in tqdm(zip(X, y), total=X.shape[0], desc="Fitting model"):
            values_io = list(xi) + [yi]
            self.rule_manager.update_rules(self.input_vars, self.output_var, values_io,
                                           self.input_names + [self.output_name])
        self.fs.add_rules(self.rule_manager.rules)

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Predict output for given input data.
        
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted output values.
        """


        assert X.shape[1:] == self.X_train_dim[1:], f"The input X dimensions {X.shape[1:]} are different\
              from the train input dimensions {self.X_train_dim[1:]}"
        
        predictions = []
        for xi in tqdm(X, total=X.shape[0], desc="Predicting"):
            # Set input values
            for name, val in zip(self.input_names, xi):
                self.fs.set_variable(name, val)

            # Perform inference
            result = self.fs.inference()
                        
            # Register rule usage and prune unused rules
            self.rule_manager.register_rule_usage()
            if self.rule_manager.prune_unused_rules():
                # Update the rules in the FuzzySystem after pruning
                assert len(self.rule_manager.rules) > 0, "All rules were pruned. Adjust pruning parameters." 
                self.fs._rules.clear()
                self.fs.add_rules(self.rule_manager.rules)
            
            # Get the prediction
            predictions.append(result[self.output_name])
        
        return np.array(predictions)

    def explain(self) -> list[str]:
        return self.rule_manager.rules

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Computes prediction error metrics: MAE, MAPE, RMSE and R2.
        Args:
            y_pred (np.ndarray): Predict output values.
            y_true (np.ndarray): True output values.
        Returns:
            dict: Dictionary with error metrics.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}

