import numpy as np
from functools import reduce
from sklearn.metrics import mean_absolute_error, r2_score
from simpful import FuzzySystem, AutoTriangle, LinguisticVariable


class FuzzyVariableManager:
    """Manages fuzzy variables (inputs and outputs)."""

    def __init__(self, num_regions:int, input_range:list, output_range:list):
        self.num_regions = num_regions
        self.input_range = input_range
        self.output_range = output_range
        self.variables = {}

    def create_variable(self, name: str, is_output: bool =False) -> LinguisticVariable:
        """Create fuzzy variable based on Auto Triangle and convert to Linguistic Variable."""
        universe = self.output_range if is_output else self.input_range
        regions = 2 * self.num_regions + 1

        # regions names
        terms = []
        for i in range(regions):
            if i < self.num_regions:
                terms.append(f"S{abs(i - self.num_regions)}")
            elif i > self.num_regions:
                terms.append(f"B{abs(i - self.num_regions)}")
            else:
                terms.append("Z")

        auto = AutoTriangle(
            n_sets=regions,
            terms=terms,
            universe_of_discourse=universe
        )

        # converts to LinguisticVariable (used by FuzzySystem)
        lv = LinguisticVariable(auto._FSlist, concept=name, universe_of_discourse=universe)
        self.variables[name] = lv
        return lv

    def get(self, name:str) -> LinguisticVariable:
        return self.variables[name]


class FuzzyRuleManager:
    """Manages a fuzzy rule base."""

    def __init__(self, prune_weight_threshold:float=0.1, prune_use_threshold:int=0, prune_window:int=15):
        """
        Args:
            prune_weight_threshold (float): Minimum weight for a rule to be considered used.
            prune_use_threshold (int): Minimum number of uses for a rule to be retained.
            prune_window (int): Number of predictions after which to evaluate rule usage.
        Returns:
            None
        """

        self.rules = []
        self.weights = []
        self.usage_count = []
        self.prune_count = 0

        # Pruning parameters
        self.prune_weight_threshold = prune_weight_threshold
        self.prune_use_threshold = prune_use_threshold
        self.prune_window = prune_window

        assert self.prune_weight_threshold >= 0, "prune_weight_threshold must be non-negative"
        assert self.prune_use_threshold >= 0, "prune_use_threshold must be non-negative"
        assert self.prune_window > 0, "prune_window must be positive"
        assert self.prune_window > self.prune_use_threshold, "prune_window must be greater than prune_use_threshold"
        
    @staticmethod
    def strong_pertinence(var:LinguisticVariable, value:float) -> tuple[str,float]:
        """Selects the term with the highest relevance to a value."""
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


    def update_rules(self, input_vars: list[LinguisticVariable], output_var: LinguisticVariable,
                      values_io:list[float], var_names:list[str]) -> None:
        """Updates rule base (same as original rules_database)."""
        terms_list, weight_list = [], []

        fuzzy_vars = input_vars + [output_var]
        for i in range(len(fuzzy_vars)):
            term, weight = self.strong_pertinence(fuzzy_vars[i], values_io[i])
            terms_list.append(term)
            weight_list.append(weight)

        new_weight = reduce(lambda x, y: x * y, weight_list)
        new_rule = self.build_rule(var_names[:-1], var_names[-1], terms_list)

        if not any(new_rule[:new_rule.find('THEN')] in item for item in self.rules) or len(self.rules) == 0:
            self.rules.append(new_rule)
            self.weights.append(new_weight)
            self.usage_count.append(0)
        else:
            arr = np.array(self.rules)
            mask = np.core.defchararray.find(arr.astype(str), new_rule[:new_rule.find('THEN')])
            old_weight = self.weights[mask[0]]
            if new_weight > old_weight:
                self.rules[mask[0]] = new_rule
                self.weights[mask[0]] = new_weight

    def register_rule_usage(self) -> None:
        """
        Increment usage count for rules with weights above a certain threshold.
        Note: This method should be called after each prediction.
        """
        
        for idx, weight in enumerate(self.weights):
            if weight > self.prune_weight_threshold:
                self.usage_count[idx] += 1

    def prune_unused_rules(self):
        """
        Remove not used rules based on a sliding window approach.
        Note: This method should be called after each prediction.
        """
        
        if self.prune_count >= self.prune_window:
            to_remove = [i for i, count in enumerate(self.usage_count) if count <= self.prune_use_threshold]
            if len(to_remove) > 0:
                print(f"Pruning {len(to_remove)} rules out of {len(self.rules)} -> {len(self.rules) - len(to_remove)} remaining.")
                for idx in sorted(to_remove, reverse=True):
                    del self.rules[idx]
                    del self.weights[idx]
                    del self.usage_count[idx]

            # After each pruning, reset usage counts
            self.usage_count = [0 for _ in self.rules]
            self.prune_count = 0
        else:
            self.prune_count += 1


class FuzzyTSModel:
    """Fuzzy model for time series forecasting."""

    def __init__(self, input_names:list, output_name:str, num_regions:int, input_range:list, output_range:list):
        self.input_names = input_names
        self.output_name = output_name
        self.var_manager = FuzzyVariableManager(num_regions, input_range, output_range)
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

        for xi, yi in zip(X, y):
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
        for xi in X:
            # Set input values
            for name, val in zip(self.input_names, xi):
                self.fs.set_variable(name, val)

            # Perform inference
            result = self.fs.inference()
                        
            # Register rule usage and prune unused rules
            #print(np.argmax(self.rule_manager.weights))
            self.rule_manager.register_rule_usage()
            self.rule_manager.prune_unused_rules()
            
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

