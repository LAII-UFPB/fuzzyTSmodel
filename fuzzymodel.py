import numpy as np
from functools import reduce
from simpful import FuzzySystem, AutoTriangle, LinguisticVariable


class FuzzyVariableManager:
    """Manages fuzzy variables (inputs and outputs)."""

    def __init__(self, num_regions, input_range, output_range):
        self.num_regions = num_regions
        self.input_range = input_range
        self.output_range = output_range
        self.variables = {}

    def create_variable(self, name: str, is_output=False):
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

    def get(self, name):
        return self.variables[name]


class FuzzyRuleManager:
    """Manages a fuzzy rule base."""

    def __init__(self):
        self.rules = []
        self.weights = []

    @staticmethod
    def strong_pertinence(var, value):
        """Selects the term with the highest relevance to a value."""
        values = var.get_values(value)
        terms = list(values.keys())
        vals = list(values.values())
        strong_term = terms[np.argmax(vals)]
        value_term = vals[np.argmax(vals)]
        return strong_term, value_term

    @staticmethod
    def build_rule(input_names, output_name, terms):
        """Creates rule string in simpful-compatible format."""
        rule_string = "IF "
        for i, name in enumerate(input_names):
            rule_string += f"({name} IS {terms[i]}) AND "
        rule_string = rule_string[:-4] + f"THEN ({output_name} IS {terms[-1]})"
        return rule_string


    def update_rules(self, input_vars, output_var, values_io, var_names):
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
        else:
            arr = np.array(self.rules)
            mask = np.core.defchararray.find(arr.astype(str), new_rule[:new_rule.find('THEN')])
            old_weight = self.weights[mask[0]]
            if new_weight > old_weight:
                self.rules[mask[0]] = new_rule
                self.weights[mask[0]] = new_weight


class FuzzyTSModel:
    """Fuzzy model for time series forecasting."""

    def __init__(self, input_names, output_name, num_regions, input_range, output_range):
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

    def fit(self, X, y):
        """Learn fuzzy rules from data."""
        
        assert X.shape[0] == y.shape[0], f"The first dimension of X should be equals to the first dimension of y,\
              instead X dimension is {X.shape} and y dimension is {y.shape} "
        

        self.X_train_dim = X.shape

        for xi, yi in zip(X, y):
            values_io = list(xi) + [yi]
            self.rule_manager.update_rules(self.input_vars, self.output_var, values_io,
                                           self.input_names + [self.output_name])
        self.fs.add_rules(self.rule_manager.rules)

    def predict(self, X):
        assert X.shape[1:] == self.X_train_dim[1:], f"The input X dimensions {X.shape[1:]} are different\
              from the train input dimensions {self.X_train_dim[1:]}"
        
        predictions = []
        for xi in X:
            for name, val in zip(self.input_names, xi):
                self.fs.set_variable(name, val)
            result = self.fs.inference()
            predictions.append(result[self.output_name])
        return np.array(predictions)

    def explain(self):
        return self.rule_manager.rules
