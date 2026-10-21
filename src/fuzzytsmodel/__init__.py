"""Geração automática de regras fuzzy (Wang-Mendel) para séries temporais."""

from .model import FuzzyTSModel
from .rules import FuzzyRuleManager
from .variables import FuzzyVariableManager, region_terms

__all__ = ["FuzzyTSModel", "FuzzyRuleManager", "FuzzyVariableManager", "region_terms"]
__version__ = "0.2.0"
