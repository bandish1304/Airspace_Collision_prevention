"""GraphSAGE baseline for edge-level collision-risk classification."""

from .data import FeatureScaler, load_pairs, pair_feature_names, pairs_to_graphs
from .model import GraphSAGEEdgeClassifier

__all__ = [
    "FeatureScaler",
    "GraphSAGEEdgeClassifier",
    "load_pairs",
    "pair_feature_names",
    "pairs_to_graphs",
]
