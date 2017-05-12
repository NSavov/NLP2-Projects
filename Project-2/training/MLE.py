from libitg import *
import numpy as np

"contains all functions from the MLE part of the LV-CRF-Roadmap ipython notebook"


def inside_algorithm(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside weight of each node"""
    inside = {}

    for v in tsort:
        BS = forest.get(v)
        if not BS:  # list is empty
            inside[v] = 1
        else:
            inside[v] = 0
            for e in BS:
                k = edge_weights[v]
                for u in e.rhs:
                    k = k* inside[u]
                inside[v] += k
    return inside


def outside_algorithm(forest: CFG, tsort:list, edge_weights: dict, inside: dict) -> dict:
    """Returns the outside weight of each node"""
    pass


def weight_function(edge, fmap, wmap) -> float:
    # why do we need the edge here if every edge has a feature map associated with it?
    return np.dot(wmap.T,fmap) # dot product of fmap and wmap  (working in log-domain)


def top_sort(forest: CFG, start_label ='S') -> list:
    """Returns ordered list of nodes according to topsort order in an acyclic forest"""
    # rules = [r for r in cfg]
    for nonterminal in forest.nonterminals:
        if nonterminal.root() == Nonterminal(start_label):
            start = nonterminal
    # find the topologically sorted sequence
    ordered = [start_label]
    for nonterminal in ordered:
        rules = forest.get(nonterminal)
        for rule in rules:
            for variable in rule.rhs:
                if not variable.is_terminal() and variable not in ordered:
                    ordered.append(variable)
    return list(reversed(ordered))

def expected_feature_vector(forest: CFG, inside: dict, outside: dict, edge_features: dict) -> dict:
    """Returns an expected feature vector (here a sparse python dictionary)"""
    pass
