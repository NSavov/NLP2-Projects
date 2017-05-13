from libitg import *
import copy
import numpy as np
import random
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
                k = edge_weights[e]
                for u in e.rhs:
                    k = k* inside[u]
                inside[v] += k
    return inside


def outside_algorithm(forest: CFG, tsort:list, edge_weights: dict, inside: dict) -> dict:
    """Returns the outside weight of each node"""

    tsort = list(reversed(tsort)) # traverse nodes top-bottom
    outside = {}
    for v in tsort:
        outside[v] = 0.0
    outside[tsort[0]] = 1.0  # root (S) is one

    for v in tsort:
        for e in forest.get(v):  # the BS (incoming edges) of node v
            for u in e.rhs:  # children of v in e
                k = edge_weights[e] * outside[v]
                for s in e.rhs:  # siblings of u in e
                    if u is not s:
                        k = k * inside[s]
                outside[u] = outside[u] + k  # accumulate outside for node u

    return outside


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
    phi = defaultdict(float)

    for e in forest:
        k = outside[e.lhs]
        for u in e.rhs:
            k = k * inside[u]
        for key, feature in edge_features[e].iteritems():
            phi[key] += k * feature

    return phi


