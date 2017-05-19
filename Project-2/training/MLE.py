from libitg import *
import copy
import numpy as np
import random
from scipy.misc import logsumexp
"contains all functions from the MLE part of the LV-CRF-Roadmap ipython notebook"


def inside_algorithm(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside weight of each node in log space. The inside at ROOT is the Z of the forest"""

    inside = {}

    for v in tsort:
        BS = forest.get(v)
        if not BS:  # list is empty
            inside[v] = 1
        else:
            ks = []
            inside[v] = 0
            for e in BS:
                k = np.log(edge_weights[e])  # include weight of own edge
                for u in e.rhs:
                    k = k + np.log(inside[u])  # product becomes sum of logs
                ks.append(k)
                inside[v] += logsumexp(np.array(k))  # sum becomes log-sum of exponents
    return inside


def outside_algorithm(forest: CFG, tsort: list, edge_weights: dict, inside: dict) -> dict:
    """Returns the outside weight of each node in log space"""

    tsort = list(reversed(tsort))  # traverse nodes top-bottom
    outside = {}
    for v in tsort:
        outside[v] = 0.0
    outside[tsort[0]] = 1.0  # root (S) is one

    for v in tsort:
        for e in forest.get(v):  # the BS (incoming edges) of node v
            for u in e.rhs:  # children of v in e
                k = np.log(edge_weights[e]) + np.log(outside[v])
                for s in e.rhs:  # siblings of u in e
                    if u is not s:
                        k = k + np.log(inside[s])  # product becomes sum of logs
                outside[u] = outside[u] + np.exp(k)  # accumulate outside for node u

    for node in outside:
        outside[node] = np.log(outside[node])  # sum becomes log-sum of exponents

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
    """Returns an expected feature vector (here a sparse python dictionary) in log space"""
    phi = defaultdict(float)

    for e in forest:
        k = outside[e.lhs]
        for u in e.rhs:
            k = k + inside[u]
        for key, feature in edge_features[e].iteritems():
            phi[key] += k * feature

    return phi


def inside_viterbi(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside max of each node in log space"""
    """Do we need this? the slides say we can use the normal inside algo, but I'm not so sure"""
    inside = {}

    for v in tsort:
        BS = forest.get(v)
        if not BS:  # list is empty
            inside[v] = 1
        else:
            ks = []
            inside[v] = 0
            for e in BS:
                k = np.log(edge_weights[e]) # include weight of own edge
                for u in e.rhs:
                    k = k + np.log(inside[u]) # product becomes sum of logs
                ks.append(k)
                inside[v] = np.amax(np.array(ks)) #  sum becomes max
    return inside


def viterbi_decoding(forest:CFG, tsort:list, edge_weights: dict, inside:dict):
    return 0



