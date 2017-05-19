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
            inside[v] = 0  # terminal. 1 -> 0 in log semicircle
        else:
            ks = []
            inside[v] = -np.inf  # additive identity 0 -> -np.inf in log semicircle
            for e in BS:
                k = np.log(edge_weights[e])  # include weight of own edge
                for u in e.rhs:
                    k = k + inside[u]  # product becomes sum of logs
                ks.append(k)
            ks.append(inside[v])
            inside[v] = logsumexp(np.array(ks))  # sum becomes log-sum of exponents of logs
    return inside


def outside_algorithm(forest: CFG, tsort: list, edge_weights: dict, inside: dict) -> dict:
    """Returns the outside weight of each node in log space"""

    tsort = list(reversed(tsort))  # traverse nodes top-bottom
    outside = {}
    for v in tsort:
        outside[v] = [-np.inf]  # 0 -> -inf in log space
    outside[tsort[0]] = 0.0  # root (S) is one, 1 -> 0 in log space

    for v in tsort:
        for e in forest.get(v):  # the BS (incoming edges) of node v
            for u in e.rhs:  # children of v in e
                k = np.log(edge_weights[e]) + outside[v]
                for s in e.rhs:  # siblings of u in e
                    if u is not s:
                        k = k + inside[s]  # product becomes sum of logs
                outside[u].append(k)  # accumulate outside for node u

    for node in outside:
        outside[node] = logsumexp(np.array(outside[node]))  # sum becomes log-sum of exponents

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
            k = k + inside[u]  # product becomes sum of logs (but they are already in log space here)
        for key, feature in edge_features[e].iteritems():
            phi[key] += k * feature

    return phi


def inside_viterbi(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside max of each node in log space"""

    inside = {}

    for v in tsort:
        BS = forest.get(v)
        if not BS:  # list is empty
            inside[v] = 0  # 1 -> 0 in log space
        else:
            ks = []
            inside[v] = -np.inf  # 0 -> -1 in log space
            for e in BS:
                k = np.log(edge_weights[e])  # include weight of own edge
                for u in e.rhs:
                    k = k + inside[u]  # product becomes sum of logs (but they are already logs)
                ks.append(k)
                ks.append(inside[v])
            inside[v] = np.amax(np.array(ks))  # sum becomes max, max becomes max in log space
    return inside


def viterbi_decoding(forest: CFG, tsort:list, edge_weights: dict, inside: dict):
    """returns the Viterbi tree of a forest"""

    # prelims
    viterbi_edges = []
    nodes = []
    root = list(reversed(tsort))[0]
    nodes.append(root)

    # construct Viterbi CFG for the target side
    while nodes:
        v = nodes.pop(0)
        scores = []
        BS = forest.get(v)
        if not BS:  # terminal
            continue
        else:
            for e in BS:  # possible edges with head v
                score = edge_weights[e]
                for u in e.rhs:  # children of v in e
                    score += inside[u]  # product becomes sum of logs (but inside is already log)
                scores.append(score)
            index = np.argmax(np.array(scores))  # index of viterbi edge
            viterbi_edges.append(BS[index])
            nodes.append(BS[index].rhs)  # new nodes to traverse

    return CFG(viterbi_edges)

def ancestral_sampling(forest: CFG, tsort: list, edge_weights: dict, inside: dict):
    return 0


