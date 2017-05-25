from libitg import *
import subprocess
import globals
from training import MLE

"contains all function necessary for target side prediction of a trained model"


def BLEU_script(num_ref: int):
    """
    Run the BLEU perl script
    :param num_ref: number of references to compare with
    :return output: BLEU scores, [total, 1-gram, 2-gram, 3-gram, 4-gram]
    """
    # get the hypothesis and reference files
    references = ""
    reference_path = globals.REF_PATH
    hypotheses_path = globals.VAL_HYPOTHESIS
    for i in range(num_ref):
        references += " " + reference_path + str(i+1)

    # run the perl script to obtain BLEU scores
    output = subprocess.check_output("perl multi-bleu.perl -lc" + references + " < " + hypotheses_path, shell=True)
    output = str(output)
    output = output[2:][:-3].split(' ')
    for i, value in enumerate(output):
        output[i] = float(value)

    return output


def viterbi_decoding(forest: CFG, tsort: list, edge_weights: dict, inside: dict):
    """returns the Viterbi tree of a forest"""

    # prelims
    sentence = ""
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
            sentence = sentence + str(v.root()) + " "
        else:
            for e in BS:  # possible edges with head v
                score = edge_weights[e]
                for u in e.rhs:  # children of v in e
                    score += inside[u]  # product becomes sum of logs (but inside is already log)
                scores.append(score)
            index = np.argmax(np.array(scores))  # index of viterbi edge
            viterbi_edges.append(BS[index])
            for node in list(reversed(BS[index].rhs)):
                nodes.insert(0, node)  # new nodes to traverse, left-depth-first or something

    sentence += "\n"

    return CFG(viterbi_edges), sentence


def ancestral_sampling(forest: CFG, tsort: list, edge_weights: dict, inside: dict):
    """Returns a likely tree from a forest with ancestral sampling of the edge probabilities"""

    # prelims
    sentence = ""
    ancestral_edges = []
    nodes = []
    root = list(reversed(tsort))[0]
    nodes.append(root)

    # perform ancestral sampling on nodes until a full tree is constructed
    while nodes:
        v = nodes.pop(0)
        probs = []
        BS = forest.get(v)
        if not BS:
            # yield of derivation
            sentence += str(v.root()) + " "
        else:
            # find the probability of each edge with head v
            for e in BS:
                prob = edge_weights[e]
                for u in e.rhs:  # children of v given the current edge
                    prob += inside[u]  # accumulate the inside values, product becomes sum in log space
                prob = prob / inside[v]  # normalize, the inside of the parent should be the sum of the children
                probs.append(prob)

            # sample uniformly from the CDF of edge probabilities
            assert 0.99 < np.sum(np.array(probs)) < 1.01, "probs are not properly normalized"
            sample = np.random.uniform(0.0, np.sum(np.array(probs)))
            cdf = 0.0
            index = 0
            for i, prob in enumerate(probs):
                cdf += prob
                if sample < cdf:
                    index = i
                    break

            # select the sampled edge and continue sampling from its rhs nodes
            ancestral_edges.append(BS[index])
            for node in list(reversed(BS[index].rhs)):
                # left-depth-first hypergraph traversal
                nodes.insert(0, node)

    sentence += "\n"

    return CFG(ancestral_edges), sentence


def minimum_bayes_risk_decoding(forest: CFG, features: dict, wmap:int, num_samples:int):
    """
    Run minimum bayes risk decoding for N samples and
    :param forest: D_i(x) forest to sample from
    :param features: features for every edge in the forest
    :param wmap: weights for every features
    :param num_samples: number of samples to take
    :return sentence: best prediction from forest
    """

    # get edge weights
    edge_weights = {}
    for edge in forest:
        # weight of each edge in the forest based on its features and the current wmap
        edge_weights[edge] = MLE.weight_function(edge, features[edge], wmap)

    # compute inside scores for this forest
    parents = MLE.get_parents_dict(forest)
    tsort = MLE.top_sort(forest, parents)
    inside = MLE.inside_algorithm(forest, tsort, edge_weights)

    # sample N times



