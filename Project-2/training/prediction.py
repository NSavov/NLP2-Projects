from libitg import *

"contains all function necessary for target side prediction of a trained model"


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
            sentence += str(v.root()) + " "
        else:
            # find the probability of each edge with head v
            for e in BS:
                prob = edge_weights[e]
                for u in e.rhs:  # children of v given the current edge
                    prob *= np.exp(inside[u])  # accumulate the exponent of the log-space inside
                prob = prob / np.exp(inside[v])  # normalize
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
                nodes.insert(0, node)

    sentence += "\n"

    return CFG(ancestral_edges), sentence


