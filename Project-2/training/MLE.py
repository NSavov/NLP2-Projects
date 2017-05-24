from libitg import *
import numpy as np
from scipy.misc import logsumexp
from training.model import weight_function
import globals
import pickle
import time

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
                k = edge_weights[e]  # include weight of own edge, they are in log space
                for u in e.rhs:
                    try:
                        k += inside[u]  # product becomes sum of logs
                    except KeyError:
                        print("Rule: ", e)
                        print("trying to compute with: ", u)
                        print("computing for: ", v)
                ks.append(k)
            ks.append(inside[v])
            inside[v] = logsumexp(np.array(ks))  # sum becomes log-sum of exponents of logs
    return inside


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
                k = edge_weights[e]  # include weight of own edge
                for u in e.rhs:
                    k = k + inside[u]  # product becomes sum of logs (but they are already logs)
                ks.append(k)
            ks.append(inside[v])
            inside[v] = np.amax(np.array(ks))  # sum becomes max, max becomes max in log space
    return inside


def outside_algorithm(forest: CFG, tsort: list, edge_weights: dict, inside: dict) -> dict:
    """Returns the outside weight of each node in log space"""

    tsort = list(reversed(tsort))  # traverse nodes top-bottom
    outside = {}
    for v in tsort:
        outside[v] = [-np.inf]  # 0 -> -inf in log space
    outside[tsort[0]] = 0.0  # root (S) is one, 1 -> 0 in log space

    for v in tsort:
        if v != tsort[0]:  # the root already has a correct outside value
            outside[v] = logsumexp(np.array(outside[v]))  # v has been fully processed, we now take the log-sum of exps
        for e in forest.get(v):  # the BS (incoming edges) of node v
            for u in e.rhs:  # children of v in e
                k = edge_weights[e] + outside[v]  # edge weights are in log space, add them to the outside
                for s in e.rhs:  # siblings of u in e
                    if u is not s:
                        k = k + inside[s]  # product becomes sum of logs
                outside[u].append(k)  # accumulate outside for node u

    return outside


def get_parents_dict(forest):
    parents = defaultdict(set)
    for rule in forest:
        for symbol in rule.rhs:
            parents[symbol].add(rule.lhs)
    return parents


def top_sort(forest: CFG, parents) -> list:
    """Returns ordered list of nodes according to topsort order in an acyclic forest"""
    # rules = [r for r in cfg]
    S = forest.terminals

    D = defaultdict(set)
    for child, all_parents in parents.items():
        for parent in all_parents:
            D[parent].add(child)

    L = []
    while S:
        u = S.pop()
        L.append(u)
        for v in parents[u]: #we get the heads of edges directly from the parents dict
            D[v].remove(u)
            if not bool(D[v]):
                S.add(v)

    return L


def expected_feature_vector(forest: CFG, inside: dict, outside: dict, edge_features: dict, root: float, edge_weights: dict) -> dict:
    """Returns an expected feature vector (here a sparse python dictionary) in log space"""
    phi = defaultdict(float)

    for e in forest:
        k = outside[e.lhs]
        for u in e.rhs:
            k = k + inside[u]  # now we have the exclusive weight for an edge, k(e), in log space
        # print("root: ", root, "k_log(e): ", k)
        if (k-root) > 0:
            print(e)
        #     print(k-root)
        #     print(edge_weights[e])
        k = np.exp(k - root)  # we normalize it and take the exponent to take it to probability space
        for key, feature in edge_features[e].items():
            phi[key] += k * feature  # now the expected feature vector is a simple product between features and k

    return phi


def get_validation_loss(val_set: list, val_feat: list, wmap:dict):

    loss = 0.0

    # process the validation set
    for forests, feats in zip(val_set, val_feat):
        Z = []

        # remove index at beginning of forest pair list
        forests = forests[-2:]

        # process each pair of forest pairs and feature pairs in the batch
        for forest, feat in zip(forests, feats):

            edge_weights = {}
            for edge in forest:
                # weight of each edge in the forest based on its features and the current wmap
                edge_weights[edge] = weight_function(edge, feat[edge], wmap)

            # compute log normalizer of the forest
            parents = get_parents_dict(forest)
            tsort = top_sort(forest, parents)
            inside = inside_algorithm(forest, tsort, edge_weights)
            root = inside[tsort[-1]]  # normalizer of the forest

            # store forest normalizer to compute log
            Z.append(root)  # the normalizer Z is the inside value at the root of the forest

        # accumulate loss
        loss += Z[1] - Z[0]

    return loss


def stochastic_gradient_descent_step(batch: list, features: list, learning_rate: float, wmap:dict,
                                     reg=False, lamb=0.1):
    """
    Performs one SGD step on a batch of featurized DiX and DiXY forests
        :param batch: list of forests
        :param features: list of features
        :param learning_rate: learning rate for update
        :param wmap: current weights
        :param reg: whether or nog to regularize the update
        :param lamb: regularizer strength parameter
        :return wmap2: updated weights
        :return loss: batch log likelihood
    """

    # prelims
    gradient = defaultdict(float)
    wmap2 = defaultdict(float)
    loss = 0.0

    # process the entire batch
    for forests, feats in zip(batch, features):
        expected_features = []
        Z = []
        i = 0

        # remove index at beginning of forest pair list
        forests = forests[-2:]

        # process each pair of forest pairs and feature pairs in the batch
        for forest, feat in zip(forests, feats):

            edge_weights = {}
            for edge in forest:
                # weight of each edge in the forest based on its features and the current wmap
                edge_weights[edge] = weight_function(edge, feat[edge], wmap)

            # compute expected feature vector for this forest
            parents = get_parents_dict(forest)
            tsort = top_sort(forest, parents)
            inside = inside_algorithm(forest, tsort, edge_weights)
            root = inside[tsort[-1]]  # normalizer of the forest
            outside = outside_algorithm(forest, tsort, edge_weights, inside)
            expected_features.append(expected_feature_vector(forest, inside, outside, feat, root, edge_weights))

            # store forest normalizer to compute log
            Z.append(root)  # the normalizer Z is the inside value at the root of the forest

            i += 1

        # consider the intersection of features of the two forests
        keys = set(expected_features[0].keys())
        keys.update(expected_features[1].keys())

        # accumulate gradients
        for key in keys:
            gradient[key] += expected_features[1][key] - expected_features[0][key]

        # accumulate loss
        loss += Z[1] - Z[0]

    # update the weights
    if reg:
        for key in gradient:
            wmap2[key] = wmap[key] + learning_rate * (gradient[key] - lamb * wmap[key])
    else:
        for key in gradient:
            wmap2[key] = wmap[key] + learning_rate * gradient[key]

    return wmap2, loss


def stochastic_gradient_descent(batch_size: int, learning_rate: float, threshold: float, max_ticks: int,
                                max_batch=np.inf, max_epochs=np.inf, reg=False, lamb=0.1):

    # intialize wmap with random floats between 0 and 1
    wmap = defaultdict(lambda: np.random.random())

    # get the correct filenames for loading from globals
    forests_file = globals.ITG_FILE_PATH
    features_file = globals.FEATURES_FILE_PATH

    # returns
    validation_loss = [0]
    average_loss = []
    weights = []

    # convergence checks
    converged = False
    ticks = 0
    epoch = 0
    t = 0
    tstar = 0

    # run for x epochs
    while not converged:
        # statistics
        start = time.time()
        num_batches = 1
        total_loss = 0.0
        epoch += 1

        if epoch > max_epochs:
            break

        print("Starting epoch " + str(epoch))

        # start reading forest files
        forests = open(forests_file, "rb")
        features = open(features_file, "rb")

        # will be true when the end of file is reached
        stop = False

        while not stop:
            # load a batch from file
            forest_batch = []
            feature_batch = []
            for i in range(batch_size):
                try:
                    forest_batch.append(pickle.load(forests))
                    feature_batch.append(pickle.load(features))
                except EOFError:
                    stop = True
                    break

            # set learning rate for this batch
            new_learning_rate = learning_rate * (1.0 + learning_rate * lamb * t)**(-1)

            # run gradient descent over batch and store loss
            wmap, loss = stochastic_gradient_descent_step(forest_batch, feature_batch,
                                                          new_learning_rate, wmap, reg, lamb)
            total_loss += loss

            print("\r" + "epoch: " + str(epoch) + ", processed batch number: " + str(num_batches) +
                  ", average loss: " + str(total_loss / num_batches) + ", batch loss: " + str(loss))

            # record number of batches in this epoch and the total number of baches - 1 as timestep
            num_batches += 1
            t += 1

            if num_batches > max_batch:
                break

        # check for convergence on the validation set
        forest_batch = []
        feature_batch = []
        for i in range(batch_size):
            forest_batch.append(pickle.load(forests))
            feature_batch.append(pickle.load(features))
        loss = get_validation_loss(forest_batch, feature_batch, wmap)
        validation_loss.append(loss)

        # stop reading files
        forests.close()
        features.close()

        print("epoch: " + str(epoch) + ", validation loss: " + str(loss))

        # check for convergence on the validation set
        if np.abs(validation_loss[-1] - validation_loss[-2]) < threshold:
            print("converge tick")
            ticks += 1

        if tstar == 0 and validation_loss[-1] - validation_loss[-2] > 0:
            tstar = t

        # after x number of under threshold likelihood differences, convergence is achieved
        if ticks > max_ticks:
            print("likelihood converged")
            converged = True

        average_loss.append(total_loss / num_batches)  # store the loss for each epoch over the entire dataset
        weights.append(wmap)  # store the wmap for each epoch over the entire dataset
        end = time.time()
        print("epoch done, time: ", str(end-start))

    return weights, average_loss, validation_loss, t, tstar




