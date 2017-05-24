from training import MLE, prediction
import pickle
import globals
import libitg

# generate some weights for one sentence
weight, avg_loss, val, t, tstar = MLE.stochastic_gradient_descent(1, 0.1, 0.1, 1, 1, 250, True, 0.5)
print(weight[-1])

# get the forest

# get the correct filenames for loading from globals
forests_file = globals.ITG_FILE_PATH
features_file = globals.FEATURES_FILE_PATH

# start reading forest files
forests = open(forests_file, "rb")
features = open(features_file, "rb")

# forest and feature of first training thing
forest = pickle.load(forests)
feature = pickle.load(features)
forest_check = forest[2]
forest = forest[1]
feature = feature[0]

# get tsort and weights
edge_weights = {}
for edge in forest:
    # weight of each edge in the forest based on its features and the current wmap
    edge_weights[edge] = MLE.weight_function(edge, feature[edge], weight[-1])

# compute tsort
parents = MLE.get_parents_dict(forest)
tsort = MLE.top_sort(forest, parents)

# try me some viterbi
inside = MLE.inside_viterbi(forest, tsort, edge_weights)
viterbi = prediction.viterbi_decoding(forest, tsort, edge_weights, inside)

print(libitg.language_of_fsa(libitg.forest_to_fsa(forest_check, libitg.Nonterminal("D_i(x,y)"))))
print(libitg.language_of_fsa(libitg.forest_to_fsa(viterbi, libitg.Nonterminal("D_i(x)"))))






