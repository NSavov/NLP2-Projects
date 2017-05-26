import numpy as np
from training.prediction import minimum_bayes_risk_decoding, BLEU_script
import pickle



# get forests, features from the test set and wmap from the trained model
test_forest = "load the forest here"
test_features = "load the features here"
wmap = "unpickle the wmap here"

alpha = [0.1, 0.3, 0.5, 0.7, 0.9]

for a in alpha:
    file = open("datamap/hypotheses", 'w')

    for i in range(20):
        sentence = minimum_bayes_risk_decoding(test_forest, test_features, wmap, 30, a)
        file.write(sentence)

    file.close()

    # make sure your refpath in globals is set correctly so you read from the correct folder!
    scores = BLEU_script(16)

    # pickle the scores!
    pickle.dump(scores, open("MBR_val_set_BLEU_" + str(a), "wb"))

