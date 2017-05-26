import numpy as np
from training.prediction import minimum_bayes_risk_decoding, BLEU_script
import pickle

file = open("datamap/hypotheses",'w')

# get forests, features from the test set and wmap from the trained model
test_forest = "load the forest here"
test_features = "load the features here"
wmap = "unpickle the wmap here"

for i in range(50):
    sentence = minimum_bayes_risk_decoding(test_forest, test_features, wmap, 30)
    file.write(sentence)

file.close()

# make sure your refpath in globals is set correctly so you read from the correct folder!
scores = BLEU_script(50)

# pickle the scores!
pickle.dump(scores, open("MBR_test_set_BLEU", "wb"))

