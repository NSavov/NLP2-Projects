#The code is also available in a notebook format in main.ipynb
#The best trained models can be downloaded from: https://my.pcloud.com/publink/show?code=XZwxbNZw8pmtXOdX1YrEoIBr0WSy4gINueX

from data import DataProcessing
from ibm import IBM
import globals
import cPickle
import os
from test import test

#Data Preparation:
data_processor = DataProcessing("training")
training_pairs = data_processor.generate_pairs(True)

data_processor = DataProcessing("validation")
validation_pairs = data_processor.generate_pairs(True)

data_processor = DataProcessing("test")
test_pairs = data_processor.generate_pairs(True)

if (globals.EMPTY_DICT_TYPE == 'training'):
    DataProcessing.init_translation_dict(training_pairs, True, globals.EMPTY_DICT_FILEPATH)
elif (globals.EMPTY_DICT_TYPE == 'validation'):
    DataProcessing.init_translation_dict(validation_pairs, True, globals.EMPTY_DICT_FILEPATH)
elif (globals.EMPTY_DICT_TYPE == 'training_validation'):
    DataProcessing.init_translation_dict(training_pairs + validation_pairs, True, globals.EMPTY_DICT_FILEPATH)

#Data retrieaval
trainPairs, valPairs, testPairs, transProbs = DataProcessing.get_data()
valAlignments = DataProcessing.get_validation_alignments(globals.VALIDATION_DIRECTORY + '/' + globals.VALIDATION_ALIGNMENTS_FILENAME)

data = []
if (globals.EMPTY_DICT_TYPE == 'training'):
    data = trainPairs
elif (globals.EMPTY_DICT_TYPE == 'validation'):
    data = valPairs
elif (globals.EMPTY_DICT_TYPE == 'training_validation'):
    data = trainPairs + valPairs


# IBM 1 Experiments
ibm1 = IBM(transProbs, model = IBM.IBM1)
transProbs_ibm1, aerTransProbs_out = ibm1.train_ibm(data, '', globals.THRESHOLD)
output_dir = globals.IBM1_MODEL_OUTPUT_DIR
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cPickle.dump(transProbs_ibm1, open(output_dir + "transProbs", "wb"))
cPickle.dump(aerTransProbs_out, open(output_dir + "aerTransProbs", "wb"))

loglike_dir = globals.IBM1_MODEL_OUTPUT_DIR + "loglikelihood/"
aer_dir = output_dir + "aer/"
if not os.path.exists(loglike_dir):
    os.makedirs(loglike_dir)
if not os.path.exists(aer_dir):
    os.makedirs(aer_dir)

test(IBM.IBM1, output_dir + "transProbs", "", "", loglike_dir)
test(IBM.IBM1, output_dir + "aerTransProbs", "", "", aer_dir)

# IBM 1 variational bayes Experiments
frenchWords = DataProcessing.get_vocabulary_size(data)

for alpha in [0.0005, 0.005, 0.05]:
    ibm1_b = IBM(transProbs, model="ibm1_bayesian", alpha = alpha, fWords = frenchWords)
    transProbsOut, unseenProbsOut, aerTransProbsOut, aerUnseenProbsOut = \
        ibm1_b.train_ibm(data, globals.THRESHOLD, valPairs = valPairs, valAlignments = valAlignments, aerEpochsThreshold=globals.EPOCHS)
    output_dir = globals.IBM1B_MODEL_OUTPUT_DIR + "/" + str(alpha) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cPickle.dump(transProbsOut, open(output_dir + "transProbs", "wb"))
    cPickle.dump(unseenProbsOut, open(output_dir + "unseenProbs", "wb"))
    cPickle.dump(aerTransProbsOut, open(output_dir + "aerTransProbs", "wb"))
    cPickle.dump(aerUnseenProbsOut, open(output_dir +  "aerUnseenProbs", "wb"))

    loglike_dir = output_dir + "loglikelihood/"
    aer_dir = globals.IBM1B_MODEL_OUTPUT_DIR + "aer/"
    if not os.path.exists(loglike_dir):
        os.makedirs(loglike_dir)
    if not os.path.exists(aer_dir):
        os.makedirs(aer_dir)

    test(IBM.IBM1B, output_dir + "transProbs", output_dir + "unseenProbs", "", loglike_dir)
    test(IBM.IBM1B, output_dir + "aerTransProbs", output_dir + "aerUnseenProbs", "", aer_dir)

# IBM 2 Experiments
methods = ['uniform', 'random', 'random', 'random']

for init_method in methods:
    ibm2 = IBM(transProbs, method=init_method, model=IBM.IBM2)
    transProbsOut, vogelProbsOut, aerTransProbsOut, aerVogelProbsOut = ibm2.train_ibm(data, globals.THRESHOLD, valPairs=valPairs, valAlignments=valAlignments, aerEpochsThreshold=globals.EPOCHS)

    output_dir = globals.IBM2_MODEL_OUTPUT_DIR + "/" + init_method + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cPickle.dump(transProbsOut, open(output_dir + "transProbs", "wb"))
    cPickle.dump(vogelProbsOut, open(output_dir  + "vogelProbs", "wb"))
    cPickle.dump(aerTransProbsOut, open(output_dir  + "aerTransProbs", "wb"))
    cPickle.dump(aerVogelProbsOut, open(output_dir + "aerVogelProbs", "wb"))

    loglike_dir = output_dir + "loglikelihood/"
    aer_dir = output_dir + "aer/"
    if not os.path.exists(loglike_dir):
        os.makedirs(loglike_dir)
    if not os.path.exists(aer_dir):
        os.makedirs(aer_dir)

    test(IBM.IBM2, output_dir + "transProbs", "", output_dir + "vogelProbs", loglike_dir)
    test(IBM.IBM2, output_dir + "aerTransProbs", "",  output_dir + "aerVogelProbs", aer_dir)

ibm2 = IBM(transProbs, model=IBM.IBM2)
ibm2.set_trans_probs(transProbs_ibm1)
transProbsOut, vogelProbsOut, aerTransProbsOut, aerVogelProbsOut = ibm2.train_ibm(data, globals.THRESHOLD, valPairs=valPairs, valAlignments=valAlignments, aerEpochsThreshold=globals.EPOCHS)

output_dir = globals.IBM2_MODEL_OUTPUT_DIR + "/" + "ibm1_pretrained" + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cPickle.dump(transProbsOut, open(output_dir + "transProbs", "wb"))
cPickle.dump(vogelProbsOut, open(output_dir + "vogelProbs", "wb"))
cPickle.dump(aerTransProbsOut, open(output_dir + "aerTransProbs", "wb"))
cPickle.dump(aerVogelProbsOut, open(output_dir + "aerVogelProbs", "wb"))

loglike_dir = output_dir + "loglikelihood/"
aer_dir = output_dir + "aer/"
if not os.path.exists(loglike_dir):
    os.makedirs(loglike_dir)
if not os.path.exists(aer_dir):
    os.makedirs(aer_dir)

test(IBM.IBM2, output_dir + "transProbs", output_dir + "unseenProbs", "", loglike_dir)
test(IBM.IBM2, output_dir + "aerTransProbs", output_dir + "aerUnseenProbs", "", aer_dir)

