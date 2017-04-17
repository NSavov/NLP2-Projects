import random
import numpy as np
import math
import time
import cPickle
import matplotlib.pyplot as plt

class IBM:

    IBM1 = 'ibm1'
    IBM2 = 'ibm2'

    def __init__(self, transProbs, vogelProbs=dict, model="ibm1", method="uniform", path="", preloaded=False):
        self.model = model
        self.method = method
        self.preloaded = preloaded

        if self.model == self.IBM1:
            if not preloaded:
                # uniform parameter initialization
                self.uniform_init(transProbs)
            else:
                self.transProbs = transProbs
        elif self.model == self.IBM2:
            if not preloaded:
                if method == "uniform":
                    # uniform parameter initialization
                    self.uniform_init(transProbs)
                elif method == "random":
                    # do some random init
                    self.random_init(transProbs)
                elif method == "ibm-1":
                    # load or train ibm-1 as init step
                    if path is not "":
                        self.transProbs = cPickle.load(open(path, 'rb'))
                    else:
                        print "provide a path to ibm-1 parameters"
            else:
                self.transProbs = transProbs
                self.vogelProbs = vogelProbs
        else:
            print "invalid model"

    def uniform_init(self, transProbs):
        trans = {}
        for key in transProbs:
            trans[key] = {}
            vocabSize = len(transProbs[key].keys())
            for secKey in transProbs[key]:
                trans[key][secKey] = 1.0 / vocabSize
        self.transProbs = trans

    def random_init(self, transProbs):
        trans = {}
        for key in transProbs:
            trans[key] = {}
            for secKey in transProbs[key]:
                trans[key][secKey] = random.random()
            normalizer = sum(trans[key].itervalues())
            trans[key] = {k: v / normalizer for k, v in trans[key].iteritems()}
        self.transProbs = trans

    @staticmethod
    def vogel_index(i, j, I, J):
        # get the Vogel count index
        return math.floor(i - (j+1.0) * I / J)


    def train_ibm(self, pairs, termination_criteria, threshold):

        # trains an ibm 1 model
        converged = False
        logLikelihood = []

        transProbs = self.transProbs  # initialize_ibm(transProbs)
        numberOfSentences = len(pairs)

        if self.model == self.IBM2:
            # initialize vogel count parameter vector
            if not self.preloaded:
                countsVogel = {}
                vogelProbs = {}
                for pair in pairs:
                    I = len(pair[0])  # english sentence length
                    J = len(pair[1])  # french sentence length
                    for i, enWord in enumerate(pair[0]):
                        for j, frWord in enumerate(pair[1]):
                            countsVogel[self.vogel_index(i, j, I, J)] = 0.0  # check: do we need j + 1 cuz null
                if self.method == "uniform":
                    length = len(countsVogel.keys())
                    for key in countsVogel:
                        vogelProbs[key] = 1.0 / length
                else:
                    for key in countsVogel:
                        vogelProbs[key] = random.random()
                    normalizer = sum(vogelProbs.itervalues())
                    vogelProbs = {k: v / normalizer for k, v in vogelProbs.iteritems()}
            else:
                vogelProbs = self.vogelProbs
                countsVogel = {k: 0.0 for k in vogelProbs.keys()}

        while not converged:
            start = time.time()
            logLike = 0

            # set all counts for the translation model to zero
            counts = {}
            countsEnglish = {}
            for key in transProbs:
                counts[key] = {}
                countsEnglish[key] = 0.0
                for secKey in transProbs[key]:
                    counts[key][secKey] = 0.0

            # Expectation - step
            print "E"
            for pair in pairs:
                I = len(pair[0])  # english sentence length
                J = len(pair[1])  # french sentence length

                if self.model == self.IBM1:
                    logLike += -(J * np.log(I+1))


                for j, fWord in enumerate(pair[1]):
                    # calculate the normalizer of the posterior probability of this french word
                    normalizer = 0.0
                    normalizerVogel = 0.0
                    for i, eWord in enumerate(pair[0]):
                        if self.model == self.IBM2:
                            normalizer += transProbs[eWord][fWord] * vogelProbs[self.vogel_index(i, j, I, J)]
                            normalizerVogel += vogelProbs[self.vogel_index(i, j, I, J)]
                        else:
                            normalizer += transProbs[eWord][fWord]

                    logLike += np.log(normalizer)
                    # get the expected counts based on the posterior probabilities
                    for i, eWord in enumerate(pair[0]):
                        if self.model == self.IBM2:
                            delta = vogelProbs[self.vogel_index(i, j, I, J)] * transProbs[eWord][fWord] / normalizer
                            countsVogel[self.vogel_index(i, j, I, J)] += delta  # do we only need to take the maximum probable likelihood?
                        else:
                            delta = transProbs[eWord][fWord] / normalizer
                        counts[eWord][fWord] += delta
                        countsEnglish[eWord] += delta                

            logLikelihood.append(logLike / numberOfSentences)
            print logLikelihood[-1]

            # check for log-likelihood convergence
            if len(logLikelihood) > 1:
                difference = logLikelihood[-1] - logLikelihood[-2]
                if difference < threshold:
                    converged = True
                    break

            # Maximization - step
            print "M"
            for eKey in transProbs:
                for fKey in transProbs[eKey]:
                    # update translation probabilities
                    transProbs[eKey][fKey] = counts[eKey][fKey] / countsEnglish[eKey]
            if self.model == self.IBM2:
                # update Vogel-based alignment probabilities
                normalizer = sum(countsVogel.itervalues())
                vogelProbs = {k: v / normalizer for k, v in countsVogel.iteritems()}
                
            end = time.time()
            print end-start


        plt.plot([x+1 for x in range(len(logLikelihood))], logLikelihood, 'ro')
        plt.show()


        if self.model == self.IBM1:
            return transProbs
        else:
            return transProbs, vogelProbs


    @staticmethod
    def get_alignments( pairs, transProbs, model="ibm1", vogelProbs=""):
        """Get the predicted alignments on sentence pairs from a trained ibm model 1 or 2"""
        alignments = []
        for k, pair in enumerate(pairs):
            alignments.append([])
            I = len(pair[0])
            J = len(pair[1])
            for j, fWord in enumerate(pair[1]):
                maxProb = 0.0
                alignment = 0
                for i, eWord in enumerate(pair[0]):
                    if eWord in transProbs:
                        if fWord in transProbs[eWord]:
                            if model == "ibm1":
                                alignProb = transProbs[eWord][fWord]
                            elif model == "ibm2":
                                alignProb = transProbs[eWord][fWord] * vogelProbs[IBM.vogel_index(i, j, I, J)]
                            else:
                                print "incorrect model"
                                break
                    if alignProb > maxProb:
                        maxProb = alignProb
                        alignment = i
                alignments[k].append(alignment)
        return alignments


    @staticmethod
    def get_AER(prediction, test):
        aer = 0

        for pair_id, pair in test.items():
            alignments_count = len(pair)
            sure_alignments = {(a[0], a[1]) for a in pair if a[-1] == 'S'}
            possible_alignments = {(a[0], a[1]) for a in pair if a[-1] == 'P'}

            predicted_alignments = {(predicted_alignment, french_ind + 1) for french_ind, predicted_alignment in enumerate(prediction[pair_id - 1])}

            intersection_A_S = predicted_alignments & sure_alignments
            intersection_A_P = predicted_alignments & possible_alignments

            aer += (1 - (len(intersection_A_S) + len(intersection_A_P))/float(alignments_count))

        aer = aer/len(test)

        return aer
