import random
import numpy as np
import math
import time
import cPickle
import matplotlib.pyplot as plt
from scipy.special import psi
import aer


class IBM:

    IBM1 = 'ibm1'
    IBM2 = 'ibm2'
    IBM1B = 'ibm1_bayesian'

    def __init__(self, transProbs, vogelProbs=dict, unseenProbs=dict, model="ibm1", method="uniform", path="", preloaded=False, alpha = 0.1):
        self.model = model
        self.method = method
        self.preloaded = preloaded

        if self.model == self.IBM1:
            if not preloaded:
                # uniform parameter initialization
                self.uniform_init(transProbs)
            else:
                self.transProbs = transProbs
        elif self.model == self.IBM1B:
            self.alpha = alpha
            self.frenchWords = 46442.0
            if not preloaded:
                self.bayes_init(transProbs)
            else:
                self.transProbs = transProbs
                self.unseenProbs = unseenProbs
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
        # uniform initialisation of the translation probabilities
        trans = {}
        for key in transProbs:
            trans[key] = {}
            vocabSize = len(transProbs[key].keys())
            for secKey in transProbs[key]:
                trans[key][secKey] = 1.0 / vocabSize
        self.transProbs = trans

    def bayes_init(self, transProbs):
        # uniform IBM1 initialisation for variational Bayes
        trans = {}
        unseen = {}
        for key in transProbs:
            trans[key] = {}
            unseen[key] = 1.0 / self.frenchWords
            for secKey in transProbs[key]:
                trans[key][secKey] = 1.0 / self.frenchWords
        self.transProbs = trans
        self.unseenProbs = unseen

    def random_init(self, transProbs):
        # random initialisation of the translation probabilities
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

    def bayesian_maximization(self, counts, normalizer,temp):
        if temp == "prostitute":
            print psi(counts + self.alpha), psi(normalizer + self.alpha * self.frenchWords)
        return np.exp(psi(counts + self.alpha) - psi(normalizer + self.alpha * self.frenchWords))

    def train_ibm(self, pairs, termination_criteria, threshold, valPairs = False, valAlignments = False, aerEpochsThreshold = 5):

        # trains an ibm 1 model
        converged = False
        logLikelihood = []

        transProbs = self.transProbs  # initialize_ibm(transProbs)
        if self.model == self.IBM1B:
            unseenProbs = self.unseenProbs
        numberOfSentences = len(pairs)
        minAer = float('inf')
        epoch = 0

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
            epoch += 1

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
                            countsVogel[self.vogel_index(i, j, I, J)] += delta
                        else:
                            delta = transProbs[eWord][fWord] / normalizer
                        counts[eWord][fWord] += delta
                        countsEnglish[eWord] += delta                

            logLikelihood.append(logLike / numberOfSentences)
            print logLikelihood[-1]

            # check for log-likelihood convergence
            if termination_criteria == 'loglike':
                if len(logLikelihood) > 1:
                    difference = logLikelihood[-1] - logLikelihood[-2]
                    if difference < threshold:
                        converged = True
                        break

            # Maximization - step
            print "M"
            for eKey in transProbs:
                if self.model == self.IBM1B:
                    unseenProbs[eKey] = self.bayesian_maximization(0, countsEnglish[eKey], eKey)
                for fKey in transProbs[eKey]:
                    # update translation probabilities
                    if not self.model == self.IBM1B:
                        transProbs[eKey][fKey] = counts[eKey][fKey] / countsEnglish[eKey]
                    else:
                        transProbs[eKey][fKey] = self.bayesian_maximization(counts[eKey][fKey], countsEnglish[eKey], eKey)

            if self.model == self.IBM2:
                # update Vogel-based alignment probabilities
                normalizer = sum(countsVogel.itervalues())
                vogelProbs = {k: v / normalizer for k, v in countsVogel.iteritems()}

            if termination_criteria == 'aer':
                if not valPairs or not valAlignments:
                    print "Invalid validation data"
                    break

                if self.model != self.IBM2:
                    vogelProbs = ''

                if self.model != self.IBM1B:
                    unseenProbs = ''

                predictions = self.get_alignments(valPairs, transProbs, unseenProbs, vogelProbs)
                aer = IBM.get_AER(predictions, valAlignments)

                print "epoch: ", epoch, " aer: ", aer

                if aer < minAer:
                    minAer = aer
                    if self.model == self.IBM2:
                        bestVogelProbs = vogelProbs
                    bestTransProbs = transProbs

                if epoch >= aerEpochsThreshold:
                    break

            end = time.time()
            print end-start

        plt.plot([x+1 for x in range(len(logLikelihood))], logLikelihood, 'ro')
        plt.show()

        if termination_criteria == 'aer':
            transProbs = bestTransProbs
            if self.model == self.IBM2:
                vogelProbs = bestVogelProbs

        if self.model == self.IBM1:
            return transProbs
        elif self.model == self.IBM1B:
            return transProbs, unseenProbs
        else:
            return transProbs, vogelProbs

    def get_alignments(self, pairs, transProbs, unseenProbs = dict, vogelProbs=""):
        """Get the predicted alignments on sentence pairs from a trained ibm model 1 or 2"""
        alignments = []
        for k, pair in enumerate(pairs):
            alignments.append(set())
            I = len(pair[0])
            J = len(pair[1])
            for j, fWord in enumerate(pair[1]):
                maxProb = 0.0
                alignment = 0
                for i, eWord in enumerate(pair[0]):
                    if eWord in transProbs:
                        if fWord in transProbs[eWord]:
                            if self.model == self.IBM1 or self.model == self.IBM1B:
                                alignProb = transProbs[eWord][fWord]
                            elif self.model == self.IBM2:
                                alignProb = transProbs[eWord][fWord] * vogelProbs[IBM.vogel_index(i, j, I, J)]
                            else:
                                print "incorrect model"
                                break
                        elif self.model == self.IBM1B:
                            alignProb = unseenProbs[eWord]
                    if alignProb > maxProb:
                        maxProb = alignProb
                        alignment = i
                if alignment is not 0:
                    alignments[k].add((alignment, j+1))

            # print alignments
        return alignments

    @staticmethod
    def get_AER(predictions, test):
        metric = aer.AERSufficientStatistics()
        # then we iterate over the corpus
        for gold, pred in zip(test, predictions):

            metric.update(sure=gold[0], probable=gold[1], predicted=pred)
        
        return metric.aer()

    # @staticmethod
    # def get_AER(prediction, test):
    #     aer = 0
    #
    #     for pair_id, test_alignment in test.items():
    #
    #         sure_alignments = {(a[0], a[1]) for a in test_alignment if a[-1] == 'S'}
    #         possible_alignments = {(a[0], a[1]) for a in test_alignment if a[-1] == 'P'}
    #
    #         predicted_alignments = {(predicted_alignment, french_ind + 1) for french_ind, predicted_alignment in enumerate(prediction[pair_id - 1])}
    #         alignments_count = len(predicted_alignments) + len(sure_alignments)
    #
    #         intersection_A_S = predicted_alignments & sure_alignments
    #         intersection_A_P = predicted_alignments & possible_alignments
    #
    #         aer += (1 - (len(intersection_A_S) + len(intersection_A_P))/float(alignments_count))
    #
    #     aer = aer/len(test)
    #
    #     return aer
