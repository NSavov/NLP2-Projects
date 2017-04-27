import random
import numpy as np
import math
import time
import cPickle
import matplotlib.pyplot as plt
from scipy.special import psi, gammaln
import aer
import os
import matplotlib.patches as mpatches


class IBM:

    IBM1 = 'ibm1'
    IBM2 = 'ibm2'
    IBM1B = 'ibm1_bayesian'

    def __init__(self, transProbs, vogelProbs=dict, unseenProbs=dict, model="ibm1", method="uniform", path="", preloaded=False, alpha = 0.01, fWords = 0):
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
            self.frenchWords = fWords
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
                        print "provide a path to ibm-1 parameters" # TODO
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

    def bayesian_maximization(self, counts, normalizer):
        return psi(counts + self.alpha) - psi(normalizer + self.alpha * self.frenchWords)

    def train_ibm(self, pairs, threshold, valPairs, valAlignments, aerEpochsThreshold):
        if self.model == IBM.IBM1:
            return self.train_ibm1(pairs, threshold, valPairs, valAlignments, aerEpochsThreshold)
        elif self.model == IBM.IBM1B:
            return self.train_ibm1b(pairs, threshold, valPairs, valAlignments, aerEpochsThreshold)
        elif self.model == IBM.IBM2:
            return self.train_ibm2(pairs, threshold, valPairs, valAlignments, aerEpochsThreshold)


    def train_ibm1b(self, pairs, threshold, valPairs, valAlignments, aerEpochsThreshold):
        """
        Train an IBM model 1, 2 or variational bayes

        Input:
            pairs: list of english-french sentence pairs
            threshold: log-likelihood/ELBO convergence threshold
            valPairs: list of english-french validation sentence pairs, for AER
            valAlignments: gold-standard alignments of the validation pairs, for AER
            aerEpochsThreshold: number of epochs when aer is the termination criteria

        Output:
            
            transProbs, unseenProbs, bestTransProbs, bestUnseenProbs

            Where:
                transProbs: translation probabilities of the trained model with maximum loglikelihood
                unseenProbs: translation probabilities for unseen french-english word pairs with maximum loglikelihood
                bestTransProb: translation probabilities of the trained model with minimum AER
                bestUnseenProbs: translation probabilities for unseen french-english word pairs with minimum AER
        """

        #initialize metrics
        logLikelihood = []
        aers = []
        minAer = float('inf')
        epoch = 0

        #initialize probabilities of the model
        transProbs = self.transProbs  # initialize_ibm(transProbs)

        unseenProbs = self.unseenProbs
        numberOfSentences = len(pairs)

        bestUnseenProbs = unseenProbs
        bestTransProbs = transProbs

        converged = False

        #computing empty counts
        countsEmpty = {}
        countsEnglishEmpty = {}
        for key in transProbs:
            countsEmpty[key] = {}
            countsEnglishEmpty[key] = 0.0
            for secKey in transProbs[key]:
                countsEmpty[key][secKey] = 0.0

        #EM Loop
        while not converged:
            start = time.time()
            epoch += 1
            logLike = 0

            # reset all counts for the translation model to zero
            counts = countsEmpty
            countsEnglish = countsEnglishEmpty

            # Expectation - step
            print "E Step"
            for pair in pairs:

                for j, fWord in enumerate(pair[1]):
                    # calculate the normalizer of the posterior probability of this french word
                    normalizer = 0.0
                    for i, eWord in enumerate(pair[0]):
                        normalizer += transProbs[eWord][fWord]

                    logLike += np.log(normalizer)

                    # get the expected counts based on the posterior probabilities
                    for i, eWord in enumerate(pair[0]):
                        delta = transProbs[eWord][fWord] / normalizer
                        counts[eWord][fWord] += delta
                        countsEnglish[eWord] += delta

            # Maximization - step
            print "M Step"
            for eKey in transProbs:
                unseenProbs[eKey] = self.bayesian_maximization(0, countsEnglish[eKey])
                for fKey in transProbs[eKey]:
                        transProbs[eKey][fKey] = self.bayesian_maximization(counts[eKey][fKey], countsEnglish[eKey])

            # ELBO estimation
            # this constitutes the second part of eq. 25 of Philip's paper,
            # following the derivation in Philip's ELBO write-up

            alpha = self.alpha
            gammaAlpha = gammaln(alpha)
            gammaAlphaSum = gammaln(alpha * self.frenchWords)
            for eWord, eProbs in transProbs.iteritems():
                lamb = 0
                for fWord, fProb in eProbs.iteritems():
                    logProb = fProb
                    transProbs[eWord][fWord] = np.exp(fProb)
                    count = counts[eWord][fWord]
                    logLike += (logProb * (-count) + gammaln(alpha + count) - gammaAlpha)
                    lamb += count
                lamb += self.frenchWords * alpha
                logLike += gammaAlphaSum - gammaln(lamb)

            # Obtaining the log likelihood at this iteration
            logLikelihood.append(logLike / numberOfSentences)

            #Obtaining the AER at this iteration
            predictions = self.get_alignments(self.model, valPairs, transProbs, unseenProbs)

            aer = IBM.get_AER(predictions, valAlignments)
            aers.append(aer)

            #Recalculating the best model so far according to AER
            if aer < minAer:
                minAer = aer
                bestUnseenProbs = unseenProbs
                bestTransProbs = transProbs

            #termination decision
            if epoch >= aerEpochsThreshold:
                if len(logLikelihood) > 1:
                    difference = logLikelihood[-1] - logLikelihood[-2]
                    if difference < threshold:
                        converged = True

            end = time.time()
            print "epoch: ", epoch, " aer: ", aer, " loglikelihood: ", logLikelihood[-1], " time: ", end - start

        self.save_metrics(logLikelihood, "loglike")
        self.save_metrics(aers, "aer")

        return transProbs, unseenProbs, bestTransProbs, bestUnseenProbs







    def init_vogel(self, pairs):
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

        return vogelProbs, countsVogel


    def train_ibm2(self, pairs, threshold, valPairs = False, valAlignments = False, aerEpochsThreshold = 5):
        """
        Train an IBM model 1, 2 or variational bayes

        Input:
            pairs: list of english-french sentence pairs
            threshold: log-likelihood/ELBO convergence threshold
            valPairs: list of english-french validation sentence pairs, for AER
            valAlignments: gold-standard alignments of the validation pairs, for AER
            aerEpochsThreshold: number of epochs when aer is the termination criteria

        Output:
            For IBM2:
                transProbs, vogelProbs, bestTransProbs, bestVogelProbs

            Where:
                transProbs: translation probabilities of the trained model with minimum likelihood
                vogelProbs: vogel jump probabilities of the trained model with minimum likelihood
                bestTransProbs: translation probabilities of the trained model with minimum AER
                bestVogelProbs: vogel jump probabilities of the trained model with minimum AER
                
        """

        logLikelihood = []
        aers = []

        transProbs = self.transProbs  # initialize_ibm(transProbs)

        numberOfSentences = len(pairs)
        minAer = float('inf')
        epoch = 0

        vogelProbs, countsVogel = self.init_vogel(pairs)

        bestVogelProbs = vogelProbs
        bestTransProbs = transProbs

        #computing empty counts
        countsEmpty = {}
        countsEnglishEmpty = {}
        for key in transProbs:
            countsEmpty[key] = {}
            countsEnglishEmpty[key] = 0.0
            for secKey in transProbs[key]:
                countsEmpty[key][secKey] = 0.0

        converged = False

        while not converged:
            start = time.time()
            logLike = 0
            epoch += 1

            # set all counts for the translation model to zero
            counts = countsEmpty
            countsEnglish = countsEnglishEmpty

            # Expectation - step
            print "E Step"
            for pair in pairs:
                I = len(pair[0])  # english sentence length
                J = len(pair[1])  # french sentence length

                for j, fWord in enumerate(pair[1]):
                    # calculate the normalizer of the posterior probability of this french word
                    normalizer = 0.0
                    normalizerVogel = 0.0
                    for i, eWord in enumerate(pair[0]):
                        normalizer += transProbs[eWord][fWord] * vogelProbs[self.vogel_index(i, j, I, J)]
                        normalizerVogel += vogelProbs[self.vogel_index(i, j, I, J)]

                    logLike += np.log(normalizer)

                    # get the expected counts based on the posterior probabilities
                    for i, eWord in enumerate(pair[0]):
                        delta = vogelProbs[self.vogel_index(i, j, I, J)] * transProbs[eWord][fWord] / normalizer
                        countsVogel[self.vogel_index(i, j, I, J)] += delta

                        counts[eWord][fWord] += delta
                        countsEnglish[eWord] += delta

            # Maximization - step
            print "M Step"
            for eKey in transProbs:
                for fKey in transProbs[eKey]:
                    transProbs[eKey][fKey] = counts[eKey][fKey] / countsEnglish[eKey]


            # update Vogel-based alignment probabilities
            normalizer = sum(countsVogel.itervalues())
            vogelProbs = {k: v / normalizer for k, v in countsVogel.iteritems()}

            logLikelihood.append(logLike / numberOfSentences)

            #Obtaining the AER at this iteration
            predictions = self.get_alignments(self.model, valPairs, transProbs, dict(), vogelProbs)

            aer = IBM.get_AER(predictions, valAlignments)
            aers.append(aer)

            #Recalculating the best model so far according to AER

            if aer < minAer:
                minAer = aer
                bestVogelProbs = vogelProbs
                bestTransProbs = transProbs

            if epoch >= aerEpochsThreshold: #termination_criteria == 'aer'
                if len(logLikelihood) > 1:
                    difference = logLikelihood[-1] - logLikelihood[-2]
                    if difference < threshold:
                        converged = True

            end = time.time()
            print "epoch: ", epoch, " aer: ", aer, " loglikelihood: ", logLikelihood[-1], " time: ", end - start


        self.save_metrics(logLikelihood, "loglike")
        self.save_metrics(aers, "aer")

        return transProbs, vogelProbs, bestTransProbs, bestVogelProbs













    def train_ibm1(self, pairs, threshold, valPairs, valAlignments, aerEpochsThreshold):
        """
        Train an IBM model 1, 2 or variational bayes

        Input:
            pairs: list of english-french sentence pairs
            threshold: log-likelihood/ELBO convergence threshold
            valPairs: list of english-french validation sentence pairs, for AER
            valAlignments: gold-standard alignments of the validation pairs, for AER
            aerEpochsThreshold: number of epochs when aer is the termination criteria

        Output:
            transProbs, transProbsAer
            
            Where:
                transProbs: translation probabilities of the trained model
                transProbsAer: translation probabilities of the trained model with the lowest AER metric value
        """

        logLikelihood = []
        aers = []

        transProbs = self.transProbs  # initialize_ibm(transProbs)

        numberOfSentences = len(pairs)
        minAer = float('inf')
        epoch = 0

        bestTransProbs = transProbs

        #computing empty counts
        countsEmpty = {}
        countsEnglishEmpty = {}
        for key in transProbs:
            countsEmpty[key] = {}
            countsEnglishEmpty[key] = 0.0
            for secKey in transProbs[key]:
                countsEmpty[key][secKey] = 0.0

        converged = False

        while not converged: #while loglikelihood hasn't converged
            start = time.time()
            logLike = 0
            epoch += 1

            # set all counts for the translation model to zero
            counts = countsEmpty
            countsEnglish = countsEnglishEmpty

            # Expectation - step
            print "E step"
            for pair in pairs:

                for j, fWord in enumerate(pair[1]):
                    # calculate the normalizer of the posterior probability of this french word
                    normalizer = 0.0
                    for i, eWord in enumerate(pair[0]):
                            normalizer += transProbs[eWord][fWord]

                    logLike += np.log(normalizer)

                    # get the expected counts based on the posterior probabilities
                    for i, eWord in enumerate(pair[0]):
                        delta = transProbs[eWord][fWord] / normalizer
                        counts[eWord][fWord] += delta
                        countsEnglish[eWord] += delta

            # Maximization - step
            print "M Step"
            for eKey in transProbs:
                for fKey in transProbs[eKey]:
                    transProbs[eKey][fKey] = counts[eKey][fKey] / countsEnglish[eKey]

            logLikelihood.append(logLike / numberOfSentences)


            #Obtaining the AER at this iteration
            predictions = self.get_alignments(self.model, valPairs, transProbs)

            aer = IBM.get_AER(predictions, valAlignments)
            aers.append(aer)

            #Recalculating the best model so far according to AER
            if aer < minAer:
                minAer = aer
                bestTransProbs = transProbs

            if epoch >= aerEpochsThreshold:
                if len(logLikelihood) > 1:
                    difference = logLikelihood[-1] - logLikelihood[-2]
                    if difference < threshold:
                        converged = True

            end = time.time()
            print "epoch: ", epoch, " aer: ", aer, " loglikelihood: ", logLikelihood[-1], " time: ", end-start

        self.save_metrics(logLikelihood, "loglike")
        self.save_metrics(aers, "aer")
        self.transProbs = transProbs
        # check for log-likelihood convergence

        return transProbs, bestTransProbs





    def save_metrics(self, data, data_label):
        """Obtain plot of aer error/ log likelihood and store it to the file system"""
        filename = ""
        delim = '_'

        filename += self.model + delim

        if self.model == self.IBM1B:
            filename += str(self.alpha) + delim

        filename += self.method + delim
        filename += data_label

        plt.figure()
        plt.plot([x + 1 for x in range(len(data))], data, 'ro')
        plt.xlabel('Iterations')


        path = "results/" + self.model
        if not os.path.exists(path):
            os.makedirs(path)

        if data_label == 'loglike':
            plt.ylabel('log likelihood')
            cPickle.dump(data, open(path + '/' + filename + '_loglikelihoods', 'wb'))

        if data_label == 'aer':
            plt.ylabel('AER')
            cPickle.dump(data, open(path + '/' + filename + '_aers', 'wb'))


        filename += '.png'

        plt.savefig(path + '/' + filename, bbox_inches='tight')


    @staticmethod
    def get_alignments(model, pairs, transProbs, unseenProbs = dict, vogelProbs=dict):
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
                            if model == IBM.IBM1 or model == IBM.IBM1B:
                                alignProb = transProbs[eWord][fWord]
                            elif model == IBM.IBM2:
                                alignProb = transProbs[eWord][fWord] * vogelProbs[IBM.vogel_index(i, j, I, J)]
                            else:
                                print "incorrect model"
                                break
                        elif model == IBM.IBM1B:
                            alignProb = unseenProbs[eWord]
                    if alignProb > maxProb:
                        maxProb = alignProb
                        alignment = i
                if alignment is not 0:
                    alignments[k].add((alignment, j+1))

        return alignments

    @staticmethod
    def get_AER(predictions, test):
        metric = aer.AERSufficientStatistics()
        # then we iterate over the corpus
        for gold, pred in zip(test, predictions):

            metric.update(sure=gold[0], probable=gold[1], predicted=pred)
        
        return metric.aer()

    @staticmethod
    def plot(data, data_label, legend_label, file_path, save = True, max_iter = 6):
        # plt.figure()

        x_offset = 0.1
        y_offset = 0.01
        axes = plt.gca()
        axes.set_xlim([1-x_offset, max_iter  + x_offset])
        # axes.set_ylim([min(data) - y_offset, max(data) + y_offset])
        p = plt.plot([x + 1 for x in range(len(data))], data, 'o-', label=legend_label, markersize=4)

        if data_label == 'loglike':
            plt.ylabel('log likelihood')

        if data_label == 'aer':
            plt.ylabel('AER')

        if data_label == 'elbo':
            plt.ylabel('ELBO')

        if data_label == "aer":
            min_ind = len(data) - 1 - data[::-1].index(min(data))
            color = p[-1].get_color()
            plt.plot([min_ind + 1], min(data), color + 'o',  markersize=7)

        plt.xlabel('Iterations')
        plt.xticks(range(1,len(data)+1))
        # plt.yticks(data)
        if save:
            plt.savefig(file_path + '.png', bbox_inches='tight')
        # plt.show()

