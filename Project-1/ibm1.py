import random
import numpy as np

class IBM1:
    def _init_(self, transProbs):
        trans = {}
        for key in transProbs:
            trans[key] = {}
            for secKey in transProbs[key]:
                trans[key][secKey] = random.uniform(0.00001, 0.9999999)
        self.transProbs = trans

    def train_ibm_1(self, pairs, criteria, threshold, val):
        # trains an ibm 1 model
        converged = False
        logLikelihood = []
        transProbs = self.transProbs # initialize_ibm_1(transProbs)
        while (not converged):
            logLike = 0

            # set all counts to zero
            counts = {}
            countsEnglish = {}
            for key in transProbs:
                counts[key] = {}
                countsEnglish[key] = 0.0
                for secKey in transProbs[key]:
                    counts[key][secKey] = 0.0

            # loop over sentences, french words and english words
            # Expectation - step
            for pair in pairs:
                logLike += np.log(1 / (len(pair[0]) + 1) ** len(pair[1]))
                for i, fWord in enumerate(pair[1]):
                    # calculate the normalizer of the posterior probability of this french word
                    normalizer = 0
                    for eWord in enumerate(pair[0]):
                        normalizer += transProbs[eWord][fWord]

                    logLike += np.log(normalizer)
                    # get the expected counts based on the posterior probabilities
                    for j, eWord in enumerate(pair[0]):
                        counts[eWord][fWord] += (transProbs[eWord][fWord] / normalizer)
                        countsEnglish[eWord] += (transProbs[eWord][fWord] / normalizer)

            logLikelihood.append(logLike)

            # check for log-likelihood convergence
            if len(logLikelihood) > 1:
                difference = logLikelihood[-1] - logLikelihood[-2]
                if difference < threshold:
                    converged = True
                    break

            # Maximization - step
            for eKey in transProbs:
                for fKey in transProbs[eKey]:
                    transProbs[eKey][fKey] = counts[eKey][fKey] / countsEnglish[eKey]

        return transProbs