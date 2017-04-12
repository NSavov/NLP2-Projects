import random
import numpy as np

class IBM:

    IBM1 = 'ibm1'
    IBM2 = 'ibm2'

    def _init_(self, transProbs, type):
        trans = {}
        for key in transProbs:
            trans[key] = {}
            vocabSize = len(transProbs[key].keys())
            for secKey in transProbs[key]:
                trans[key][secKey] = 1.0 / vocabSize
        self.transProbs = trans
        self.type = type
        
        
    def randomize(self, transProbs):
        trans = {}
        for key in transProbs:
            trans[key] = {}
            vocabSize = len(transProbs[key].keys())
            for secKey in transProbs[key]:
                trans[key][secKey] = 1.0 / vocabSize
        self.transProbs = trans
        

    def train_ibm(self, pairs, termination_criteria, threshold, transProbs = False):
        # trains an ibm 1 model
        converged = False
        logLikelihood = []
        if not transProbs:
            transProbs = self.transProbs # initialize_ibm_1(transProbs)
        else:
            self.randomize(transProbs)
            transProbs = self.transProbs
            
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
            print "E"
            for pair in pairs:

                # if type == self.IBM2:
                #     logLike += #log of IBM2 alignment probability


                logLike += -(len(pair[1]) * np.log(len(pair[0])+1))
                for fWord in pair[1]:
                    # calculate the normalizer of the posterior probability of this french word
                    normalizer = 0
                    for eWord in pair[0]:
                        normalizer += transProbs[eWord][fWord]

                    logLike += np.log(normalizer)
                    # get the expected counts based on the posterior probabilities
                    for eWord in pair[0]:
                        counts[eWord][fWord] += (transProbs[eWord][fWord] / normalizer)
                        countsEnglish[eWord] += (transProbs[eWord][fWord] / normalizer)

                        # if self.type == IBM2:
                        #     #add alignment counts here

            logLikelihood.append(logLike)
            print logLike

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
                    if self.type == self.IBM1:
                        transProbs[eKey][fKey] = counts[eKey][fKey] / countsEnglish[eKey]

                    # elif self.type == self.IBM2:
                    #     transProbs[eKey][fKey] = ## do magic with IBM2 counts here

        return transProbs