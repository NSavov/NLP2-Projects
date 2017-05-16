from training.model import *
from gensim.models import Word2Vec
import functools
import pickle
"contains all necessary functionality to construct a feature vector fmap"
"this is an extension of the simple feature framework given in the notebook, found in the model file"


def complex_features(edge: Rule, src_fsa: FSA, source: dict, target: dict, bi_probs: dict, src_em: Word2Vec,
                     eps=Terminal('-EPS-'), sparse_del=False, sparse_ins=False, sparse_trans=False,
                     sparse_bigrams=False, fmap=False) -> dict:
    """
    Featurises an edge given:
        * edge: rule and spans
        * src_fsa: src sentence as an FSA
        * source: chinese -> english translation probabilities
        * target: english -> chinese translation probabilities
        * bi_probs: source side skip-bi-gram probabilities
        * src_em: memory efficient (embeddings.wv) source word embeddings
        * eps: epsilon rule symbol
        * sparse functionality
        * fmap: simple features map or False
        * TODO: target sentence length n
        * TODO: extract IBM1 dense features
    :returns a feature map (sparse dict) with features:
        * features from the simple_features function
        * source language word embeddings
        * skip-bi-grams; dense or sparse
    """

    # start with the simple features
    if not fmap:
        fmap = simple_features(edge, src_fsa, source, target, eps, sparse_del, sparse_ins, sparse_trans)

    # average outside word embeddings (rest of sentence)
    (s1, s2), (t1, t2) = get_bispans(edge.lhs)
    previous = []
    after = []

    for i in range(s1): # words before the span under consideration
        try:
            previous.append(src_em[get_source_word(src_fsa, i, i+1)])
        except KeyError:
            pass
    for k in range(s2, src_fsa.nb_states()): # words after the span under consideration
        try:
            after.append(src_em[get_source_word(src_fsa,k,k+1)])
        except KeyError:
            pass

    after = functools.reduce(np.add(), after) / len(after)
    previous = functools.reduce(np.add(), previous) / len(previous)

    for j in range(previous.size):
        # one feature for each dimension in the word embedding
        fmap["outside:before" + str(j)] += previous[j]
        fmap["outside:after" + str(j)] += after[j]

    # inside word embeddings and skip-bi-grams
    inside = []
    skip_bigrams = []
    for i in range(s1, s2):
        inside.append(src_em[get_source_word(src_fsa, i, i+1)])
        for k in range(i+1, s2):
            skip_bigrams.append([get_source_word(src_fsa, i, i+1), get_source_word(src_fsa, k, k+1)])

    # average inside word embeddings
    inside = functools.reduce(np.add(), inside) / len(inside)
    for j in range(inside.size):
        fmap["inside:lhs" + str(j)] += inside[j]

    # dense skip-bi-grams, product over bi-gram probabilities
    if len(skip_bigrams) > 0:
        fmap["skip-bigram"] = 1.0
    for bigram in skip_bigrams:
        fmap["skip-bigram"] *= bi_probs[bigram[0]][bigram[1]]
        if sparse_bigrams:  # sparse
            fmap["bigram:%s/%s" % (bigram[0], bigram[1])] += 1.0

    if len(edge.rhs) == 2:  # binary rule
        #TODO
        pass

    return fmap


def get_word_embeddings(file: str, iterations=100, save=True, name="Word2Vec") -> Word2Vec:
    """
    get words embeddings from a file containing a list of sentences
    uses the skip-gram architecture as it is suitable for smaller datasets
    uses horizontal softmax, as it yields better results for little seen words
    embeds a 100 dimensional word vector as the small size of the dataset prohibits larger embeddings
    """

    # get sentences from file
    with open(file, encoding='utf8') as f:
        sentences = [line.split() for line in f.read().splitlines()]

    # train a word to vec model
    model = Word2Vec(sentences, size=100, workers=8, iter=iterations, min_count=1, hs=1, negative=0, sg=1)
    model.train(sentences)

    # storage of the word vectors
    if save:
        model.save(name)
        return model
    else:
        return model


def get_bigram_probabilities(file: str, save=True, name1="uniProbs", name2="jointProbs", name3="biProbs"):
    """get uni-gram, skip-bi-joint and skip-bi-gram probabilities from a corpus"""
    # get sentences from file
    with open(file, encoding='utf8') as f:
        sentences = [line.split() for line in f.read().splitlines()]

    # prelims
    uni_probs = defaultdict(float)
    bi_joint_probs = defaultdict(float)
    bi_probs = defaultdict(float)
    words = 0.0
    bigrams = 0.0

    # extract counts from the given corpus
    for sentence in sentences:
        words += len(sentence)
        for i, word in enumerate(sentence):
            uni_probs[word] += 1.0  # count word occurrence
            for j in range(i+1, len(sentence)):
                bigrams += 1
                bi_probs[word][sentence[j]] += 1.0  # count skip bigrams

    # determine probabilities from the counts
    for key in uni_probs:
        uni_probs[key] = uni_probs[key] / words  # uni-gram probabilities
        for key2 in bi_joint_probs[key]:
            bi_joint_probs[key][key2] = bi_joint_probs[key][key2] / bigrams  # co-occurence probabilities skip-bi-grams
            bi_probs[key][key2] = bi_joint_probs[key][key2] / uni_probs[key] # skip-bi-gram probabilities

    if save:
        pickle.dump(uni_probs, open(name1, "wb"))
        pickle.dump(bi_joint_probs, open(name2, "wb"))
        pickle.dump(bi_probs, open(name3, "wb"))
        return uni_probs, bi_joint_probs, bi_probs
    else:
        return uni_probs, bi_joint_probs, bi_probs





