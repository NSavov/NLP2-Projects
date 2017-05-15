from training.model import *
from gensim.models import Word2Vec
import functools

"contains all necessary functionality to construct a feature vector fmap"
"this is an extension of the simple feature framework given in the notebook, found in the model file"


def complex_features(edge: Rule, src_fsa: FSA, source: dict, target: dict, src_em: Word2Vec,
                     eps=Terminal('-EPS-'), sparse_del=False, sparse_ins=False, sparse_trans=False, fmap=False) -> dict:
    """
    Featurises an edge given:
        * edge: rule and spans
        * src_fsa: src sentence as an FSA
        * eps: epsilon rule symbol
        * sparse functionality
        * fmap: simple features map or False
        * TODO: target sentence length n
        * TODO: extract IBM1 dense features
    :returns a feature map (sparse dict) with features:
        * features from the simple_features function
        * source language word embeddings
        * other cool stuff
        *
        *
        *
        *
        *
        *
    """

    # start with the simple features
    if not fmap:
        fmap = simple_features(edge, src_fsa, source, target, eps, sparse_del, sparse_ins, sparse_trans)

    # average outside word embeddings (rest of sentence)
    (s1, s2), (t1, t2) = get_bispans(edge.lhs)
    previous = []
    after = []
    for i in range(s1):
        previous.append(src_em[get_source_word(src_fsa, i, i+1)])
    for k in range(s2, src_fsa.nb_states()):
        after.append(src_em[get_source_word(src_fsa,k,k+1)])
    after = functools.reduce(np.add(), after) / len(after)
    previous = functools.reduce(np.add(), previous) / len(previous)
    for j in range(previous.size):
        fmap["outside:before" + str(j)] += previous[j]
        fmap["outside:after" + str(j)] += after[j]

    # average inside word embeddings
    inside = []
    for i in range(s1, s2):
        inside.append(src_em[get_source_word(src_fsa, i, i+1)])
    inside = functools.reduce(np.add(), inside) / len(inside)
    for j in range(inside.size):
        fmap["inside:lhs" + str(j)] += inside[j]

    if len(edge.rhs) == 2:  # binary rule
        #TODO
        len = 1


    return fmap


def get_word_embeddings(file: str, iterations=100, save=True, name="Word2Vec") -> Word2Vec:
    """get words embeddings from a file containing a list of sentences"""
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
