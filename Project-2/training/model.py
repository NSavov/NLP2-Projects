from libitg import *
from training.features import *
from gensim.models import Word2Vec
import math
from data import Data
import numpy as np
import globals

'contains all functions from the model part of the LV-CRF-Roadmap ipython notebook'


def get_source_word(fsa: FSA, origin: int, destination: int) -> str:
    """Returns the python string representing a source word from origin to destination (assuming there's a single one)"""
    labels = list(fsa.labels(origin, destination))
    assert len(
        labels) == 1, 'Use this function only when you know the path is unambiguous, found %d labels %s for (%d, %d)' % (
    len(labels), labels, origin, destination)
    return labels[0]


def get_target_word(symbol: Symbol):
    """Returns the python string underlying a certain terminal (thus unwrapping all span annotations)"""
    if not symbol.is_terminal():
        raise ValueError('I need a terminal, got %s of type %s' % (symbol, type(symbol)))
    return symbol.root().obj()


def get_bispans(symbol: Span):
    """
    Returns the bispans associated with a symbol.

    The first span returned corresponds to paths in the source FSA (typically a span in the source sentence),
     the second span returned corresponds to either
        a) paths in the target FSA (typically a span in the target sentence)
        or b) paths in the length FSA
    depending on the forest where this symbol comes from.
    """
    if not isinstance(symbol, Span):
        raise ValueError('I need a span, got %s of type %s' % (symbol, type(symbol)))
    s, start2, end2 = symbol.obj()  # this unwraps the target or length annotation
    _, start1, end1 = s.obj()  # this unwraps the source annotation
    return (start1, end1), (start2, end2)


def simple_features(edge: Rule, src_fsa: FSA, source: dict, target: dict, eps=Terminal('-EPS-'),
                    sparse_del=False, sparse_ins=False, sparse_trans=False) -> dict:
    """
    Featurises an edge given
        * edge: rule and spans
        * src_fsa: src sentence as an FSA
        * eps: epsilon terminal symbol
        * source: chinese -> english translation probabilities
        * target: english -> chinese translation probabilities
        * sparse feature vector options
    :returns a feature map containing the following features:
        * left and right source rhs deletion indicator for binary rules
        * left and right target rhs insertion indicator for binary rules
        * monotone and inverted source rhs indicator for binary rules
        * terminal deletion, insertion and translation indicator
        * terminal deletion, insertion and translation IBM1 probabilities
        * top rule (S -> X), binary and terminal indicator
        * source and target span lengths
        * UNK translation, insertion and deletion indicators
    crucially, note that the target sentence y is not available!
    """
    fmap = defaultdict(float)

    if len(edge.rhs) == 2:  # binary rule
        fmap['type:binary'] += 1.0
        # here we could have sparse features of the source string as a function of spans being concatenated
        (ls1, ls2), (lt1, lt2) = get_bispans(edge.rhs[0])  # left of RHS
        (rs1, rs2), (rt1, rt2) = get_bispans(edge.rhs[1])  # right of RHS

        # source and target span lengths as features
        fmap["length:src"] += (ls2 - ls1) + (rs2 - rs1)
        fmap["length:tgt"] += (lt2 - lt1) + (rt2 - rt1)

        if lt1 == lt2:  # deletion of source left child
            fmap['deletion:lbs'] += 1.0
        if rt1 == rt2:  # deletion of source right child
            fmap['deletion:rbs'] += 1.0
        if ls1 == ls2:  # insertion of target left child
            fmap['insertion:lbt'] += 1.0
        if rs1 == rs2:  # insertion of target right child
            fmap['insertion:rbt'] += 1.0
        if ls2 == rs1:  # monotone
            fmap['monotone'] += 1.0
        if ls1 == rs2:  # inverted
            fmap['inverted'] += 1.0
    else:  # unary
        symbol = edge.rhs[0]

        # source and target span lengths as features
        (s1, s2), (t1, t2) = get_bispans(symbol)
        fmap["length:src"] += s2 - s1
        fmap["length:tgt"] += t2 - t1

        if symbol.is_terminal():  # terminal rule
            fmap['type:terminal'] += 1.0
            # we could have IBM1 log probs for the translation pair or ins/del
            if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
                # for sure there is a source word
                src_word = get_source_word(src_fsa, s1, s2)
                fmap['type:deletion'] += 1.0

                if src_word == "UNK":
                    # UNK -> epsilon rule
                    fmap['type:UNK_del'] += 1.0
                else:
                    # dense versions (for initial development phase)
                    # use IBM1 prob of null aligning to chinese source word
                    fmap['ibm1:del:logprob'] += math.log(target["<NULL>"][src_word])

                # sparse version
                if sparse_del:
                    fmap['del:%s' % src_word] += 1.0
            else:
                # for sure there's a target word
                tgt_word = get_target_word(symbol)
                if s1 == s2:  # has not consumed any source word, must be an eps rule
                    fmap['type:insertion'] += 1.0

                    if tgt_word == "UNK":
                        # epsilon -> UNK rule
                        fmap['type:UNK_ins'] += 1.0
                    else:
                        # dense version
                        # use IBM1 prob of null aligning to english target word
                        fmap['ibm1:ins:logprob'] += math.log(source["<NULL>"][tgt_word])

                    # sparse version
                    if sparse_ins:
                        fmap['ins:%s' % tgt_word] += 1.0
                else:
                    # for sure there's a source word
                    src_word = get_source_word(src_fsa, s1, s2)

                    fmap['type:translation'] += 1.0

                    if src_word == 'UNK' and tgt_word == 'UNK':
                        # UNK -> UNK rule
                        fmap['type:UNK2UNK'] += 1.0
                    elif src_word == 'UNK':
                        # UNK -> target rule
                        fmap['type:UNK2t'] += 1.0
                    elif tgt_word == 'UNK':
                        # target -> UNK rule
                        fmap['type:s2UNK'] += 1.0
                    else:
                        # dense version
                        # use IBM1 prob for source to target and target to source translation
                        fmap['ibm1:s2t:logprob'] += math.log(source[src_word][tgt_word])
                        fmap['ibm1:t2s:logprob'] += math.log(target[tgt_word][src_word])

                    # sparse version
                    if sparse_trans:
                        fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0
        else:  # S -> X
            fmap['top'] += 1.0
    return fmap


def featurize_edges(forest: CFG, is_complex: bool, src_fsa: FSA, source: dict, target: dict, bi_probs: dict, bi_joint: dict,
                     src_em: Word2Vec, eps=Terminal('-EPS-'), sparse_del=False, sparse_ins=False, sparse_trans=False,
                     sparse_bigrams=False, fmap=False) -> dict:
    """Featurize the edges of a forest, yielding either a dict with some simple features or a more complex
    feature dict that contains word embeddings and skip-bi-grams"""
    edge2fmap = dict()
    for edge in forest:
        if is_complex:
            # generate complex feature vector
            edge2fmap[edge] = complex_features(edge, src_fsa, source, target, bi_probs, bi_joint, src_em, eps,
                                               sparse_del, sparse_ins, sparse_trans, sparse_bigrams, fmap)
        else:
            # generate only a simple feature vector
            edge2fmap[edge] = simple_features(edge, src_fsa, source, target, eps, sparse_del, sparse_ins, sparse_trans)
    return edge2fmap


def weight_function(edge, fmap, wmap) -> float:
    """
    computes the dot product of the feature vector and weights of the given edge
    both the fmap and wmap are defaultdicts, of which corresponding keys are multiplied
    and summed over.
    """
    w = 0.0
    for key in fmap:
        w += fmap[key] * wmap[key]
    return w

import msvcrt as m
def generate_features(source_lexicon, target_lexicon, bi_probs: dict, bi_joint: dict, src_em: Word2Vec, corpus_file_path = globals.TRAINING_SET_SELECTED_FILE_PATH):
    itgs = Data.read_forests()
    features = []

    with open(corpus_file_path, encoding='utf8') as f:
        corpus_lines = f.read().splitlines()

    selected_sentences = [corpus_lines[entry[0]] for entry in itgs]
    for i, forest in enumerate(itgs):
        translation_pair = selected_sentences[i].split(' ||| ')
        chinese_sentence = translation_pair[0]
        src_fsa = make_fsa(chinese_sentence)
        # print(features)
        # input()
        try:
            sentence_features  = [featurize_edges(forest[1], False, src_fsa, source_lexicon, target_lexicon, bi_probs, bi_joint, src_em),
                                  featurize_edges(forest[2], False, src_fsa, source_lexicon, target_lexicon, bi_probs, bi_joint, src_em)]
            features.append(sentence_features)
        except KeyError:
            continue
