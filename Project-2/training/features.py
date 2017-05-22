from training.model import *
from gensim.models import Word2Vec
import functools
import pickle
import time
"contains all necessary functionality to construct a feature vector fmap"
"this is an extension of the simple feature framework given in the notebook, found in the model file"


def complex_features(edge: Rule, src_fsa: FSA, source: dict, target: dict, bi_probs: dict, bi_joint: dict,
                     src_em: Word2Vec, eps=Terminal('-EPS-'), sparse_del=False, sparse_ins=False, sparse_trans=False,
                     sparse_bigrams=False, fmap=False) -> dict:
    """
    Featurises an edge given:
        * edge: rule and spans
        * src_fsa: src sentence as an FSA
        * source: chinese -> english translation probabilities
        * target: english -> chinese translation probabilities
        * bi_probs: source side skip-bi-gram probabilities
        * bi_joint: source side skip-bi-joint probabilities
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

    # measuring_time = []
    # start_time = time.clock()
    # start with the simple features
    if not fmap:
        fmap = simple_features(edge, src_fsa, source, target, eps, sparse_del, sparse_ins, sparse_trans)
    # measuring_time.append(start_time - time.clock())

    # average outside word embeddings (rest of sentence)
    if len(edge.rhs) == 2 or edge.rhs[0].is_terminal():
        (s1, s2), (_, _) = get_bispans(edge.lhs)
        skip_bigrams = []
        if len(edge.rhs) == 2:
            (sl1, sl2), (_, _) = get_bispans(edge.rhs[0])
            (sr1, sr2), (_, _) = get_bispans(edge.rhs[1])
            skip_bigrams_left = []
            skip_bigrams_right = []

        # previous = []
        # after = []
        # start_time = time.clock()
        # for i in range(s1):  # words before the span under consideration
        #     try:
        #         previous.append(src_em[get_source_word(src_fsa, i, i+1)])
        #     except KeyError:
        #         pass
        # for k in range(s2, src_fsa.nb_states()-1):  # words after the span under consideration
        #     try:
        #         after.append(src_em[get_source_word(src_fsa, k, k+1)])
        #     except KeyError:
        #         pass

        # if after:
        #     # average word embedding after span
        #     after = functools.reduce(lambda x, y: x+y, after) / len(after)
        # if previous:
        #     # average word embedding before span
        #     previous = functools.reduce(lambda x, y: x+y, previous) / len(previous)

        # for j in range(len(previous)):
        #     # one feature for each dimension in the word embedding
        #     fmap["outside:before" + str(j)] += previous[j]
        # for j in range(len(after)):
        #     fmap["outside:after" + str(j)] += after[j]

        # measuring_time.append(start_time - time.clock())
        # start_time = time.clock()
        # inside word embeddings and skip-bi-grams
        # inside = []

        for i in range(s1, s2):
            # try:
            #     inside.append(src_em[get_source_word(src_fsa, i, i+1)])
            # except KeyError:
            #     pass
            for k in range(i+1, s2):
                skip_bigrams.append([get_source_word(src_fsa, i, i+1), get_source_word(src_fsa, k, k+1)])
                if len(edge.rhs) == 2:
                    if k < sl2:
                        skip_bigrams_left.append([get_source_word(src_fsa, i, i+1), get_source_word(src_fsa, k, k+1)])
                    if i + 1 > sr1:
                        skip_bigrams_right.append([get_source_word(src_fsa, i, i+1), get_source_word(src_fsa, k, k+1)])

        # measuring_time.append(start_time - time.clock())
        # start_time = time.clock()

        # average inside word embeddings
        # if inside:
        #     inside = functools.reduce(lambda x, y: x+y, inside) / len(inside)

        # for j in range(len(inside)):
        #     fmap["inside:lhs" + str(j)] += inside[j]
        # measuring_time.append(start_time - time.clock())
        # start_time = time.clock()

        # dense skip-bi-grams, product over bi-gram probabilities
        if len(skip_bigrams) > 0:
            fmap["skip-bigram"] = 1.0
            fmap["skip-joint"] = 1.0
            if len(edge.rhs) == 2:
                fmap["skip-bigram-left"] = 1.0
                fmap["skip-joint-left"] = 1.0
                fmap["skip-bigram-right"] = 1.0
                fmap["skip-joint-right"] = 1.0

            for bigram in skip_bigrams:
                if bigram[0] != "-UNK-" and bigram[1] != "-UNK-":
                    # left hand side bigram probabilities
                    fmap["skip-bigram"] *= bi_probs[bigram[0]][bigram[1]]
                    fmap["skip-joint"] *= bi_joint[bigram[0]][bigram[1]]
                    if sparse_bigrams:  # sparse
                        fmap["bigram:%s/%s" % (bigram[0], bigram[1])] += 1.0

            if len(edge.rhs) == 2:
                # right hand side bigram probabilities
                for bigram in skip_bigrams_left:
                    if bigram[0] != "-UNK-" and bigram[1] != "-UNK-":
                        fmap["skip-bigram-left"] *= bi_probs[bigram[0]][bigram[1]]
                        fmap["skip-joint-left"] *= bi_joint[bigram[0]][bigram[1]]
                        if sparse_bigrams:  # sparse
                            fmap["bigram-left:%s/%s" % (bigram[0], bigram[1])] += 1.0
                for bigram in skip_bigrams_right:
                    if bigram[0] != "-UNK-" and bigram[1] != "-UNK-":
                        fmap["skip-bigram-right"] = bi_probs[bigram[0]][bigram[1]]
                        fmap["skip-joint-right"] = bi_joint[bigram[0]][bigram[1]]
                        if sparse_bigrams:  # sparse
                            fmap["bigram-right:%s/%s" % (bigram[0], bigram[1])] += 1.0

        # measuring_time.append(start_time - time.clock())
    else:
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
    """get skip-bi-joint and skip-bi-gram probabilities from a corpus"""
    # get sentences from file
    with open(file, encoding='utf8') as f:
        sentences = [line.split() for line in f.read().splitlines()]

    # prelims
    bi_joint_probs = defaultdict(lambda : defaultdict(float))
    bi_probs = defaultdict(lambda : defaultdict(float))
    bigrams = defaultdict(float)
    total_bigrams = 1.0

    # extract counts from the given corpus
    for sentence in sentences:
        for i, word in enumerate(sentence):
            for j in range(i+1, len(sentence)):
                bigrams[word] += 1.0 # count total bigrams for that word
                total_bigrams += 1.0
                bi_joint_probs[word][sentence[j]] += 1.0  # count skip bigrams

    # determine probabilities from the counts
    for key in bi_joint_probs:
        for key2 in bi_joint_probs[key]:
            bi_probs[key][key2] = bi_joint_probs[key][key2] / bigrams[key]  # skip-bi-gram conditional probabilities
            bi_joint_probs[key][key2] = bi_joint_probs[key][key2] / total_bigrams  # skip-bi-gram joint probabilities

    if save:
        pickle.dump(dict(bi_joint_probs), open(name2, "wb"))
        pickle.dump(dict(bi_probs), open(name3, "wb"))
        return bi_joint_probs, bi_probs
    else:
        return bi_joint_probs, bi_probs
