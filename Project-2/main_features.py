from training.features import *
from gensim.models import Word2Vec
import training.model as model

"""script for generating features from forests"""
EMBED = False
BIGRAM = False
embedpath = "chEmbeddings100"
bipath = "biProbs"
bijoinpath = "jointProbs"

if EMBED:
    print('embed')
    chEmbeddings = get_word_embeddings("datamap/chinese.zh-en", iterations=500, name="chEmbeddings100")
    enEmbeddings = get_word_embeddings("datamap/english.zh-en", iterations=500, name="enEmbeddings100")

    print(enEmbeddings.wv.similarity("man", "woman"))
    print(enEmbeddings.wv.similarity("big", "large"))
    print(enEmbeddings.most_similar(positive=["man"]))
    print(enEmbeddings.most_similar(positive=["woman"]))
    print(enEmbeddings.most_similar(positive=["fish"]))
    print(enEmbeddings.most_similar(positive=["big"]))
    print(enEmbeddings.most_similar(positive=["the"]))
    print(enEmbeddings.most_similar(positive=["problem"]))
else:
    # if globals.USE_COMPLEX_FEATURES:
    #     chEmbeddings = Word2Vec.load(embedpath)
    #     chEmbeddings = chEmbeddings.wv
    # else:
    chEmbeddings = 0

if BIGRAM:
    print("bigram")
    bi_joint_probs, bi_probs = get_bigram_probabilities("datamap/chinese.zh-en")
else:
    if globals.USE_COMPLEX_FEATURES:
        bi_joint_probs = pickle.load(open(bijoinpath, 'rb'))
        bi_probs = pickle.load(open(bipath, 'rb'))
    else:
        bi_joint_probs = 0
        bi_probs = 0


# for key in bi_probs:
#     density = 0.0
#     for key2 in bi_probs[key]:
#         if bi_probs[key][key2] > 0.2:
#             print(key, key2, bi_probs[key][key2])

# TODO extract the features

source_lexicon, target_lexicon = Data.generate_IBM_lexicons()
# itgs = Data.read_forests()

features = generate_features_all(source_lexicon, target_lexicon, bi_probs,
                             bi_joint_probs, chEmbeddings)