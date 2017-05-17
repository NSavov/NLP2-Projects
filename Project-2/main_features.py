from training.features import *
from gensim.models import Word2Vec

"""script for generating features from forests"""
EMBED = False
BIGRAM = True
embedpath = "chEmbeddings100"
bipath = "biProbs"
bijoinpath = "jointProbs"

if EMBED:
    chEmbeddings = get_word_embeddings("data/chinese.zh-en", iterations=500, name="chEmbeddings100")
    enEmbeddings = get_word_embeddings("data/english.zh-en", iterations=500, name="enEmbeddings100")

    print(enEmbeddings.wv.similarity("man", "woman"))
    print(enEmbeddings.wv.similarity("big", "large"))
    print(enEmbeddings.most_similar(positive=["man"]))
    print(enEmbeddings.most_similar(positive=["woman"]))
    print(enEmbeddings.most_similar(positive=["fish"]))
    print(enEmbeddings.most_similar(positive=["big"]))
    print(enEmbeddings.most_similar(positive=["the"]))
    print(enEmbeddings.most_similar(positive=["problem"]))
else:
    chEmbeddings = Word2Vec.load(embedpath)

if BIGRAM:
    print("bigram")
    bi_joint_probs, bi_probs = get_bigram_probabilities("data/english.zh-en")
else:
    bi_joint_probs = pickle.load(open(bijoinpath, 'rb'))
    bi_probs = pickle.load(open(bipath, 'rb'))


for key in bi_probs:
    density = 0.0
    for key2 in bi_probs[key]:
        density += bi_probs[key][key2]
    print(density)

# TODO extract the features

