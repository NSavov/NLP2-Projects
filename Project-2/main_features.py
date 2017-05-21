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
    chEmbeddings = Word2Vec.load(embedpath)

if BIGRAM:
    print("bigram")
    bi_joint_probs, bi_probs = get_bigram_probabilities("datamap/chinese.zh-en")
else:
    bi_joint_probs = pickle.load(open(bijoinpath, 'rb'))
    bi_probs = pickle.load(open(bipath, 'rb'))


# for key in bi_probs:
#     density = 0.0
#     for key2 in bi_probs[key]:
#         if bi_probs[key][key2] > 0.2:
#             print(key, key2, bi_probs[key][key2])

# TODO extract the features

source_lexicon, target_lexicon = Data.generate_IBM_lexicons()
# itgs = Data.read_forests()

features = []
USE_COMPLEX_FEATURES = False

corpus_file_path = globals.TRAINING_SET_SELECTED_FILE_PATH
with open(corpus_file_path, encoding='utf8') as f:
    corpus_lines = f.read().splitlines()

for i in range(7):
    subset_file_path = globals.ITG_SET_SELECTED_FILE_PATH[:-5] + str(i+1) + globals.ITG_SET_SELECTED_FILE_PATH[-5:]
    itgs = Data.read_forests(subset_file_path)
    features.extend(model.generate_features(itgs, source_lexicon, target_lexicon, USE_COMPLEX_FEATURES,  bi_probs, bi_joint_probs, chEmbeddings.wv, corpus_lines))

