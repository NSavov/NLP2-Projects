
# coding: utf-8

# # Neural IBM1
# 
# NLP2 2016/2017 Project 3

# In[ ]:




# In[1]:

# first run a few imports:
# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')
import tensorflow as tf
import numpy as np
from pprint import pprint
import pickle


# In[2]:

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
tf.__version__


# In[3]:

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape=[2, 4], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape=[2, 2, 2, 1], name='b')
sess = tf.Session()
c = a[:,0:-1]
print(sess.run(a))
# print(sess.run(b))
# c = tf.matmul(a,b)
# print(sess.run(c))
# b = tf.reshape(a, [4,2])
# c = tf.reshape(b, [2,2,2])
print(sess.run(c))


# In[4]:

# Creates a graph.
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))


# ### Let's first load some data

# In[16]:

# the paths to our training and validation data, English side
train_e_path = 'data/training/hansards.36.2.e.gz'
train_f_path = 'data/training/hansards.36.2.f.gz'
dev_e_path = 'data/validation/dev.e.gz'
dev_f_path = 'data/validation/dev.f.gz'
dev_wa = 'data/validation/dev.wa.nonullalign'
test_e_path = 'data/test/test.e.gz'
test_f_path = 'data/test/test.f.gz'
test_wa = 'data/test/test.wa.nonullalign'


# In[6]:

# check utils.py if you want to see how smart_reader and bitext_reader work in detail
from utils import smart_reader, bitext_reader

    
def bitext_reader_demo(src_path, trg_path):
    """Demo of the bitext reader."""

    # create a reader
    src_reader = smart_reader(src_path)
    trg_reader = smart_reader(trg_path)
    bitext = bitext_reader(src_reader, trg_reader)

    # to see that it really works, try this:
    print(next(bitext))
    print(next(bitext))
    print(next(bitext))
    print(next(bitext))  


bitext_reader_demo(train_e_path, train_f_path)


# In[7]:

# To see how many sentences are left if you filter by length, you can do this:

def demo_number_filtered_sentence_pairs(src_path, trg_path):
    src_reader = smart_reader(src_path)
    trg_reader = smart_reader(trg_path)
    max_length = 30
    bitext = bitext_reader(src_reader, trg_reader, max_length=max_length)   
    num_sentences = sum([1 for _ in bitext])
    print("There are {} sentences with max_length = {}".format(num_sentences, max_length))

  
demo_number_filtered_sentence_pairs(train_e_path, train_f_path)


# ### Now, let's create a vocabulary!
# 
# We first define a class `Vocabulary` that helps us convert tokens (words) into numbers. This is useful later, because then we can e.g. index a word embedding table using the ID of a word.

# In[8]:

# check vocabulary.py to see how the Vocabulary class is defined
from vocabulary import OrderedCounter, Vocabulary


# Now let's try out our Vocabulary class:

# In[9]:

def vocabulary_demo():
    # We used up a few lines in the previous example, so we set up
    # our data generator again.
    corpus = smart_reader(train_e_path)  

    # Let's create a vocabulary given our (tokenized) corpus
    vocabulary = Vocabulary(corpus=corpus)
    print("Original vocabulary size: {}".format(len(vocabulary)))

    # Now we only keep the highest-frequency words
    vocabulary_size=1000
    vocabulary.trim(vocabulary_size)
    print("Trimmed vocabulary size: {}".format(len(vocabulary)))

    # Now we can get word indexes using v.get_word_id():
    for t in ["<PAD>", "<UNK>", "the"]:
        print("The index of \"{}\" is: {}".format(t, vocabulary.get_token_id(t)))

    # And the inverse too, using v.i2t:
    for i in range(10):
        print("The token with index {} is: {}".format(i, vocabulary.get_token(i)))

    # Now let's try to get a word ID for a word not in the vocabulary
    # we should get 1 (so, <UNK>)
    for t in ["!@!_not_in_vocab_!@!"]:
        print("The index of \"{}\" is: {}".format(t, vocabulary.get_token_id(t)))

    
vocabulary_demo()


# Now let's create the vocabularies that we use further on.

# In[18]:

# Using only 1000 words will result in many UNKs, but
# it will make training a lot faster. 
# If you have a fast computer, a GPU, or a lot of time,
# try with 10000 instead.
max_tokens=4000

corpus_e = smart_reader(train_e_path)    
vocabulary_e = Vocabulary(corpus=corpus_e, max_tokens=max_tokens)
pickle.dump(vocabulary_e, open("vocabulary_e.pkl", mode="wb"))
print("English vocabulary size: {}".format(len(vocabulary_e)))

corpus_f = smart_reader(train_f_path)    
vocabulary_f = Vocabulary(corpus=corpus_f, max_tokens=max_tokens)
pickle.dump(vocabulary_f, open("vocabulary_f.pkl", mode="wb"))
print("French vocabulary size: {}".format(len(vocabulary_f)))
print()


def sample_words(vocabulary, n=5):
    """Print a few words from the vocabulary."""
    for _ in range(n):
        token_id = np.random.randint(0, len(vocabulary) - 1)
        print(vocabulary.get_token(token_id))


print("A few English words:")
sample_words(vocabulary_e, n=5)
print()

print("A few French words:")
sample_words(vocabulary_f, n=5)


# ### Mini-batching
# 
# With our vocabulary, we still need a method that converts a whole sentence to a sequence of IDs.
# And, to speed up training, we would like to get a so-called mini-batch at a time: multiple of such sequences together. So our function takes a corpus iterator and a vocabulary, and returns a mini-batch of shape [Batch, Time], where the first dimension indexes the sentences in the batch, and the second the time steps in each sentence. 

# In[19]:

from utils import iterate_minibatches, prepare_data


# Let's try it out!

# In[20]:

src_reader = smart_reader(train_e_path)
trg_reader = smart_reader(train_f_path)
bitext = bitext_reader(src_reader, trg_reader)


for batch_id, batch in enumerate(iterate_minibatches(bitext, batch_size=4)):

    print("This is the batch of data that we will train on, as tokens:")
    pprint(batch)
    print()

    x, y = prepare_data(batch, vocabulary_e, vocabulary_f)

    print("These are our inputs (i.e. words replaced by IDs):")
    print(x)
    print()

    print("These are the outputs (the foreign sentences):")
    print(y)
    print()

    if batch_id == 0:
        break  # stop after the first batch, this is just a demonstration


# Now, notice the following:
# 
# 1. Every English sequence starts with a 4, the ID for < NULL \>.
# 2. The longest sequence in the batch contains no padding symbols. Any sequences shorter, however, will have padding zeros.
# 
# With our input pipeline in place, now let's create a model.

# ### Building our model

# In[21]:

# check neuralibm1.py for the Model code
from neuralibm1context import NeuralIBM1ModelContext
from neuralibm1 import NeuralIBM1Model


# ### Training the model
# 
# Now that we have a model, we need to train it. To do so we define a Trainer class that takes our model as an argument and trains it, keeping track of some important information.
# 
# 

# In[ ]:

# check neuralibm1trainer.py for the Trainer code
from neuralibm1trainer import NeuralIBM1Trainer


# Now we instantiate a model and start training.

# In[ ]:

import time
import pickle
tf.reset_default_graph()

with tf.Session() as sess:
    # some hyper-parameters
    # tweak them as you wish
    batch_size=8  # on CPU, use something much smaller e.g. 1-16
    max_length=30
    lr = 0.001
    lr_decay = 0.0  # set to 0.0 when using Adam optimizer (default)
    emb_dim = 64
    mlp_dim = 128

    # our model
    model = NeuralIBM1ModelContext(
    x_vocabulary=vocabulary_e, y_vocabulary=vocabulary_f, 
    batch_size=batch_size, emb_dim=emb_dim, mlp_dim=mlp_dim, session=sess)

    # our trainer
    trainer = NeuralIBM1Trainer(
    model, train_e_path, train_f_path, 
    dev_e_path, dev_f_path, dev_wa,
    test_e_path, test_f_path, test_wa,
    num_epochs=5, batch_size=batch_size, 
    max_length=max_length, lr=lr, lr_decay=lr_decay, session=sess)

    # now first TF needs to initialize all the variables
    print("Initializing variables..")
    sess.run(tf.global_variables_initializer())

    start = time.time()
    
    # now we can start training!
    print("Training started..")
    trainer.train()
    print("Training took: " + str(time.time() - start))
    pickle.dump([trainer.epoch_loss, trainer.val_loss, trainer.val_aer, trainer.test_aer], open("NeuralIBM1ContextTrainer_5e_10000v", "wb"))
    print("Trainer saved")


# In[ ]:



