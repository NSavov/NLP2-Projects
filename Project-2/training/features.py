from training.model import *

"contains all necessary functionality to construct a feature vector fmap"
"this is an extension of the simple feature framework given in the notebook, found in the model file"


def complex_features(edge: Rule, src_fsa: FSA, eps=Terminal('-EPS-'),
                    sparse_del=False, sparse_ins=False, sparse_trans=False, fmap=False) -> dict:
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
        fmap = simple_features(edge, src_fsa,eps,sparse_del,sparse_ins,sparse_trans)
        
    return fmap