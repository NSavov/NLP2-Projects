from ibm import IBM
import globals
import cPickle
from data import DataProcessing

def test(model, trans_filepath, unseen_filepath, vogel_filepath, output_dir):
    data_processor = DataProcessing("test")
    test_pairs = data_processor.generate_pairs(True)

    trans_probs = cPickle.load(open(trans_filepath, 'r'))

    unseen_probs = dict
    vogel_probs = dict

    if model == IBM.IBM1B:
        unseen_probs = cPickle.load(open(unseen_filepath, 'r'))

    if model == IBM.IBM2:
        vogel_probs = cPickle.load(open(vogel_filepath, 'r'))


    alignments = IBM.get_alignments(model, test_pairs, trans_probs, unseenProbs=unseen_probs, vogelProbs=vogel_probs)
    test_alignments = DataProcessing.get_validation_alignments(output_dir  + globals.TEST_ALIGNMENTS_FILENAME)

    aer = IBM.get_AER(alignments, test_alignments)
    DataProcessing.save_as_naacl(alignments, model)

    return aer