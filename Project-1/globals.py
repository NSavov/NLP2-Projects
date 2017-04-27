# Constants
### DATASET FILES ###

#Data paths to read:
TRAINING_DIRECTORY = 'training'
TRAINING_ENGLISH_FILENAME = 'hansards.36.2.e'
TRAINING_FRENCH_FILENAME = 'hansards.36.2.f'

VALIDATION_DIRECTORY = 'validation'
VALIDATION_ENGLISH_FILENAME = 'dev.e'
VALIDATION_FRENCH_FILENAME = 'dev.f'
VALIDATION_ALIGNMENTS_FILENAME = 'dev.wa.nonullalign'

TEST_DIRECTORY = 'testing'
TEST_ENGLISH_FILENAME = 'test.e'
TEST_FRENCH_FILENAME = 'test.f'
TEST_ALIGNMENTS_FILENAME = 'test.wa.nonullalign'

#Data paths to write to and load from the processed data:

EMPTY_DICT_TYPE = "training_validation" # Available options:
                    # training
                    # validation
                    # training_validation


TRAIN_PAIRS_FILEPATH = "training_pairs"
VALIDATION_PAIRS_FILEPATH = "validation_pairs"
TEST_PAIRS_FILEPATH = "validation_pairs"
EMPTY_DICT_FILEPATH = EMPTY_DICT_TYPE + "_empty_dictionary"

TEST_ALIGNMENTS_OUTPUT_IBM1 = "ibm1.mle.naacl"
TEST_ALIGNMENTS_OUTPUT_IBM1B = "ibm1.vb.naacl"
TEST_ALIGNMENTS_OUTPUT_IBM2 = "ibm2.mle.naacl"
### MODEL PARAMETERS ###

EPOCHS = 1
THRESHOLD = 10
