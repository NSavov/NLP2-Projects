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

EMPTY_DICT_TYPE = 'training_validation' # Available options:
                    # training
                    # validation
                    # training_validation


TRAIN_DATA_FILEPATH = "training_pairs"
VAL_DATA_FILEPATH = "validation_pairs"
EMPTY_DICT_FILEPATH = EMPTY_DICT_TYPE + '_empty_dictionary'


### MODEL PARAMETERS ###

EPOCHS = 5
THRESHOLD = 1
