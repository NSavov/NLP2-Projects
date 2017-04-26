import cPickle
import globals
import aer

class DataProcessing:
    def __init__(self, dataset_type):
        #initialize the object with the files of the selected dataset
        #dataset_type should be "training",  "test" or "validation"
        self.dataset_type = dataset_type

        if (dataset_type == "training"):
            self.english_file = globals.TRAINING_DIRECTORY + "/" + globals.TRAINING_ENGLISH_FILENAME
            self.french_file = globals.TRAINING_DIRECTORY + "/" + globals.TRAINING_FRENCH_FILENAME
            self.alignments = ""
            print("Preprocessing training set")
        elif (dataset_type == "validation"):
            self.english_file = globals.VALIDATION_DIRECTORY + "/" + globals.VALIDATION_ENGLISH_FILENAME
            self.french_file = globals.VALIDATION_DIRECTORY + "/" + globals.VALIDATION_FRENCH_FILENAME
            self.alignments = globals.VALIDATION_DIRECTORY + "/" + globals.VALIDATION_ALIGNMENTS_FILENAME
            print("Preprocessing validation set")
        elif (dataset_type == "test"):
            self.english_file = globals.TEST_DIRECTORY + "/" + globals.TEST_ENGLISH_FILENAME
            self.french_file = globals.TEST_DIRECTORY + "/" + globals.TEST_FRENCH_FILENAME
            self.alignments = globals.TEST_DIRECTORY + "/" + globals.TEST_ALIGNMENTS_FILENAME
            print("Preprocessing test set")
        else:
            print("ERROR: pick either \"training\",  \"test\" or \"validation\"")

    # def read_alignments(self):
    #
    #     with open(self.alignments) as f:
    #         alignment_strings = f.read().splitlines()
    #
    #     alignments = {}
    #     for alignment_str in alignment_strings:
    #         alignment_str = alignment_str.split(" ")
    #         alignment = [ int(alignment_str[1]), int(alignment_str[2]), alignment_str[3] ]
    #
    #         pair_id = int(alignment_str[0])
    #         if pair_id in alignments.keys():
    #             alignments[pair_id].append(alignment)
    #         else:
    #             alignments[pair_id] = [alignment]
    #
    #     return alignments

    def generate_pairs(self, should_dump):
        # returns list of of french-enghlish sentences pairs in the dataset

        with open(self.english_file) as f:
            sentences_english = f.read().splitlines()
        with open(self.french_file) as f:
            sentences_french = f.read().splitlines()

        paired = []
        for i, sentence_english in enumerate(sentences_english):
            paired.append([("null " + sentence_english.decode('utf-8')).split(" ")[0:-1],
                           sentences_french[i].decode('utf-8').split(" ")[0:-1]])

        self.paired = paired

        if should_dump:
            cPickle.dump(paired, open(str(self.dataset_type + "_pairs"), 'wb'))
        return paired

    def init_local_translation_dict(self, should_dump):
        # returns an empty translation dictionary corresponding to the local pairs
        return DataProcessing.init_translation_dict(self.paired, globals.EMPTY_DICT_FILEPATH, should_dump)

    @staticmethod
    def init_translation_dict(pairs, should_dump=False, filename = ""):
        #returns an empty translation dictionary corresponding to the pairs passed
        translation = {}
        for i,pair in enumerate(pairs):
            for enWord in pair[0]:
                if enWord not in translation:
                    translation[enWord] = {}
                for frWord in pair[1]:
                    translation[enWord][frWord] = 0

        if should_dump:
            cPickle.dump(translation, open(str(filename), 'wb'))

        return translation

    @staticmethod
    def get_data():
        #reads the training, validation pairs and the empty translation probabilities dictionary
        #from files and returns it
        trainPairs = cPickle.load(open(str(globals.TRAIN_PAIRS_FILEPATH), 'rb'))
        valPairs = cPickle.load(open(str(globals.VALIDATION_PAIRS_FILEPATH), 'rb'))
        testPairs = cPickle.load(open(str(globals.TEST_PAIRS_FILEPATH), 'rb'))
        transProbs = cPickle.load(open(str(globals.EMPTY_DICT_FILEPATH), 'rb'))

        return (trainPairs, valPairs, testPairs, transProbs)

    # @staticmethod
    # def get_validation_alignments(path):
    #     validation_alignments = aer.read_naacl_alignments(path)
    #     return validation_alignments