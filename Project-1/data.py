import cPickle
import globals

class DataProcessing:
    def __init__(self, directory_to_preprocess):
        self.directory_to_preprocess = directory_to_preprocess
        if (directory_to_preprocess == "training"):
            self.english_file = directory_to_preprocess + "/hansards.36.2.e"
            self.french_file = directory_to_preprocess + "/hansards.36.2.f"
            self.alignments = ""
            print("Preprocessing training set")
        elif (directory_to_preprocess == "validation"):
            self.english_file = directory_to_preprocess + "/dev.e"
            self.french_file = directory_to_preprocess + "/dev.f"
            self.alignments = directory_to_preprocess + "/dev.wa.nonullalign"
            print("Preprocessing validation set")
        else:
            print("ERROR: pick either \"training\" or \"validation\"")

    def read_alignments(self):
        with open(self.alignments) as f:
            alignment_strings = f.read().splitlines()

        alignments = {}
        for alignment_str in alignment_strings:
            alignment_str = alignment_str.split(" ")
            alignment = [ int(alignment_str[1]), int(alignment_str[2]), alignment_str[3] ]

            pair_id = int(alignment_str[0])
            if pair_id in alignments.keys():
                alignments[pair_id].append(alignment)
            else:
                alignments[pair_id] = [alignment]

        return alignments

    def generate_pairs(self, should_dump):
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
            cPickle.dump(paired, open(self.directory_to_preprocess + "_pairs", 'wb'))
        return paired

    def init_translation_dict(self, should_dump):
        return DataProcessing.init_translation_dict(self.paired, should_dump, self.directory_to_preprocess + globals.DICT_FILENAME)

    @staticmethod
    def init_translation_dict(paired, should_dump=False , filename = ''):
        translation = {}
        for i,pair in enumerate(paired):
            for enWord in pair[0]:
                if enWord not in translation:
                    translation[enWord] = {}
                for frWord in pair[1]:
                    translation[enWord][frWord] = 0

        if should_dump:
            cPickle.dump(translation, open(filename, 'wb'))

        return translation

    @staticmethod
    def get_data():
        trainPairs = cPickle.load(open(globals.TRAIN_DATA_FILENAME, 'rb'))
        valPairs = cPickle.load(open(globals.VAL_DATA_FILENAME, 'rb'))
        transProbs = cPickle.load(open(globals.TRAIN_DICT_FILENAME, 'rb'))

        return (trainPairs, valPairs, transProbs)