###DATA PROCESSING PARAMETERS

LEXICON_TOP_N = 5
LEXICON_TOP_NULL = 2
LEXICON_TOP_N_TO_NULL = 1
DNX_N = 10
DIX_I = 3
percentage_of_one_occurence_words_to_UNK = 0.8
UNK = True
USE_COMPLEX_FEATURES = True
USE_SPARSE_FEATURES = True

SUBSET = 1   # 1-Nuno; 2-Tom; 3-Nedko

###DATA FILE PATHS

LEXICON_FILE_PATH = 'lexicon'

FULL_LEXICON_ZH_EN_DICT_FILE_PATH = 'datamap/%unseen' + str(percentage_of_one_occurence_words_to_UNK) + '_translation_probs_ZH_to_EN.mem'
FULL_LEXICON_EN_ZH_DICT_FILE_PATH = 'datamap/%unseen' + str(percentage_of_one_occurence_words_to_UNK) + '_translation_probs_EN_to_ZH.mem'

CONSTRAINED_LEXICON_ZH_EN_DICT_FILE_PATH = 'datamap/top' + str(LEXICON_TOP_N) + '_topNULL' + str(LEXICON_TOP_NULL) + '_%unseen' + str(percentage_of_one_occurence_words_to_UNK) + '_translation_probs_ZH_to_EN.mem'
CONSTRAINED_LEXICON_EN_ZH_DICT_FILE_PATH = 'datamap/top' + str(LEXICON_TOP_N) + '_topNULL' + str(LEXICON_TOP_NULL) + '_%unseen' + str(percentage_of_one_occurence_words_to_UNK) +  '_translation_probs_EN_to_ZH.mem'
LEXICON_CONVERTION_ENABLED = True

TRAINING_SET_FULL_FILE_PATH = 'datamap/training.zh-en'
ITG_SET_FULL_FILE_PATH = 'datamap/itg.zh-en'


if UNK:
    unk = '_UNK'
else:
    unk = '_noUNK'


TRAINING_SUBSET_1_FILE_PATH = 'datamap/training_subset1_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.zh-en'
TRAINING_SUBSET_2_FILE_PATH = 'datamap/training_subset2_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.zh-en'
TRAINING_SUBSET_3_FILE_PATH = 'datamap/training_subset3_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.zh-en'

# forest and features from the training set
ITG_FILE_PATH = "datamap/training_forests.itgs"
FEATURES_FILE_PATH = "datamap/training_features_complex_sparse.ftrs"

# pickle object position information files
FORESTS_CURSOR_POSITION_FILE = "datamap/training_forests.b"
FEATURES_CURSOR_POSITION_FILE = "datamap/training_features_complex_sparse_byte_position.b"

# forests and features from the validation set
VAL_FOREST_PATH = "datamap/validation"
VAL_FEATURES_PATH = ""

# hypothesis and reference file path for BLEU
VAL_HYPOTHESIS = "datamap/hypotheses"
REF_PATH = "datamap/references_val/reference"

# feature ablation for experiments
ABLATION = False  # pick: False, lexical, "segmentation", "translation", "order"
# segmentation features: they tell us something about the structure of the tree, including insertions and deletions
SEG_LIST = ['type:binary', "length:src", "length:tgt", 'type:terminal', 'deletion:lbs', 'deletion:rbs', 'insertion:lbt',
            'insertion:rbt', 'type:deletion', 'type:-UNK-_del','ibm1:del:logprob', 'type:insertion', 'type:-UNK-_ins']
# they tell us something about lexical translation, including translations to and from unk
TRANS_LIST = [ 'ibm1:ins:logprob', 'type:translation', 'type:-UNK-2-UNK-', 'type:-UNK-2t', 'type:s2-UNK-',
               'ibm1:s2t:logprob', 'ibm1:t2s:logprob']
# they tell us something about the word order. skip-bigrams are seen as source side word order information
ORDER_LIST = ['monotone', 'inverted', "skip-bigram", "skip-joint", "skip-bigram-left", "skip-joint-left",
              "skip-bigram-right", "skip-joint-right"]

# NEDKO stuff
CHINESE_TRAINING_SET_SELECTED_FILE_PATH = 'datamap/references_val/chinese_val.zh'
ENGLISH_TRAINING_SET_SELECTED_FILE_PATH = 'datamap/references_val/reference1'

VALIDATION_ITG_FILE_PATH = 'datamap/itg_validation_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_topNtoNULL'+str(LEXICON_TOP_N_TO_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.itgs'
TEST_ITG_FILE_PATH = 'datamap/itg_test_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_topNtoNULL'+str(LEXICON_TOP_N_TO_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.itgs'

VALIDATION_CORPUS = 'datamap/references_val/corpus.zh-en'

SELECTED_VALIDATION_ITG_FILE_PATH = VALIDATION_ITG_FILE_PATH





