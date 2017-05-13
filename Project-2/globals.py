###DATA PROCESSING PARAMETERS

LEXICON_TOP_N = 5
DNX_N = 5

###DATA FILE PATHS

LEXICON_FILE_PATH = 'lexicon'
LEXICON_DICT_FILE_PATH = 'data/top5_translation_probs_ZH_to_EN.mem'
LEXICON_CONVERTION_ENABLED = True

LEXICON_DICT_IBM_FILE_PATH = 'data/lexicon_ibm_dict'

TRAINING_SET_FULL_FILE_PATH = 'data/training.zh-en'
TRAINING_SUBSET_1_FILE_PATH = 'data/training_subset1_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'.zh-en'
TRAINING_SUBSET_2_FILE_PATH = 'data/training_subset2_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'.zh-en'
TRAINING_SUBSET_3_FILE_PATH = 'data/training_subset3_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'.zh-en'


TRAINING_SET_SELECTED_FILE_PATH = TRAINING_SUBSET_1_FILE_PATH

ITG_SET_FULL_FILE_PATH = 'data/itg.zh-en'
ITG_SUBSET_1_FILE_PATH = 'data/itg_subset1_size'+str(LEXICON_TOP_N)+'.zh-en'
ITG_SUBSET_2_FILE_PATH = 'data/itg_subset2_size'+str(LEXICON_TOP_N)+'.zh-en'
ITG_SUBSET_3_FILE_PATH = 'data/itg_subset3_size'+str(LEXICON_TOP_N)+'.zh-en'

ITG_SET_SELECTED_FILE_PATH = ITG_SET_FULL_FILE_PATH