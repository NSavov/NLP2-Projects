###DATA PROCESSING PARAMETERS

LEXICON_TOP_N = 5
LEXICON_TOP_NULL = 2
LEXICON_TOP_N_TO_NULL = 1
DNX_N = 10
DIX_I = 3
percentage_of_one_occurence_words_to_UNK = 0.8
UNK = True
USE_COMPLEX_FEATURES = True

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

ITG_SUBSET_1_FILE_PATH = 'datamap/itg_subset1_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_topNtoNULL'+str(LEXICON_TOP_N_TO_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.itgs'
ITG_SUBSET_2_FILE_PATH = 'datamap/itg_subset2_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_topNtoNULL'+str(LEXICON_TOP_N_TO_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.itgs'
ITG_SUBSET_3_FILE_PATH = 'datamap/itg_subset3_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_topNtoNULL'+str(LEXICON_TOP_N_TO_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.itgs'

FEATURES_FILE_PATH = 'datamap/features_subset'+str(SUBSET)+'_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_topNtoNULL'+str(LEXICON_TOP_N_TO_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.ftrs'


if SUBSET == 1:
    TRAINING_SET_SELECTED_FILE_PATH = TRAINING_SUBSET_1_FILE_PATH
    ITG_SET_SELECTED_FILE_PATH = ITG_SUBSET_1_FILE_PATH
elif SUBSET == 2:
    TRAINING_SET_SELECTED_FILE_PATH = TRAINING_SUBSET_2_FILE_PATH
    ITG_SET_SELECTED_FILE_PATH = ITG_SUBSET_2_FILE_PATH
else:
    TRAINING_SET_SELECTED_FILE_PATH = TRAINING_SUBSET_3_FILE_PATH
    ITG_SET_SELECTED_FILE_PATH = ITG_SUBSET_3_FILE_PATH
