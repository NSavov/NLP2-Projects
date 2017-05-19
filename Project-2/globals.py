###DATA PROCESSING PARAMETERS

LEXICON_TOP_N = 5
LEXICON_TOP_NULL = 2
LEXICON_TOP_N_TO_NULL = 1
DNX_N = 10
DIX_I = 3
percentage_of_one_occurence_words_to_UNK = 0.5
UNK = True

SUBSET = 1   # 1-Nuno; 2-Nedko; 3-Tom





###DATA FILE PATHS

LEXICON_FILE_PATH = 'lexicon'
LEXICON_DICT_FILE_PATH = 'data/top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL) + '_%unseen' + str(percentage_of_one_occurence_words_to_UNK) + '_translation_probs_ZH_to_EN.mem'
LEXICON_CONVERTION_ENABLED = True

TRAINING_SET_FULL_FILE_PATH = 'data/training.zh-en'
ITG_SET_FULL_FILE_PATH = 'data/itg.zh-en'


if UNK:
    unk = '_UNK'
else:
    unk = '_noUNK'


TRAINING_SUBSET_1_FILE_PATH = 'data/training_subset1_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.zh-en'
TRAINING_SUBSET_2_FILE_PATH = 'data/training_subset2_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.zh-en'
TRAINING_SUBSET_3_FILE_PATH = 'data/training_subset3_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+'_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.zh-en'

ITG_SUBSET_1_FILE_PATH = 'data/itg_subset1_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+ '_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.itgs'
ITG_SUBSET_2_FILE_PATH = 'data/itg_subset2_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+ '_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.itgs'
ITG_SUBSET_3_FILE_PATH = 'data/itg_subset3_size'+str(DNX_N)+'_top'+str(LEXICON_TOP_N)+ '_topNULL'+str(LEXICON_TOP_NULL)+'_%unseen'+str(percentage_of_one_occurence_words_to_UNK)+unk+'.itgs'


if SUBSET == 1:
    TRAINING_SET_SELECTED_FILE_PATH = TRAINING_SUBSET_1_FILE_PATH
    ITG_SET_SELECTED_FILE_PATH = ITG_SUBSET_1_FILE_PATH
elif SUBSET == 2:
    TRAINING_SET_SELECTED_FILE_PATH = TRAINING_SUBSET_2_FILE_PATH
    ITG_SET_SELECTED_FILE_PATH = ITG_SUBSET_2_FILE_PATH
else:
    TRAINING_SET_SELECTED_FILE_PATH = TRAINING_SUBSET_3_FILE_PATH
    ITG_SET_SELECTED_FILE_PATH = ITG_SUBSET_3_FILE_PATH
