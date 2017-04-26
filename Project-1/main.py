# insert this in a Jupyter notebook to update the py files without reimporting:
# %load_ext autoreload
# %autoreload 2

from data import DataProcessing
from ibm import IBM
import globals

#Data Preparation:
data_processor = DataProcessing("training")
training_pairs = data_processor.generate_pairs(True)

data_processor = DataProcessing("validation")
validation_pairs = data_processor.generate_pairs(True)

data_processor = DataProcessing("test")
paired_val = data_processor.generate_pairs(True)

if (globals.EMPTY_DICT_TYPE == 'training'):
    DataProcessing.init_translation_dict(training_pairs, True, globals.EMPTY_DICT_FILEPATH)
elif (globals.EMPTY_DICT_TYPE == 'validation'):
    DataProcessing.init_translation_dict(validation_pairs, True, globals.EMPTY_DICT_FILEPATH)
elif (globals.EMPTY_DICT_TYPE == 'training_validation'):
    DataProcessing.init_translation_dict(training_pairs + validation_pairs, True, globals.EMPTY_DICT_FILEPATH)

# trainPairs, valPairs, transProbs = DataProcessing.get_data()

# # IBM 1
# ibm = IBM(transProbs, 'ibm1')
# transProbs = ibm.train_ibm(trainPairs, '', globals.THRESHOLD)
#
# # IBM 2
# ibm = IBM(transProbs, 'ibm2')
# transProbs, vogelProbs = ibm.train_ibm(trainPairs, '', globals.THRESHOLD)
#
