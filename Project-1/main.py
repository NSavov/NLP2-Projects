# insert this in a Jupyter notebook to update the py files without reimporting:
# %load_ext autoreload
# %autoreload 2

from data import DataProcessing
from ibm1 import IBM1
import globals

#Data Preparation:
data_processor = DataProcessing("training")
paired_train = data_processor.generate_pairs(True)

data_processor = DataProcessing("validation")
paired_val = data_processor.generate_pairs(True)

tr = data_processor.init_translation_dict(paired_train, True)

#Working with IBM 1

trainPairs, valPairs, transProbs = DataProcessing.get_data()
ibm1 = IBM1(transProbs)
transProbs = ibm1.train_ibm_1(trainPairs, '', globals.THRESHOLD, [])
