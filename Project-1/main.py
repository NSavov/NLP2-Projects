# insert this in a Jupyter notebook to update the py files without reimporting:
# %load_ext autoreload
# %autoreload 2

from data import DataProcessing
from ibm import IBM
import globals

#Data Preparation:
data_processor = DataProcessing("training")
paired_train = data_processor.generate_pairs(True)
tr = data_processor.init_translation_dict(True)

data_processor = DataProcessing("validation")
paired_val = data_processor.generate_pairs(True)



trainPairs, valPairs, transProbs = DataProcessing.get_data()

# IBM 1
ibm = IBM(transProbs, 'ibm1')
transProbs = ibm.train_ibm(trainPairs, '', globals.THRESHOLD)

# IBM 2
ibm = IBM(transProbs, 'ibm2')
transProbs, vogelProbs = ibm.train_ibm(trainPairs, '', globals.THRESHOLD)

