from datamap import Data

lexicon = Data.read_lexicon_dict()
# print(lexicon['-EPS-'])
trees = Data.generate_trees(lexicon = lexicon)
