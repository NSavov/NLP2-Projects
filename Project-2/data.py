import pickle
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
import libitg
import globals
import operator

class Data:

    @staticmethod
    def generate_IBM_lexicons(file_path=globals.LEXICON_FILE_PATH, top_n=globals.LEXICON_TOP_N):
        with open(file_path, encoding='utf8') as f:
            dictionary_lines = f.read().splitlines()

        translation_probs_ZH_to_EN = {}
        translation_probs_EN_to_ZH = {}

        for line in dictionary_lines:
            entries = line.split(' ')
            if (entries[0] not in translation_probs_ZH_to_EN):
                translation_probs_ZH_to_EN[entries[0]] = {}
            if (entries[1] not in translation_probs_EN_to_ZH):
                translation_probs_EN_to_ZH[entries[1]] = {}
            if (entries[2] != "NA"):
                translation_probs_ZH_to_EN[entries[0]][entries[1]] = float(entries[2])
            if (entries[3] != "NA"):
                translation_probs_EN_to_ZH[entries[1]][entries[0]] = float(entries[3])

        top_n_translation_probs_ZH_to_EN = {}
        top_n_translation_probs_EN_to_ZH = {}

        for entry in translation_probs_ZH_to_EN:
            new_entry = dict(
                sorted(translation_probs_ZH_to_EN[entry].items(), key=operator.itemgetter(1), reverse=True)[:top_n])
            top_n_translation_probs_ZH_to_EN[entry] = new_entry

        for entry in translation_probs_EN_to_ZH:
            new_entry = dict(
                sorted(translation_probs_EN_to_ZH[entry].items(), key=operator.itemgetter(1), reverse=True)[:top_n])
            top_n_translation_probs_EN_to_ZH[entry] = new_entry


        return top_n_translation_probs_ZH_to_EN, top_n_translation_probs_EN_to_ZH


    @staticmethod
    def generate_lexicon(file_path = globals.LEXICON_FILE_PATH, top_n=globals.LEXICON_TOP_N, should_dump = True, dump_filename = globals.LEXICON_DICT_FILE_PATH):
        # lexicon, top_n_translation_probs_EN_to_ZH = Data.generate_IBM_lexicons(file_path, top_n)
        # for key in lexicon:
        #     lexicon[key] = list(lexicon[key].keys())

        lexicon_file = open(file_path, 'rb')

        lexicon = {}
        probs = {}
        for line in lexicon_file:
            line = line.decode().strip()
            splitted = line.split(' ')
            if splitted[0] not in lexicon:
                lexicon[splitted[0]] = []
                probs[splitted[0]] = []

            if splitted[2] == 'NA':
                continue
            lexicon[splitted[0]].append(splitted[1])
            probs[splitted[0]].append(float(splitted[2]))

        for key in lexicon:
            lexicon[key] = [x for (y, x) in sorted(zip(probs[key], lexicon[key]), reverse=True)]
            lexicon[key] = lexicon[key][0:top_n]

        if should_dump:
            pickle.dump(lexicon, open(dump_filename, 'wb'))

        return lexicon

    @staticmethod
    def convert_lexicon(lexicon):
        for key in lexicon:
            lexicon[key] = list(lexicon[key].keys())
        return lexicon

    @staticmethod
    def read_lexicon_dict(lexicon_dict_file_path =globals.LEXICON_DICT_FILE_PATH, convert=globals.LEXICON_CONVERTION_ENABLED):
        lexicon = pickle.load(open(lexicon_dict_file_path, 'rb'))

        if convert:
            lexicon = Data.convert_lexicon(lexicon)
        return lexicon

    @staticmethod
    def generate_trees(training_file_path=globals.TRAINING_SET_SELECTED_FILE_PATH, N=globals.DNX_N, lexicon_dict_file_path=globals.LEXICON_DICT_FILE_PATH, should_dump = True, dump_file_path = globals.ITG_SET_SELECTED_FILE_PATH, lexicon = {}):
        training_file = open(training_file_path, 'rb')

        if not bool(lexicon):
            lexicon = pickle.load(open(lexicon_dict_file_path, 'rb'))
        print("test1")
        src_cfg = libitg.make_source_side_itg(lexicon)



        i = 0

        trees = []

        for line in training_file:

            if i >= 100:
                break
            i += 1
            line = line.decode().strip()
            translation_pair = line.split(' ||| ')
            chinese_sentence = translation_pair[0]
            english_sentence = translation_pair[1]

            print(chinese_sentence.split())
            print(english_sentence)

            # generate Dx
            src_fsa = libitg.make_fsa(chinese_sentence)
            src_forest = libitg.earley(src_cfg, src_fsa,
                                       start_symbol=Nonterminal('S'),
                                       sprime_symbol=Nonterminal("D(x)"))


            Dx = libitg.make_target_side_itg(src_forest, lexicon)

            # generate Dxy
            tgt_fsa = libitg.make_fsa(english_sentence)
            Dxy = libitg.earley(Dx, tgt_fsa,
                                start_symbol=Nonterminal("D(x)"),
                                sprime_symbol=Nonterminal('D(x,y)'))

            # generate Dnx
            length_fsa = libitg.LengthConstraint(N, strict=False)
            Dnx = libitg.earley(Dx, length_fsa,
                                start_symbol=Nonterminal("D(x)"),
                                sprime_symbol=Nonterminal("D_n(x)"))

            trees.append([Dnx, Dxy])

            # print(Dx)
            #     lst = libitg.language_of_fsa(libitg.forest_to_fsa(Dnx, Nonterminal('D_n(x)')))

        if should_dump:
            pickle.dump(trees, open(dump_file_path, 'wb'))

        return trees

