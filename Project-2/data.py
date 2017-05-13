import pickle
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
import libitg
import globals

class Data:

    @staticmethod
    def generate_lexicon(file_path = globals.LEXICON_FILE_PATH, top_n=globals.LEXICON_TOP_N, should_dump = True, dump_filename = globals.LEXICON_DICT_FILE_PATH):
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
    def read_lexicon_dict(lexicon_dict_file_path =globals.LEXICON_DICT_FILE_PATH):
        lexicon = pickle.load(open(lexicon_dict_file_path, 'rb'))
        return lexicon

    @staticmethod
    def generate_trees(training_file_path=globals.TRAINING_SET_SELECTED_FILE_PATH, N=globals.DNX_N, lexicon_dict_file_path=globals.LEXICON_DICT_FILE_PATH, should_dump = True, dump_file_path = globals.ITG_SET_SELECTED_FILE_PATH):
        training_file = open(training_file_path, 'rb')


        lexicon = pickle.load(open(lexicon_dict_file_path, 'rb'))
        print("test1")
        src_cfg = libitg.make_source_side_itg(lexicon)

        i = 0

        trees = []

        for line in training_file:

            if i >= 1:
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

            trees.append([Dx, Dxy])

            # print(Dx)
            #     lst = libitg.language_of_fsa(libitg.forest_to_fsa(Dnx, Nonterminal('D_n(x)')))

        if should_dump:
            pickle.dump(trees, open(dump_file_path, 'wb'))

        return trees

