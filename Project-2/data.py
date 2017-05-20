import pickle
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
import libitg
import globals
import operator
import time


class Data:

    @staticmethod
    def generate_IBM_lexicons(file_path=globals.LEXICON_FILE_PATH, should_dump = True, dump_file_path_zh_en = globals.FULL_LEXICON_ZH_EN_DICT_FILE_PATH, dump_file_path_en_zh = globals.FULL_LEXICON_EN_ZH_DICT_FILE_PATH):
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

        if should_dump:
            pickle.dump(translation_probs_ZH_to_EN, open(dump_file_path_zh_en, 'wb'))
            pickle.dump(translation_probs_EN_to_ZH, open(dump_file_path_en_zh, 'wb'))

        return translation_probs_ZH_to_EN, translation_probs_EN_to_ZH


    @staticmethod
    def constrain_IBM_lexicons(translation_probs_ZH_to_EN, translation_probs_EN_to_ZH, top_n=globals.LEXICON_TOP_N, should_dump = True, dump_file_path_zh_en = globals.FULL_LEXICON_ZH_EN_DICT_FILE_PATH, dump_file_path_en_zh = globals.FULL_LEXICON_EN_ZH_DICT_FILE_PATH):
        top_n_translation_probs_ZH_to_EN = {}
        top_n_translation_probs_EN_to_ZH = {}

        for entry in translation_probs_ZH_to_EN:
            new_entry = dict(sorted( translation_probs_ZH_to_EN[entry].items(), key=lambda element: (element[1], element[0]), reverse=True)[:top_n])
            top_n_translation_probs_ZH_to_EN[entry] = new_entry

        for entry in translation_probs_EN_to_ZH:
            new_entry = dict( sorted(translation_probs_EN_to_ZH[entry].items(), key=lambda element: (element[1], element[0]), reverse=True)[:top_n])
            top_n_translation_probs_EN_to_ZH[entry] = new_entry

        if should_dump:
            pickle.dump(top_n_translation_probs_ZH_to_EN, open(dump_file_path_zh_en, 'wb'))
            pickle.dump(top_n_translation_probs_EN_to_ZH, open(dump_file_path_en_zh, 'wb'))

        return top_n_translation_probs_ZH_to_EN, top_n_translation_probs_EN_to_ZH



    @staticmethod
    def generate_lexicon(file_path = globals.LEXICON_FILE_PATH, top_n=globals.LEXICON_TOP_N, should_dump = True, dump_filename = globals.CONSTRAINED_LEXICON_ZH_EN_DICT_FILE_PATH):
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
            items = list(lexicon[key].items())
            strings = [item[0] for item in items]
            probs = [item[1] for item in items]
            lexicon[key] = [x for (y, x) in sorted(zip(probs, strings), reverse=True)]
        return lexicon





    @staticmethod
    def read_lexicon_dict(lexicon_dict_file_path =globals.CONSTRAINED_LEXICON_ZH_EN_DICT_FILE_PATH, convert=globals.LEXICON_CONVERTION_ENABLED, unk_enabled = globals.UNK):
        lexicon = pickle.load(open(lexicon_dict_file_path, 'rb'))

        if convert:
            lexicon = Data.convert_lexicon(lexicon)

            if unk_enabled == True:
                lexicon['-UNK-'] = ['-UNK-']

            for key in lexicon:
                if key != '<NULL>':
                    lexicon[key] += ['-EPS-']

            lexicon['-EPS-'] = []
            if unk_enabled == True:
                lexicon['-EPS-'] += ['-UNK-']
        # lexicon['-EPS-'] = list(lexicon['<NULL>'])

        return lexicon





    @staticmethod
    def generate_trees(training_file_path=globals.TRAINING_SET_SELECTED_FILE_PATH,
                       I=globals.DIX_I, lexicon_dict_file_path=globals.CONSTRAINED_LEXICON_ZH_EN_DICT_FILE_PATH, should_dump = True,
                       dump_file_path = globals.ITG_SET_SELECTED_FILE_PATH, lexicon = {}):

        start_time = time.clock()

        with open(training_file_path, encoding='utf8') as f:
            paired_sentences = f.read().splitlines()

        training_file = open(training_file_path, 'rb')

        if not bool(lexicon):
            lexicon = pickle.load(open(lexicon_dict_file_path, 'rb'))

        src_cfg = libitg.make_source_side_itg(lexicon)

        number_of_training_sentences = len(paired_sentences)
        empty = 0

        trees = []

        for i, line in enumerate(training_file):


            if i + 1 > 100:
                break

            line = line.decode().strip()
            translation_pair = line.split(' ||| ')
            chinese_sentence = translation_pair[0]
            english_sentence = translation_pair[1]

            top_sentence_word_translations = list(lexicon['<NULL>'])

            for source_word in chinese_sentence.split(' '):
                top_sentence_word_translations += lexicon[source_word][:globals.LEXICON_TOP_N_TO_NULL]

            top_sentence_word_translations = list(set(top_sentence_word_translations))
            if '-EPS-' in top_sentence_word_translations:
                top_sentence_word_translations.remove('-EPS-')
            # print(top5_sentence_word_translations)
            # print(len(top5_sentence_word_translations))
            lexicon['-EPS-'] = top_sentence_word_translations
            # print(lexicon['-EPS-'])

            # generate Dx
            src_fsa = libitg.make_fsa(chinese_sentence)
            _Dx = libitg.earley(src_cfg, src_fsa,
                                start_symbol=Nonterminal('S'),
                                sprime_symbol=Nonterminal("D(x)"))

            # Dx = libitg.make_target_side_itg(_Dx, lexicon)
            eps_count_fsa = libitg.InsertionConstraint(I)


            _Dix = libitg.earley(_Dx,
                                 eps_count_fsa,
                                 start_symbol=Nonterminal('D(x)'),
                                 sprime_symbol=Nonterminal('D_i(x)'),
                                 eps_symbol=None)  # Note I've disabled special treatment of -EPS-
            # we project it just like before
            Dix = libitg.make_target_side_itg(_Dix, lexicon)

            # print(_Dix)
            # print('+++++++++++')
            # print('+++++++++++')
            # print('+++++++++++')
            # print('+++++++++++')
            # print('+++++++++++')
            # print(_Dix)
            # print('-----------')
            # print('-----------')
            # print('-----------')
            # print('-----------')
            # print('-----------')
            # print(Dix)
            # print('***********')
            # print('***********')
            # print('***********')
            # print('***********')
            # print('***********')

            # generate Dixy
            tgt_fsa = libitg.make_fsa(english_sentence)
            Dixy = libitg.earley(Dix, tgt_fsa, start_symbol=Nonterminal("D_i(x)"), sprime_symbol=Nonterminal('D_i(x,y)'))

            if(len(Dixy) == 0):
                empty += 1

            #if (i % 10 == 0 or i + 1 == number_of_training_sentences):
            print('\r' + 'Elapsed time: ' + str('{:0.0f}').format(time.clock() - start_time) + 's. Parsing Forests... ' +
                  str('{:0.5f}').format(100.0*(i+1)/number_of_training_sentences) +
                  '% forests processed so far, out of ' + str(number_of_training_sentences) + '. ' +
                  str('{:0.5f}').format(100.0*(empty)/(i+1)) + '% (' + str(empty) + ') empty forests so far, out of ' +
                  str((i+1)) + '.', end='')

            if(len(Dixy) == 0):
                continue


            # Dxy = libitg.earley(Dx, tgt_fsa,
            #                     start_symbol=Nonterminal("D(x)"),
            #                     sprime_symbol=Nonterminal('D(x,y)'))

            # generate Dnx
            # length_fsa = libitg.LengthConstraint(N, strict=False)
            # Dnx = libitg.earley(Dx, length_fsa,
            #                     start_symbol=Nonterminal("D(x)"),
            #                     sprime_symbol=Nonterminal("D_n(x)"))

            trees.append([i, Dix, Dixy])

        if should_dump:
            pickle.dump(trees, open(dump_file_path, 'wb'))

        return trees

    @staticmethod
    def read_forests(file_path=globals.ITG_SET_SELECTED_FILE_PATH):
        trees =  pickle.load(open(file_path, 'rb'))
        return trees

    @staticmethod
    def filter_forests(forests):
    # removes the indices information from the list of forests
        return [forest[-2:] for forest in forests]

