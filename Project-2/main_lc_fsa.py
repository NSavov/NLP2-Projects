from .main_ITG import *

def make_length_constraining_fsa(sigma, n):
    """Converts a sentence (string) to an FSA (labels are python str objects)"""
    fsa = FSA()
    fsa.add_state(initial=True)
    for i in range(1, n + 1):
        fsa.add_state()
        fsa.make_final(i)
        #         fsa.add_arc(i-1,i, wildcard)
        for ind, word in enumerate(sigma):
            # create a destination state
            fsa.add_arc(i - 1, i, word)  # label the arc with the current word

    return fsa


def make_multisent_fsa(strings):
    """Converts a list of sentences(strings) to an FSA"""
    fsa = FSA()
    i = 0
    for string in strings:
        fsa.add_state(initial=True)
        for word in string.split():
            fsa.add_state()  # create a destination state
            fsa.add_arc(i, i + 1, word)  # label the arc with the current word
            i += 1
        fsa.make_final(fsa.nb_states() - 1)
    return fsa


def make_multisent_det_fsa(strings):
    """Converts a list of sentences(strings) to an FSA"""
    fsa = FSA()
    i = 0
    fsa.add_state(initial=True)
    for string in strings:
        current_state = 0

        for word in string.split():
            if fsa.destination(current_state, word) == -1:
                fsa.add_state()  # create a destination state
                i += 1
                fsa.add_arc(current_state, i, word)  # label the arc with the current word
                current_state = i
            else:
                current_state = fsa.destination(current_state, word)
        fsa.make_final(current_state)
    return fsa


e_vocabulary = set.union(*list(lexicon.values()))
n = 10
lc = LengthConstraint(n)
lc_fsa = make_length_constraining_fsa(e_vocabulary, n)
ms_fsa = make_multisent_det_fsa(["a little dog"])
print(ms_fsa)
lc_forest = earley(projected_forest, lc, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x,y)'))
ms_forest = earley(projected_forest, ms_fsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x,y)'))
print(lc_forest)