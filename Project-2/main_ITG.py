from ITG.Earley import *
from ITG.ITG import *
from ITG.Symbol import *

lexicon = defaultdict(set)
lexicon['le'].update(['the', '-EPS-'])  # we will assume that `le` can be deleted
lexicon['-EPS-'].update(['a', 'the'])  # we will assume that `the` and `a` can be inserted
lexicon['e'].add('and')
lexicon['chien'].add('dog')
lexicon['noir'].update(['black', 'noir'])
lexicon['blanc'].add('white')
lexicon['petit'].update(['small', 'little'])
lexicon['petite'].update(['small', 'little'])

src_cfg = make_source_side_itg(lexicon)

src_str = 'petit chien'
src_fsa = make_fsa(src_str)
print(src_fsa)
print(src_fsa)
print(src_cfg)

forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)"))

# This is D(x)
projected_forest = make_target_side_itg(forest, lexicon)

print(projected_forest)
len(projected_forest)

tgt_str = 'little dog'
tgt_fsa = make_fsa(tgt_str)
print(tgt_fsa)

# This is D(x, y)
ref_forest = earley(projected_forest, tgt_fsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x,y)'))
print(ref_forest)


def topological_sort(cfg, start_label='S'):
    # find the initial nonterminal
    for nonterminal in projected_forest.nonterminals:
        if nonterminal.root() == Nonterminal(start_label):
            start = nonterminal
    # find the topologically sorted sequence
    ordered = [start]
    for nonterminal in ordered:
        rules = cfg.get(nonterminal)
        for rule in rules:
            for variable in rule.rhs:
                if not variable.is_terminal() and variable not in ordered:
                    ordered.append(variable)
    return ordered


topological_sort(projected_forest, start_label='S')