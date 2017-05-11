from .CFG import *
from .FSA import *

def read_lexicon(path):
    """
    Read translation dictionary from a file (one word pair per line) and return a dictionary
    mapping x \in \Sigma to a set of y \in \Delta
    """
    lexicon = defaultdict(set)
    with open(path) as istream:
        for n, line in enumerate(istream):
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) != 2:
                raise ValueError('I expected a word pair in line %d, got %s' % (n, line))
            x, y = words
            lexicon[x].add(y)
    return lexicon


def make_source_side_itg(lexicon, s_str='S', x_str='X') -> CFG:
    """Constructs the source side of an ITG from a dictionary"""
    S = Nonterminal(s_str)
    X = Nonterminal(x_str)

    def iter_rules():
        yield Rule(S, [X])  # Start: S -> X
        yield Rule(X, [X, X])  # Segment: X -> X X
        for x in lexicon.keys():
            yield Rule(X, [Terminal(x)])  # X - > x

    return CFG(iter_rules())

def make_target_side_itg(source_forest: CFG, lexicon: dict) -> CFG:
    """Constructs the target side of an ITG from a source forest and a dictionary"""
    def iter_rules():
        for lhs, rules in source_forest.items():
            for r in rules:
                if r.arity == 1:  # unary rules
                    if r.rhs[0].is_terminal():  # terminal rules
                        x_str = r.rhs[0].root().obj()  # this is the underlying string of a Terminal
                        targets = lexicon.get(x_str, set())
                        if not targets:
                            pass  # TODO: do something with unknown words?
                        else:
                            for y_str in targets:
                                yield Rule(r.lhs, [r.rhs[0].translate(y_str)])  # translation
                    else:
                        yield r  # nonterminal rules
                elif r.arity == 2:
                    yield r  # monotone
                    if r.rhs[0] != r.rhs[1]:  # avoiding some spurious derivations by blocking invertion of identical spans
                        yield Rule(r.lhs, [r.rhs[1], r.rhs[0]])  # inverted
                else:
                    raise ValueError('ITG rules are unary or binary, got %r' % r)
    return CFG(iter_rules())