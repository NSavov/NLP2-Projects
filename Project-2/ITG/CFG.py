from .Symbol import *
from collections import defaultdict


class Rule(object):
    """
    A rule is a container for a LHS symbol and a sequence of RHS symbols.
    """

    def __init__(self, lhs: Symbol, rhs: list):
        """
        A rule takes a LHS symbol and a list/tuple of RHS symbols
        """
        assert isinstance(lhs, Symbol), 'LHS must be an instance of Symbol'
        assert len(rhs) > 0, 'If you want an empty RHS, use an epsilon Terminal'
        assert all(isinstance(s, Symbol) for s in rhs), 'RHS must be a sequence of Symbol objects'
        self._lhs = lhs
        self._rhs = tuple(rhs)

    def __eq__(self, other):
        return type(self) == type(other) and self._lhs == other._lhs and self._rhs == other._rhs

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self._lhs, self._rhs))

    def __str__(self):
        return '%s ||| %s' % (self._lhs, ' '.join(str(s) for s in self._rhs))

    def __repr__(self):
        return 'Rule(%r, %r)' % (self._lhs, self._rhs)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def arity(self):
        return len(self._rhs)




class CFG:
    """
    A CFG is nothing but a container for rules.
    We group rules by LHS symbol and keep a set of terminals and nonterminals.
    """

    def __init__(self, rules=[]):
        self._rules = []
        self._rules_by_lhs = defaultdict(list)
        self._terminals = set()
        self._nonterminals = set()
        # organises rules
        for rule in rules:
            self._rules.append(rule)
            self._rules_by_lhs[rule.lhs].append(rule)
            self._nonterminals.add(rule.lhs)
            for s in rule.rhs:
                if s.is_terminal():
                    self._terminals.add(s)
                else:
                    self._nonterminals.add(s)

    @property
    def nonterminals(self):
        return self._nonterminals

    @property
    def terminals(self):
        return self._terminals

    def __len__(self):
        return len(self._rules)

    def __getitem__(self, lhs):
        return self._rules_by_lhs.get(lhs, frozenset())

    def get(self, lhs, default=frozenset()):
        """rules whose LHS is the given symbol"""
        return self._rules_by_lhs.get(lhs, default)

    def can_rewrite(self, lhs):
        """Whether a given nonterminal can be rewritten.

        This may differ from ``self.is_nonterminal(symbol)`` which returns whether a symbol belongs
        to the set of nonterminals of the grammar.
        """
        return len(self[lhs]) > 0

    def __iter__(self):
        """iterator over rules (in arbitrary order)"""
        return iter(self._rules)

    def items(self):
        """iterator over pairs of the kind (LHS, rules rewriting LHS)"""
        return self._rules_by_lhs.items()

    def __str__(self):
        lines = []
        for lhs, rules in self.items():
            for rule in rules:
                lines.append(str(rule))
        return '\n'.join(lines)