# -*- coding: utf-8 -*-

#
# This is a small quaint interpreter made using a nice recursive
# descent parsing library called funcparserlib, purportedly resembling
# Haskell's parser combinator libraries such as Parsec.  One thing
# this library lacks, however, is a builtin solution to the common
# problem of rearranging infix subexpressions with accordance to their
# operators' precedence and associativity.  This module implements one
# possible solution for this problem.
#
# Such functionality is also useful if your language allows for
# operators with custom user-defined precedence, like Haskell does.
#
# This program implements a version of the famous shunting yard
# algorithm that receives an already constructed and transformed
# abstract syntax tree (AST) as input and returns another AST as
# output.
#
# The algorithm is explained in some detail in a block comment above
# the «rewrite precedence» part.
#
# Note that it does not handle prefix operators, postfix operators,
# function calls etc.  Adding it would be trivial.  I suggest
# ensuring that function calls have the top priority and prefix or
# postfix operators are next to top using grammar rules.
#
# To run the built-in tests, execute:
#
# py.test shunting.py
#
# py.test can be found in python-pytest/python3-pytest packages on
# Debian and Ubuntu, or installed with `pip install pytest`.
#

from __future__ import print_function
import operator
import sys
from pprint import pprint, pformat

import py.test

import funcparserlib
from funcparserlib.parser import (some, a, many, skip, finished,
                                  pure, maybe, with_forward_decls)
from funcparserlib.util import pretty_tree


#
# Lexer.
#

EOF = "\0"  # to simplify matters.


@py.test.mark.parametrize("given, expected", [
    ("1 + 2 * (30 + 4)",
     ["1", "+", "2", "*", "(", "30", "+", "4", ")", EOF]),
    ("1", ["1", EOF]),
    ("1+", ["1", "+", EOF]),
])
def test_tokenizer(given, expected):
    assert list(tokenize(given)) == expected


def tokenize(s):
    idx, num = -1, ""
    for c in s:
        if c.isdigit():
            num += c
            continue
        if len(num) > 0:
            yield num
            num = ""
        if c.isspace():
            continue
        elif c in "()+-*^&|":
            yield c
    if len(num) > 0:
        yield num
    yield EOF


#
# AST.
#

class ASTNode(object):
    def __str__(self):
        return str(self._value)


class Function(ASTNode):
    def __init__(self, name, args):
        assert (type(args) == list and
                all(isinstance(x, ASTNode) for x in args))
        self._args = args
        self._name = name

    def name(self):
        return self._name

    def args(self):
        return self._args

    def __eq__(self, other):
        return (type(other) == Function and
                self._name == other._name and
                self._args == other._args)

    def __str__(self):
        return "(%s %s %s)" % (str(self._args[0]),
                               self._name,
                               str(self._args[1]))

    def __call__(self):
        funcs = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "^": operator.pow,
            "|": operator.or_,
            "&": operator.and_,
        }
        return funcs[self._name](*[arg() for arg in self._args])


class Value(ASTNode):
    def value(self):
        return self._value

    def __eq__(self, other):
        return (type(other) == type(self) and
                self._value == other._value)

    def __call__(self):
        return self._value


class Number(Value):
    def __init__(self, value):
        self._value = int(value)

    def __str__(self):
        return str(self._value)


#
# Parsing utils.
#

# Will be used to tag different «tails» after we eliminate left
# recursion.  When Python-3, use the standard enum module.
NumberMark, ParenMark = 0, 1


def make_number(arg):
    return Number(arg)


def make_expr(args):
    left, mark, rest = args[:-2], args[-2], args[-1]
    if mark == NumberMark:
        item = left[0]
    elif mark == ParenMark:
        item = left[1]
    if rest is None:
        return item
    larg, fun, rarg = item, rest[0], rest[1]
    return Function(fun, [larg, rarg])


#
# Parser.
#

lparen = some(lambda tok: tok == "(")
rparen = some(lambda tok: tok == ")")
op = some(lambda tok: tok in "+-*^&|")
eof = some(lambda tok: tok == EOF)
number = some(lambda tok: tok.isdigit()) >> make_number
paren_expr = with_forward_decls(
    lambda: lparen + expr + rparen
)

# *Mark here are not really required, but if you are going to do
# anything complex that requires that you discern between different
# parsing paths, marks are often give you least hassle.
expr = with_forward_decls(
    lambda:
    (number + pure(NumberMark) + expr_rest |
     paren_expr + pure(ParenMark) + expr_rest) >> make_expr)

# This one allows us to add more complex expressions like function
# application and ternary operators to the above definition with ease.
# Otherwise terms such as `apply = expr lparen many(expr) rpanen`
# would be impossible to add, always leading to infinite left recursion.
expr_rest = maybe(op + expr)

toplev = expr + skip(eof)


@py.test.mark.parametrize("given, parser, expected", [
    ("1", number, Number("1")),
    ("+", op, "+"),
    ("-", op, "-"),
    ("*", op, "*"),
    ("^", op, "^"),
])
def test_parse_primitives(given, parser, expected):
    data = parser.parse(list(tokenize(given))[:-1])
    assert data == expected


@py.test.mark.parametrize("given, expected", [
    ("1", Number("1")),
    ("1 + 2", Function("+", [Number("1"), Number("2")])),
    ("1 * 2 + 3",
     Function("*", [Number("1"), Function("+", [Number("2"), Number(3)])])),
    ("(1 + 2)",
     Function("+", [
         Number("1"),
         Number("2")
     ])),
    ("(1 + 2) * (3 + 4)",
     Function("*", [
        Function("+", [
            Number("1"),
            Number("2")]),
        Function("+", [
            Number("3"),
            Number("4")
        ])
     ])),
    ("1 * 2 + 3 * 4",
     Function("*", [
         Number(1),
         Function("+", [
             Number(2),
             Function("*", [
                 Number(3),
                 Number(4)
             ])
         ])
     ])),
])
def test_parse_toplev(given, expected):
    data = toplev.parse(list(tokenize(given)))
    assert data == expected


#
# Rewriting precedence.
#

LeftAssoc, RightAssoc = 1, 2
PRECEDENCE_DATA = {
    "^": (7, RightAssoc),
    "*": (6, LeftAssoc),
    "+": (5, LeftAssoc),
    "-": (5, LeftAssoc),
    "&": (4, RightAssoc),
    "|": (3, RightAssoc),
}


def get_precedence(fun):
    if type(fun) != Function:
        return PRECEDENCE_DATA[fun]
    return PRECEDENCE_DATA[fun.name()]


#
# Explanation of the algorithm.
#
# NOTE on using this implementation with your grammar: it assumes that
# your grammar is right-recursive.  That is, 1 + 1 + 1 is parsed
# exactly as 1 + (1 + 1).
#
# Each step is basically checking the next operator in line to see if
# its precedence is less than, equal to, or greater than the
# precedence of the operator on top of the stack.
#
# If it is greater, put the left argument of the next operator and the
# operator itself onto the stack.
#
# If it is the same, check if the stack-top operator is
# left-associative.  If so, unroll the AST (more on unrolling below).
#
# If it is less, unroll.
#
# Therefore, the invariant on the stack is that the precedence of its
# operators is in increasing order, or non-strictly increasing if they
# are right-associative.
#
# Unrolling means popping operators and arguments from their stacks
# reconstructing the AST back until the precedence of the top-stack
# operator becomes low enough (lower than the next operator in AST).
# The result of unrolling is then pushed onto the argument stack.
#

def rewrite_precedence(ast):
    """
    Uses a version of the shunting yard algorithm to rearrange AST.
    Looks for Function nodes, and applies the algorithm to unbroken
    chains of other Function calls within their arguments.

    Since our grammar is rigth-recursive, we leave the left op arg
    as is, and descend on the right branch.
    """
    if isinstance(ast, Value):
        return ast
    assert type(ast) == Function
    op_stack, arg_stack = [], []
    rewritten = rp_worker(ast, op_stack, arg_stack)
    assert (op_stack, arg_stack == [], [])
    return rewritten


def unroll_ast(op_stack, arg_stack, minprec=0):
    assert (len(op_stack) >= 1 and
            len(arg_stack) >= 2 and
            len(arg_stack) == len(op_stack) + 1)

    op = op_stack.pop()
    rarg = arg_stack.pop()
    larg = arg_stack.pop()
    curfun = Function(op, [larg, rarg])
    startprec = get_precedence(op)

    def need_unroll_more():
        tprec, tass = get_precedence(op_stack[-1])
        if tprec > minprec:
            return True
        if tass == LeftAssoc and tprec == minprec:
            return True
        return False

    while len(op_stack) > 0 and need_unroll_more():
        op = op_stack.pop()
        arg = arg_stack.pop()
        curfun = Function(op, [arg, curfun])
    if minprec == 0:
        assert op_stack == [] and arg_stack == []
    return curfun


# Here all the magic happens.
def rp_worker(ast, op_stack, arg_stack):
    if type(ast) != Function:
        arg_stack.append(rewrite_precedence(ast))
        return unroll_ast(op_stack, arg_stack)

    check_op_stack_invariant(op_stack)

    left, right = ast.args()

    if len(op_stack) == 0:
        op_stack.append(ast.name())
        arg_stack.append(rewrite_precedence(left))
        return rp_worker(right, op_stack, arg_stack)

    assert (len(op_stack) >= 1 and
            len(arg_stack) >= 1 and
            len(arg_stack) == len(op_stack))

    lprec, lass = get_precedence(op_stack[-1])
    cprec, cass = get_precedence(ast)

    if lprec > cprec or lass == LeftAssoc and lprec == cprec:
        # Time to unroll our stacks.  Note the minprec limit: we only
        # unroll operations with precedence above or equal to that of
        # the next-in-line operator.
        arg_stack.append(rewrite_precedence(left))
        newleft = unroll_ast(op_stack, arg_stack, minprec=cprec)
        arg_stack.append(newleft)
        op_stack.append(ast.name())
    else:
        # Nothing happens.
        op_stack.append(ast.name())
        arg_stack.append(rewrite_precedence(left))
    return rp_worker(right, op_stack, arg_stack)


def check_op_stack_invariant(op_stack):
    for i, op in enumerate(op_stack[:-1]):
        cprec, cass = get_precedence(op)
        nprec, nass = get_precedence(op_stack[i + 1])
        if nass == LeftAssoc:
            assert cprec < nprec
        else:
            assert cprec <= nprec


@py.test.mark.parametrize("ops, args, expected", [
    (["-", "*"], [Number(x) for x in range(1, 4)],
     Function("-", [
         Number(1),
         Function("*", [
             Number(2),
             Number(3)
         ])
     ])),

    (["-", "-"], [Number(x) for x in range(1, 4)],
     Function("-", [
         Number(1),
         Function("-", [
             Number(2),
             Number(3)
         ])
     ])),

    (["-", "+"], [Number(x) for x in range(1, 4)],
     Function("-", [
         Number(1),
         Function("+", [
             Number(2),
             Number(3),
         ])
     ])),
])
def test_unroll_ast(ops, args, expected):
    res = unroll_ast(ops[:], args[:])
    print("unrolled", res)
    print("expected", expected)
    assert unroll_ast(ops, args) == expected


@py.test.mark.parametrize("given, expected", [
    ("1", "1"),
    ("1 + 2", "1 + 2"),
    ("1 + 2 + 3", "(1 + 2) + 3"),
    ("1 + 2 + 3 + 4", "((1 + 2) + 3) + 4"),
    ("1 * 2 + 3", "(1 * 2) + 3"),
    ("1 + 2 * 3", "1 + (2 * 3)"),
    ("1 * 2 + 3 * 4", "(1 * 2) + (3 * 4)"),
    ("1 + 2 * 3 + 4", "(1 + (2 * 3)) + 4"),

    ("1 ^ 2 ^ 3", "1 ^ (2 ^ 3)"),
    ("1 ^ 2 ^ 3 ^ 4", "1 ^ (2 ^ (3 ^ 4))"),

    ("1 - 2 - 3", "(1 - 2) - 3"),
    ("1 - 2 + 3", "(1 - 2) + 3"),
    ("1 + 2 - 3", "(1 + 2) - 3"),
    ("1 + 2 - 3 - 4 + 5", "(((1 + 2) - 3) - 4) + 5"),
    ("1 - 2 - 3 - 4", "((1 - 2) - 3) - 4"),
    ("1 + 2 - 3 + 4 - 5", "(((1 + 2) - 3) + 4) - 5"),
    ("1 * 2 - 3", "(1 * 2) - 3"),
    ("1 * 2 - 3 - 4", "(((1 * 2) - 3) - 4)"),
    ("1 - 2 * 3 - 4", "(1 - (2 * 3)) - 4"),
    ("1 - 2 - 3 * 4", "(1 - 2) - (3 * 4)"),
    ("1 - 2 * 3 * 4", "1 - ((2 * 3) * 4)"),
    ("1 - 2 - 3 * 4 * 5", "(1 - 2) - ((3 * 4) * 5)"),

    ("(1 - 2) * 3 - 4", "((1 - 2) * 3) - 4"),
    ("1 + 2 - (3 + 4) - 5", "((1 + 2) - (3 + 4)) - 5"),

    ("1 - 2 * 3 ^ 4 * 5 - 6", "(1 - ((2 * (3 ^ 4)) * 5)) - 6"),
])
def test_parse_precedence(given, expected):
    res = rewrite_precedence(toplev.parse(list(tokenize(given))))
    expected = toplev.parse(list(tokenize(expected)))
    print("unrolled", toplev.parse(list(tokenize(given))))
    print("result  ", res)
    print("expected", expected)
    assert res == expected


#
# Testing against Python's own precedence implementation.
#

def gen_expr():
    import random
    tokens = []
    # We leave out 0 because it often gives uninteresting results.
    gen_int = lambda: str(random.randint(1, 20))
    # We omit ^ because ^ is xor in python, and also because
    # expressions such as 7 ^ 7 ^ 7 ^ 7 take forever to compute and a
    # Universe of RAM to store the result.
    gen_op = lambda: random.choice("+-*&|")
    for i in range(10):
        tokens.append(gen_int())
        tokens.append(gen_op())
    tokens.append(gen_int())
    return " ".join(tokens)
expressions = (gen_expr() for _ in range(1000))


@py.test.mark.parametrize("expression", expressions)
def test_generate(expression):
    result = rewrite_precedence(
        toplev.parse(list(tokenize(expression))))
    print(expression, str(result))
    assert result() == eval(expression)
