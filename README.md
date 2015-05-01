# Motivation.

While one might argue that the task of parsing with regard for
operator precedence is solved to death, amount of copy-able Python
code which works on pre-parsed AST (and not on a token stream) seemed
scarce.

## Why not use a real parsing lib that implements the Shunting Yard algo?

First of all, `funcparserlib` is my favourite recursive descent
parsing library so far, and I value its recursive descent property for
the ability to reuse parts of the parser or call them directly.
Operator precedence parsing is the only thing it currently lacks.

Also, sometimes you might want to define your own operators with their
own precedence, e. g. Haskell and camlp4.  This requires you to define
a separate step after parsing just to settle down precedence, which
has to be extracted from the already parsed AST.

## Why not submit a patch to funcparserlib?

I believe that AST rewriting is outside of the scope of the compact
and concise `funcparserlib` library.

## Why not make a package?

The code is too small to deserve being a package.  Plus, it is
designed to be copypasteable.

# Explanation of the algorithm.

Copypasted from the source.

Explanation of the algorithm.

NOTE on using this implementation with your grammar: it assumes that
your grammar is right-recursive.  That is, `1 + 1 + 1` is parsed
exactly as `1 + (1 + 1)`.

Each step is basically checking the next operator in line to see if
its precedence is less than, equal to, or greater than the
precedence of the operator on top of the stack.

If it is greater, put the left argument of the next operator and the
operator itself onto the stack.

If it is the same, check if the stack-top operator is
left-associative.  If so, unroll the AST (more on unrolling below).

If it is less, unroll.

Therefore, the invariant on the stack is that the precedence of its
operators is in increasing order, or non-strictly increasing if they
are right-associative.

Unrolling means popping operators and arguments from their stacks
reconstructing the AST back until the precedence of the top-stack
operator becomes low enough (lower than the next operator in AST).
The result of unrolling is then pushed onto the argument stack.

# Testing.

Tests are embedded within the module.

There is a test which compares this implementation to the Python's own
one by `eval`'ing the expressions.  1000 random cases are generated
for each run.
