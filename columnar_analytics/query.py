"""Single-file and multi-file SQL query layer (public orchestration).

Public API:

* :func:`query_file` -- run a ``SELECT ... FROM input [WHERE ...]`` query
  against one columnar file and return a :class:`~columnar_analytics.format.Table`
* :func:`query_files` -- run a statement against a table-name to path
  mapping, optionally chaining several equi-joins
* :func:`explain_file` / :func:`explain_files` -- parse, bind and plan the
  same statements without executing them; only file metadata is read
* :class:`QuerySyntaxError` -- lexical / grammatical errors
* :class:`QueryValidationError` -- unknown columns, wrong table name,
  type-incompatible comparisons, invalid aggregate use

Every entry point shares one preparation pipeline
(:mod:`columnar_analytics.prepare`): statement parsing, source resolution
in FROM/JOIN order, referenced-source metadata, schema binding,
required-column collection and v2 row-group selection all run once and
drive both execution (:mod:`columnar_analytics.executor`) and the explain
plan (:mod:`columnar_analytics.plan`).  This module only orchestrates:
prepare, then either read and run, or render the plan.

The accepted grammar (keywords case-insensitive)::

    query       := SELECT [DISTINCT] select_item (',' select_item)*
                   FROM ident [WHERE expr] [GROUP BY ident (',' ident)*]
                   [HAVING expr]
                   [ORDER BY order_item (',' order_item)*]
                   [LIMIT uint]
    select_item := '*' | ident | agg_name '(' [DISTINCT] ('*' | ident) ')'
                   | scalar_expr AS alias
    order_item  := (ident | agg_name '(' [DISTINCT] ('*' | ident) ')')
                   [ASC | DESC] [NULLS FIRST | NULLS LAST]
    expr       := or_expr
    or_expr     := and_expr (OR and_expr)*
    and_expr    := cmp_expr (AND cmp_expr)*
    cmp_expr    := not_factor (cmp_op not_factor)?
    not_factor  := NOT not_factor | postfix
    postfix     := scalar_expr (IS [NOT] NULL)?
    scalar_expr := term (('+' | '-') term)*
    term        := factor (('*' | '/') factor)*
    factor      := ('+' | '-') factor | atom
    atom        := '(' expr ')' | operand
                | CASE WHEN expr THEN scalar_expr
                  (WHEN expr THEN scalar_expr)*
                  (ELSE scalar_expr)? END
    operand     := ident | literal
    literal     := TRUE | FALSE | [+-]? int64 | [+-]? finite float64
                   | single-quoted utf8

The supported aggregates are ``COUNT(*)``, ``COUNT(ident)`` and
``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` applied to one column.  Each of
``COUNT`` / ``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` also accepts
``DISTINCT`` before its single column argument (``COUNT(DISTINCT ident)``
etc.): rows are first selected by WHERE, then within each group (or the
single global group) NULL arguments are ignored and the remaining values
are deduplicated by the column's type -- float64 ``0.0`` and ``-0.0``
count as the same value -- before the aggregate runs.  ``COUNT(DISTINCT
x)`` returns the number of distinct non-NULL values (0 on empty input,
a non-nullable int64); the other DISTINCT aggregates return NULL when no
distinct non-NULL value remains and otherwise keep the result type,
nullability, comparison rules and overflow / non-finite
:class:`QueryValidationError` behaviour of their plain counterparts.
DISTINCT here only deduplicates the aggregate's argument values; it does
not change ``SELECT DISTINCT`` row deduplication, and the argument stays
a single column reference (no ``*``, expressions, multiple arguments or
nested aggregates).  A misplaced DISTINCT keyword, a missing argument,
``*``, an expression or multiple arguments inside a DISTINCT call raise
:class:`QuerySyntaxError` before any file is opened.  DISTINCT
aggregates may appear wherever plain aggregates may (SELECT, HAVING,
ORDER BY), and textually identical calls share one computation.  Without
GROUP BY the projection may contain aggregate expressions only; with
GROUP BY it may additionally contain the grouped columns.  A bare ``*``
must not be mixed with aggregates or grouping.  Aggregates are not
allowed inside WHERE, may not be nested, and aliases are not supported
for bare columns or aggregate calls.

Scalar arithmetic (parentheses, unary ``+``/``-``, binary ``+`` ``-``
``*`` ``/``; precedence: parentheses, unary, ``*``/``/``, ``+``/``-``)
is available over int64 / float64 columns and numeric literals in SELECT
and WHERE.  A computed SELECT expression must be named with ``AS`` (the
alias follows the usual identifier rules and must be unique among the
output names); bare columns and aggregate calls keep their existing
result names.  int64 ``+``/``-``/``*`` on two int64 operands yields
int64; any float64 operand or a division yields float64, and unary
operators preserve the operand type.  Any NULL operand makes the result
NULL; the output column is nullable iff a participating column is
nullable (pure constants are not).  int64 overflow, division by zero and
non-finite float64 results raise :class:`QueryValidationError`.  A
numeric expression used directly as a boolean condition is rejected with
:class:`QueryValidationError` as well.  Non-aggregate queries execute
WHERE, then compute the sort keys, stably sort, apply LIMIT and only
then evaluate the SELECT expressions, so filtered or truncated rows
never trigger division-by-zero or overflow in the projection; a
DISTINCT query instead evaluates the projection right after WHERE,
deduplicates the result rows, and only then sorts and applies LIMIT.
ORDER BY may name a SELECT alias (an alias shadows an input column of
the same name) and sorts on the expression's result type with the usual
NULL placement and stability rules.  Aggregate queries do not accept
scalar expressions: GROUP BY keys and aggregate arguments stay plain
column references, and mixing a scalar expression into an aggregate
query raises :class:`QueryValidationError`.

``SELECT DISTINCT`` removes duplicate rows from a non-aggregate query:
the projection may be a star, plain columns or ``AS``-aliased scalar
expressions exactly as without DISTINCT, and deduplication compares the
full projected result row column by column after WHERE.  Two NULLs in
the same result column compare equal, NULL never equals a non-NULL
value, bool / utf8 / numeric values keep their existing comparison
semantics, and float64 ``0.0`` and ``-0.0`` compare equal.  Without
ORDER BY the surviving rows keep the order of the first occurrence of
each distinct result row; ORDER BY then sorts the deduplicated rows and
LIMIT keeps the first N.  The projection expressions are evaluated
only for rows that passed WHERE (and, with ORDER BY, before the sort),
so a division by zero, int64 overflow or non-finite float64 result that
actually occurs still raises :class:`QueryValidationError`.  ORDER BY
of a DISTINCT query may name only bare columns or explicit aliases that
are part of the projection; an unprojected or otherwise unknown name
raises :class:`QueryValidationError`, and the sort direction, NULL
placement and stability rules are unchanged.  DISTINCT combined with
GROUP BY, HAVING or any aggregate projection raises
:class:`QueryValidationError`; argument-level ``DISTINCT`` inside an
aggregate call (``COUNT(DISTINCT ...)`` etc.) is a separate feature
described above and stays an aggregate query.  A missing projection, a
repeated DISTINCT or a misplaced DISTINCT keyword raises
:class:`QuerySyntaxError` before any file is opened.

The row-processing stages (WHERE, the ORDER BY sort keys, the SELECT
expressions and the projection feeding SELECT DISTINCT) share one batched
column-vector evaluator: each stage hands the next a set of valid row
indices, and an expression is evaluated one batch of column values at a
time instead of row by row.  The batched semantics are exactly the
per-row semantics described above -- three-valued logic, NULL propagation,
per-row CASE short-circuit, the int64 / float64 promotion and error rules
and the stage order (so filtered or LIMIT-cut rows never evaluate the
expressions of a later stage) are unchanged.  Joins, grouping, aggregation
and HAVING keep their existing result organisation; the data they hand to
the expression stages follows the same batched semantics.

Searched CASE expressions
(``CASE WHEN cond THEN result [WHEN ...] [ELSE result] END``; the simple
``CASE value WHEN ...`` form is not supported and at least one WHEN is
required) are scalar operands and may be nested inside more arithmetic,
comparisons and other CASE expressions.  For each row the WHEN conditions
are tested in written order and only TRUE matches; FALSE and UNKNOWN fall
through, an explicit ELSE supplies the fallback and a missing ELSE yields
NULL.  Only the chosen result is evaluated, so division by zero, int64
overflow or a non-finite float64 inside an unhit branch never raises; a
chosen result that errors still raises :class:`QueryValidationError`.
Every reachable THEN and an explicit ELSE must share one type family:
int64 and float64 may mix and unify to float64, while bool and utf8 must
match each other exactly (any other mix raises
:class:`QueryValidationError`).  The result column is nullable if any
reachable result is nullable or the ELSE is omitted.  WHEN conditions use
the same boolean type check and three-valued logic as WHERE; CASE stays a
scalar expression, so it is rejected in aggregate arguments, GROUP BY and
the projection of an aggregate query just like the other scalar
expressions.

Operator precedence (highest first) is NOT, comparison, AND, OR; IS [NOT]
NULL is a postfix of its scalar operand.  Comparison operands may be
column references, literals or scalar arithmetic expressions.  WHERE
follows SQL three-valued logic: a normal
comparison against NULL yields UNKNOWN, UNKNOWN propagates through the
logical operators, and only TRUE rows are returned.

Discrete set membership is expressed with ``operand [NOT] IN (list)`` in
WHERE and CASE WHEN conditions; in HAVING the operand stays limited to
grouping columns, aggregate calls and literals.  The list is non-empty and
holds bool, int64, float64 or utf8 literals plus a NULL allowed only inside
the list -- no column references, expressions or aggregate calls; missing
parentheses, an empty list, a trailing comma, an illegal item or a NOT/IN
ordering error are :class:`QuerySyntaxError` raised before any file is
opened.  The operand is type-checked against every non-NULL option with
the comparison rules (int64 and float64 may mix; bool and utf8 match only
their own family), so an incompatible list raises
:class:`QueryValidationError`; an all-NULL list is legal.  A NULL operand
is UNKNOWN; otherwise a match on a non-NULL option is TRUE for IN / FALSE
for NOT IN, and with no match the predicate is UNKNOWN when the list
contains NULL and FALSE / TRUE otherwise.  Operand division-by-zero,
int64 overflow and non-finite float64 results raise
:class:`QueryValidationError` just like other scalar expressions.  The
explain tree renders the predicate as ``{"kind": "in", "negated": ...,
"operand": ..., "options": [...]}`` with options in written order.  For
single-source and all-INNER-chain v2 scans a top-level AND ``column IN
(constants)`` leaf joins the statistics pushdown: a group is skipped when
its statistics prove no non-NULL candidate can occur there; NOT IN and
expression operands are never pushed.

After WHERE, rows are grouped in GROUP BY column order (a NULL key forms
its own group); without an explicit ORDER BY groups come out in the order
of their first selected row.  An optional HAVING clause between GROUP BY
and ORDER BY filters the formed groups with the same boolean expression
grammar and three-valued logic as WHERE: its conditions combine grouping
columns, aggregate calls and type-compatible literals with NOT / AND / OR
/ comparisons / IS [NOT] NULL, only groups whose condition is TRUE are
kept, and aggregates named in HAVING need not appear in SELECT.  Without
GROUP BY HAVING filters the single global aggregate row (still produced
when WHERE selected no rows); with GROUP BY an empty selection yields
zero groups.  HAVING may not name ungrouped plain columns, ``*``, CASE,
scalar arithmetic or nested aggregates; it is rejected on a completely
non-aggregate query.  ORDER BY may only name a selected result; ASC is the
default, NULLs sort last regardless of direction unless NULLS FIRST /
NULLS LAST is given, and groups equal on every sort key keep that
first-row order.  LIMIT then keeps the first N groups.  Without GROUP BY
and aggregates an aggregate query over zero selected rows still yields one
output row (COUNT 0, the other aggregates NULL), unless HAVING removes it;
with GROUP BY it yields zero rows.

Multi-file queries (:func:`query_files`) add a deterministic chain of zero
or more equi-joins to the grammar::

    query := SELECT ... FROM from_table
             { join_kind JOIN next_table
               ON (earlier_table.col = next_table.col
                   | next_table.col = earlier_table.col) } ...
    join_kind := INNER | LEFT | RIGHT | FULL OUTER

``sources`` maps table names to file paths; only the tables referenced by
the statement are read.  Each join step introduces exactly one not-yet-used
table (aliases and repeated tables are rejected) and its single ON equality
connects a qualified column of that new table with a qualified column of any
table introduced earlier (FROM or a previous join); the two sides of the
equality are interchangeable.  Compound or non-equality ON conditions,
other join types and ``FULL`` without ``OUTER`` are syntax errors raised
before any file is opened.  In any statement that contains a JOIN every
column reference outside ``COUNT(*)`` must be qualified as ``table.column``
(either part may be a double-quoted identifier).  ``SELECT *`` emits the
referenced schemas in FROM/JOIN order with ``table.column`` names; explicit
projections keep their qualified names and aggregates keep the uppercase
``FUNC(table.column)`` labels.

Each step is evaluated with the materialised intermediate result as its
left input and the newly read table as its right input.  Join keys may
share a type or mix int64 with float64; NULL keys never match.  INNER JOIN
emits every matching combination, LEFT JOIN also emits unmatched left rows
with the right-side values set to NULL (the new table's result columns are
nullable), RIGHT JOIN emits all new-table rows in new-file order (one new
row's matches expand in intermediate/current row order) padding the whole
intermediate side with NULL (every earlier table's result columns become
nullable), and FULL OUTER JOIN emits the LEFT JOIN output and then appends
the unmatched new rows in new-file order (every result column is nullable).
INNER / LEFT / FULL expand in the current intermediate row order, with the
matching new rows in new-file order within one current row; RIGHT expands
in new-file row order with the matches in current row order.  Nullability
is derived per step: once a step makes an earlier table's columns
nullable, later steps keep them nullable.  WHERE / GROUP BY / HAVING /
projection / DISTINCT / stable sorting / LIMIT run only after the whole
chain has been built, with their usual semantics.

The multi-file entry points (:func:`query_files`, :func:`explain_files`)
take an optional ``join_strategy`` of ``"hash"`` or ``"sort_merge"``; an
explicit strategy is applied to every join step.  Both strategies accept
the same statements and return identical column descriptions, values, NULL
placement and row order; they differ only in how matching right rows are
located (a right-key hash index versus sorting both sides on the key and
merging equal-key runs).  With the argument omitted the historical
default join path is used and the explain plan carries no strategy field;
with an explicit strategy every plan Join operator gains a ``strategy``
field of ``HASH`` or ``SORT_MERGE``.  A JOIN-less statement accepts either
strategy but gains no extra operator, and an invalid strategy raises
:class:`ValueError` before any source file is opened.

The algorithm selection is decoupled from everything described here:
:mod:`columnar_analytics.join` owns the strategy registry and validation,
the per-algorithm match lookup, key semantics, match expansion, outer NULL
padding, deterministic row order and post-step nullability.  Each algorithm
only produces a match relation; the join layer assembles the result and the
query layer keeps parsing, qualified-name binding, key/type validation,
grouping, sorting, plan description and export.  Adding an algorithm is a
registry entry there -- no part of the query, explain or export path is
copied or changed.

The explain plan lists the referenced sources and their Scan operators in
FROM/JOIN order with ``required_columns`` attributed to each source, then
one Join operator per step in the same order, recording the step type and
the two normalised qualified keys (the earlier table on the left, the
freshly introduced table on the right).

Row-group-partitioned (v2) files are read selectively: every query decodes
only the column blocks its plan actually references (a join chain
additionally reads each table's ON keys).  Single-file statements
(:func:`query_file` and JOIN-less :func:`query_files`) and multi-file
statements whose whole join chain is INNER push the AND-connected,
type-compatible comparisons and IS [NOT] NULL conditions of WHERE down to
the per-group statistics.  In a join chain only a top-level AND leaf that
references exactly one source (a qualified column of that source compared
to a type-compatible literal, on either side, or that column's IS [NOT]
NULL) is attributed to that source; each source's own leaves are combined
with AND independently.  A chain containing any LEFT / RIGHT / FULL OUTER
step performs no row-group pruning at all (an OUTER step can pad a source
with NULLs, so no group can be proven absent), v1 sources are always read
in full, and in a mixed v1/v2 all-INNER chain only the v2 sources are
pruned.

A row group is skipped only when a source's statistics prove its pushed
leaves cannot all be TRUE at once for any of the group's rows; OR, NOT,
CASE, arithmetic, column-to-column or cross-source comparisons and
undecidable ranges are never pushed, and neither block nor replace the
eligible leaves next to them.  Surviving rows are still filtered row by
row with the full WHERE condition over the complete join result, so NULL
three-valued logic, duplicate key combinations, aggregates, DISTINCT,
ordering, LIMIT and the deterministic unordered row order are unchanged.
When every group is excluded the statement simply sees zero rows, so
empty results, ``COUNT(*)``, other global aggregates and HAVING keep
their usual semantics.  Both format versions return identical column
descriptions, values, row order, sorting, LIMIT, join and export results
for the same data and statement.  Only the blocks of the columns and row
groups actually selected are decompressed or decoded: a block belonging
to an excluded group is never read even when corrupt, while a corrupt
block of a selected group raises
:class:`~columnar_analytics.format.ColumnarFormatError`.

In the explain plan a v2 Scan operator of an eligible statement
(single-source, or an all-INNER chain) carries, right after
``required_columns``, the fields ``row_groups_total``,
``row_groups_selected`` and ``pushed_condition`` (the leaves attributed
to that source combined, in SQL appearance order, as one condition tree
with qualified column names, or null when no leaf was pushed; with no
pushed leaf selected equals total).  v1 Scan operators and the Scan
operators of a chain containing an OUTER step keep their historical
shape; the reported counts always match the groups execution reads.

Internally the query layer is split into single-responsibility stages:
pure SQL parsing (:mod:`columnar_analytics.parser`), source and
table-name resolution (:mod:`columnar_analytics.resolve`), schema
binding (:mod:`columnar_analytics.binder`) over the shared bound
expression IR (:mod:`columnar_analytics.expr`), row-group statistics
pushdown (:mod:`columnar_analytics.pushdown`), the shared preparation
pipeline (:mod:`columnar_analytics.prepare`) that ties parsing, source
selection, referenced-metadata binding, required-column collection and
row-group selection into one result, execution
(:mod:`columnar_analytics.executor`) and explain-plan construction
(:mod:`columnar_analytics.plan`).  This module keeps only the public
orchestration: every entry point -- query, explain and, through them,
export -- first runs that one preparation and then lets the one prepared
result drive the data reads or the plan, so the single-file,
multi-file, explain and export paths can never drift apart.
"""

from __future__ import annotations

from typing import Any

from .errors import QuerySyntaxError, QueryValidationError
from .executor import _execute_prepared, _run_query
from .format import Table
from .parser import _Parser, _tokenize
from .plan import _build_explain
from .prepare import _prepare_multi, _prepare_single

# Compatibility re-export: the export layer resolves the referenced source
# paths through this module (a parse-only step that opens no file), exactly
# as before the preparation split.
from .resolve import _referenced_source_paths

__all__ = [
    "QuerySyntaxError",
    "QueryValidationError",
    "explain_file",
    "explain_files",
    "query_file",
    "query_files",
]


# ---------------------------------------------------------------------------
# Public query entry points
# ---------------------------------------------------------------------------


def query_file(path: Any, sql: str) -> Table:
    """Run ``sql`` against the single columnar file ``path``.

    Returns a :class:`~columnar_analytics.format.Table` with columns in
    projection order.  Rows are filtered by WHERE, then grouped (aggregate
    queries may drop groups via HAVING) or sorted by ORDER BY, then capped
    by LIMIT, then projected; without ORDER BY the file's original row
    order (or the first-selected-row group order) is kept.  The statement
    is parsed before the file is touched, so purely grammatical errors
    surface as :class:`QuerySyntaxError` regardless of whether ``path``
    exists.  Unknown columns, duplicate result columns, the wrong table
    name, type-incompatible predicates, ungrouped columns, illegal
    aggregate arguments, invalid HAVING conditions or int64 SUM overflow
    raise :class:`QueryValidationError`;
    malformed files raise :class:`~columnar_analytics.format.ColumnarFormatError`;
    other I/O failures propagate as :class:`OSError`.
    """
    prepared = _prepare_single(path, sql)
    return _execute_prepared(prepared)


def query_table(table: Table, sql: str) -> Table:
    """Apply ``sql`` (``FROM input``) to an in-memory table."""
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    return _run_query(table, select)


def query_files(sources: Any, sql: str, join_strategy: Any = None) -> Table:
    """Run ``sql`` against the tables named by ``sources``.

    ``sources`` maps table names to columnar file paths; only the tables
    referenced by the statement are read.  The FROM table may be followed by
    a deterministic chain of zero or more equi-joins (``INNER JOIN`` /
    ``LEFT JOIN`` / ``RIGHT JOIN`` / ``FULL OUTER JOIN``); each step joins
    the intermediate result to one freshly introduced table on a single
    ``a.col = b.col`` condition.  See the module docstring for the exact
    grammar and semantics.

    ``join_strategy`` optionally selects the join algorithm applied to
    every step: ``"hash"`` (a right-key hash lookup) or ``"sort_merge"``
    (sort both sides on the join key and merge the runs).  Both strategies
    accept the same statements and return identical columns, values and row
    order; when omitted the implicit default join path is used.  A statement
    without a JOIN accepts either strategy, which then adds no operator and
    changes nothing.  A non-string or otherwise invalid ``join_strategy``
    raises :class:`ValueError` before any file is opened.

    A non-mapping or empty ``sources``, non-string keys or non-path values
    raise :class:`ValueError` before any file is touched.  Lexical and
    grammatical errors (including join-keyword, ON and alias problems)
    raise :class:`QuerySyntaxError` before any file is read.  Unknown or
    duplicate tables, ON conditions that do not connect a new table to an
    earlier one, unknown or unqualified columns, and type-incompatible join
    keys raise :class:`QueryValidationError`; malformed files raise
    :class:`~columnar_analytics.format.ColumnarFormatError`; other I/O
    failures propagate :class:`OSError`.
    """
    prepared = _prepare_multi(sources, sql, join_strategy)
    return _execute_prepared(prepared)


# ---------------------------------------------------------------------------
# EXPLAIN: parse, bind and plan without touching the data section
# ---------------------------------------------------------------------------


def explain_file(path: Any, sql: str) -> dict:
    """Produce the query plan for ``sql`` against one columnar file.

    Like :func:`query_file` for parsing, source selection and binding, but
    only the file metadata is read: the data section is never read,
    decompressed or decoded, so data CRCs and value-level statistics are
    not verified.  The returned value is a JSON-serialisable ordered dict
    with the fixed top-level keys ``sources``, ``operators`` and
    ``output``.

    :class:`QuerySyntaxError` is raised before the file is touched;
    binding problems raise :class:`QueryValidationError`; malformed
    metadata or a declared-size mismatch raise
    :class:`~columnar_analytics.format.ColumnarFormatError`; other I/O
    failures propagate as :class:`OSError`.
    """
    prepared = _prepare_single(path, sql)
    return _build_explain(
        prepared.source_tuples(),
        prepared.schema,
        prepared.select,
        prepared.bound,
        steps=(),
        scan_extras=prepared.scan_extras,
        required_by_source=prepared.required_by_source(),
    )


def explain_files(sources: Any, sql: str, join_strategy: Any = None) -> dict:
    """Produce the query plan for ``sql`` against the mapped tables.

    Like :func:`query_files` for parsing, source resolution and binding,
    but only the metadata of the referenced files is read: unreferenced
    sources are never opened and no data section is ever touched.  The
    returned value is a JSON-serialisable ordered dict with the fixed
    top-level keys ``sources``, ``operators`` and ``output``.

    ``join_strategy`` accepts the same ``"hash"`` / ``"sort_merge"``
    values as :func:`query_files`; an explicit strategy applies to every
    join step and each of the plan's ``Join`` operators carries a
    ``strategy`` field (``HASH`` / ``SORT_MERGE``) matching the strategy
    the query and export entries would use.  Sources and Scans are listed
    in FROM/JOIN order, one Join operator follows per step, and each
    source's ``required_columns`` cover only its own referenced columns.
    In a single-source statement or an all-INNER chain every v2 Scan
    additionally carries, right after ``required_columns``,
    ``row_groups_total``, ``row_groups_selected`` and
    ``pushed_condition`` (the statistics-pushed leaves attributed to that
    source, combined in SQL appearance order with qualified names, or
    null; without a pushed leaf selected equals total); v1 Scans and the
    Scans of a chain containing an OUTER step keep their three-field
    shape, and the counts mirror exactly the groups execution would read.
    A statement without a JOIN accepts either strategy but gains no join
    operator.  An invalid ``join_strategy`` raises :class:`ValueError`
    before any file is touched.

    A non-mapping or empty ``sources``, non-string keys or non-path
    values raise :class:`ValueError` before any file is touched;
    :class:`QuerySyntaxError` is raised before files are read as well.
    Unknown tables or columns and other binding problems raise
    :class:`QueryValidationError`; malformed metadata or a declared-size
    mismatch raise :class:`~columnar_analytics.format.ColumnarFormatError`;
    other I/O failures propagate as :class:`OSError`.
    """
    prepared = _prepare_multi(sources, sql, join_strategy)
    return _build_explain(
        prepared.source_tuples(),
        prepared.schema,
        prepared.select,
        prepared.bound,
        steps=prepared.steps,
        strategy=prepared.strategy,
        scan_extras=prepared.scan_extras,
        required_by_source=prepared.required_by_source(),
    )
