import jax
import jax.numpy as jnp
import pytest
from vertex import gp

def test_primitive_set():
    pset = gp.make_simple_primitive_set(input_count=2, max_len=10)
    assert pset.num_terminals == 2
    assert pset.num_ops > 0
    arities = pset.get_arities_array()
    assert len(arities) == pset.total_symbols

def test_evaluation():
    # Expression: x0 + x1
    # Prefix: + x0 x1
    # IDs: x0=0, x1=1
    # Ops start at 2.
    # Let's say:
    # + : 2
    # - : 3
    # * : 4
    # / : 5
    # sin: 6
    # cos: 7

    # We need to verify exact IDs from the set.
    pset = gp.make_simple_primitive_set(input_count=2, max_len=10)
    # "add" is first op => ID 2.

    # Genome: [2, 0, 1, -1, ...]
    genome = jnp.full((10,), -1, dtype=jnp.int32)
    genome = genome.at[0].set(2)
    genome = genome.at[1].set(0)
    genome = genome.at[2].set(1)

    eval_fn = gp.build_evaluator(pset)

    inputs = jnp.array([3.0, 5.0])
    result = eval_fn(genome, inputs)

    assert result == 8.0

def test_subtree_end():
    pset = gp.make_simple_primitive_set(input_count=2, max_len=10)
    arities = pset.get_arities_array()

    # Tree: + x0 x1 -> length 3
    genome = jnp.array([2, 0, 1, -1, -1, -1])
    end = gp.find_subtree_end(genome, 0, arities)
    assert end == 3

    # Tree: + (+ x0 x1) x1 -> length 5
    # [+, +, x0, x1, x1]
    genome2 = jnp.array([2, 2, 0, 1, 1, -1])
    end2 = gp.find_subtree_end(genome2, 0, arities)
    assert end2 == 5

    # Subtree at index 1 (+ x0 x1)
    end_sub = gp.find_subtree_end(genome2, 1, arities)
    assert end_sub == 4 # 1 + 3 = 4

def test_crossover():
    key = jax.random.PRNGKey(0)
    pset = gp.make_simple_primitive_set(input_count=1, max_len=10)
    arities = pset.get_arities_array()

    # P1: + x0 x0 ([2, 0, 0])
    p1 = jnp.full((10,), -1, dtype=jnp.int32)
    p1 = p1.at[:3].set(jnp.array([2, 0, 0]))

    # P2: * x0 x0 ([4, 0, 0]) (assuming * is ID 4)
    p2 = jnp.full((10,), -1, dtype=jnp.int32)
    p2 = p2.at[:3].set(jnp.array([4, 0, 0])) # mul is 3rd op?
    # ops: add, sub, mul. IDs: 1, 2, 3 (if 1 input).
    # input_count=1.
    # 0: x0
    # 1: add
    # 2: sub
    # 3: mul

    p1 = p1.at[:3].set(jnp.array([1, 0, 0]))
    p2 = p2.at[:3].set(jnp.array([3, 0, 0]))

    # Crossover
    # If we swap root: P1 becomes * x0 x0.
    # If we swap leaf: P1 becomes + x0 x0 (no change if leaf is same).

    # Let's try to force a crossover.
    # Since we use random inside, we just check if output is valid.

    c1 = gp.crossover_one_point(key, p1, p2, arities)

    # Check validity
    end = gp.find_subtree_end(c1, 0, arities)
    assert end > 0
