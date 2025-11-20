import jax
import jax.numpy as jnp
import pytest
from vertex import gp, engine

def test_evolution_simple_regression():
    # Problem: Find expression that approximates target(x) = x * x + x
    # Inputs: x
    # Terminals: x (id 0)
    # Ops: +, -, *, / (ids 1..4)

    input_count = 1
    max_len = 15
    pset = gp.make_simple_primitive_set(input_count, max_len)

    # Target data
    xs = jnp.linspace(-1, 1, 10)
    ys = xs * xs + xs

    # Evaluator
    eval_one = gp.build_evaluator(pset)

    def fitness_fn(genome):
        # MSE
        # We need to vmap evaluation over data points

        def predict(x):
            # Create input array (size 1)
            inp = jnp.array([x])
            return eval_one(genome, inp)

        preds = jax.vmap(predict)(xs)
        # Check for NaNs or Infs
        mse = jnp.mean((preds - ys)**2)
        return jnp.where(jnp.isfinite(mse), mse, 1e9)

    # Params
    pop_size = 50
    generations = 20
    key = jax.random.PRNGKey(42)

    # Init population
    k_init, k_evolve = jax.random.split(key)
    population = gp.initialize_population(k_init, pop_size, max_len, pset, max_depth=4)

    # Build step
    step_fn = engine.evolve(
        k_evolve,
        pop_size,
        generations,
        pset,
        fitness_fn,
        mutation_rate=0.2,
        crossover_rate=0.7,
        tournament_size=3
    )

    # JIT the loop
    scan_fn = jax.jit(jax.lax.scan, static_argnums=(0,))

    # Run
    # scan(f, init, xs)
    # We iterate 'generations' times.
    # We pass dummy inputs to scan? Or use range.

    initial_carry = (k_evolve, population)
    final_carry, history = scan_fn(step_fn, initial_carry, jnp.arange(generations))

    best_fitnesses, best_genomes = history

    print(f"Best fitness history: {best_fitnesses}")

    # Check if fitness improved
    assert best_fitnesses[-1] < best_fitnesses[0] or best_fitnesses[-1] < 1.0

    # The problem is simple, so it should find a good solution or at least improve.
    # x*x + x might be hard if constants are not available (we didn't add Ephemeral Constants).
    # But we have x * x + x.
    # (add (mul x x) x)
