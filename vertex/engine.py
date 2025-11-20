import jax
import jax.numpy as jnp
import chex
from vertex import gp
from functools import partial

def evolve(
    key: chex.PRNGKey,
    pop_size: int,
    generations: int,
    pset: gp.PrimitiveSet,
    evaluator: callable, # Function that takes (genome) -> fitness (scalar)
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    tournament_size: int = 3,
    max_len: int = 64
):
    # Initialize population (Python side)
    # We can't jit this part easily if it uses python loops.
    # But we can pass the initial population.

    # For the full loop, we want JIT.
    # We need to express the evolution step as a scan or loop.

    # Prepare static data
    arities = pset.get_arities_array()
    mut_table, mut_counts = pset.get_mutation_tables()

    # Batched evaluation
    # evaluator should take a single genome. We vmap it.
    evaluate_pop = jax.vmap(evaluator)

    def step(carry, _):
        key, population = carry

        # 1. Evaluate
        fitnesses = evaluate_pop(population)

        # Log best
        best_idx = jnp.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_genome = population[best_idx]

        # 2. Selection & Reproduction
        # We need to generate a new population of size `pop_size`.
        # We can do this by generating `pop_size` offspring.

        # For each offspring, we decide op: crossover, mutation, or copy.
        # Or standard: select 2 parents, crossover -> 2 children.

        # Let's just map over range(pop_size) to produce one child each.

        def generate_offspring(k, i):
            k_op, k_sel, k_cross, k_mut = jax.random.split(k, 4)

            # Select parents
            p1_idx = gp.tournament_selection(k_sel, fitnesses, tournament_size)
            # Need second parent for crossover
            k_sel2, _ = jax.random.split(k_sel) # simplified split
            p2_idx = gp.tournament_selection(k_sel2, fitnesses, tournament_size)

            p1 = population[p1_idx]
            p2 = population[p2_idx]

            # Probabilistically choose operator
            rand_val = jax.random.uniform(k_op)

            # If < crossover_rate: Crossover
            # Else if < crossover_rate + mutation_rate: Mutation
            # Else: Reproduction (Copy)

            def do_crossover():
                return gp.crossover_one_point(k_cross, p1, p2, arities)

            def do_mutation():
                # Mutate p1
                return gp.mutation_point(k_mut, p1, arities, mut_table, mut_counts)

            def do_copy():
                return p1

            # Branch
            # Note: JAX if/else
            child = jax.lax.cond(
                rand_val < crossover_rate,
                do_crossover,
                lambda: jax.lax.cond(
                    rand_val < (crossover_rate + mutation_rate),
                    do_mutation,
                    do_copy
                )
            )

            return child

        # Generate new population keys
        keys = jax.random.split(key, pop_size)

        new_population = jax.vmap(generate_offspring)(keys, jnp.arange(pop_size))

        return (keys[0], new_population), (best_fitness, best_genome)

    return step
