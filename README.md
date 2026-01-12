# Vertex

> **Von-Neumann Evolutionary Engine for Tracking Data**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Powered-9cf)](https://github.com/google/jax)

**Vertex** is a high-performance, JAX-based Genetic Programming (GP) engine. Designed for speed and scalability, it leverages JAX's JIT compilation and hardware acceleration (GPU/TPU) to evolve programs efficiently. While initially conceived for analyzing particle physics tracking data, Vertex serves as a general-purpose, differentiable-friendly evolutionary engine.

## 🚀 Key Features

*   **⚡ JAX Powered:** Fully JIT-compilable evolution loops. Run your entire GP pipeline on GPUs or TPUs.
*   **🧬 Linear Genome Representation:** Uses fixed-length integer arrays with Prefix (Polish) notation, optimized for vectorization.
*   **📚 Stack-Based Evaluation:** Robust stack machine evaluator implemented with `jax.lax.scan` for efficient program execution.
*   **🔧 Customizable Primitives:** Easily define custom function sets and terminals.
*   **🔀 Genetic Operators:** Built-in support for Tournament Selection, One-Point Crossover, and Point Mutation.
*   **🔄 End-to-End Compilation:** The entire evolutionary process, from evaluation to reproduction, can be compiled into a single XLA kernel.

## 📦 Installation

Clone the repository and install the package:

```bash
git clone https://github.com/yourusername/vertex.git
cd vertex
pip install .
```

## 🛠️ Usage

Here is a complete example of Symbolic Regression to approximate $f(x) = x^2$.

```python
import jax
import jax.numpy as jnp
from vertex import gp, engine

def main():
    # 1. Define Primitive Set
    # Input count = 1 (x), max genome length = 64
    max_len = 64
    pset = gp.make_simple_primitive_set(input_count=1, max_len=max_len)

    # 2. Define Evaluator
    # We want to approximate f(x) = x^2
    X = jnp.linspace(-1, 1, 20).reshape(-1, 1) # 20 samples, 1 feature
    Y = X ** 2

    # gp.build_evaluator returns a function `eval_one(genome, input_values)`
    eval_one_fn = gp.build_evaluator(pset)

    # Define fitness function (Mean Squared Error)
    def compute_fitness(genome):
        # Evaluate genome on all data points using vmap
        preds = jax.vmap(eval_one_fn, in_axes=(None, 0))(genome, X)
        error = jnp.mean((preds - Y.flatten()) ** 2)
        return error

    # 3. Initialize Population
    seed = 42
    key = jax.random.PRNGKey(seed)
    pop_size = 1000
    generations = 50

    # Initialize population (returns numpy array, convert to JAX array)
    population = jnp.array(gp.initialize_population(key, pop_size, max_len, pset))

    # 4. Create Evolution Step
    # engine.evolve returns a step function compatible with jax.lax.scan
    evolve_step = engine.evolve(
        key, pop_size, generations, pset, compute_fitness,
        mutation_rate=0.1, crossover_rate=0.8
    )

    # 5. Run Evolution Loop (JIT compiled)
    @jax.jit
    def run_evolution(key, population):
        # scan runs the loop `generations` times.
        final_carry, history = jax.lax.scan(evolve_step, (key, population), None, length=generations)
        return final_carry, history

    print("Compiling and running evolution...")
    (final_key, final_pop), (history_fitness, history_genomes) = run_evolution(key, population)

    print("Evolution complete.")
    print(f"Best fitness over generations: {history_fitness}")
    print(f"Final best fitness: {history_fitness[-1]}")

if __name__ == "__main__":
    main()
```

## 📂 Project Structure

*   **`vertex/gp.py`**: Core Genetic Programming logic.
    *   `PrimitiveSet`: Manages functions and terminals.
    *   `build_evaluator`: Generates the JAX-compatible stack machine evaluator.
    *   `initialize_population`: Creates random initial genomes.
    *   Genetic Operators: `crossover_one_point`, `mutation_point`, `tournament_selection`.
*   **`vertex/engine.py`**: Evolution orchestration.
    *   `evolve`: High-level function that creates the evolution `step` function for `jax.lax.scan`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
