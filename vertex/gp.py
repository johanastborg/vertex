import jax
import jax.numpy as jnp
from typing import Callable, List, NamedTuple, Tuple, Optional, Dict
import chex
from functools import partial

# Type definitions
Gene = jnp.int32
Genome = chex.Array  # Shape (max_len, )

class Primitive(NamedTuple):
    name: str
    arity: int
    func: Callable[[jnp.ndarray], jnp.ndarray]

class PrimitiveSet:
    def __init__(self, input_count: int, max_len: int):
        self.primitives: List[Primitive] = []
        self.input_count = input_count
        self.max_len = max_len # Max genome length

    def add_op(self, name: str, arity: int, func: Callable):
        self.primitives.append(Primitive(name, arity, func))

    @property
    def num_ops(self):
        return len(self.primitives)

    @property
    def num_terminals(self):
        return self.input_count

    @property
    def total_symbols(self):
        return self.num_terminals + self.num_ops

    def get_arities_array(self) -> jnp.ndarray:
        # Returns array of arities for all symbols [terminals..., ops...]
        arities = [0] * self.input_count + [p.arity for p in self.primitives]
        return jnp.array(arities, dtype=jnp.int32)

    def get_mutation_tables(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Returns (arity_map, candidates)
        # We need to map Arity -> Array of symbol IDs with that arity.
        # Since jagged arrays are hard, we can use a padded array.
        # Max arity?
        max_arity = max([p.arity for p in self.primitives]) if self.primitives else 0

        # Map: arity -> list of IDs
        groups: Dict[int, List[int]] = {0: list(range(self.input_count))}
        for i, p in enumerate(self.primitives):
            a = p.arity
            if a not in groups: groups[a] = []
            groups[a].append(self.input_count + i)

        # Convert to padded array
        # shape: (max_arity + 1, max_candidates_per_arity)
        max_cands = max(len(g) for g in groups.values())

        table = jnp.full((max_arity + 1, max_cands), -1, dtype=jnp.int32)
        counts = jnp.zeros((max_arity + 1,), dtype=jnp.int32)

        for a, ids in groups.items():
            table = table.at[a, :len(ids)].set(jnp.array(ids))
            counts = counts.at[a].set(len(ids))

        return table, counts

def make_simple_primitive_set(input_count: int, max_len: int = 64):
    pset = PrimitiveSet(input_count, max_len)
    pset.add_op("add", 2, jnp.add)
    pset.add_op("sub", 2, jnp.subtract)
    pset.add_op("mul", 2, jnp.multiply)

    def protected_div(x, y):
        return jnp.where(jnp.abs(y) > 1e-6, x / y, 1.0)
    pset.add_op("div", 2, protected_div)

    pset.add_op("sin", 1, jnp.sin)
    pset.add_op("cos", 1, jnp.cos)
    return pset

# --- Evaluation ---

@chex.dataclass
class StackState:
    stack: chex.Array
    ptr: int

def build_evaluator(pset: PrimitiveSet, max_stack_size: int = 64):
    arities_arr = pset.get_arities_array()
    input_count = pset.input_count
    ops = pset.primitives

    def execute_op_impl(op_id, args):
        branches = []
        for prim in ops:
            arity = prim.arity
            func = prim.func
            def make_branch(f, a):
                def branch(stack_vals):
                    operands = [stack_vals[i] for i in range(a)]
                    return f(*operands)
                return branch
            branches.append(make_branch(func, arity))

        return jax.lax.switch(op_id, branches, args)

    def eval_one(genome: Genome, input_values: chex.Array):
        # Use helper to find valid length first
        valid_len = find_subtree_end(genome, 0, arities_arr)

        # Invert for RPN
        # But we only care about valid part: genome[:valid_len]
        # We can pad the rest with -1.
        # Note: jnp.flip reverses the *whole* array.
        # If genome is [+, a, b, -1, -1], reverse is [-1, -1, b, a, +].
        # We need to skip the leading -1s.

        # Hack: assume we scan the whole reversed array, ignoring -1s.
        rpn_genome = genome[::-1]

        def step(carry, gene):
            stack, ptr = carry.stack, carry.ptr

            is_valid = gene >= 0
            is_terminal = (gene < input_count) & is_valid
            is_op = (gene >= input_count) & is_valid

            arity = arities_arr[jnp.maximum(gene, 0)]

            max_arity_global = max([p.arity for p in ops]) if ops else 0
            gather_indices = ptr - 1 - jnp.arange(max_arity_global)
            gather_indices = jnp.maximum(gather_indices, 0)
            args = stack[gather_indices]

            op_idx = jnp.maximum(gene - input_count, 0)
            op_res = execute_op_impl(op_idx, args)

            term_val = input_values[jnp.maximum(gene, 0)]

            res = jnp.where(is_terminal, term_val, op_res)

            delta = jnp.where(is_terminal, 1, 1 - arity)
            delta = jnp.where(is_valid, delta, 0)

            new_ptr = ptr + delta
            write_idx = jnp.where(is_terminal, ptr, ptr - arity)

            new_stack = jnp.where(
                is_valid,
                stack.at[write_idx].set(res),
                stack
            )

            return StackState(stack=new_stack, ptr=new_ptr), None

        initial_stack = jnp.zeros(max_stack_size, dtype=jnp.float32)
        final_state, _ = jax.lax.scan(step, StackState(stack=initial_stack, ptr=0), rpn_genome)

        return final_state.stack[0]

    return eval_one

# --- Traversal / Analysis ---

def find_subtree_end(genome: Genome, start_idx: int, arities: chex.Array) -> int:
    length = genome.shape[0]

    def cond_fun(val):
        idx, count, done = val
        return (idx < length) & (~done)

    def body_fun(val):
        idx, count, done = val
        gene = genome[idx]
        # Treat -1 (padding) as arity 0 but it shouldn't happen in valid tree logic?
        # If we hit padding while looking for children, the tree is invalid.
        # We just stop or continue?
        arity = jnp.where(gene >= 0, arities[jnp.maximum(gene, 0)], 0)

        # If padding encountered while count > 0, it's an error in tree structure (truncated).
        # But for this function we just proceed.

        new_count = count - 1 + arity
        is_finished = (new_count == 0)
        return (idx + 1, new_count, is_finished)

    final_val = jax.lax.while_loop(cond_fun, body_fun, (start_idx, 1, False))
    end_idx, _, _ = final_val
    return end_idx

# --- Genetic Operators ---

def initialize_population(key: chex.PRNGKey, pop_size: int, max_len: int, pset: PrimitiveSet, min_depth=2, max_depth=5):
    import numpy as np

    pop = np.full((pop_size, max_len), -1, dtype=np.int32)

    for i in range(pop_size):
        tree = []
        def gen_node(depth):
            if depth >= max_depth:
                term_idx = np.random.randint(0, pset.num_terminals)
                tree.append(term_idx)
            else:
                if np.random.rand() < 0.5 and depth >= min_depth:
                     term_idx = np.random.randint(0, pset.num_terminals)
                     tree.append(term_idx)
                else:
                    op_idx = np.random.randint(0, pset.num_ops)
                    prim = pset.primitives[op_idx]
                    tree.append(pset.input_count + op_idx)
                    for _ in range(prim.arity):
                        gen_node(depth + 1)

        gen_node(0)
        l = len(tree)
        if l <= max_len:
            pop[i, :l] = tree
        else:
            pop[i, :max_len] = tree[:max_len]

    return jnp.array(pop)

def crossover_one_point(key: chex.PRNGKey, p1: Genome, p2: Genome, arities: chex.Array):
    len1 = find_subtree_end(p1, 0, arities)
    len2 = find_subtree_end(p2, 0, arities)

    k1, k2 = jax.random.split(key)

    # Random split points
    idx1 = jax.random.randint(k1, (), 0, len1)
    idx2 = jax.random.randint(k2, (), 0, len2)

    end1 = find_subtree_end(p1, idx1, arities)
    end2 = find_subtree_end(p2, idx2, arities)

    # Copy loop
    def copy_chunk(start_src, end_src, src, start_dst, dst):
        length = end_src - start_src
        def body(i, d):
            val = src[start_src + i]
            return jax.lax.cond(
                (start_dst + i) < d.shape[0],
                lambda: d.at[start_dst + i].set(val),
                lambda: d
            )
        return jax.lax.fori_loop(0, length, body, dst)

    child = jnp.full_like(p1, -1)

    # P1 pre
    child = copy_chunk(0, idx1, p1, 0, child)

    len_pre = idx1
    len_sub = end2 - idx2

    # P2 sub
    child = copy_chunk(idx2, end2, p2, len_pre, child)

    # P1 post
    child = copy_chunk(end1, p1.shape[0], p1, len_pre + len_sub, child)

    return child

def mutation_point(key: chex.PRNGKey, genome: Genome, arities: chex.Array, mut_table: jnp.ndarray, mut_counts: jnp.ndarray):
    # Select a valid gene
    valid_len = find_subtree_end(genome, 0, arities)
    idx = jax.random.randint(key, (), 0, valid_len)

    gene = genome[idx]
    arity = arities[jnp.maximum(gene, 0)]

    # Select replacement with same arity
    # candidates = mut_table[arity] (slice)
    # count = mut_counts[arity]

    count = mut_counts[arity]
    candidates = mut_table[arity]

    cand_idx = jax.random.randint(key, (), 0, count)
    new_gene = candidates[cand_idx]

    # Mutate
    return genome.at[idx].set(new_gene)

def tournament_selection(key: chex.PRNGKey, fitnesses: chex.Array, tournament_size: int):
    # Returns index of selected individual
    pop_size = fitnesses.shape[0]
    indices = jax.random.randint(key, (tournament_size,), 0, pop_size)
    # We assume lower fitness is better (error)? Or higher is better?
    # Let's assume Minimize Error (lower is better).
    # If maximize, change logic.
    # GP usually minimizes error.

    selected_fitnesses = fitnesses[indices]
    best_idx_in_tournament = jnp.argmin(selected_fitnesses)
    return indices[best_idx_in_tournament]
