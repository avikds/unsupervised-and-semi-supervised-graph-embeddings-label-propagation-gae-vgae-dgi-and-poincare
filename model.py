"""
Unsupervised and Semi-Supervised Graph Embeddings: Label Propagation, GAE/VGAE, DGI and Poincare

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - contextual_sbm
import numpy as np

def contextual_sbm(n=1000, k=5, avg_degree=5.0, homophily=0.8, mu=2.0, d=64, degree_shape=2.5, seed=0):
    # Initialize a single random number generator.
    rng = np.random.default_rng([seed, 23])

    # 1. Generate balanced ground-truth labels and randomly permute them.
    y = rng.permutation(np.arange(n) % k)

    # 2. Generate degree propensities.
    if degree_shape > 0:
        theta = rng.pareto(degree_shape, n) + 1.0

        # Cap extreme degree propensities and normalize their mean to one.
        theta = np.minimum(theta, np.sqrt(n) / 2.0)
        theta = theta / theta.mean()
    else:
        # No degree correction.
        theta = np.ones(n)

    # 3. Calculate within-class and between-class edge probabilities.
    p_in = avg_degree * homophily / (n / k)
    p_out = avg_degree * (1.0 - homophily) / (n - n / k)

    # Assign the appropriate block probability to every node pair.
    block_rates = np.where(y[:, None] == y[None, :], p_in, p_out)

    # Apply degree correction and clip probabilities to [0, 1].
    P = block_rates * theta[:, None] * theta[None, :]
    P = np.clip(P, 0.0, 1.0)

    # Remove self-loop probabilities.
    np.fill_diagonal(P, 0.0)

    # 4. Sample edges using the true edge probabilities.
    # Sample only the upper triangular part to avoid duplicate edges.
    A = np.triu(rng.random((n, n)) < P, 1)

    # Make the adjacency matrix symmetric and convert it to float64.
    A = (A + A.T).astype(np.float64)

    # 5. Generate one Gaussian class mean per class.
    M = rng.normal(size=(k, d))

    # Normalize every class mean to unit length.
    M = M / np.linalg.norm(M, axis=1, keepdims=True)

    # 6. Generate node features using their corresponding class means.
    X = mu * M[y] + rng.normal(size=(n, d))

    # Convert node features to float32.
    X = X.astype(np.float32)

    # Return the graph, features, labels, and true edge probabilities.
    return {
        "A": A,
        "X": X,
        "y": y.astype(np.int64),
        "P": P
    }

# Step 2 - edge_homophily
def edge_homophily(A, y):
    # Find all undirected edges by considering only the upper triangle.
    src, dst = np.where(np.triu(A > 0, k=1))

    # Handle graphs with no undirected edges.
    if len(src) == 0:
        return 0.0

    # Count edges whose endpoints have the same ground-truth label.
    same_label = (y[src] == y[dst])

    # Return the fraction of same-label edges as a Python float.
    return float(np.mean(same_label))


def few_labels(y, n_per_class, seed=0):
    # Initialize the random number generator with the specified seed.
    rng = np.random.default_rng([seed, 29])

    # Store the randomly selected labeled node indices.
    labeled = []

    # Draw exactly n_per_class nodes from each class, in class order.
    for c in range(int(y.max()) + 1):
        class_nodes = np.flatnonzero(y == c)

        selected = rng.choice(
            class_nodes,
            size=n_per_class,
            replace=False
        )

        labeled.append(selected)

    # Concatenate the selected indices while preserving class order.
    labeled = np.concatenate(labeled).astype(np.int64)

    # Create a Boolean mask identifying selected nodes.
    selected_mask = np.zeros(len(y), dtype=bool)
    selected_mask[labeled] = True

    # Collect all remaining node indices in increasing order.
    rest = np.flatnonzero(~selected_mask).astype(np.int64)

    # Return labeled and unlabeled node indices.
    return labeled, rest

# Step 3 - balanced_tree
def balanced_tree(b, depth):
    # Validate the branching factor and tree depth.
    if b < 1:
        raise ValueError("The branching factor b must be at least 1.")
    if depth < 0:
        raise ValueError("The tree depth must be non-negative.")

    # Calculate the total number of nodes across all levels.
    N = sum(b ** level for level in range(depth + 1))

    # Initialize the adjacency matrix, parent array, and level array.
    A = np.zeros((N, N), dtype=np.float64)
    parent = np.full(N, -1, dtype=np.int64)
    level = np.zeros(N, dtype=np.int64)

    # Assign parents and levels using breadth-first node numbering.
    for v in range(1, N):
        # In breadth-first numbering, each node's parent is (v - 1) // b.
        p = (v - 1) // b

        parent[v] = p
        level[v] = level[p] + 1

        # Add an undirected edge between the node and its parent.
        A[v, p] = 1.0
        A[p, v] = 1.0

    # Return the adjacency matrix, parent array, and level array.
    return A, parent, level


def tree_distances(parent):
    # Convert the parent array to int64.
    parent = np.asarray(parent, dtype=np.int64)

    # Get the total number of nodes.
    N = len(parent)

    # Initialize the all-pairs distance matrix.
    D = np.zeros((N, N), dtype=np.int64)

    # Calculate distances using the parent-before-child ordering.
    for v in range(1, N):
        p = parent[v]

        # Distance from v to every previously numbered node u:
        # d(v, u) = d(parent(v), u) + 1.
        D[v, :v] = D[p, :v] + 1

        # Copy distances to the upper triangle to ensure symmetry.
        D[:v, v] = D[v, :v]

    # Return the symmetric all-pairs shortest-path distance matrix.
    return D

