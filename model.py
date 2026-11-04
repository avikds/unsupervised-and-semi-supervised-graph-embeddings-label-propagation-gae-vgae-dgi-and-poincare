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

# Step 4 - label_propagation
def label_propagation(W, labeled, y_labeled, k, n_iter=100):
    # Convert inputs to NumPy arrays.
    W = np.asarray(W, dtype=np.float64)
    labeled = np.asarray(labeled, dtype=np.int64)
    y_labeled = np.asarray(y_labeled, dtype=np.int64)

    # Get the number of nodes.
    n = W.shape[0]

    # Initialize the label-score matrix with zeros.
    F = np.zeros((n, k), dtype=np.float64)

    # Assign one-hot label vectors to the labeled nodes.
    F[labeled, y_labeled] = 1.0

    # Calculate the degree of each node.
    degree = W.sum(axis=1)

    # Construct the row-normalized transition matrix.
    # Rows with zero degree remain all zeros.
    P = np.zeros_like(W, dtype=np.float64)
    np.divide(
        W,
        degree[:, None],
        out=P,
        where=degree[:, None] != 0
    )

    # Repeatedly propagate label information through the graph.
    for _ in range(n_iter):
        # Average the neighboring nodes' label scores.
        F = P @ F

        # Keep the originally labeled nodes fixed.
        F[labeled] = 0.0
        F[labeled, y_labeled] = 1.0

    return F


def dirichlet_energy(W, F):
    # Convert inputs to floating-point NumPy arrays.
    W = np.asarray(W, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)

    # Calculate the degree matrix.
    degree = W.sum(axis=1)
    D = np.diag(degree)

    # Construct the graph Laplacian.
    L = D - W

    # Calculate the Dirichlet energy:
    # 2 * trace(F.T @ (D - W) @ F).
    energy = 2.0 * np.trace(F.T @ L @ F)

    # Return the result as a Python float.
    return float(energy)

# Step 5 - harmonic_labels
def harmonic_labels(W, labeled, y_labeled, k):
    # Convert inputs to NumPy arrays with the required data types.
    W = np.asarray(W, dtype=np.float64)
    labeled = np.asarray(labeled, dtype=np.int64)
    y_labeled = np.asarray(y_labeled, dtype=np.int64)

    # Get the total number of nodes.
    n = W.shape[0]

    # Identify unlabeled nodes in increasing index order.
    mask = np.ones(n, dtype=bool)
    mask[labeled] = False
    unlabeled = np.flatnonzero(mask)

    # Initialize the label-score matrix.
    F = np.zeros((n, k), dtype=np.float64)

    # Assign one-hot label vectors to the labeled nodes.
    F[labeled, y_labeled] = 1.0

    # If every node is labeled, return the initialized matrix.
    if len(unlabeled) == 0:
        return F

    # Construct one-hot label vectors for the labeled nodes.
    Y_L = F[labeled]

    # Calculate the full graph degree vector.
    # Important: degrees must include connections to both labeled
    # and unlabeled nodes.
    degree = W.sum(axis=1)

    # Extract the unlabeled-to-unlabeled adjacency submatrix.
    W_UU = W[np.ix_(unlabeled, unlabeled)]

    # Extract the unlabeled-to-labeled adjacency submatrix.
    W_UL = W[np.ix_(unlabeled, labeled)]

    # Construct the unlabeled graph Laplacian block.
    # D_UU uses the full graph degrees, not just W_UU row sums.
    L_UU = np.diag(degree[unlabeled]) - W_UU

    # Add a small ridge term to keep the system numerically solvable.
    L_UU += 1e-9 * np.eye(len(unlabeled), dtype=np.float64)

    # Construct the right-hand side of the linear system.
    rhs = W_UL @ Y_L

    # Solve:
    # (D_UU - W_UU + 1e-9 I) F_U = W_UL Y_L
    F_U = np.linalg.solve(L_UU, rhs)

    # Assign the computed harmonic scores to unlabeled nodes.
    F[unlabeled] = F_U

    return F

# Step 6 - class_mass_normalize
def class_mass_normalize(F, y_labeled, k):
    # Convert inputs to NumPy arrays.
    F = np.asarray(F, dtype=np.float64)
    y_labeled = np.asarray(y_labeled, dtype=np.int64)

    # 1. Estimate class proportions using add-one smoothing.
    # q[c] = (count of class c + 1) / (number of labeled nodes + k).
    class_counts = np.bincount(y_labeled, minlength=k)
    q = (class_counts + 1.0) / (len(y_labeled) + k)

    # 2. Calculate the total predicted mass of each class.
    m = F.sum(axis=0)

    # 3. Prevent division by zero for classes with negligible mass.
    m = np.maximum(m, 1e-12)

    # Rescale each class column to match its estimated class proportion.
    G = F * (q / m)[None, :]

    # Return a new normalized prediction matrix.
    return G


def node_accuracy(F, y, idx):
    # Convert inputs to NumPy arrays.
    F = np.asarray(F)
    y = np.asarray(y)
    idx = np.asarray(idx, dtype=np.int64)

    # Predict node labels using the highest-scoring class.
    predictions = np.argmax(F, axis=1)

    # Handle an empty evaluation index array.
    if len(idx) == 0:
        return 0.0

    # Calculate accuracy on the specified nodes.
    accuracy = np.mean(predictions[idx] == y[idx])

    # Return accuracy as a Python float.
    return float(accuracy)

