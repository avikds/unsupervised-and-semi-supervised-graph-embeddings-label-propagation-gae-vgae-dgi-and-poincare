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

# Step 7 - label_spreading
def label_spreading(W, labeled, y_labeled, k, alpha=0.9):
    # Convert inputs to NumPy arrays with the required data types.
    W = np.asarray(W, dtype=np.float64)
    labeled = np.asarray(labeled, dtype=np.int64)
    y_labeled = np.asarray(y_labeled, dtype=np.int64)

    # Get the number of nodes.
    n = W.shape[0]

    # Initialize the label matrix with zeros.
    Y = np.zeros((n, k), dtype=np.float64)

    # Assign one-hot label vectors to the labeled nodes.
    Y[labeled, y_labeled] = 1.0

    # Calculate the degree of each node.
    degree = W.sum(axis=1)

    # Calculate inverse square roots of node degrees.
    # Isolated nodes receive a factor of zero.
    inv_sqrt_degree = np.zeros(n, dtype=np.float64)

    np.divide(
        1.0,
        np.sqrt(degree),
        out=inv_sqrt_degree,
        where=degree > 0
    )

    # Construct the symmetrically normalized adjacency matrix:
    # S = D^(-1/2) @ W @ D^(-1/2).
    S = (
        inv_sqrt_degree[:, None]
        * W
        * inv_sqrt_degree[None, :]
    )

    # Construct the identity matrix.
    I = np.eye(n, dtype=np.float64)

    # Construct the coefficient matrix:
    # I - alpha * S.
    M = I - alpha * S

    # Solve the linear system:
    # (I - alpha * S) F = (1 - alpha) * Y.
    F = np.linalg.solve(M, (1.0 - alpha) * Y)

    return F


def spreading_objective(W, F, Y, mu):
    # Convert inputs to NumPy arrays with float64 precision.
    W = np.asarray(W, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Get the number of nodes.
    n = W.shape[0]

    # Calculate the degree of each node.
    degree = W.sum(axis=1)

    # Calculate inverse square roots of node degrees.
    # Isolated nodes receive a factor of zero.
    inv_sqrt_degree = np.zeros(n, dtype=np.float64)

    np.divide(
        1.0,
        np.sqrt(degree),
        out=inv_sqrt_degree,
        where=degree > 0
    )

    # Construct the symmetrically normalized adjacency matrix.
    S = (
        inv_sqrt_degree[:, None]
        * W
        * inv_sqrt_degree[None, :]
    )

    # Construct the normalized graph Laplacian.
    L = np.eye(n, dtype=np.float64) - S

    # Calculate the graph smoothness term:
    # trace(F.T @ (I - S) @ F).
    smoothness = np.trace(F.T @ L @ F)

    # Calculate the label-fitting term:
    # mu * sum_i ||F_i - Y_i||^2.
    fidelity = mu * np.sum((F - Y) ** 2)

    # Combine both terms and return a Python float.
    return float(smoothness + fidelity)

# Step 8 - kmeans
def kmeans(Z, k, seed=0, n_init=5, n_iter=50):
    # Convert the embedding matrix to float64.
    Z = np.asarray(Z, dtype=np.float64)

    # Validate the input dimensions and parameters.
    if Z.ndim != 2:
        raise ValueError("Z must be a two-dimensional array.")

    n, d = Z.shape

    if n == 0 or d == 0:
        raise ValueError("Z must contain at least one point and one feature.")

    if k < 1 or k > n:
        raise ValueError("k must be between 1 and the number of points.")

    if n_init < 1 or n_iter < 1:
        raise ValueError("n_init and n_iter must be positive.")

    # Use one random number generator shared across all restarts.
    rng = np.random.default_rng(seed)

    best_labels = None
    best_inertia = np.inf

    # Run multiple independent K-means restarts.
    for _ in range(n_init):

        # -------------------------------------------------
        # 1. K-means++ initialization.
        # -------------------------------------------------

        centers = np.empty((k, d), dtype=np.float64)

        # Select the first center uniformly at random.
        first_idx = rng.integers(n)
        centers[0] = Z[first_idx]

        # Select the remaining centers using squared distances.
        for c in range(1, k):

            # Calculate squared distances to the nearest existing center.
            d2 = np.min(
                np.sum(
                    (Z[:, None, :] - centers[None, :c, :]) ** 2,
                    axis=2
                ),
                axis=1
            )

            # Select the next center using the K-means++ probability.
            total_distance = d2.sum()

            if total_distance == 0:
                # All points coincide with existing centers.
                next_idx = rng.integers(n)
            else:
                probabilities = d2 / total_distance
                next_idx = rng.choice(n, p=probabilities)

            centers[c] = Z[next_idx]

        # -------------------------------------------------
        # 2. Lloyd's algorithm.
        # -------------------------------------------------

        # Initialize assignments so the first iteration runs.
        labels = np.full(n, -1, dtype=np.int64)

        for _ in range(n_iter):

            # Calculate squared distances from every point to every center.
            distances = np.sum(
                (Z[:, None, :] - centers[None, :, :]) ** 2,
                axis=2
            )

            # Assign each point to its nearest center.
            new_labels = np.argmin(distances, axis=1).astype(np.int64)

            # Check whether any point changed its cluster assignment.
            unchanged = np.array_equal(labels, new_labels)

            labels = new_labels

            # Update every non-empty cluster center to its mean.
            for c in range(k):
                members = Z[labels == c]

                if len(members) > 0:
                    centers[c] = members.mean(axis=0)

                # Empty clusters retain their previous centers.

            # Stop early if cluster assignments have not changed.
            if unchanged:
                break

        # -------------------------------------------------
        # 3. Calculate within-cluster sum of squares.
        # -------------------------------------------------

        inertia = np.sum((Z - centers[labels]) ** 2)

        # Keep the restart with the lowest inertia.
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    # Return the best clustering as an int64 array.
    return best_labels.astype(np.int64)


def adjusted_rand_index(a, b):
    # Convert inputs to NumPy arrays.
    a = np.asarray(a)
    b = np.asarray(b)

    # Both label arrays must be one-dimensional and have equal length.
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("a and b must be one-dimensional arrays of equal length.")

    n = len(a)

    # Handle empty or single-element arrays.
    if n < 2:
        return 1.0

    # Convert arbitrary label values into contiguous integer indices.
    _, a_inverse = np.unique(a, return_inverse=True)
    _, b_inverse = np.unique(b, return_inverse=True)

    n_a = a_inverse.max() + 1
    n_b = b_inverse.max() + 1

    # -------------------------------------------------
    # 1. Construct the contingency table.
    # -------------------------------------------------

    contingency = np.zeros((n_a, n_b), dtype=np.int64)

    np.add.at(
        contingency,
        (a_inverse, b_inverse),
        1
    )

    # Calculate the row sums and column sums.
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    # Helper: calculate the number of unordered pairs.
    def combinations_2(x):
        return x * (x - 1) / 2.0

    # -------------------------------------------------
    # 2. Calculate the pair-count statistics.
    # -------------------------------------------------

    # Sum of combinations of size two in each contingency cell.
    index = combinations_2(contingency).sum()

    # Sum of combinations of size two in the row and column totals.
    sum_rows = combinations_2(row_sums).sum()
    sum_cols = combinations_2(col_sums).sum()

    # Total number of unordered pairs.
    total_pairs = combinations_2(n)

    # Expected index under random independent assignments.
    expected_index = (sum_rows * sum_cols) / total_pairs

    # -------------------------------------------------
    # 3. Calculate the Adjusted Rand Index.
    # -------------------------------------------------

    numerator = index - expected_index

    denominator = 0.5 * (sum_rows + sum_cols) - expected_index

    # Handle degenerate partitions.
    if np.isclose(denominator, 0.0, atol=1e-12):
        return 1.0

    ari = numerator / denominator

    # Return the result as a Python float.
    return float(ari)

# Step 9 - laplacian_eigenmaps
def laplacian_eigenmaps(W, dim, tau=0.0):
    # Convert the adjacency matrix to float64.
    W = np.asarray(W, dtype=np.float64)

    # Get the number of nodes.
    N = W.shape[0]

    # Add tau / N to every entry of the adjacency matrix.
    # This provides regularization and can connect disconnected components.
    Wt = W + tau / N

    # Calculate the degree of each node.
    degree = Wt.sum(axis=1)

    # Calculate D^(-1/2).
    # Use zero for nodes having zero degree.
    inv_sqrt_degree = np.zeros(N, dtype=np.float64)

    np.divide(
        1.0,
        np.sqrt(degree),
        out=inv_sqrt_degree,
        where=degree > 0
    )

    # Construct the symmetric normalized adjacency matrix:
    # S = D^(-1/2) @ Wt @ D^(-1/2).
    S = (
        inv_sqrt_degree[:, None]
        * Wt
        * inv_sqrt_degree[None, :]
    )

    # Construct the symmetric normalized graph Laplacian:
    # L_sym = I - D^(-1/2) @ Wt @ D^(-1/2).
    L_sym = np.eye(N, dtype=np.float64) - S

    # Compute all eigenvalues and eigenvectors.
    # np.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    # Skip the first eigenvector and select the next 'dim' eigenvectors.
    selected_eigenvectors = eigenvectors[:, 1:dim + 1]

    # Map the selected eigenvectors back through D^(-1/2).
    Z = inv_sqrt_degree[:, None] * selected_eigenvectors

    # Return the embedding matrix and all ascending eigenvalues.
    return Z, eigenvalues

# Step 10 - deepwalk_similarity
def window_weights(T, window="dynamic"):
    # Validate the window size.
    if T < 1:
        raise ValueError("T must be a positive integer.")

    # Fixed window: every distance receives equal probability.
    if window == "fixed":
        return np.full(T, 1.0 / T, dtype=np.float64)

    # Dynamic window:
    # A window size is sampled uniformly from {1, 2, ..., T}.
    # A context at distance q is included whenever the sampled
    # window size is at least q.
    elif window == "dynamic":
        q = np.arange(1, T + 1, dtype=np.float64)

        # P(Q = q) is proportional to the number of window sizes
        # that include distance q, namely T - q + 1.
        weights = T - q + 1.0

        # Normalize the weights so that they sum to one.
        weights /= weights.sum()

        return weights.astype(np.float64)

    else:
        raise ValueError("window must be either 'dynamic' or 'fixed'.")


def deepwalk_similarity(W, T, window="dynamic"):
    # Convert the adjacency matrix to float64.
    W = np.asarray(W, dtype=np.float64)

    # Validate the adjacency matrix.
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square adjacency matrix.")

    # Get the number of nodes.
    n = W.shape[0]

    # Obtain the context-window distance weights.
    weights = window_weights(T, window)

    # Calculate the degree of each node.
    degree = W.sum(axis=1)

    # Construct the row-normalized transition matrix:
    # P = D^(-1) @ W.
    # Isolated nodes retain all-zero rows.
    P = np.zeros_like(W, dtype=np.float64)

    np.divide(
        W,
        degree[:, None],
        out=P,
        where=degree[:, None] != 0
    )

    # Initialize the DeepWalk similarity matrix.
    similarity = np.zeros((n, n), dtype=np.float64)

    # Start with the first transition-matrix power.
    power = P.copy()

    # Compute the weighted sum:
    # S(W) = sum_{q=1}^T P(Q=q) * P^q.
    for q in range(T):
        similarity += weights[q] * power

        # Calculate the next transition-matrix power.
        power = power @ P

    return similarity

# Step 11 - netmf_embedding
def netmf_embedding(W, T=5, b=1.0, dim=16):
    # Convert the adjacency matrix to float64.
    W = np.asarray(W, dtype=np.float64)

    # Get the number of nodes.
    n = W.shape[0]

    # 1. Compute the DeepWalk similarity matrix.
    # Reuse the deepwalk_similarity function from Step 3.
    S = deepwalk_similarity(W, T)

    # Calculate the graph volume (sum of all adjacency entries).
    vol = float(W.sum())

    # 2. Calculate the inverse node degrees.
    degree = W.sum(axis=1)

    # Use zero for the inverse degree of isolated nodes.
    inv_degree = np.zeros(n, dtype=np.float64)

    np.divide(
        1.0,
        degree,
        out=inv_degree,
        where=degree > 0
    )

    # Construct the NetMF matrix:
    # M = (vol / b) * S * D^(-1).
    # Right multiplication by D^(-1) scales columns.
    M = (vol / b) * S * inv_degree[None, :]

    # Apply the element-wise logarithm:
    # L = log(max(M, 1)).
    L = np.log(np.maximum(M, 1.0))

    # 3. Symmetrize the resulting matrix.
    L = (L + L.T) / 2.0

    # Compute the eigenvalue decomposition of the symmetric matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # 4. Select the eigenpairs with the largest absolute eigenvalues.
    indices = np.argsort(np.abs(eigenvalues))[::-1][:dim]

    lam = eigenvalues[indices]
    V = eigenvectors[:, indices]

    # Construct the source embeddings:
    # Zs = V * sqrt(abs(lam)).
    Zs = V * np.sqrt(np.abs(lam))[None, :]

    # Construct the target embeddings:
    # Zt = Zs * sign(lam).
    Zt = Zs * np.sign(lam)[None, :]

    # Return source and target embeddings.
    return Zs, Zt

# Step 12 - edm_fit
from tinygrad.nn.optim import Adam


def edm_fit(params, sup=None, recon=None, reg=None, alpha=1.0, beta=1.0, gamma=0.0, batch_fn=None, steps=200, lr=0.01, seed=0):

    # Determine which loss terms are active.
    use_sup = sup is not None and alpha != 0
    use_recon = recon is not None and beta != 0
    use_reg = reg is not None and gamma != 0

    # At least one loss term must be active.
    if not (use_sup or use_recon or use_reg):
        raise ValueError("At least one weighted loss term must be active.")

    params = list(params)

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Initialize Adam optimizer.
    optimizer = Adam(
        params,
        lr=lr,
        b1=0.9,
        b2=0.999,
        eps=1e-8
    )

    # Define the training step.
    def train_step(*batch):
        optimizer.zero_grad()

        loss = None

        # Supervised loss.
        if use_sup:
            loss = alpha * sup()

        # Graph reconstruction loss.
        if use_recon:
            recon_loss = beta * recon(*batch)
            loss = recon_loss if loss is None else loss + recon_loss

        # Regularization loss.
        if use_reg:
            reg_loss = gamma * reg()
            loss = reg_loss if loss is None else loss + reg_loss

        # Backpropagation.
        loss.backward()

        # Update parameters using Adam.
        optimizer.step()

        return loss.realize()

    # Compile the training step.
    jit_train_step = TinyJit(train_step)

    history = []

    # IMPORTANT:
    # tinygrad requires training mode to be enabled
    # before calling optimizer.step().
    previous_training = Tensor.training
    Tensor.training = True

    try:
        for t in range(steps):

            # Generate a fresh minibatch for every iteration.
            if use_recon and batch_fn is not None:
                batch_arrays = batch_fn(rng)

                # Convert NumPy arrays into realized Tensors.
                batch = tuple(
                    Tensor(x).realize()
                    for x in batch_arrays
                )
            else:
                batch = ()

            # Execute one JIT-compiled training iteration.
            loss = jit_train_step(*batch)

            # Record the loss at the required iterations.
            if t == 0 or (t + 1) % 50 == 0 or t == steps - 1:
                history.append(
                    (int(t), float(loss.item()))
                )

    finally:
        # Restore the previous tinygrad training state.
        Tensor.training = previous_training

    return history

# Step 13 - GCNEncoder
def gcn_normalize(A):
    # Convert the adjacency matrix to a NumPy array.
    A = np.asarray(A, dtype=np.float64)

    # Get the number of nodes.
    n = A.shape[0]

    # Add self-loops to every node.
    A_tilde = A + np.eye(n, dtype=np.float64)

    # Calculate the degree of each node.
    degree = A_tilde.sum(axis=1)

    # Calculate inverse square-root degrees.
    inv_sqrt_degree = np.zeros(n, dtype=np.float64)

    np.divide(
        1.0,
        np.sqrt(degree),
        out=inv_sqrt_degree,
        where=degree > 0
    )

    # Apply symmetric degree normalization:
    # A_hat = D^(-1/2) @ (A + I) @ D^(-1/2).
    A_hat = (
        inv_sqrt_degree[:, None]
        * A_tilde
        * inv_sqrt_degree[None, :]
    )

    # Return the normalized adjacency matrix as float32.
    return A_hat.astype(np.float32)


def glorot(rng, a, b):
    # Calculate the Glorot-uniform initialization limit.
    limit = np.sqrt(6.0 / (a + b))

    # Generate uniformly distributed weights.
    weights = rng.uniform(
        -1.0,
        1.0,
        size=(a, b)
    ) * limit

    # Convert weights to float32.
    weights = weights.astype(np.float32)

    # Convert to a contiguous, realized tinygrad Tensor.
    return Tensor(weights).contiguous().realize()


class GCNEncoder:
    def __init__(self, in_dim, hidden, out_dim, seed=0):
        # Initialize a single random number generator.
        rng = np.random.default_rng(seed)

        # Initialize the first GCN weight matrix.
        self.W1 = glorot(rng, in_dim, hidden)

        # Initialize the second GCN weight matrix using the same RNG.
        self.W2 = glorot(rng, hidden, out_dim)

    def parameters(self):
        # Return all trainable parameters.
        return [self.W1, self.W2]

    def hidden_layer(self, Ahat, X):
        # First graph convolution followed by ReLU activation.
        H = (Ahat @ X @ self.W1).relu()

        return H

    def __call__(self, Ahat, X):
        # Compute the hidden representations.
        H = self.hidden_layer(Ahat, X)

        # Second graph convolution produces node embeddings.
        Z = Ahat @ H @ self.W2

        return Z

# Step 14 - link_split
def link_split(A, val_frac=0.05, test_frac=0.10, seed=0):
    # Convert the adjacency matrix to a NumPy array.
    A = np.asarray(A)

    # Get the number of nodes.
    N = A.shape[0]

    # Initialize the random number generator with the specified seed.
    rng = np.random.default_rng([seed, 31])

    # -------------------------------------------------
    # 1. Extract and shuffle undirected positive edges.
    # -------------------------------------------------

    # Extract only the upper-triangular edges (i < j).
    edges = np.column_stack(
        np.nonzero(np.triu(A, k=1))
    ).astype(np.int64)

    # Get the total number of undirected edges.
    E_count = len(edges)

    # Shuffle the edge rows using the specified RNG.
    edges = edges[rng.permutation(E_count)]

    # -------------------------------------------------
    # 2. Split positive edges into validation, test,
    #    and training sets.
    # -------------------------------------------------

    # Determine the number of validation and test edges.
    nv = int(E_count * val_frac)
    nt = int(E_count * test_frac)

    # Validation positives are the first nv shuffled edges.
    val_pos = edges[:nv].copy()

    # Test positives are the next nt shuffled edges.
    test_pos = edges[nv:nv + nt].copy()

    # Remaining edges are used for training.
    train = edges[nv + nt:].copy()

    # -------------------------------------------------
    # 3. Construct the training adjacency matrix.
    # -------------------------------------------------

    # Initialize an empty symmetric float64 adjacency matrix.
    A_train = np.zeros((N, N), dtype=np.float64)

    # Insert only the training edges.
    if len(train) > 0:
        A_train[train[:, 0], train[:, 1]] = 1.0
        A_train[train[:, 1], train[:, 0]] = 1.0

    # -------------------------------------------------
    # 4. Sample negative edges.
    # -------------------------------------------------

    # Check whether enough non-edges are available.
    total_possible_edges = N * (N - 1) // 2
    available_negatives = total_possible_edges - E_count

    if nv + nt > available_negatives:
        raise ValueError("Not enough non-edges available for validation and test negatives.")

    # Maintain a set of already selected negative edges.
    used_negatives = set()

    # Generate validation negatives first, followed by test negatives.
    negative_sets = []

    for count in (nv, nt):

        negatives = []

        # Continue sampling until the required number is reached.
        while len(negatives) < count:

            # Draw two node indices using the specified RNG.
            a, b = rng.integers(0, N, 2)

            # Skip self-loops.
            if a == b:
                continue

            # Ensure that the edge is represented as (i, j), i < j.
            i, j = (a, b) if a < b else (b, a)

            # Skip existing edges and previously selected negatives.
            if A[i, j] != 0 or (i, j) in used_negatives:
                continue

            # Keep this negative edge.
            used_negatives.add((i, j))
            negatives.append((i, j))

        # Convert the negative edges into an int64 array.
        negative_sets.append(
            np.asarray(negatives, dtype=np.int64).reshape(-1, 2)
        )

    # Separate validation and test negative edges.
    val_neg, test_neg = negative_sets

    # -------------------------------------------------
    # 5. Return all data splits.
    # -------------------------------------------------

    return {
        "A_train": A_train,
        "train": train.astype(np.int64),
        "val_pos": val_pos.astype(np.int64),
        "test_pos": test_pos.astype(np.int64),
        "val_neg": val_neg.astype(np.int64),
        "test_neg": test_neg.astype(np.int64)
    }

# Step 15 - link_auc
def link_auc(pos, neg):
    # Convert the positive and negative scores to NumPy arrays.
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)

    # Validate the input arrays.
    if pos.ndim != 1 or neg.ndim != 1:
        raise ValueError("pos and neg must be one-dimensional arrays.")

    n_pos = len(pos)
    n_neg = len(neg)

    # AUC is undefined if either class has no observations.
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Combine positive and negative scores.
    scores = np.concatenate((pos, neg))

    # Store the original positions of positive scores.
    positive_mask = np.concatenate((
        np.ones(n_pos, dtype=bool),
        np.zeros(n_neg, dtype=bool)
    ))

    # Sort scores in ascending order using a stable sort.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    # Identify the starting positions of groups of tied scores.
    starts = np.concatenate((
        np.array([0]),
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
    ))

    # Identify the ending positions of each tied-score group.
    ends = np.concatenate((
        starts[1:],
        np.array([len(scores)])
    ))

    # Calculate average ranks for tied groups.
    # Ranks are one-based.
    average_ranks = (starts + 1 + ends) / 2.0

    # Assign the average ranks to all observations.
    sorted_ranks = np.repeat(
        average_ranks,
        ends - starts
    )

    # Restore the ranks to their original observation order.
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = sorted_ranks

    # Calculate the sum of ranks belonging to positive scores.
    rank_sum_positive = ranks[positive_mask].sum()

    # Calculate the Mann-Whitney U statistic.
    U = rank_sum_positive - n_pos * (n_pos + 1) / 2.0

    # Calculate AUC.
    auc = U / (n_pos * n_neg)

    return float(auc)


def pair_auc(S, pos_pairs, neg_pairs):
    # Convert inputs to NumPy arrays.
    S = np.asarray(S)
    pos_pairs = np.asarray(pos_pairs, dtype=np.int64)
    neg_pairs = np.asarray(neg_pairs, dtype=np.int64)

    # Validate the pair array dimensions.
    if pos_pairs.ndim != 2 or pos_pairs.shape[1] != 2:
        raise ValueError("pos_pairs must have shape (m, 2).")

    if neg_pairs.ndim != 2 or neg_pairs.shape[1] != 2:
        raise ValueError("neg_pairs must have shape (m, 2).")

    # Extract predicted scores for positive edges.
    pos_scores = S[pos_pairs[:, 0], pos_pairs[:, 1]]

    # Extract predicted scores for negative edges.
    neg_scores = S[neg_pairs[:, 0], neg_pairs[:, 1]]

    # Calculate AUC using the previously implemented function.
    return link_auc(pos_scores, neg_scores)

