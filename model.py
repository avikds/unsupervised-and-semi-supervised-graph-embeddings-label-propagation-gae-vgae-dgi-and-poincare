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

# Step 16 - gae_full_loss
def gae_full_loss(Z, A, pos_weight=None):
    # Get the number of nodes.
    n = A.shape[0]

    # Validate the adjacency matrix dimensions.
    if A.shape != (n, n):
        raise ValueError("A must be a square adjacency matrix.")

    # Calculate the positive-class weight if not provided.
    if pos_weight is None:
        positive_count = float(A.sum().item())
        total_entries = n * n

        if positive_count == 0:
            raise ValueError("A must contain at least one positive entry.")

        pos_weight = (total_entries - positive_count) / positive_count

    # Inner-product decoder: W_hat = Z @ Z.T.
    logits = Z @ Z.T

    # Weighted sigmoid cross-entropy.
    # Use logsigmoid for numerical stability.
    positive_loss = pos_weight * A * logits.logsigmoid()

    negative_loss = (1.0 - A) * (-logits).logsigmoid()

    # Average over all N^2 entries.
    loss = -(positive_loss + negative_loss).mean()

    return loss


def gae_sampled_loss(Z, pos, neg):
    # Extract source and destination indices for positive edges.
    pos_src = pos[:, 0]
    pos_dst = pos[:, 1]

    # Calculate inner-product logits for positive edges.
    pos_logits = (
        Z[pos_src] * Z[pos_dst]
    ).sum(axis=1)

    # Extract source and destination indices for negative edges.
    neg_src = neg[:, 0]
    neg_dst = neg[:, 1]

    # Calculate inner-product logits for negative edges.
    neg_logits = (
        Z[neg_src] * Z[neg_dst]
    ).sum(axis=1)

    # Positive-edge reconstruction loss.
    positive_loss = -pos_logits.logsigmoid().mean()

    # Negative-edge reconstruction loss.
    negative_loss = -(-neg_logits).logsigmoid().mean()

    # Return the sum of the two mean losses.
    return positive_loss + negative_loss

# Step 17 - train_gae
def train_gae(split, X, hidden=32, dim=16, loss="sampled", steps=200, lr=0.01, seed=0):

    # Get the number of nodes.
    N = X.shape[0]

    # 1. Prepare the training adjacency matrix and node features.
    # Only A_train is allowed to reach the encoder.
    Ahat = Tensor(
        gcn_normalize(split["A_train"])
    ).realize()

    Xt = Tensor(
        np.asarray(X, dtype=np.float32)
    ).realize()

    # 2. Create the two-layer GCN encoder.
    enc = GCNEncoder(
        X.shape[1],
        hidden,
        dim,
        seed=seed
    )

    # 3. Convert training positive edges into an int32 Tensor.
    pos_array = np.asarray(
        split["train"],
        dtype=np.int32
    )

    pos = Tensor(pos_array).contiguous().realize()

    # Number of positive training edges.
    m = len(split["train"])

    # 4. Configure the reconstruction objective.
    batch_fn = None

    if loss == "sampled":

        # Generate m random negative node pairs for every step.
        # The shape and dtype remain constant for TinyJit.
        def batch_fn(rng):
            neg = rng.integers(
                0,
                N,
                size=(m, 2),
                dtype=np.int32
            )

            return (neg,)

        # Sampled positive-negative reconstruction loss.
        def recon(neg):
            Z = enc(Ahat, Xt)

            return gae_sampled_loss(
                Z,
                pos,
                neg
            )

    elif loss == "full":

        # Convert the training adjacency matrix into a float32 Tensor.
        At = Tensor(
            np.asarray(split["A_train"], dtype=np.float32)
        ).realize()

        # Full weighted reconstruction loss.
        def recon():
            Z = enc(Ahat, Xt)

            return gae_full_loss(
                Z,
                At
            )

    elif loss == "full_unweighted":

        # Convert the training adjacency matrix into a float32 Tensor.
        At = Tensor(
            np.asarray(split["A_train"], dtype=np.float32)
        ).realize()

        # Full reconstruction loss with positive weight equal to 1.
        def recon():
            Z = enc(Ahat, Xt)

            return gae_full_loss(
                Z,
                At,
                pos_weight=1.0
            )

    else:
        raise ValueError(
            "loss must be 'sampled', 'full', or 'full_unweighted'."
        )

    # 5. Train the GAE using the shared GraphEDM trainer.
    # alpha = 0 removes the supervised loss.
    # beta = 1 assigns full weight to graph reconstruction.
    hist = edm_fit(
        enc.parameters(),
        recon=recon,
        alpha=0.0,
        beta=1.0,
        batch_fn=batch_fn,
        steps=steps,
        lr=lr,
        seed=seed
    )

    # 6. Compute the final node embeddings.
    Z = enc(Ahat, Xt).realize().numpy()

    # Return embeddings, encoder, and training history.
    return {
        "Z": np.asarray(Z, dtype=np.float32),
        "enc": enc,
        "hist": hist
    }

# Step 18 - train_vgae
def gaussian_kl(mu, logvar):
    # Calculate the KL divergence between:
    # N(mu, diag(exp(logvar))) and N(0, I).

    # Element-wise Gaussian KL contribution:
    # mu^2 + exp(logvar) - 1 - logvar.
    kl = (
        mu * mu
        + logvar.exp()
        - 1.0
        - logvar
    )

    # Sum over latent dimensions and average over nodes.
    return 0.5 * kl.sum(axis=1).mean()


def train_vgae(split, X, hidden=32, dim=16, kl_weight=1.0, steps=200, lr=0.01, seed=0):

    # Get the number of nodes.
    N = X.shape[0]

    # 1. Prepare the training adjacency matrix and node features.
    # Only A_train is allowed to reach the encoder.
    Ahat = Tensor(
        gcn_normalize(split["A_train"])
    ).realize()

    Xt = Tensor(
        np.asarray(X, dtype=np.float32)
    ).realize()

    # Create the shared two-layer GCN encoder.
    enc = GCNEncoder(
        X.shape[1],
        hidden,
        dim,
        seed=seed
    )

    # Initialize the additional log-variance weight matrix
    # using an independent random number generator.
    rng_w3 = np.random.default_rng(seed + 1000)

    W3 = glorot(
        rng_w3,
        hidden,
        dim
    )

    # 2. Convert training positive edges into an int32 Tensor.
    pos = Tensor(
        np.asarray(split["train"], dtype=np.int32)
    ).contiguous().realize()

    # Number of positive training edges.
    m = len(split["train"])

    # -------------------------------------------------
    # Define the shared variational encoder.
    # -------------------------------------------------

    def latent_stats():
        # First shared GCN hidden representation.
        # H = A_hat @ ReLU(A_hat @ X @ W1).
        H = Ahat @ enc.hidden_layer(Ahat, Xt)

        # Mean of the Gaussian latent distribution.
        mu = H @ enc.W2

        # Log-variance, clipped for numerical stability.
        logvar = (H @ W3).clip(-10.0, 10.0)

        return mu, logvar

    # -------------------------------------------------
    # 3. Generate fresh negative edges and noise.
    # -------------------------------------------------

    def batch_fn(rng):
        # Sample m negative node pairs.
        neg = rng.integers(
            0,
            N,
            size=(m, 2),
            dtype=np.int32
        )

        # Sample Gaussian noise for reparameterization.
        eps = rng.normal(
            0.0,
            1.0,
            size=(N, dim)
        ).astype(np.float32)

        return neg, eps

    # -------------------------------------------------
    # 4. Variational reconstruction loss.
    # -------------------------------------------------

    def recon(neg, eps):
        # Compute the current Gaussian parameters.
        mu, logvar = latent_stats()

        # Reparameterization trick:
        # z = mu + sigma * eps.
        sigma = (0.5 * logvar).exp()

        z = mu + sigma * eps

        # Calculate sampled reconstruction loss.
        return gae_sampled_loss(
            z,
            pos,
            neg
        )

    # -------------------------------------------------
    # 5. KL regularization loss.
    # -------------------------------------------------

    def reg():
        # Obtain the current Gaussian parameters.
        mu, logvar = latent_stats()

        # Return the KL divergence divided by N,
        # as explicitly required by the instructions.
        return gaussian_kl(mu, logvar) / N

    # -------------------------------------------------
    # 6. Train using the shared GraphEDM trainer.
    # -------------------------------------------------

    hist = edm_fit(
        [enc.W1, enc.W2, W3],
        alpha=0.0,
        beta=1.0,
        gamma=kl_weight,
        recon=recon,
        reg=reg,
        batch_fn=batch_fn,
        steps=steps,
        lr=lr,
        seed=seed
    )

    # -------------------------------------------------
    # 7. Extract the final latent distribution.
    # -------------------------------------------------

    mu_t, logvar_t = latent_stats()

    # Convert final Tensor values into NumPy arrays.
    mu_np = mu_t.realize().numpy()
    logvar_np = logvar_t.realize().numpy()

    mu_np = np.asarray(mu_np, dtype=np.float32)
    logvar_np = np.asarray(logvar_np, dtype=np.float32)

    # Calculate the mean KL contribution of each
    # latent dimension across all nodes.
    kl_per_dim = 0.5 * np.mean(
        mu_np ** 2
        + np.exp(logvar_np)
        - 1.0
        - logvar_np,
        axis=0
    )

    # Return the final variational parameters,
    # per-dimension KL values, and training history.
    return {
        "mu": mu_np,
        "logvar": logvar_np,
        "kl_per_dim": kl_per_dim,
        "hist": hist
    }

# Step 19 - dgi_loss
def readout(Z):
    # Calculate the mean embedding across all nodes.
    mean_embedding = Z.mean(axis=0)

    # Apply sigmoid to obtain the graph-level summary vector.
    summary = mean_embedding.sigmoid()

    return summary


def dgi_loss(Z, Z_neg, Wb, fake_summary=False):
    # Calculate the summary vector from the real graph.
    # The positive term always uses this summary.
    s = readout(Z)

    # Choose the summary vector for the negative term.
    if fake_summary:
        # Use the summary of the corrupted graph.
        s_neg = readout(Z_neg)
    else:
        # Use the real graph summary, as in the original DGI.
        s_neg = s

    # Bilinear discriminator for real node embeddings:
    # logits_i = Z_i.T @ Wb @ s.
    real_logits = ((Z @ Wb) * s).sum(axis=1)

    # Bilinear discriminator for corrupted node embeddings:
    # logits_i = Z_neg_i.T @ Wb @ s_neg.
    neg_logits = ((Z_neg @ Wb) * s_neg).sum(axis=1)

    # Positive loss: maximize log D(Z_i, s).
    positive_loss = real_logits.logsigmoid().mean()

    # Negative loss: maximize log(1 - D(Z_neg_i, s_neg)).
    negative_loss = (-neg_logits).logsigmoid().mean()

    # Return the averaged Deep Graph Infomax objective.
    loss = -0.5 * (positive_loss + negative_loss)

    return loss

# Step 20 - train_dgi
def train_dgi(A, X, dim=256, steps=200, lr=0.001, seed=0):

    # Get the number of nodes.
    N = X.shape[0]

    # 1. Precompute the normalized adjacency matrix and node features.
    Ahat = Tensor(
        gcn_normalize(A)
    ).realize()

    Xt = Tensor(
        np.asarray(X, dtype=np.float32)
    ).realize()

    # 2. Create the GCN encoder.
    enc = GCNEncoder(
        X.shape[1],
        dim,
        dim,
        seed=seed
    )

    # Initialize the bilinear discriminator weight matrix.
    rng_wb = np.random.default_rng(seed + 1000)

    Wb = glorot(
        rng_wb,
        dim,
        dim
    )

    # Define the embedding function.
    # ReLU is applied to the final GCN output.
    def embed(features):
        return enc(Ahat, features).relu()

    # 3. Generate a fresh random permutation of node features
    # at every training iteration.
    def batch_fn(rng):
        perm = rng.permutation(N).astype(np.int32)

        # Return the permutation as a one-element tuple.
        return (perm,)

    # 4. Define the DGI reconstruction/contrastive objective.
    def recon(perm):
        # Embeddings from the original graph features.
        Z = embed(Xt)

        # Embeddings from shuffled (corrupted) node features.
        Z_neg = embed(Xt[perm])

        # Calculate the Deep Graph Infomax loss.
        return dgi_loss(
            Z,
            Z_neg,
            Wb
        )

    # 5. Train the encoder and discriminator together.
    # No supervised labels or graph reconstruction are used.
    hist = edm_fit(
        enc.parameters() + [Wb],
        alpha=0.0,
        beta=1.0,
        recon=recon,
        batch_fn=batch_fn,
        steps=steps,
        lr=lr,
        seed=seed
    )

    # 6. Compute the final embeddings using the original features.
    Z = embed(Xt).realize().numpy()

    # Return the embeddings and training history.
    return {
        "Z": np.asarray(Z, dtype=np.float32),
        "hist": hist
    }

# Step 21 - linear_probe
def linear_probe(Z, y, train_idx, test_idx, l2=1e-3, steps=300, lr=0.05, seed=0):
    # Convert inputs to NumPy arrays.
    Z = np.asarray(Z, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    # Determine the number of input dimensions and classes.
    n, d = Z.shape
    k = int(y.max()) + 1

    # 1. Standardize each embedding dimension using all nodes.
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)

    Z_std = ((Z - mean) / (std + 1e-6)).astype(np.float32)

    # Extract standardized training and testing embeddings.
    Z_train = Tensor(
        Z_std[train_idx]
    ).contiguous().realize()

    Z_test = Tensor(
        Z_std[test_idx]
    ).contiguous().realize()

    # Convert training labels into one-hot encoded vectors.
    Y_train = np.eye(k, dtype=np.float32)[y[train_idx]]

    Y_train_t = Tensor(
        Y_train
    ).contiguous().realize()

    # 2. Initialize the softmax regression parameters.
    # Both weights and bias are initialized to zero.
    W = Tensor.zeros(d, k).contiguous().realize()
    b = Tensor.zeros(k).contiguous().realize()

    # Define the supervised cross-entropy loss.
    def sup():
        # Compute class logits.
        logits = Z_train @ W + b

        # Calculate numerically stable log probabilities.
        log_probs = logits.log_softmax(axis=1)

        # Mean cross-entropy over the training nodes.
        return -(Y_train_t * log_probs).sum(axis=1).mean()

    # Define L2 regularization on the weight matrix only.
    def reg():
        return (W * W).sum()

    # Train softmax regression using the shared GraphEDM trainer.
    edm_fit(
        [W, b],
        sup=sup,
        reg=reg,
        alpha=1.0,
        gamma=l2,
        steps=steps,
        lr=lr,
        seed=seed
    )

    # 3. Evaluate the trained classifier on the test nodes.
    test_logits = Z_test @ W + b

    # Obtain predicted class labels.
    predictions = test_logits.argmax(axis=1).numpy()

    # Calculate test accuracy as a Python float.
    accuracy = np.mean(
        predictions == y[test_idx]
    )

    return float(accuracy)


def smoothed_features(A, X, hops=2):
    # Convert the node feature matrix to a NumPy array.
    X = np.asarray(X)

    # Validate the number of smoothing iterations.
    if hops < 0:
        raise ValueError("hops must be non-negative.")

    # Zero hops returns the original features unchanged.
    if hops == 0:
        return X

    # Calculate the normalized adjacency matrix.
    # This includes self-loops through gcn_normalize().
    Ahat = gcn_normalize(A)

    # Initialize the smoothed features.
    H = X.copy()

    # Repeatedly apply normalized graph convolution.
    for _ in range(hops):
        H = Ahat @ H

    return H

# Step 22 - train_gcn_classifier
def train_gcn_classifier(A, X, y, labeled, hidden=32, dim=16, beta=0.0, wd=5e-4, steps=200, lr=0.01, seed=0):

    # Convert inputs to NumPy arrays.
    A = np.asarray(A)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    labeled = np.asarray(labeled, dtype=np.int64)

    # Get the number of nodes and classes.
    N = X.shape[0]
    k = int(y.max()) + 1

    # Validate the labeled node indices.
    if len(labeled) == 0:
        raise ValueError("At least one labeled node is required.")

    # 1. Prepare the normalized adjacency matrix and node features.
    Ahat = Tensor(
        gcn_normalize(A)
    ).realize()

    Xt = Tensor(X).realize()

    # 2. Create the GCN encoder.
    enc = GCNEncoder(
        X.shape[1],
        hidden,
        dim,
        seed=seed
    )

    # Initialize the classification weight matrix independently.
    Wc = glorot(
        np.random.default_rng(seed + 1000),
        dim,
        k
    )

    # Build a full-size one-hot target matrix.
    # Unlabeled nodes have all-zero target rows.
    Y = np.zeros((N, k), dtype=np.float32)

    Y[labeled] = np.eye(k, dtype=np.float32)[y[labeled]]

    Yt = Tensor(Y).realize()

    # 3. Define the supervised classification loss.
    def sup():
        # Generate node embeddings using the GCN encoder.
        Z = enc(Ahat, Xt)

        # Calculate class logits for every node.
        logits = Z @ Wc

        # Compute log probabilities using softmax.
        log_probs = logits.log_softmax(axis=1)

        # Calculate mean cross-entropy over labeled nodes only.
        return -(Yt * log_probs).sum() / len(labeled)

    # 4. Define L2 regularization on the GCN encoder.
    # The classification head Wc is intentionally excluded.
    def reg():
        return (
            (enc.W1 * enc.W1).sum()
            + (enc.W2 * enc.W2).sum()
        )

    # -------------------------------------------------
    # 5. Configure optional graph reconstruction.
    # -------------------------------------------------

    recon = None
    batch_fn = None

    if beta != 0.0:

        # Extract all undirected positive edges from the
        # upper triangular part of the original adjacency matrix.
        pos_array = np.column_stack(
            np.nonzero(np.triu(A, k=1))
        ).astype(np.int32)

        # Convert positive edges into a realized int32 Tensor.
        pos = Tensor(pos_array).contiguous().realize()

        # Number of undirected positive edges.
        m = len(pos_array)

        # Generate m negative pairs for each training iteration.
        def batch_fn(rng):
            neg = rng.integers(
                0,
                N,
                size=(m, 2),
                dtype=np.int32
            )

            return (neg,)

        # Define the sampled graph reconstruction loss.
        def recon(neg):
            Z = enc(Ahat, Xt)

            return gae_sampled_loss(
                Z,
                pos,
                neg
            )

    # -------------------------------------------------
    # 6. Train using the shared GraphEDM trainer.
    # -------------------------------------------------

    edm_fit(
        enc.parameters() + [Wc],
        sup=sup,
        recon=recon,
        reg=reg,
        alpha=1.0,
        beta=beta,
        gamma=wd,
        batch_fn=batch_fn,
        steps=steps,
        lr=lr,
        seed=seed
    )

    # -------------------------------------------------
    # 7. Calculate final logits for all nodes.
    # -------------------------------------------------

    Z = enc(Ahat, Xt)

    logits = (Z @ Wc).realize().numpy()

    # Return the (N, k) NumPy array of class logits.
    return np.asarray(logits, dtype=np.float32)

# Step 23 - sample_context_pairs
def context_cdf(S):
    # Convert the similarity matrix to float64.
    S = np.asarray(S, dtype=np.float64)

    # Make a copy so that the original matrix is not modified.
    S = S.copy()

    # Get the number of nodes.
    N = S.shape[0]

    # Calculate the sum of every row.
    row_sums = S.sum(axis=1)

    # Handle isolated nodes (rows with zero total probability).
    # Assign a self-loop so that the node's only context is itself.
    isolated = row_sums == 0.0
    S[isolated] = 0.0
    S[np.flatnonzero(isolated), np.flatnonzero(isolated)] = 1.0

    # Recalculate row sums after handling isolated nodes.
    row_sums = S.sum(axis=1)

    # Normalize every row into a probability distribution.
    S = S / row_sums[:, None]

    # Calculate the row-wise cumulative distribution function.
    cdf = np.cumsum(S, axis=1)

    return cdf


def sample_context_pairs(cdf, m, rng, neg_frac=0.5):
    # Convert the CDF to a NumPy array.
    cdf = np.asarray(cdf, dtype=np.float64)

    # Get the number of nodes.
    N = cdf.shape[0]

    # 1. Draw centre nodes.
    i = rng.integers(0, N, m)

    # 2. Draw uniform random numbers for inverse-CDF sampling.
    u = rng.random(m)

    # 3. Draw random numbers to determine positive/negative pairs.
    coin = rng.random(m)

    # 4. Draw uniformly random context nodes for negative pairs.
    other = rng.integers(0, N, m)

    # Sample positive contexts using inverse-CDF sampling.
    # Each context is selected according to its row probability.
    context = (cdf[i] < u[:, None]).sum(axis=1)

    # Ensure that the selected context index is within [0, N-1].
    context = np.clip(context, 0, N - 1)

    # Select negative samples when coin < neg_frac.
    is_negative = coin < neg_frac

    # Initialize context indices.
    j = context.copy()

    # Replace positive contexts with uniformly sampled nodes
    # for the negative samples.
    j[is_negative] = other[is_negative]

    # Assign +1 to positive pairs and -1 to negative pairs.
    gamma = np.where(is_negative, -1.0, 1.0)

    # Return indices as int32 and labels as float32.
    return (
        i.astype(np.int32),
        j.astype(np.int32),
        gamma.astype(np.float32)
    )

# Step 24 - train_planetoid
def train_planetoid(S, X, y, labeled, zc_dim=16, hidden=8, pre=500, joint=200, wd=0.05, m=2000, lr=0.01, seed=0):

    # Convert inputs to NumPy arrays.
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    labeled = np.asarray(labeled, dtype=np.int64)

    # Get the number of nodes, feature dimensions, and classes.
    N, D = X.shape
    k = int(y.max()) + 1

    if len(labeled) == 0:
        raise ValueError("At least one labeled node is required.")

    # Initialize the random number generator.
    rng0 = np.random.default_rng(seed)

    # -------------------------------------------------
    # 1. Initialize the lookup table and weight matrices.
    # -------------------------------------------------

    # Initialize the node context embedding lookup table.
    Zc = Tensor(
        rng0.normal(
            0.0,
            0.1,
            size=(N, zc_dim)
        ).astype(np.float32)
    ).contiguous().realize()

    # Initialize the feature encoder weight matrix.
    W1 = glorot(rng0, D, hidden)

    # Initialize the classification weight matrix.
    # Input dimensions = context dimensions + hidden dimensions.
    W2 = glorot(rng0, zc_dim + hidden, k)

    # Convert node features to a realized Tensor.
    Xt = Tensor(X).contiguous().realize()

    # -------------------------------------------------
    # 2. Prepare the DeepWalk context distribution.
    # -------------------------------------------------

    # Convert the DeepWalk similarity matrix into row-wise CDFs.
    cdf = context_cdf(S)

    # -------------------------------------------------
    # 3. Define the context-pair sampling function.
    # -------------------------------------------------

    def batch_fn(rng):
        # Sample context pairs using the precomputed CDF.
        # Returns:
        # i: center node indices
        # j: context node indices
        # gamma: +1 for positive pairs, -1 for negative pairs.
        return sample_context_pairs(
            cdf,
            m,
            rng
        )

    # -------------------------------------------------
    # 4. Define the context reconstruction loss.
    # -------------------------------------------------

    def context_loss(i, j, gamma):
        # Calculate the dot product between center and context embeddings.
        scores = (Zc[i] * Zc[j]).sum(axis=1)

        # Apply positive/negative labels and calculate
        # the numerically stable logistic loss.
        return -(scores * gamma).logsigmoid().mean()

    # -------------------------------------------------
    # 5. Stage 1: Context pretraining.
    # -------------------------------------------------

    # Train only the context embedding lookup table.
    edm_fit(
        [Zc],
        recon=context_loss,
        alpha=0.0,
        beta=1.0,
        batch_fn=batch_fn,
        steps=pre,
        lr=lr,
        seed=seed
    )

    # Save a NumPy copy of the pretrained context embeddings.
    Zc_pre = Zc.realize().numpy().copy()

    # -------------------------------------------------
    # 6. Prepare supervised classification targets.
    # -------------------------------------------------

    # Construct a full-size one-hot label matrix.
    # Unlabeled nodes have zero target rows.
    Y = np.zeros((N, k), dtype=np.float32)

    Y[labeled] = np.eye(k, dtype=np.float32)[y[labeled]]

    Yt = Tensor(Y).contiguous().realize()

    # -------------------------------------------------
    # 7. Define the supervised classification loss.
    # -------------------------------------------------

    def sup():
        # Feature-based encoder without graph convolution.
        H = (Xt @ W1).relu()

        # Concatenate context embeddings and feature embeddings.
        Z_combined = Zc.cat(H, dim=1)

        # Calculate classification logits.
        logits = Z_combined @ W2

        # Compute log probabilities for all nodes.
        log_probs = logits.log_softmax(axis=1)

        # Calculate cross-entropy only on labeled nodes.
        return -(Yt * log_probs).sum() / len(labeled)

    # -------------------------------------------------
    # 8. Define regularization.
    # -------------------------------------------------

    def reg():
        # Regularize the feature encoder and classification head.
        # The context lookup table Zc is not regularized here.
        return (
            (W1 * W1).sum()
            + (W2 * W2).sum()
        )

    # -------------------------------------------------
    # 9. Stage 2: Joint training.
    # -------------------------------------------------

    # Jointly optimize the context embeddings,
    # feature encoder, and classification head.
    edm_fit(
        [Zc, W1, W2],
        sup=sup,
        recon=context_loss,
        reg=reg,
        alpha=1.0,
        beta=1.0,
        gamma=wd,
        batch_fn=batch_fn,
        steps=joint,
        lr=lr,
        seed=seed
    )

    # -------------------------------------------------
    # 10. Compute final logits for all nodes.
    # -------------------------------------------------

    H_final = (Xt @ W1).relu()

    Z_combined = Zc.cat(H_final, dim=1)

    logits = (Z_combined @ W2).realize().numpy()

    # Return final classification logits and context embeddings.
    return {
        "logits": np.asarray(logits, dtype=np.float32),
        "Zc": Zc.realize().numpy().copy(),
        "Zc_pre": Zc_pre
    }

# Step 25 - poincare_distance
def poincare_distance(u, v, eps=1e-7):
    # Calculate squared Euclidean norms of both points.
    u_norm_sq = (u * u).sum(axis=-1)
    v_norm_sq = (v * v).sum(axis=-1)

    # Calculate the squared Euclidean distance.
    diff_norm_sq = ((u - v) * (u - v)).sum(axis=-1)

    # Compute the Poincare distance argument:
    # x = 1 + 2 * ||u - v||^2 /
    #     ((1 - ||u||^2) * (1 - ||v||^2)).
    denominator = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)

    x = 1.0 + 2.0 * diff_norm_sq / denominator

    # Clamp x from below to ensure numerical stability.
    x = x.clip(1.0 + eps, float("inf"))

    # Calculate acosh(x):
    # acosh(x) = log(x + sqrt(x - 1) * sqrt(x + 1)).
    distance = (
        x + (x - 1.0).sqrt() * (x + 1.0).sqrt()
    ).log()

    return distance


def euclidean_distance(u, v, eps=1e-7):
    # Calculate the squared Euclidean distance.
    squared_distance = ((u - v) * (u - v)).sum(axis=-1)

    # Add epsilon before taking the square root.
    return (squared_distance + eps).sqrt()


def tree_softmax_loss(Z, i, j, neg, dist):
    # Get the number of positive edges and latent dimensions.
    m = i.shape[0]
    dim = Z.shape[1]

    # Extract source and destination embeddings for positive edges.
    Z_i = Z[i]
    Z_j = Z[j]

    # Calculate distances for positive edges.
    # Shape: (m,).
    positive_dist = dist(Z_i, Z_j)

    # Extract embeddings of negative destination nodes.
    # Shape: (m, K, dim).
    Z_neg = Z[neg]

    # Expand source embeddings to support broadcasting.
    # Shape: (m, 1, dim).
    Z_i_expanded = Z_i.reshape(m, 1, dim)

    # Calculate distances for all negative samples.
    # Shape: (m, K).
    negative_dist = dist(Z_i_expanded, Z_neg)

    # Construct the distance matrix.
    # Column 0 contains the true neighbor distance.
    # Remaining columns contain negative neighbor distances.
    all_distances = positive_dist.reshape(m, 1).cat(
        negative_dist,
        dim=1
    )

    # Convert distances into logits by negating them.
    logits = -all_distances

    # Apply softmax and calculate the negative log-likelihood
    # of the true neighbor (column 0).
    loss = -logits.log_softmax(axis=1)[:, 0].mean()

    return loss

# Step 26 - riemannian_update
def project_to_ball(Z, eps=1e-5):
    # Calculate the Euclidean norm of every embedding row.
    norm = (Z * Z).sum(axis=1).sqrt().reshape(-1, 1)

    # Calculate the scaling factor.
    # Rows inside the ball remain unchanged.
    # Rows with norm >= 1 - eps are projected onto the boundary.
    scale = (
        (1.0 - eps) / norm.clip(eps, float("inf"))
    ).clip(0.0, 1.0)

    # Return a new projected Tensor without modifying the input.
    return Z * scale


def riemannian_update(Z, grad, lr, eps=1e-5):
    # Calculate the squared Euclidean norm of each embedding row.
    norm_sq = (Z * Z).sum(axis=1).reshape(-1, 1)

    # Calculate the inverse Poincare metric factor:
    # (1 - ||z||^2)^2 / 4.
    metric_factor = ((1.0 - norm_sq) ** 2) / 4.0

    # Scale the Euclidean gradient according to the
    # Riemannian metric and perform the gradient update.
    Z_new = Z - lr * metric_factor * grad

    # Project the updated embeddings back into the Poincare ball.
    return project_to_ball(Z_new, eps=eps)

# Step 27 - train_tree_embedding
def sample_tree_negatives(rng, A, u, K):
    # Convert inputs to NumPy arrays.
    A = np.asarray(A)
    u = np.asarray(u, dtype=np.int64).reshape(-1)

    # Get the number of nodes.
    N = len(A)

    # Draw initial negative samples uniformly.
    neg = rng.integers(
        0,
        N,
        size=(len(u), K)
    )

    # Identify invalid samples:
    # 1. The sampled node is the source node itself.
    # 2. The sampled node is a direct neighbor of the source.
    invalid = (
        (neg == u[:, None])
        | (A[u[:, None], neg] != 0)
    )

    # Redraw exactly the invalid entries until all samples are valid.
    while np.any(invalid):
        count = int(invalid.sum())

        neg[invalid] = rng.integers(
            0,
            N,
            size=count
        )

        # Recheck the newly sampled entries.
        invalid = (
            (neg == u[:, None])
            | (A[u[:, None], neg] != 0)
        )

    return neg


def train_tree_embedding(A, dim=2, geometry="poincare", steps=2000, lr=10.0, K=20, burn=200, burn_factor=0.1, seed=0):

    # Convert the adjacency matrix to a NumPy array.
    A = np.asarray(A)

    # Validate the embedding geometry.
    if geometry not in ("poincare", "euclidean"):
        raise ValueError(
            "geometry must be either 'poincare' or 'euclidean'."
        )

    # Get the number of nodes.
    N = len(A)

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # 1. Initialize node embeddings uniformly near the origin.
    Z_init = rng.uniform(
        -1e-3,
        1e-3,
        size=(N, dim)
    ).astype(np.float32)

    # Convert to a contiguous, realized tinygrad Tensor.
    Z = Tensor(Z_init).contiguous().realize()

    # 2. Extract all directed edges from the adjacency matrix.
    i_np, j_np = np.nonzero(A)

    # Convert edge indices into int32 Tensors.
    i = Tensor(
        i_np.astype(np.int32)
    ).contiguous().realize()

    j = Tensor(
        j_np.astype(np.int32)
    ).contiguous().realize()

    # Number of directed edges.
    m = len(i_np)

    if N == 0 or m == 0:
        raise ValueError("A must contain at least one node and one edge.")

    # Select the appropriate distance function.
    if geometry == "poincare":
        dist = poincare_distance
    else:
        dist = euclidean_distance

    # -------------------------------------------------
    # 3. Define one TinyJit training step.
    # -------------------------------------------------

    def train_step(neg, lr_t):

        # Calculate the tree softmax loss.
        loss = tree_softmax_loss(
            Z,
            i,
            j,
            neg,
            dist
        )

        # Calculate the Euclidean gradient.
        # Scale it by the directed edge count divided by N.
        grad = loss.gradient(Z)[0] * (m / N)

        # Update embeddings according to the selected geometry.
        if geometry == "poincare":

            # Riemannian SGD followed by projection.
            Z.assign(
                riemannian_update(
                    Z,
                    grad,
                    lr_t
                )
            ).realize()

        else:

            # Ordinary Euclidean gradient descent.
            Z.assign(
                Z - lr_t * grad
            ).realize()

        # Return the scalar loss.
        return loss.realize()

    # Compile the training step.
    jit_step = TinyJit(train_step)

    # -------------------------------------------------
    # 4. Training loop.
    # -------------------------------------------------

    history = []

    for t in range(steps):

        # Sample negative destinations for every edge source.
        neg_np = sample_tree_negatives(
            rng,
            A,
            i_np,
            K
        ).astype(np.int32)

        # Convert negative samples into a realized Tensor.
        neg = Tensor(neg_np).contiguous().realize()

        # Apply learning-rate burn-in.
        # Use a smaller learning rate during the first burn steps.
        current_lr = (
            lr * burn_factor
            if t < burn
            else lr
        )

        # Pass the learning rate as a one-element float32 Tensor.
        lr_t = Tensor(
            np.asarray([current_lr], dtype=np.float32)
        ).contiguous().realize()

        # Execute one JIT-compiled training iteration.
        loss = jit_step(neg, lr_t)

        # Record the loss at the required iterations.
        if t == 0 or (t + 1) % 500 == 0 or t == steps - 1:
            history.append(
                (int(t), float(loss.item()))
            )

    # -------------------------------------------------
    # 5. Return final embeddings and loss history.
    # -------------------------------------------------

    Z_numpy = Z.realize().numpy()

    return (
        np.asarray(Z_numpy, dtype=np.float32),
        history
    )

# Step 28 - reconstruction_metrics
def pairwise_distances(Z, geometry):
    # Convert the embedding matrix to float64.
    Z = np.asarray(Z, dtype=np.float64)

    # Calculate pairwise squared Euclidean distances.
    diff = Z[:, None, :] - Z[None, :, :]
    squared_distances = np.sum(diff ** 2, axis=-1)

    # Euclidean geometry.
    if geometry == "euclidean":
        D = np.sqrt(squared_distances)

    # Poincare hyperbolic geometry.
    elif geometry == "poincare":
        # Calculate squared norms of all embeddings.
        norm_sq = np.sum(Z ** 2, axis=1)

        # Calculate the denominator:
        # (1 - ||u||^2) * (1 - ||v||^2).
        denominator = (
            (1.0 - norm_sq[:, None])
            * (1.0 - norm_sq[None, :])
        )

        # Calculate the argument of arccosh.
        x = 1.0 + 2.0 * squared_distances / denominator

        # Clip the argument from below at 1 for numerical stability.
        x = np.maximum(x, 1.0)

        # Calculate the Poincare distance.
        D = np.arccosh(x)

    else:
        raise ValueError(
            "geometry must be either 'euclidean' or 'poincare'."
        )

    # Return the float64 pairwise distance matrix.
    return np.asarray(D, dtype=np.float64)


def reconstruction_metrics(D, A):
    # Convert inputs to NumPy arrays.
    D = np.asarray(D, dtype=np.float64)
    A = np.asarray(A)

    # Get the number of nodes.
    N = A.shape[0]

    # Validate input dimensions.
    if A.shape != (N, N) or D.shape != (N, N):
        raise ValueError("A and D must be square matrices of the same shape.")

    # Store all neighbour ranks and per-node average precisions.
    all_ranks = []
    average_precisions = []

    # Evaluate each node independently.
    for u in range(N):

        # Identify the neighbours of node u.
        # Self-connections are excluded.
        neighbours_mask = (A[u] > 0)
        neighbours_mask[u] = False

        neighbours = np.flatnonzero(neighbours_mask)

        # Skip nodes without neighbours.
        if len(neighbours) == 0:
            continue

        # Identify all non-neighbours, excluding node u itself.
        nonneighbours_mask = (A[u] <= 0)
        nonneighbours_mask[u] = False

        nonneighbours = np.flatnonzero(nonneighbours_mask)

        # Distance from node u to its neighbours.
        neighbour_distances = D[u, neighbours]

        # Distance from node u to all non-neighbours.
        nonneighbour_distances = D[u, nonneighbours]

        # Rank each neighbour:
        # 1 + number of non-neighbours strictly closer to u.
        ranks = (
            1
            + np.sum(
                nonneighbour_distances[:, None]
                < neighbour_distances[None, :],
                axis=0
            )
        )

        # Collect all neighbour ranks for mean-rank calculation.
        all_ranks.extend(ranks.tolist())

        # Sort neighbour ranks in increasing order.
        sorted_ranks = np.sort(ranks)

        # Calculate Average Precision for node u.
        num_neighbours = len(sorted_ranks)

        k = np.arange(1, num_neighbours + 1)

        # Position of the kth neighbour in the combined ordering.
        positions = sorted_ranks + k - 1

        # Precision at each relevant neighbour position.
        precisions = k / positions

        # Mean Average Precision for node u.
        ap = np.mean(precisions)

        average_precisions.append(float(ap))

    # Handle graphs without any nodes having neighbours.
    if len(all_ranks) == 0:
        return 0.0, 0.0

    # Mean rank across all neighbour edges.
    mean_rank = float(np.mean(all_ranks))

    # Mean Average Precision across nodes having neighbours.
    mean_ap = float(np.mean(average_precisions))

    return mean_rank, mean_ap

# Step 29 - classical_mds
def classical_mds(D, dim):
    # Convert the distance matrix to float64.
    D = np.asarray(D, dtype=np.float64)

    # Validate the distance matrix.
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix.")

    # Get the number of nodes.
    N = D.shape[0]

    if dim < 0:
        raise ValueError("dim must be non-negative.")

    # Handle an empty distance matrix or zero-dimensional embedding.
    Z = np.zeros((N, dim), dtype=np.float64)

    if N == 0 or dim == 0:
        return Z

    # 1. Square all pairwise distances.
    D2 = D ** 2

    # 2. Construct the centering matrix:
    # J = I - (1/N) * 11.T.
    J = np.eye(N, dtype=np.float64) - np.ones(
        (N, N), dtype=np.float64
    ) / N

    # 3. Construct the double-centered Gram matrix:
    # B = -0.5 * J @ D2 @ J.
    B = -0.5 * J @ D2 @ J

    # Ensure numerical symmetry.
    B = (B + B.T) / 2.0

    # 4. Compute the eigendecomposition.
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort eigenvalues in decreasing order.
    indices = np.argsort(eigenvalues)[::-1]

    # Select at most 'dim' largest eigenvalues.
    num_components = min(dim, N)
    indices = indices[:num_components]

    selected_values = eigenvalues[indices]
    selected_vectors = eigenvectors[:, indices]

    # Treat negative eigenvalues as zero.
    selected_values = np.maximum(selected_values, 0.0)

    # 5. Construct the MDS coordinates:
    # Z = V @ sqrt(Lambda).
    #
    # Eigenvectors are scaled column-wise by sqrt(eigenvalues).
    Z[:, :num_components] = (
        selected_vectors * np.sqrt(selected_values)[None, :]
    )

    return Z


def average_distortion(D_emb, D_true):
    # Convert both distance matrices to float64.
    D_emb = np.asarray(D_emb, dtype=np.float64)
    D_true = np.asarray(D_true, dtype=np.float64)

    # Validate matrix dimensions.
    if D_emb.ndim != 2 or D_true.ndim != 2:
        raise ValueError("Both distance matrices must be two-dimensional.")

    if D_emb.shape != D_true.shape:
        raise ValueError("Distance matrices must have identical shapes.")

    if D_true.shape[0] != D_true.shape[1]:
        raise ValueError("Distance matrices must be square.")

    # Get the number of nodes.
    N = D_true.shape[0]

    # Use only the strict upper triangle to avoid duplicate pairs.
    upper_i, upper_j = np.triu_indices(N, k=1)

    # Handle matrices with fewer than two nodes.
    if len(upper_i) == 0:
        return 0.0

    # Extract embedding and true distances.
    emb_distances = D_emb[upper_i, upper_j]
    true_distances = D_true[upper_i, upper_j]

    # Calculate pairwise distance ratios.
    rho = emb_distances / true_distances

    # Calculate the mean ratio.
    c = rho.mean()

    # Handle the degenerate case where all embedding distances are zero.
    if c == 0.0:
        return 0.0

    # Calculate average relative distortion after optimal uniform rescaling.
    distortion = np.mean(np.abs(rho / c - 1.0))

    # Return a Python float.
    return float(distortion)

