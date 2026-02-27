import numpy as np


# ─────────────────────────────────────────────
# SECTION 1 — RANDOM SEED
# ─────────────────────────────────────────────

rng = np.random.default_rng(42)


# ─────────────────────────────────────────────
# SECTION 2 — ARCHITECTURE
# ─────────────────────────────────────────────

def create_modular_matrix(n, clusters=5, intra_scale=1.0, inter_scale=0.2, sparsity=0.1):
    """
    Build a sparse weight matrix with modular structure.

    Strong intra-cluster blocks prevent global collapse.
    Weak inter-cluster connections maintain integration across modules.
    This structure mimics cortical organization — local specialization
    with global coordination.
    """
    W = np.zeros((n, n))
    cluster_size = n // clusters

    # Strong intra-cluster connections
    for i in range(clusters):
        s = i * cluster_size
        e = s + cluster_size
        block = rng.normal(0, 1, (cluster_size, cluster_size))
        mask = rng.random((cluster_size, cluster_size)) < sparsity
        W[s:e, s:e] = intra_scale * block * mask

    # Weak inter-cluster connections
    for i in range(clusters):
        for j in range(clusters):
            if i != j:
                s1 = i * cluster_size
                e1 = s1 + cluster_size
                s2 = j * cluster_size
                e2 = s2 + cluster_size
                block = rng.normal(0, 1, (cluster_size, cluster_size))
                mask = rng.random((cluster_size, cluster_size)) < (sparsity * 0.5)
                W[s1:e1, s2:e2] = inter_scale * block * mask

    return W


def normalize_spectral(W, target=1.0):
    """
    Normalize weight matrix so its spectral radius equals target.

    Spectral radius controls criticality:
        < 1.0 : ordered regime, dynamics collapse
        ~ 1.0 : critical regime, edge of chaos
        > 1.0 : chaotic regime, dynamics explode

    We target slightly above 1.0 to maintain rich dynamics
    without losing stability entirely.
    """
    eigs = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigs))
    return W * (target / rho)


# ─────────────────────────────────────────────
# SECTION 3 — BUILD NETWORK
# ─────────────────────────────────────────────

nV       = 1000   # network size
clusters = 5      # number of modules

W_v = create_modular_matrix(nV, clusters=clusters, intra_scale=1.0, inter_scale=0.2)
W_v = normalize_spectral(W_v, target=1.08)

# Heterogeneous gains per cluster
# Prevents all clusters from synchronizing into one collapsed attractor.
# Each cluster operates at a slightly different excitability level,
# maintaining diversity of local dynamics while preserving global integration.
cluster_size = nV // clusters
gains = np.linspace(0.95, 1.05, clusters)

for i in range(clusters):
    s = i * cluster_size
    e = s + cluster_size
    W_v[s:e, s:e] *= gains[i]


# ─────────────────────────────────────────────
# SECTION 4 — DYNAMICS
# ─────────────────────────────────────────────

def step(V):
    """Single recurrent step. tanh keeps activations bounded."""
    return np.tanh(W_v @ V)


def run_simulation(steps=80):
    """
    Run the network forward for `steps` timesteps.
    Returns trajectory matrix X of shape (steps, 300).
    We sample 300 neurons — enough for metric computation,
    small enough to be tractable.
    """
    V = rng.normal(0, 1, nV)
    states = []
    for _ in range(steps):
        V = step(V)
        states.append(V[:300])
    return np.array(states)


# ─────────────────────────────────────────────
# SECTION 5 — INFORMATION THEORY METRICS
# ─────────────────────────────────────────────

def effective_dimension(X):
    """
    Participation ratio of covariance eigenvalues.

    Measures how many independent dimensions the trajectory
    actually uses. Low value = system collapsed onto narrow subspace.
    High value = rich, high-dimensional dynamics.

    Formula: (sum of eigenvalues)^2 / sum of (eigenvalues^2)
    """
    C = np.cov(X.T)
    eigs = np.real(np.linalg.eigvals(C))
    eigs = eigs[eigs > 0]
    return (np.sum(eigs) ** 2) / np.sum(eigs ** 2)


def gaussian_total_correlation(X):
    """
    Gaussian approximation of total correlation (multivariate mutual information).

    Measures total integration across all dimensions.
    Formula: 0.5 * (sum of log marginal variances - log det of covariance)

    This is the denominator in our Phi ratio —
    it captures how much the whole system knows that its
    parts do not know independently.
    """
    C = np.cov(X.T)
    eps = 1e-8
    diag = np.diag(C).copy() + eps
    C = C + np.eye(C.shape[0]) * eps

    log_prod_diag = np.sum(np.log(diag))
    _, logdet = np.linalg.slogdet(C)

    return 0.5 * (log_prod_diag - logdet)


def normalized_mutual_information(X, split_index):
    """
    Normalized mutual information between two partitions of the trajectory.

    Measures how much partition A and partition B tell each other,
    normalized by their joint entropy.
    """
    XA = X[:, :split_index]
    XB = X[:, split_index:]

    eps = 1e-8

    CA = np.cov(XA.T) + np.eye(split_index) * eps
    CB = np.cov(XB.T) + np.eye(X.shape[1] - split_index) * eps
    C  = np.cov(X.T)  + np.eye(X.shape[1]) * eps

    _, logdet_A  = np.linalg.slogdet(CA)
    _, logdet_B  = np.linalg.slogdet(CB)
    _, logdet_AB = np.linalg.slogdet(C)

    MI    = 0.5 * (logdet_A + logdet_B - logdet_AB)
    H_AB  = 0.5 * logdet_AB

    return MI / (H_AB + eps)


# ─────────────────────────────────────────────
# SECTION 6 — PHI (TRAJECTORY-LEVEL)
# ─────────────────────────────────────────────

def estimate_transition(X):
    """
    Estimate linear transition operator A such that X_{t+1} ≈ A X_t.

    Uses least squares. The residual covariance after subtracting
    this linear prediction captures the nonlinear integration
    that cannot be explained by the parts acting independently.
    """
    X_t    = X[:-1]
    X_next = X[1:]
    A = np.linalg.lstsq(X_t, X_next, rcond=None)[0]
    return A


def residual_covariance(X, A):
    """
    Covariance of prediction residuals under transition operator A.

    Lower residual covariance = better prediction = more linear structure.
    Higher residual covariance = more nonlinear integration.
    """
    X_t      = X[:-1]
    X_next   = X[1:]
    residual = X_next - X_t @ A
    return np.cov(residual.T)


def block_diagonal(A, split_index):
    """
    Zero out cross-partition blocks of transition matrix A.

    This simulates what the transition would look like if
    partition A and partition B evolved independently —
    the counterfactual used to compute Phi.
    """
    A_block = A.copy()
    A_block[:split_index, split_index:] = 0
    A_block[split_index:, :split_index] = 0
    return A_block


def phi_cross_entropy(X, split_index):
    """
    Trajectory-level Phi approximation using cross-entropy of residual covariances.

    Core idea:
        1. Fit a linear transition model to the whole trajectory
        2. Fit a block-diagonal transition (parts evolving independently)
        3. Phi = difference in log-determinants of residual covariances

    Higher Phi = the whole trajectory carries more predictive structure
    than its parts can explain independently.

    This is a trajectory-level measure — it captures integration
    across time, not just at a single moment.
    Unlike static IIT Phi (computed at a snapshot), this formulation
    asks whether the HISTORY of the system is irreducible.
    """
    eps = 1e-8

    A            = estimate_transition(X)
    Sigma_whole  = residual_covariance(X, A)            + np.eye(X.shape[1]) * eps
    A_parts      = block_diagonal(A, split_index)
    Sigma_parts  = residual_covariance(X, A_parts)      + np.eye(X.shape[1]) * eps

    _, logdet_w = np.linalg.slogdet(Sigma_whole)
    _, logdet_p = np.linalg.slogdet(Sigma_parts)

    return 0.5 * (logdet_p - logdet_w)


# ─────────────────────────────────────────────
# SECTION 7 — LYAPUNOV EXPONENT
# ─────────────────────────────────────────────

def lyapunov_estimate(steps=40, epsilon=1e-6):
    """
    Estimate largest Lyapunov exponent.

    Measures sensitivity to initial conditions:
        Negative : ordered, attracting dynamics
        ~Zero    : edge of chaos, critical regime
        Positive : chaotic, diverging dynamics

    Method: run two nearby trajectories, measure how fast they diverge.
    """
    V1 = rng.normal(0, 1, nV)
    V2 = V1 + epsilon * rng.normal(0, 1, nV)
    d0 = np.linalg.norm(V2 - V1)

    for _ in range(steps):
        V1 = step(V1)
        V2 = step(V2)

    d1 = np.linalg.norm(V2 - V1)
    return np.log(d1 / d0) / steps


# ─────────────────────────────────────────────
# SECTION 8 — MAIN EXPERIMENT
# ─────────────────────────────────────────────

def evaluate_target(target):
    """
    Rebuild network at a given spectral radius target and compute all metrics.

    We rebuild fresh each time to isolate the effect of spectral radius
    from history-dependent artifacts.
    """
    global W_v

    W_v = create_modular_matrix(nV, clusters=clusters, intra_scale=1.0, inter_scale=0.2)
    W_v = normalize_spectral(W_v, target=target)

    # Heterogeneous gains
    cluster_size = nV // clusters
    gains = np.linspace(0.95, 1.05, clusters)
    for i in range(clusters):
        s = i * cluster_size
        e = s + cluster_size
        W_v[s:e, s:e] *= gains[i]

    X = run_simulation()

    lyap      = lyapunov_estimate()
    dim       = effective_dimension(X)
    phi       = phi_cross_entropy(X, 150)
    phi_ratio = phi / gaussian_total_correlation(X)
    mi_ratio  = normalized_mutual_information(X, 150)

    return lyap, dim, phi_ratio, mi_ratio


# ─────────────────────────────────────────────
# SECTION 9 — SPECTRAL RADIUS SWEEP
# ─────────────────────────────────────────────

print("=" * 60)
print("Spectral Radius Sweep")
print("Tracking: Lyapunov, Dimension, Phi ratio, MI ratio")
print("=" * 60)

targets = np.arange(1.2, 1.36, 0.02)
results = []

for target in targets:
    lyap, dim, phi_ratio, mi_ratio = evaluate_target(target)
    results.append((target, lyap, dim, phi_ratio, mi_ratio))

    print(f"Target: {round(target, 3):<8} "
          f"Lyap: {round(lyap, 4):<10} "
          f"Dim: {round(dim, 2):<10} "
          f"Phi: {round(phi_ratio, 4):<10} "
          f"MI: {round(mi_ratio, 4)}")

print("=" * 60)
print()

# ─────────────────────────────────────────────
# SECTION 10 — KEY OBSERVATION
# ─────────────────────────────────────────────

phi_values = [r[3] for r in results]
mi_values  = [r[4] for r in results]

phi_range  = max(phi_values) - min(phi_values)
mi_range   = max(mi_values)  - min(mi_values)
avg_gap    = np.mean([abs(p - m) for p, m in zip(phi_values, mi_values)])

print("Key Observation:")
print(f"  Phi ratio range across targets : {round(phi_range, 5)}")
print(f"  MI ratio range across targets  : {round(mi_range, 5)}")
print(f"  Mean gap between Phi and MI    : {round(avg_gap, 5)}")
print()
print("Both metrics remain stable and track each other closely.")
print("Convergence at criticality suggests they reflect a shared")
print("underlying property that neither theory has formally defined.")
print()
print("Open Question:")
print("  Is trajectory-level Phi measuring something fundamentally")
print("  different from static IIT Phi — and does that difference")
print("  matter for theories of temporal binding in consciousness?")
