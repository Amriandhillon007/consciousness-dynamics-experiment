import numpy as np

# ─────────────────────────────────────────────
# SECTION 1 — RANDOM SEED
# ─────────────────────────────────────────────
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# SECTION 2 — ARCHITECTURE
# ─────────────────────────────────────────────

def create_modular_matrix(n, clusters=5, intra_scale=1.0, inter_scale=0.2, sparsity=0.1):
    W = np.zeros((n, n))
    cluster_size = n // clusters
    for i in range(clusters):
        s = i * cluster_size
        e = s + cluster_size
        block = rng.normal(0, 1, (cluster_size, cluster_size))
        mask = rng.random((cluster_size, cluster_size)) < sparsity
        W[s:e, s:e] = intra_scale * block * mask
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
    eigs = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigs))
    if rho == 0:
        return W
    return W * (target / rho)

# ─────────────────────────────────────────────
# SECTION 3 — BUILD NETWORK
# ─────────────────────────────────────────────

nV_default = 1000
clusters_default = 5
nV = nV_default
clusters = clusters_default
W_v = None


def build_network(target=1.08):
    global W_v, nV, clusters
    W_v = create_modular_matrix(nV, clusters=clusters, intra_scale=1.0, inter_scale=0.2)
    W_v = normalize_spectral(W_v, target=target)
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
    return np.tanh(W_v @ V)


def run_simulation(steps=80):
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
    C = np.cov(X.T)
    eigs = np.real(np.linalg.eigvals(C))
    eigs = eigs[eigs > 0]
    if eigs.size == 0:
        return 0.0
    return (np.sum(eigs) ** 2) / np.sum(eigs ** 2)


def gaussian_total_correlation(X):
    C = np.cov(X.T)
    eps = 1e-8
    diag = np.diag(C).copy() + eps
    C = C + np.eye(C.shape[0]) * eps
    log_prod_diag = np.sum(np.log(diag))
    _, logdet = np.linalg.slogdet(C)
    return 0.5 * (log_prod_diag - logdet)


def normalized_mutual_information(X, split_index):
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
    X_t    = X[:-1]
    X_next = X[1:]
    A = np.linalg.lstsq(X_t, X_next, rcond=None)[0]
    return A


def residual_covariance(X, A):
    X_t      = X[:-1]
    X_next   = X[1:]
    residual = X_next - X_t @ A
    return np.cov(residual.T)


def block_diagonal(A, split_index):
    A_block = A.copy()
    A_block[:split_index, split_index:] = 0
    A_block[split_index:, :split_index] = 0
    return A_block


def phi_cross_entropy(X, split_index):
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
    build_network(target=target)
    X = run_simulation(steps=80)
    lyap      = lyapunov_estimate()
    dim       = effective_dimension(X)
    phi       = phi_cross_entropy(X, 150)
    phi_ratio = phi / gaussian_total_correlation(X)
    mi_ratio  = normalized_mutual_information(X, 150)
    return lyap, dim, phi_ratio, mi_ratio


# If run as script, provide a quick CLI option
if __name__ == '__main__':
    print('Module executed as script; call functions or use runner.')
