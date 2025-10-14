import numpy as np
import math
from scipy.optimize import minimize

def HoeffdingUpperBound(Z: np.ndarray, b: float, delta: float) -> float:
    m = Z.size
    if m == 0:
        raise ValueError("Z must contain at least one element.")
    mean_Z = np.mean(Z)
    bound = b * np.sqrt(np.log(1 / delta) / (2 * m))
    return mean_Z + bound

def PredictHoeffding(Z: np.ndarray, b: float, delta: float, k: int) -> float:
    if len(Z) == 0:
        raise ValueError("Z must contain at least one element.")
    sample_mean = np.mean(Z)
    confidence_term = b * np.sqrt(math.log(1 / delta) / (2 * k))
    prediction = sample_mean + confidence_term
    return prediction

def HoeffdingCandidateObjective(theta, X0, Y0, X1, Y1, D, delta, epsilon, b, k):
    # Make sure X0, Y0, X1, Y1 have the same length
    min_size = min(len(X0), len(X1))
    X0 = X0[:min_size]
    Y0 = Y0[:min_size]
    X1 = X1[:min_size]
    Y1 = Y1[:min_size]

    # Vectorized computation for single feature:
    Z = (X0 * theta[0] - Y0) - (X1 * theta[0] - Y1)

    # Compute Hoeffding upper bound
    ub = max(
        PredictHoeffding(Z, b, delta / 2, k),
        PredictHoeffding(-Z, b, delta / 2, k)
    )

    if ub <= epsilon:
        # Compute MSE on full D
        X_vals = np.array([x for x, _, _ in D])
        Y_vals = np.array([y for _, y, _ in D])
        preds = X_vals * theta[0]  # scalar multiplication since 1D feature
        mse = np.mean((preds - Y_vals) ** 2)
        print(f"✅ Acceptable theta={theta} → MSE={mse:.6f}, UB={ub:.6f}")
        return mse
    else:
        penalty = b**2 + ub - epsilon
        print(f"❌ Rejected theta={theta} → Penalty={penalty:.6f}, UB={ub:.6f} > epsilon={epsilon}")
        return penalty

def HoeffdingDiscrimUpperBound(theta, D, delta, epsilon, b):
    type_0 = [(X, Y) for X, Y, T in D if T == 0]
    type_1 = [(X, Y) for X, Y, T in D if T == 1]

    min_size = min(len(type_0), len(type_1))

    X0 = np.array([X for X, Y in type_0[:min_size]])
    Y0 = np.array([Y for X, Y in type_0[:min_size]])
    X1 = np.array([X for X, Y in type_1[:min_size]])
    Y1 = np.array([Y for X, Y in type_1[:min_size]])

    Z = (X0 * theta[0] - Y0) - (X1 * theta[0] - Y1)

    upperbound = max(
        PredictHoeffding(Z, b, delta / 2, min_size),
        PredictHoeffding(-Z, b, delta / 2, min_size)
    )
    return upperbound

def NDLR(D, delta, epsilon, b):
    np.random.shuffle(D)
    split_index = int(len(D) * 0.2)
    D1 = D[:split_index]
    D2 = D[split_index:]

    type_0 = [(X, Y) for X, Y, T in D1 if T == 0]
    type_1 = [(X, Y) for X, Y, T in D1 if T == 1]

    X0 = np.array([X for X, Y in type_0])
    Y0 = np.array([Y for X, Y in type_0])
    X1 = np.array([X for X, Y in type_1])
    Y1 = np.array([Y for X, Y in type_1])

    feature_dim = 1
    theta_init = np.random.uniform(-0.1, 0.1, size=feature_dim)
    objective = lambda theta: HoeffdingCandidateObjective(theta, X0, Y0, X1, Y1, D1, delta, epsilon, b, len(D2))
    result = minimize(objective, theta_init, method='Nelder-Mead', options={'maxiter':500})


    if not result.success:
        print("Optimization failed!")
        print("Message:", result.message)
        print("Iterations:", result.nit)
        print("Function evaluations:", result.nfev)
        return "No Solution Found"

    theta_opt = result.x
    bound_value = HoeffdingDiscrimUpperBound(theta_opt, D2, delta, epsilon, b)
    print(f"HoeffdingDiscrimUpperBound for theta_opt: {bound_value}")

    if bound_value <= epsilon:
        return theta_opt
    else:
        print("Hoeffding bound condition failed")
        return "No Solution Found"
