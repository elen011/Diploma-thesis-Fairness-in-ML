import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

def TTestUpperBound(Z, delta):
    Z = np.array(Z)
    m = len(Z)
    mean_Z = np.mean(Z)
    sample_var = np.var(Z, ddof=1)
    sample_std = np.sqrt(sample_var)
    df = m - 1
    t_crit = t.ppf(1 - delta, df)
    upper_bound = mean_Z + t_crit * (sample_std / np.sqrt(m))
    return upper_bound

def PredictTTest(Z, delta, k):
    Z = np.array(Z)
    m = len(Z)
    mean_Z = np.mean(Z)
    sample_var = np.var(Z, ddof=1)
    sample_std = np.sqrt(sample_var)
    df = k - 1
    t_crit = t.ppf(1 - delta, df)
    predicted_upper_bound = mean_Z + t_crit * sample_std * np.sqrt(1/k)
    safety_factor = 1.1
    predicted_upper_bound *= safety_factor
    return predicted_upper_bound

def TTestCandidateObjective(theta, X0, Y0, X1, Y1, D, delta, epsilon, b, k, lambda_):
    min_size = min(len(X0), len(X1))
    X0 = X0[:min_size]
    Y0 = Y0[:min_size]
    X1 = X1[:min_size]
    Y1 = Y1[:min_size]

    theta = np.array(theta).flatten()
    
    # Prediction = X.dot(theta) works for both 1D and 2D X
    pred0 = X0.dot(theta)
    pred1 = X1.dot(theta)
    
    Z = np.abs(pred0 - Y0) - np.abs(pred1 - Y1)

    ub = max(
        PredictTTest(Z, delta / 2, k),
        PredictTTest(-Z, delta / 2, k)
    )

    if ub <= epsilon:
        X_vals = np.array([x for x, _, _ in D])
        Y_vals = np.array([y for _, y, _ in D])
        preds = X_vals.dot(theta)
        mae = np.mean(np.abs(preds - Y_vals))
        abs_mean_z = np.mean(np.abs(Z))
        result = mae + lambda_ * abs_mean_z
        print(f"✅ Fair theta={theta} → MAE+Penalty={result:.6f}, UB={ub:.6f}")
        return result
    else:
        penalty = b**2 + ub + (lambda_ - 1) * epsilon
        print(f"❌ Unfair theta={theta} → Penalty={penalty:.6f}, UB={ub:.6f} > epsilon={epsilon}")
        return penalty

def TTestDiscrimUpperBound(theta, D, delta, epsilon):
    type_0 = [(X, Y) for X, Y, T in D if T == 0]
    type_1 = [(X, Y) for X, Y, T in D if T == 1]

    min_size = min(len(type_0), len(type_1))

    X0 = np.array([X for X, Y in type_0[:min_size]])
    Y0 = np.array([Y for X, Y in type_0[:min_size]])
    X1 = np.array([X for X, Y in type_1[:min_size]])
    Y1 = np.array([Y for X, Y in type_1[:min_size]])

    theta = np.array(theta).flatten()
    pred0 = X0.dot(theta)
    pred1 = X1.dot(theta)

    Z = np.abs(pred0 - Y0) - np.abs(pred1 - Y1)

    upperbound = max(
        TTestUpperBound(Z, delta / 2),
        TTestUpperBound(-Z, delta / 2)
    )
    return upperbound

def QNDLR(D, delta, epsilon, b, lambda_):
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

    # Detect feature dimension based on first X:
    feature_dim = X0[0].shape[0] if len(X0) > 0 and len(X0[0].shape) > 0 else 1

    theta_init = np.random.uniform(-0.1, 0.1, size=feature_dim)

    objective = lambda theta: TTestCandidateObjective(theta, X0, Y0, X1, Y1, D, delta, epsilon, b, len(D2), lambda_)
    result = minimize(objective, theta_init, method='Nelder-Mead', options={'maxiter': 500})

    if not result.success:
        print("Optimization failed!")
        print("Message:", result.message)
        print("Iterations:", result.nit)
        print("Function evaluations:", result.nfev)
        return "No Solution Found"

    theta_opt = result.x
    bound_value = TTestDiscrimUpperBound(theta_opt, D2, delta, epsilon)
    print(f"TTestDiscrimUpperBound for theta_opt: {bound_value}")

    if bound_value <= epsilon:
        return theta_opt
    else:
        print("TTest bound condition failed")
        return "No Solution Found"


def QNDLR_with_test_eval(D, delta, epsilon, b, lambda_):
    np.random.shuffle(D)


    n = len(D)
    split1 = int(n * 0.2)   # 20% training
    split2 = int(n * 0.6)   # next 40% for bounding
    D1 = D[:split1]         # optimization
    D2 = D[split1:split2]   # fairness constraint
    D3 = D[split2:]         # hold-out test set

    # Split D1 for training
    type_0 = [(X, Y) for X, Y, T in D1 if T == 0]
    type_1 = [(X, Y) for X, Y, T in D1 if T == 1]

    X0 = np.array([X for X, Y in type_0])
    Y0 = np.array([Y for X, Y in type_0])
    X1 = np.array([X for X, Y in type_1])
    Y1 = np.array([Y for X, Y in type_1])

    feature_dim = X0.shape[1] if len(X0.shape) > 1 else 1
    theta_init = np.random.uniform(-0.1, 0.1, size=feature_dim)

    # Optimize on D1 using D2 to evaluate fairness
    objective = lambda theta: TTestCandidateObjective(theta, X0, Y0, X1, Y1, D2, delta, epsilon, b, len(D2), lambda_)
    result = minimize(objective, theta_init, method='Nelder-Mead', options={'maxiter': 500})

    if not result.success:
        print("Optimization failed!")
        return "No Solution Found"

    theta_opt = result.x

    bound_value = TTestDiscrimUpperBound(theta_opt, D2, delta, epsilon)
    print(f"✅ Fairness bound (D2) = {bound_value:.6f} vs ε = {epsilon}")

    if bound_value > epsilon:
        print("❌ Bound constraint violated on D2")
        return "No Solution Found"

    # ➕ Evaluate on D3 (generalization check)
    type_0_test = [(X, Y) for X, Y, T in D3 if T == 0]
    type_1_test = [(X, Y) for X, Y, T in D3 if T == 1]

    X0_test = np.array([X for X, Y in type_0_test])
    Y0_test = np.array([Y for X, Y in type_0_test])
    X1_test = np.array([X for X, Y in type_1_test])
    Y1_test = np.array([Y for X, Y in type_1_test])

    pred0 = X0_test.dot(theta_opt)
    pred1 = X1_test.dot(theta_opt)

    mae_0 = np.mean(np.abs(pred0 - Y0_test))
    mae_1 = np.mean(np.abs(pred1 - Y1_test))
    mae_diff = abs(mae_0 - mae_1)

    print(f"📊 Evaluation on D3 (test set): MAE group 0 = {mae_0:.4f}, group 1 = {mae_1:.4f}, MAE diff = {mae_diff:.4f}")

    return theta_opt
