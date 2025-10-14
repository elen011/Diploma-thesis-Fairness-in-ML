# %%
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

# %%
def TTestUpperBound(Z, delta):

    Z = np.array(Z)
    m = len(Z)
    mean_Z = np.mean(Z)
    sample_var = np.var(Z, ddof=1)  # ddof=1 for sample variance
    sample_std = np.sqrt(sample_var)
    
    # Degrees of freedom
    df = m - 1
    
    # t critical value for one-sided test at confidence level delta
    t_crit = t.ppf(1 - delta, df)
    
    # Calculate upper confidence bound
    upper_bound = mean_Z + t_crit * (sample_std / np.sqrt(m))
    
    return upper_bound

# %%
def PredictTTest(Z, delta, k):

    Z = np.array(Z)
    m = len(Z)
    mean_Z = np.mean(Z)
    sample_var = np.var(Z, ddof=1)
    sample_std = np.sqrt(sample_var)
    
    # Degrees of freedom for new sample
    df = k - 1
    
    # Critical t-value for new sample size at confidence level delta
    t_crit = t.ppf(1 - delta, df)
    
    # Conservative scaling factor to avoid underestimation:
    # Here we use the original critical t value but scale the standard error by the ratio sqrt(m/k),
    # which predicts the new standard error for sample size k.
    # To be conservative (over-predict), we might add a small buffer or keep the original critical t.
    
    # Predicted upper bound:
    predicted_upper_bound = mean_Z + t_crit * sample_std * np.sqrt(1/k)
    
    # Optionally, multiply error term by a safety factor > 1 to over-predict, for example 1.1
    safety_factor = 1.1
    predicted_upper_bound *= safety_factor
    
    return predicted_upper_bound

# %%
def TTestCandidateObjective(theta, X0, Y0, X1, Y1, D, delta, epsilon, b, k, lambda_):
   

    # Match group sizes
    min_size = min(len(X0), len(X1))
    X0 = X0[:min_size]
    Y0 = Y0[:min_size]
    X1 = X1[:min_size]
    Y1 = Y1[:min_size]

    # Fairness gap (pairwise prediction difference)
    Z = (X0 * theta[0] - Y0) - (X1 * theta[0] - Y1)

    # Hoeffding upper bound on discrimination
    ub = max(
        PredictTTest(Z, delta / 2, k),
        PredictTTest(-Z, delta / 2, k)
    )

    if ub <= epsilon:
        # Compute MSE
        X_vals = np.array([x for x, _, _ in D])
        Y_vals = np.array([y for _, y, _ in D])
        preds = X_vals * theta[0]
        mse = np.mean((preds - Y_vals) ** 2)

        # Fairness regularization: average |Z|
        abs_mean_z = np.mean(np.abs(Z))

        result = mse + lambda_ * abs_mean_z
        print(f"✅ Fair theta={theta} → MSE+Penalty={result:.6f}, UB={ub:.6f}")
        return result
    else:
        # Fixed penalty when unfair
        penalty = b**2 + ub + (lambda_ - 1) * epsilon
        print(f"❌ Unfair theta={theta} → Penalty={penalty:.6f}, UB={ub:.6f} > epsilon={epsilon}")
        return penalty

# %%
def TTestDiscrimUpperBound(theta, D, delta, epsilon):
    type_0 = [(X, Y) for X, Y, T in D if T == 0]
    type_1 = [(X, Y) for X, Y, T in D if T == 1]

    min_size = min(len(type_0), len(type_1))

    X0 = np.array([X for X, Y in type_0[:min_size]])
    Y0 = np.array([Y for X, Y in type_0[:min_size]])
    X1 = np.array([X for X, Y in type_1[:min_size]])
    Y1 = np.array([Y for X, Y in type_1[:min_size]])

    Z = (X0 * theta[0] - Y0) - (X1 * theta[0] - Y1)

    upperbound = max(
        TTestUpperBound(Z, delta / 2),
        TTestUpperBound(-Z, delta / 2)
    )
    return upperbound

# %%
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

    feature_dim = 1
    theta_init = np.random.uniform(-0.1, 0.1, size=feature_dim)
    objective = lambda theta: TTestCandidateObjective(theta, X0, Y0, X1, Y1, D, delta, epsilon, b, len(D2), lambda_) 
    result = minimize(objective, theta_init, method='Nelder-Mead', options={'maxiter':500})


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

# %%



