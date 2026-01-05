import numpy as np
import pandas as pd

def generate_synthetic_marketing_data(n=3000, seed=42):
    np.random.seed(seed)

    income = np.random.normal(50, 15, n)
    loyalty = np.random.uniform(0, 1, n)

    logits = 0.08 * income + 2.5 * loyalty - 6
    prob_treat = 1 / (1 + np.exp(-logits))
    treatment = np.random.binomial(1, prob_treat)

    true_ate = 5
    noise = np.random.normal(0, 2, n)

    spend = 0.5 * income + 10 * loyalty + true_ate * treatment + noise

    df = pd.DataFrame({
        "income": income,
        "loyalty": loyalty,
        "treatment": treatment,
        "spend": spend
    })

    return df, true_ate
