"""Cold-start exploration. Greedy ranking starves new suppliers (their observed rate is a prior),
so they never build a track record. Compare Greedy vs epsilon-greedy vs Thompson sampling."""
import numpy as np
import pandas as pd


def simulate(sup_cat: pd.DataFrame, T=3000, seed=3) -> pd.DataFrame:
    """One category, one lead per step, 1 supplier chosen per lead. Reward ~ Bernoulli(true_quality)."""
    out = []
    true = sup_cat.true_quality.values
    prior = sup_cat.response_rate.values
    new = (sup_cat.age_days.values < 30)
    n = len(true)
    best = true.max()
    for name in ["Greedy", "Epsilon-greedy (10%)", "Thompson sampling"]:
        rng = np.random.default_rng(seed)
        strength = np.where(new, 2.0, 10.0)  # new suppliers: weak prior = high uncertainty
        a = prior * strength + 1e-9
        b = (1 - prior) * strength + 1e-9
        cum_reward = cum_regret = 0.0
        new_exposure = 0
        star_new = new & (true > 0.75)  # genuinely good new suppliers worth discovering
        star_exposure = 0
        for t in range(T):
            if name == "Greedy":
                i = int(np.argmax(a / (a + b)))
            elif name.startswith("Epsilon"):
                i = int(rng.integers(n)) if rng.random() < 0.1 else int(np.argmax(a / (a + b)))
            else:
                i = int(np.argmax(rng.beta(a, b)))
            r = rng.random() < true[i]
            a[i] += r
            b[i] += 1 - r
            cum_reward += r
            cum_regret += best - true[i]
            new_exposure += new[i]
            star_exposure += star_new[i]
            if t % 100 == 99:
                out.append({"policy": name, "leads": t + 1, "cum_reward": cum_reward, "cum_regret": cum_regret,
                            "new_supplier_share_%": 100 * new_exposure / (t + 1),
                            "good_new_supplier_share_%": 100 * star_exposure / (t + 1)})
    return pd.DataFrame(out)
