"""Supplier ranker trained on (simulated) connection outcomes vs. a hand-weighted heuristic."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from .data import connect_prob, loc_level

FEATS = ["m_category", "loc", "response_rate", "verified", "log_age", "p_genuine", "urgent", "completeness", "is_new"]


def heuristic(df: pd.DataFrame) -> pd.Series:
    """Baseline: the v1 weighted-sum rule."""
    return 0.55 * df.m_category + 0.20 * df["loc"] + 0.15 * df.response_rate + 0.10 * df.verified


def candidates(enq: pd.DataFrame, sup: pd.DataFrame, k=8, seed=1) -> pd.DataFrame:
    """For each enquiry sample k suppliers of its (true) category, attach features and simulated truth."""
    rng = np.random.default_rng(seed)
    by_cat = {c: g for c, g in sup.groupby("category")}
    rows = []
    for r in enq.itertuples():
        g = by_cat[r.category]
        for s in g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)].itertuples():
            loc = loc_level(r.city, r.state, s.city, s.state)
            rows.append((r.enquiry_id, s.supplier_id, 1.0, loc, s.response_rate, s.verified,
                         np.log1p(s.age_days), r.p_genuine, r.urgent, r.completeness, int(s.age_days < 30),
                         float(connect_prob(r.genuine, loc, s.true_quality, s.verified, r.urgent)), r.genuine,
                         s.true_quality))
    c = pd.DataFrame(rows, columns=["enquiry_id", "supplier_id", "m_category", "loc", "response_rate", "verified",
                                    "log_age", "p_genuine", "urgent", "completeness", "is_new", "true_p", "genuine", "true_quality"])
    c["connected"] = (rng.random(len(c)) < c.true_p).astype(int)
    return c


def ndcg_at_k(df: pd.DataFrame, score_col: str, k=3) -> float:
    """NDCG@k using the ground-truth connect probability as graded relevance."""
    out = []
    disc = 1 / np.log2(np.arange(2, k + 2))
    for _, g in df.groupby("enquiry_id"):
        got = g.sort_values(score_col, ascending=False).true_p.values[:k]
        ideal = np.sort(g.true_p.values)[::-1][:k]
        out.append((got * disc[:len(got)]).sum() / (ideal * disc[:len(ideal)]).sum())
    return float(np.mean(out))


def fit_and_eval(train_c: pd.DataFrame, test_c: pd.DataFrame, seed=45):
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, random_state=seed)
    m.fit(train_c[FEATS], train_c.connected)
    t = test_c.copy()
    t["ml"] = m.predict_proba(t[FEATS])[:, 1]
    t["heur"] = heuristic(t)
    t["rand"] = np.random.default_rng(seed).random(len(t))
    res = pd.DataFrame({"ranker": ["Random", "Heuristic (v1 weights)", "Learned (GBM)"],
                        "NDCG@3": [ndcg_at_k(t, "rand"), ndcg_at_k(t, "heur"), ndcg_at_k(t, "ml")]})
    return m, res, t
