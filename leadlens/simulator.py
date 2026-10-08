"""Marketplace A/B simulator: current 'broadcast' routing vs LeadLens tiered routing.

ASSUMPTION-DRIVEN: outcomes come from `data.connect_prob`; buyer clarification behaviour is a
user-set assumption. It sizes trade-offs and guardrails - it is not evidence of real-world lift.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import connect_prob


@dataclass
class Assumptions:
    hot_t: float = 0.70
    warm_t: float = 0.45
    k_hot: int = 5
    k_warm: int = 3
    k_control: int = 5
    k_after_clarify: int = 3
    clarify_resp_genuine: float = 0.60   # P(genuine buyer answers the clarifying question)
    clarify_resp_junk: float = 0.20      # P(non-genuine answers)
    avg_connect_pos_decay: float = 0.18


def _expected(genuine, urgent, k, tq=0.65, loc=0.6, ver=0.6):
    """Expected connections when a lead is sent to k typical suppliers (positions 0..k-1)."""
    return float(sum(connect_prob(genuine, loc, tq, ver, urgent, pos) for pos in range(k)))


def run(te: pd.DataFrame, a: Assumptions, n_boot=300, seed=11) -> dict:
    rng = np.random.default_rng(seed)
    g, u, p = te.genuine.values, te.urgent.values, te.p_genuine.values
    n = len(te)
    # --- control: every lead to k_control suppliers
    ctrl_deliv = np.full(n, a.k_control, float)
    ctrl_conn = np.array([_expected(g[i], u[i], a.k_control) for i in range(n)])
    # --- treatment: tiered
    hot, warm = p >= a.hot_t, (p >= a.warm_t) & (p < a.hot_t)
    cold = ~(hot | warm)
    tr_deliv = np.where(hot, a.k_hot, np.where(warm, a.k_warm, 0)).astype(float)
    tr_conn = np.zeros(n)
    lost_genuine = np.zeros(n)
    for i in range(n):
        if hot[i]:
            tr_conn[i] = _expected(g[i], u[i], a.k_hot)
        elif warm[i]:
            tr_conn[i] = _expected(g[i], u[i], a.k_warm)
        else:  # cold: clarify first
            resp = a.clarify_resp_genuine if g[i] else a.clarify_resp_junk
            tr_deliv[i] = resp * a.k_after_clarify
            tr_conn[i] = resp * _expected(g[i], u[i], a.k_after_clarify)
            lost_genuine[i] = g[i] * (1 - resp)
    junk_ctrl = ((1 - g) * ctrl_deliv).sum() / ctrl_deliv.sum()
    junk_tr = ((1 - g) * tr_deliv).sum() / tr_deliv.sum()

    def stats(idx):
        return {"delivered": tr_deliv[idx].sum() / ctrl_deliv[idx].sum() - 1,
                "conn_per_delivered": (tr_conn[idx].sum() / tr_deliv[idx].sum()) / (ctrl_conn[idx].sum() / ctrl_deliv[idx].sum()) - 1,
                "total_conn": tr_conn[idx].sum() / ctrl_conn[idx].sum() - 1,
                "junk_share": (((1 - g[idx]) * tr_deliv[idx]).sum() / tr_deliv[idx].sum()) -
                              (((1 - g[idx]) * ctrl_deliv[idx]).sum() / ctrl_deliv[idx].sum())}
    boots = pd.DataFrame([stats(rng.integers(0, n, n)) for _ in range(n_boot)])
    ci = boots.quantile([0.025, 0.5, 0.975]).T
    ci.columns = ["low", "mid", "high"]
    return {"summary": ci, "tier_mix": {"Hot": hot.mean(), "Warm": warm.mean(), "Cold": cold.mean()},
            "junk_control": junk_ctrl, "junk_treatment": junk_tr,
            "genuine_lost_share": lost_genuine.sum() / g.sum()}


def sweep_hot_threshold(te, base: Assumptions, ts=(0.5, 0.6, 0.7, 0.8)) -> pd.DataFrame:
    rows = []
    for t in ts:
        a = Assumptions(**{**base.__dict__, "hot_t": t, "warm_t": min(base.warm_t, t - 0.05)})
        r = run(te, a, n_boot=40)["summary"]["mid"]
        rows.append({"hot_threshold": t, "Δ leads delivered": r["delivered"], "Δ connects / lead": r["conn_per_delivered"],
                     "Δ total connects": r["total_conn"], "Δ junk share (pp)": 100 * r["junk_share"]})
    return pd.DataFrame(rows)


def best_policy(te, volume_floor=-0.05):
    """Grid search the routing policy: maximise connects-per-delivered-lead subject to a guardrail that
    total connections (supplier value / revenue proxy) fall by no more than `volume_floor`."""
    rows = []
    for hot_t in (0.6, 0.7, 0.8):
        for warm_t in (0.3, 0.4, 0.5):
            for k_warm in (3, 4, 5):
                for k_clar in (3, 5):
                    a = Assumptions(hot_t=hot_t, warm_t=warm_t, k_warm=k_warm, k_after_clarify=k_clar)
                    s = run(te, a, n_boot=30)["summary"]["mid"]
                    rows.append({"hot_t": hot_t, "warm_t": warm_t, "k_warm": k_warm, "k_after_clarify": k_clar,
                                 "Δ connects/lead": s["conn_per_delivered"], "Δ total connects": s["total_conn"],
                                 "Δ junk share (pp)": 100 * s["junk_share"]})
    g = pd.DataFrame(rows)
    ok = g[g["Δ total connects"] >= volume_floor].sort_values("Δ connects/lead", ascending=False)
    return g, ok
