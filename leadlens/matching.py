"""Supplier matching (v1 heuristic, used for the live demo) + routing policy."""
import pandas as pd

W = {"category": 0.55, "location": 0.20, "responsiveness": 0.15, "verified": 0.10}


def tier(p: float) -> str:
    return "Hot" if p >= 0.7 else "Warm" if p >= 0.45 else "Cold"


def match(suppliers: pd.DataFrame, cat_probs: dict, city: str, state: str, top_k=5) -> pd.DataFrame:
    s = suppliers.copy()
    s["m_category"] = s["category"].map(lambda c: cat_probs.get(c, 0.0))
    s = s[s["m_category"] > 0].copy()
    s["m_location"] = [1.0 if c == city else 0.6 if st == state else 0.2 for c, st in zip(s.city, s.state)]
    s["score"] = 100 * (W["category"] * s["m_category"] + W["location"] * s["m_location"]
                        + W["responsiveness"] * s["response_rate"] + W["verified"] * s["verified"])
    return s.sort_values("score", ascending=False).head(top_k)[
        ["supplier_id", "name", "category", "city", "verified", "response_rate", "score"]].reset_index(drop=True)


def routing_policy(p_genuine: float, n_matches: int) -> str:
    t = tier(p_genuine)
    if t == "Hot":
        return f"HOT → send to top {min(5, n_matches)} suppliers instantly + push notification"
    if t == "Warm":
        return f"WARM → send to top {min(3, n_matches)} suppliers"
    return "COLD → hold; ask buyer a clarifying question first (see below), then release to top 3"
