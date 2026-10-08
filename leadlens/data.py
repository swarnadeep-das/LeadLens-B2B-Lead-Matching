"""Synthetic B2B marketplace world (seeded). NOT real IndiaMART data.

Contains: English + Hinglish buyer enquiries (with typos), a supplier base with a
cold-start segment, and a ground-truth outcome model (`connect_prob`) used to simulate
supplier-buyer call connections. Because outcomes are simulated, every downstream result
is *assumption-driven* and is labelled as such in the app and docs.
"""
import numpy as np
import pandas as pd

CATS = {
    "Industrial Valves": ["ball valve", "gate valve", "butterfly valve", "SS valve", "relief valve"],
    "Packaging Boxes": ["corrugated box", "carton box", "mono carton", "packaging box", "printed box"],
    "Solar Panels": ["solar panel", "mono perc module", "solar inverter", "rooftop solar kit", "solar module"],
    "CNC Machines": ["CNC lathe", "VMC machine", "CNC milling machine", "CNC router", "turning center"],
    "Cotton Fabric": ["cotton fabric", "poplin fabric", "denim fabric", "knitted fabric", "greige cloth"],
    "Basmati Rice": ["basmati rice", "1121 basmati", "steam rice", "sella rice", "long grain rice"],
    "LED Lights": ["LED bulb", "LED panel light", "street light", "LED tube light", "flood light"],
    "Electrical Cables": ["copper cable", "armoured cable", "house wire", "FRLS wire", "power cable"],
    "Safety Equipment": ["safety helmet", "safety shoes", "fire extinguisher", "safety harness", "hand gloves"],
    "Pharma Packaging": ["blister pack", "pharma bottle", "aluminium foil", "pharma carton", "HDPE container"],
}
CITIES = [("Delhi", "Delhi"), ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Surat", "Gujarat"),
          ("Ahmedabad", "Gujarat"), ("Chennai", "Tamil Nadu"), ("Coimbatore", "Tamil Nadu"),
          ("Bengaluru", "Karnataka"), ("Kolkata", "West Bengal"), ("Ludhiana", "Punjab"),
          ("Jaipur", "Rajasthan"), ("Hyderabad", "Telangana")]
STATE = dict(CITIES)
UNITS = ["pcs", "units", "kg", "tons", "meters"]

EN_GOOD = ["Need {q} {u} of {p} for {use}. Delivery to {c}. Please share best price and specs.",
           "Looking for {p}, quantity {q} {u}, required {when}. Location {c}. Send quotation.",
           "We are a {biz} in {c}. Require {p} - {q} {u} monthly. Kindly share catalogue and rates.",
           "Interested in bulk order of {p}, approx {q} {u}, budget around Rs {b}. Need {when}."]
HI_GOOD = ["{p} chahiye {q} {u}, {c} me delivery karni hai. Rate aur specs bhejo.",
           "Mujhe {p} ka rate batao, quantity {q} {u}, {when} chahiye. Location {c}.",
           "Hum {c} ke {biz} hain. {p} {q} {u} har mahine chahiye. Catalogue bhejiye.",
           "Bulk order hai {p} ka, around {q} {u}, budget Rs {b}. {c} me {when}."]
VAGUE_EN = ["price?", "need {p}", "send details", "rate of {p}", "interested", "call me", "{p} price list pls"]
VAGUE_HI = ["rate batao", "{p} chahiye", "details bhejo", "{p} ka price?", "call karo", "available hai?"]
BIZ = ["trader", "manufacturer", "contractor", "retailer", "exporter", "distributor"]
USE = ["our factory", "a new project", "resale", "export order", "a government tender", "our showroom"]
WHEN_EN = ["this week", "within 15 days", "next month", "urgently", "immediately"]
WHEN_HI = ["is hafte", "15 din me", "agle mahine", "urgent", "abhi"]
URGENT_WORDS = ("urgent", "immediate", "this week", "asap", "abhi", "is hafte", "jaldi")


def _typo(rng, s):
    w = s.split()
    i = int(rng.integers(len(w)))
    if len(w[i]) > 4:
        j = int(rng.integers(1, len(w[i]) - 2))
        w[i] = w[i][:j] + w[i][j + 1] + w[i][j] + w[i][j + 2:]
    return " ".join(w)


def make_enquiries(n=6000, seed=45) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = list(CATS)
    rows = []
    for i in range(n):
        cat = cats[rng.integers(len(cats))]
        p = CATS[cat][rng.integers(5)]
        city = CITIES[rng.integers(len(CITIES))][0]
        hi = rng.random() < 0.35
        vague = rng.random() < 0.38
        verified = int(rng.random() < 0.55)
        prior = int(rng.poisson(1.5 if verified else 0.6))
        has_b = bool((not vague) and rng.random() < 0.4)
        if vague:
            text = (VAGUE_HI if hi else VAGUE_EN)[rng.integers(6 if hi else 7)].format(p=p)
        else:
            text = (HI_GOOD if hi else EN_GOOD)[rng.integers(4)].format(
                q=int(rng.choice([50, 100, 200, 500, 1000, 5000])), u=rng.choice(UNITS), p=p,
                use=USE[rng.integers(6)], c=city, biz=BIZ[rng.integers(6)],
                when=(WHEN_HI if hi else WHEN_EN)[rng.integers(5)], b=int(rng.choice([50000, 100000, 500000, 1000000])))
        if rng.random() < 0.22:
            text = _typo(rng, text)
        urgent = int(any(w in text.lower() for w in URGENT_WORDS))
        z = -1.4 + 2.0 * (not vague) + 0.9 * verified + 0.35 * min(prior, 4) + 0.6 * has_b + rng.normal(0, 0.8)
        genuine = int(rng.random() < 1 / (1 + np.exp(-z)))
        rows.append((i, text, cat, city, STATE[city], verified, prior, int(has_b), "Hinglish" if hi else "English",
                     urgent, genuine))
    return pd.DataFrame(rows, columns=["enquiry_id", "text", "category", "city", "state", "buyer_verified",
                                       "prior_enquiries", "has_budget", "language", "urgent", "genuine"])


def make_suppliers(n=300, seed=7) -> pd.DataFrame:
    """~25% are 'new' (<30 days). Observed response_rate for new suppliers is the category prior
    (cold start); `true_quality` is the hidden propensity that actually drives outcomes."""
    rng = np.random.default_rng(seed)
    cats = list(CATS)
    rows = []
    for i in range(n):
        cat = cats[i % len(cats)]
        city = CITIES[rng.integers(len(CITIES))][0]
        age = int(rng.exponential(150)) + 1
        q = float(rng.beta(4, 2))
        obs = float(np.clip(q + rng.normal(0, 0.08), 0.05, 0.99)) if age >= 30 else 0.65
        rows.append((f"S{i:03d}", f"{city} {cat.split()[0]} Works {i}", cat, city, STATE[city],
                     int(rng.random() < 0.6), round(obs, 3), age, round(q, 3)))
    return pd.DataFrame(rows, columns=["supplier_id", "name", "category", "city", "state", "verified",
                                       "response_rate", "age_days", "true_quality"])


def loc_level(buyer_city, buyer_state, s_city, s_state):
    return 1.0 if buyer_city == s_city else 0.6 if buyer_state == s_state else 0.2


def connect_prob(genuine, loc, true_quality, verified, urgent, position=0):
    """Ground-truth P(call connects). Genuine buyers + responsive, nearby, verified suppliers
    connect more; responsiveness matters more for urgent needs; later list positions get fewer calls."""
    z = (-2.3 + 1.8 * np.asarray(genuine) + 1.9 * (np.asarray(true_quality) - 0.5) * (1 + np.asarray(urgent))
         + 0.6 * (np.asarray(loc) - 0.6) + 0.4 * np.asarray(verified) - 0.18 * np.asarray(position))
    return 1 / (1 + np.exp(-z))
