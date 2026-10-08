"""NLP + scoring models, with an ablation so each component has to earn its place."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from . import slots as S
from .data import STATE


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    sl = d["text"].map(S.extract)
    for k in S.SLOTS:
        d[f"slot_{k}"] = sl.map(lambda x, k=k: int(x[k]))
    d["completeness"] = sl.map(S.completeness)
    d["n_words"] = d["text"].str.split().str.len()
    return d


SIGNALS = ["buyer_verified", "prior_enquiries"]
SLOTF = ["completeness"] + [f"slot_{k}" for k in S.SLOTS] + ["n_words"]


def _dense(d, cols):
    scale = {"prior_enquiries": 4, "n_words": 25}
    return csr_matrix(np.column_stack([d[c].astype(float) / scale.get(c, 1) for c in cols]))


@dataclass
class Bundle:
    cat_vec: TfidfVectorizer
    cat_clf: LogisticRegression
    q_vec: TfidfVectorizer
    q_clf: LogisticRegression
    metrics: dict
    ablation: pd.DataFrame
    test: pd.DataFrame
    train: pd.DataFrame


def _fit_quality(tr, te):
    """Ablation: signals only -> + slot completeness -> + text n-grams."""
    out, models = [], {}
    qv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    mats = {"A. Buyer signals only": (lambda d, f: _dense(d, SIGNALS)),
            "B. + Slot completeness": (lambda d, f: _dense(d, SIGNALS + SLOTF)),
            "C. + Text n-grams (full)": (lambda d, f: hstack([_dense(d, SIGNALS + SLOTF),
                                                               qv.fit_transform(d["text"]) if f else qv.transform(d["text"])]).tocsr())}
    for name, fn in mats.items():
        clf = LogisticRegression(max_iter=3000).fit(fn(tr, True), tr["genuine"])
        p = clf.predict_proba(fn(te, False))[:, 1]
        out.append({"model": name, "auc": roc_auc_score(te["genuine"], p)})
        models[name] = (clf, fn, p)
    return pd.DataFrame(out), models["B. + Slot completeness"], qv


def train(df: pd.DataFrame, seed=45) -> Bundle:
    d = add_features(df)
    tr, te = train_test_split(d, test_size=0.25, random_state=seed, stratify=d["category"])
    # char n-grams make the classifier robust to typos and Hinglish spellings
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    cc = LogisticRegression(max_iter=3000).fit(cv.fit_transform(tr["text"]), tr["category"])
    te = te.assign(pred_cat=cc.predict(cv.transform(te["text"])))
    abl, (qclf, qfn, p), qv = _fit_quality(tr, te)
    te = te.assign(p_genuine=p)
    tr = tr.assign(p_genuine=qclf.predict_proba(qfn(tr, False))[:, 1])
    vague = te["completeness"] <= 0.25
    top = te.nlargest(int(0.3 * len(te)), "p_genuine")
    m = {"category_acc_all": accuracy_score(te["category"], te["pred_cat"]),
         "category_acc_detailed": accuracy_score(te.loc[~vague, "category"], te.loc[~vague, "pred_cat"]),
         "category_acc_vague": accuracy_score(te.loc[vague, "category"], te.loc[vague, "pred_cat"]),
         "category_acc_hinglish": accuracy_score(te.loc[te.language == "Hinglish", "category"],
                                                 te.loc[te.language == "Hinglish", "pred_cat"]),
         "quality_auc": float(abl.auc.iloc[1]), "base_genuine_rate": float(te.genuine.mean()),
         "top30_genuine_rate": float(top.genuine.mean()), "n_test": len(te)}
    b = Bundle(cv, cc, qv, qclf, m, abl, te, tr)
    b._qfn = qfn
    return b


def score_enquiry(b: Bundle, text, city, verified, prior, has_budget=False) -> dict:
    row = add_features(pd.DataFrame([{"text": text, "city": city, "state": STATE.get(city, ""),
                                      "buyer_verified": int(verified), "prior_enquiries": prior,
                                      "has_budget": int(has_budget)}]))
    probs = b.cat_clf.predict_proba(b.cat_vec.transform(row["text"]))[0]
    order = np.argsort(probs)[::-1]
    p = float(b.q_clf.predict_proba(b._qfn(row, False))[0, 1])
    return {"category": b.cat_clf.classes_[order[0]], "category_conf": float(probs[order[0]]),
            "category_probs": dict(zip(b.cat_clf.classes_[order[:3]], probs[order[:3]].round(3))),
            "p_genuine": p, "completeness": float(row["completeness"].iloc[0]),
            "slots": S.extract(text), "clarify": S.clarifying_question(text),
            "language": "Hinglish" if S.is_hinglish(text) else "English"}


def threshold_table(te: pd.DataFrame, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        sel = te[te.p_genuine >= t]
        rows.append({"threshold": t, "leads_kept_%": 100 * len(sel) / len(te),
                     "precision_%": 100 * sel.genuine.mean() if len(sel) else np.nan,
                     "genuine_recall_%": 100 * sel.genuine.sum() / te.genuine.sum()})
    return pd.DataFrame(rows)


def calibration_table(te: pd.DataFrame, bins=5) -> pd.DataFrame:
    q = pd.qcut(te.p_genuine, bins, duplicates="drop")
    g = te.groupby(q, observed=True).agg(predicted=("p_genuine", "mean"), actual=("genuine", "mean"), n=("genuine", "size"))
    return g.reset_index(drop=True)
