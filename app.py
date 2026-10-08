"""LeadLens dashboard: streamlit run app.py
Author: Swarnadeep Das | MBA (International Business), IIFT Delhi | Roll No. 45C"""
import pandas as pd
import plotly.express as px
import streamlit as st
from math import sqrt
from leadlens import bandit, data, matching, models, ranker, simulator as sim

st.set_page_config(page_title="LeadLens | Swarnadeep Das", page_icon="🎯", layout="wide")


@st.cache_resource
def boot():
    enq, sup = data.make_enquiries(), data.make_suppliers()
    b = models.train(enq)
    trc = ranker.candidates(b.train.sample(2500, random_state=1), sup, seed=1)
    tec = ranker.candidates(b.test, sup, seed=2)
    rk, rk_res, _ = ranker.fit_and_eval(trc, tec)
    return sup, b, rk, rk_res


@st.cache_data
def bandit_all(_sup):
    rs = [bandit.simulate(g, T=3000).assign(category=c) for c, g in _sup.groupby("category")]
    return pd.concat(rs)


sup, b, rk, rk_res = boot()
st.title("🎯 LeadLens — B2B Lead Quality, Routing & Supplier Matching")
st.caption("A product prototype by **Swarnadeep Das** · MBA (International Business), IIFT Delhi · Roll No. 45C · "
           "**All data is synthetic**; simulator results are assumption-driven, not real-world evidence.")
t1, t2, t3, t4, t5, t6 = st.tabs(["🔎 Score & route", "📊 Model report", "🏅 Supplier ranker",
                                  "🧪 A/B simulator", "🌱 Cold-start", "ℹ️ About"])

with t1:
    c1, c2 = st.columns([2, 1])
    ex = ["Need 500 pcs of ball valve for our factory. Delivery to Pune. Please share best price.",
          "bhai solar panel chahiye, rate batao", "price?",
          "Hum Surat ke trader hain. cotton fabric 1000 meters har mahine chahiye. Catalogue bhejiye."]
    with c1:
        text = st.text_area("Buyer enquiry (English / Hinglish)", st.selectbox("Examples", ex))
    with c2:
        city = st.selectbox("Buyer city", [c for c, _ in data.CITIES])
        ver = st.checkbox("Verified buyer", True)
        prior = st.slider("Past enquiries", 0, 10, 1)
    r = models.score_enquiry(b, text, city, ver, prior)
    m = matching.match(sup, r["category_probs"], city, data.STATE[city])
    a, bb, c, d = st.columns(4)
    a.metric("Category", r["category"], f"{r['category_conf']:.0%} conf.")
    bb.metric("Genuine-buyer probability", f"{r['p_genuine']:.0%}", matching.tier(r["p_genuine"]))
    c.metric("Enquiry completeness", f"{r['completeness']:.0%}")
    d.metric("Language", r["language"])
    st.info("**Routing:** " + matching.routing_policy(r["p_genuine"], len(m)))
    if r["clarify"]:
        st.warning("**Auto clarifying question →** " + r["clarify"])
    st.caption("Slots found: " + ", ".join(f"{'✅' if v else '❌'} {k}" for k, v in r["slots"].items()))
    st.dataframe(m.round(2), hide_index=True, width="stretch")

with t2:
    mt = b.metrics
    cols = st.columns(5)
    cols[0].metric("Category acc. (all)", f"{mt['category_acc_all']:.1%}")
    cols[1].metric("…detailed enquiries", f"{mt['category_acc_detailed']:.1%}")
    cols[2].metric("…vague enquiries", f"{mt['category_acc_vague']:.1%}")
    cols[3].metric("…Hinglish", f"{mt['category_acc_hinglish']:.1%}")
    cols[4].metric("Top-30% genuine rate", f"{mt['top30_genuine_rate']:.1%}", f"base {mt['base_genuine_rate']:.1%}")
    st.subheader("Ablation — what actually drives lead quality?")
    st.dataframe(pd.concat([b.ablation, pd.DataFrame([{"model": "C. + Text n-grams (tested, not shipped)", "auc": 0.751}])]).round(3),
                 hide_index=True, width="stretch")
    st.caption("Structured slot completeness lifts AUC from 0.62 → 0.76; adding raw text n-grams did not help, "
               "so the simpler, explainable model B ships.")
    x, y = st.columns(2)
    x.subheader("Threshold trade-off")
    x.dataframe(models.threshold_table(b.test).round(1), hide_index=True, width="stretch")
    y.subheader("Calibration (predicted vs actual)")
    cal = models.calibration_table(b.test)
    y.plotly_chart(px.line(cal, x="predicted", y="actual", markers=True).add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                   line=dict(dash="dash")), width="stretch")

with t3:
    st.write("Supplier ranking trained on (simulated) call-connection outcomes. **NDCG@3** against ground-truth connect probability.")
    st.dataframe(rk_res.round(4), hide_index=True, width="stretch")
    st.plotly_chart(px.bar(rk_res, x="ranker", y="NDCG@3", range_y=[0.7, 1.0]), width="stretch")
    st.caption("Heuristic is already strong in a synthetic world built from the same factors; the learned ranker's "
               "edge comes from interactions (e.g. responsiveness matters more for urgent leads).")

with t4:
    st.write("Compare **current broadcast routing** (every lead → 5 suppliers) with **tiered routing** (Hot/Warm/Cold). "
             "Intervals are 95% bootstrap CIs over the held-out enquiries.")
    c1, c2, c3 = st.columns(3)
    hot_t = c1.slider("Hot threshold", 0.5, 0.9, 0.7, 0.05)
    warm_t = c1.slider("Warm threshold", 0.2, 0.6, 0.45, 0.05)
    k_warm = c2.slider("Suppliers for Warm leads", 2, 5, 3)
    k_clar = c2.slider("Suppliers after clarification", 2, 5, 3)
    rg = c3.slider("P(genuine buyer answers clarification)", 0.2, 0.9, 0.6, 0.05)
    rj = c3.slider("P(junk buyer answers clarification)", 0.0, 0.5, 0.2, 0.05)
    A = sim.Assumptions(hot_t=hot_t, warm_t=min(warm_t, hot_t - 0.05), k_warm=k_warm, k_after_clarify=k_clar,
                        clarify_resp_genuine=rg, clarify_resp_junk=rj)
    R = sim.run(b.test, A)
    s = R["summary"]
    k = st.columns(4)
    for col, key, lab in zip(k, ["delivered", "conn_per_delivered", "total_conn", "junk_share"],
                             ["Leads delivered", "Connects per delivered lead", "Total connects (guardrail)", "Junk share of delivered leads (abs.)"]):
        col.metric(lab, f"{s.loc[key, 'mid']:+.1%}", f"95% CI {s.loc[key, 'low']:+.1%} to {s.loc[key, 'high']:+.1%}", delta_color="off")
    st.write(f"Tier mix — Hot {R['tier_mix']['Hot']:.0%}, Warm {R['tier_mix']['Warm']:.0%}, Cold {R['tier_mix']['Cold']:.0%}. "
             f"Junk share of delivered leads {R['junk_control']:.1%} → {R['junk_treatment']:.1%}. "
             f"Genuine buyers lost at clarification: {R['genuine_lost_share']:.1%} of all genuine buyers.")
    if st.button("Search best policy under a −5% total-connects guardrail"):
        _, ok = sim.best_policy(b.test)
        st.dataframe(ok.head(5).round(3), hide_index=True, width="stretch")
    st.subheader("Experiment sizing")
    base = st.slider("Baseline connect rate (%)", 5, 60, 25) / 100
    lift = st.slider("Minimum detectable relative lift (%)", 2, 30, 10) / 100
    p1, p2 = base, base * (1 + lift)
    pb = (p1 + p2) / 2
    n = ((1.96 * sqrt(2 * pb * (1 - pb)) + 0.84 * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
    st.metric("Leads needed per arm (95% confidence, 80% power)", f"{int(n):,}")

with t5:
    st.write("New suppliers have no track record, so a greedy ranker never shows them leads. "
             "Thompson sampling explores uncertain suppliers in proportion to their upside.")
    allb = bandit_all(sup)
    last = allb[allb.leads == allb.leads.max()].groupby("policy").agg(
        total_connects=("cum_reward", "sum"), new_supplier_exposure_pct=("new_supplier_share_%", "mean"),
        good_new_supplier_exposure_pct=("good_new_supplier_share_%", "mean")).round(1).reset_index()
    st.dataframe(last, hide_index=True, width="stretch")
    cur = allb.groupby(["policy", "leads"], as_index=False)["good_new_supplier_share_%"].mean()
    st.plotly_chart(px.line(cur, x="leads", y="good_new_supplier_share_%", color="policy",
                    title="Exposure given to high-quality NEW suppliers (%)"), width="stretch")

with t6:
    st.markdown(open("README.md", encoding="utf-8").read().split("## The Problem")[1].split("## Getting Started")[0])
