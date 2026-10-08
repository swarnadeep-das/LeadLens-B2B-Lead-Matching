import pytest
from leadlens import bandit, data, matching, models, ranker, simulator as sim, slots


@pytest.fixture(scope="module")
def world():
    enq, sup = data.make_enquiries(), data.make_suppliers()
    return enq, sup, models.train(enq)


def test_data_reproducible():
    assert data.make_enquiries(300).equals(data.make_enquiries(300))


def test_slots_and_clarifying_question_language():
    s = slots.extract("Need 500 pcs of ball valve for our factory. Delivery to Pune. urgently")
    assert s["quantity"] and s["city"] and s["timeline"]
    assert slots.completeness(slots.extract("price?")) == 0
    assert "kitni" in slots.clarifying_question("rate batao bhai chahiye")      # Hinglish asks in Hinglish
    assert slots.clarifying_question("price?").startswith("To get you")


def test_models(world):
    m = world[2].metrics
    assert m["category_acc_detailed"] > 0.95 and m["category_acc_hinglish"] > 0.7
    assert m["quality_auc"] > 0.7 and m["top30_genuine_rate"] > m["base_genuine_rate"]
    a = world[2].ablation.set_index("model").auc
    assert a["B. + Slot completeness"] > a["A. Buyer signals only"] + 0.05      # slots earn their place


def test_detailed_beats_vague_and_typo_robust(world):
    b = world[2]
    good = models.score_enquiry(b, "Need 500 pcs of ball valve for our factory. Delivery to Pune. Share price.", "Pune", 1, 2)
    vague = models.score_enquiry(b, "price?", "Pune", 0, 0)
    typo = models.score_enquiry(b, "Need 500 pcs of bal lvae for our factory. Delivery to Pune.", "Pune", 1, 2)
    assert good["p_genuine"] > vague["p_genuine"] and typo["category"] == "Industrial Valves"


def test_ranker_beats_random_and_not_worse_than_heuristic(world):
    enq, sup, b = world
    trc = ranker.candidates(b.train.sample(1500, random_state=1), sup, seed=1)
    tec = ranker.candidates(b.test.head(500), sup, seed=2)
    _, res, _ = ranker.fit_and_eval(trc, tec)
    r = res.set_index("ranker")["NDCG@3"]
    assert r["Learned (GBM)"] > r["Random"] and r["Learned (GBM)"] >= r["Heuristic (v1 weights)"] - 0.01


def test_simulator_tradeoffs(world):
    b = world[2]
    r = sim.run(b.test, sim.Assumptions(), n_boot=50)
    s = r["summary"]["mid"]
    assert s["conn_per_delivered"] > 0 and s["junk_share"] < 0          # quality up, junk down
    assert abs(sum(r["tier_mix"].values()) - 1) < 1e-9
    assert s["delivered"] < 0                                           # ...at the cost of volume


def test_bandit_explores_new_suppliers(world):
    sup = world[1]
    g = sup[sup.category == "Cotton Fabric"]
    d = bandit.simulate(g, T=1500)
    last = d[d.leads == 1500].set_index("policy")
    assert last.loc["Thompson sampling", "new_supplier_share_%"] > last.loc["Greedy", "new_supplier_share_%"]


def test_matching_sorted(world):
    m = matching.match(world[1], {"Solar Panels": 1.0}, "Delhi", "Delhi")
    assert (m.category == "Solar Panels").all() and m.score.is_monotonic_decreasing
