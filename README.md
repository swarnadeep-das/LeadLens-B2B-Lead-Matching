# 🎯 LeadLens — B2B Lead Quality, Smart Routing & Supplier Matching

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Author:** Swarnadeep Das · Roll No. 45C · MBA (International Business), IIFT Delhi
> An end-to-end **product + ML** project for a B2B marketplace: the models, the routing policy, the PRD, the A/B-test simulator and the trade-offs a PM must decide.

> **Disclaimer:** all data is **synthetic** (seeded, reproducible). It is not IndiaMART data and this is an independent student project with no affiliation to any company. Model metrics describe the synthetic world. **Simulator outputs are assumption-driven (they size trade-offs); they are not evidence of real-world lift.**

## The Problem
B2B marketplaces monetise suppliers who pay to receive buyer enquiries. IndiaMART's own Q1 FY26 release reports ~29 million unique business enquiries in a quarter (+17% YoY) and 218K paying suppliers ([source](https://corporate.indiamart.com/2025/07/18/indiamart-intermesh-limited-announces-strong-q1fy26-results-with-12-revenue-growth)). At that scale, three things decide marketplace health:
1. **Lead quality** – vague or low-intent enquiries waste supplier time and erode renewal intent.
2. **Matching** – a good enquiry sent to unresponsive or distant suppliers dies unanswered.
3. **Supply liquidity** – new suppliers get no leads, so never build a track record (cold start).

LeadLens is a working prototype that addresses all three, and shows the **volume-vs-quality trade-off** honestly.

## What I Built
| Module | What it does | Why this approach |
|---|---|---|
| `slots.py` | Extracts quantity / city / timeline / spec-or-use / budget from free text; computes a completeness score; auto-writes a **clarifying question in English or Hinglish** | Deterministic, auditable, zero serving cost; turns a "bad lead" into a better one |
| `models.py` | Category classifier (char n-grams → robust to typos & Hinglish) + genuine-buyer scorer | Explainable linear models; **ablation** shows what really drives quality |
| `ranker.py` | Supplier ranker trained on connection outcomes vs. a hand-weighted heuristic | Captures interactions (e.g. responsiveness matters more when urgent) |
| `simulator.py` | A/B simulator: current broadcast routing vs. tiered Hot/Warm/Cold routing, with bootstrap CIs, guardrails and a policy search | Lets a PM see the trade-off *before* an experiment |
| `bandit.py` | Thompson-sampling exploration for new suppliers | Fixes cold start at ~zero cost to total connects |
| `app.py` | 6-tab Streamlit product demo | Everything explorable by a non-engineer |

## Key Findings (synthetic world, held-out 1,500 enquiries)
**1. What drives lead quality? Structure beats raw text.**
| Model | ROC-AUC |
|---|---|
| A. Buyer signals only (verified, history) | 0.622 |
| **B. + slot completeness (shipped)** | **0.761** |
| C. + raw text n-grams | 0.751 (no gain → not shipped) |

Product insight: *asking buyers for structured details is worth more than a fancier language model.* The top-30% scored leads are 87% genuine vs a 64% base rate.

**2. Language robustness.** Category accuracy: 100% on detailed enquiries, 80% on Hinglish, 45% on vague one-liners (e.g. "price?") – which is exactly why vague leads trigger a clarifying question instead of a guess.

**3. Tiered routing improves quality but costs volume – the central PM trade-off.**
With default thresholds (Hot ≥ 0.70, Warm ≥ 0.45; Hot→5 suppliers, Warm→3, Cold→clarify first), vs. sending every lead to 5 suppliers (95% bootstrap CIs):
| Metric | Change |
|---|---|
| Connects per delivered lead | **+14.2%** (13.0% to 15.7%) |
| Junk share of delivered leads | **35.7% → 25.1%** |
| Leads delivered | −28.3% |
| **Total connects (guardrail)** | **−18.1%** ⚠️ |

A naive launch would hurt supplier value (and revenue proxy). The policy search finds settings that keep total connects within −5% (e.g. Warm leads also sent to 5 suppliers, clarification path to 5) while still lifting connects per lead (+5.6%) and cutting junk share by 5.3 pp. **The recommendation is a guardrailed rollout, not a flip-the-switch launch** – see [`docs/PRD.md`](docs/PRD.md).

**4. Learned ranker.** NDCG@3 vs ground-truth connect probability: Random 0.789 · Heuristic 0.944 · Learned GBM 0.967. (The heuristic is already strong because the synthetic world is built from the same factors; the gain comes from interaction effects.)

**5. Cold start.** Across 10 categories × 3,000 leads, greedy ranking gives new suppliers **0%** exposure. Thompson sampling gives them ~21% (≈19.5% to genuinely good new suppliers) while total connects fall by only ~0.1% (27,733 vs 27,760).

## Product Docs
- [`docs/PRD.md`](docs/PRD.md) – problem, personas, user stories, metric tree, guardrails, rollout, risks
- [`docs/STRATEGY.md`](docs/STRATEGY.md) – business context (sourced), hypotheses and prioritisation
- [`docs/VIVA.md`](docs/VIVA.md) – how to explain every design decision

## Getting Started
```bash
git clone https://github.com/swarnadeep-das/LeadLens-B2B-Lead-Matching.git
cd LeadLens-B2B-Lead-Matching
pip install -r requirements.txt
streamlit run app.py      # or: docker build -t leadlens . && docker run -p 8501:8501 leadlens
pytest -q                 # 8 tests, also run in CI
```

## Project Structure
```
app.py                 Streamlit product demo (6 tabs)
leadlens/data.py       Synthetic English+Hinglish enquiries, suppliers, ground-truth outcome model
leadlens/slots.py      Slot extraction, completeness, clarifying questions
leadlens/models.py     Category + lead-quality models, ablation, calibration, thresholds
leadlens/ranker.py     Learned supplier ranker + NDCG evaluation
leadlens/simulator.py  A/B simulator with bootstrap CIs and guardrail policy search
leadlens/bandit.py     Cold-start exploration (Greedy / ε-greedy / Thompson)
leadlens/matching.py   v1 heuristic matcher + routing policy
docs/                  PRD, strategy, viva notes
tests/                 pytest suite
```

## Limitations (what I would do next)
- Synthetic data; real enquiries have richer Hinglish, voice/WhatsApp input, fraud patterns and attachments.
- "Genuine" labels are simulated; in production they come from call-connect, supplier feedback and orders – with label-noise handling.
- Clarification response rates are assumptions; the first real experiment should measure them.
- Fairness: ensure new/small buyers aren't systematically down-ranked; monitor by segment.
- Next: multilingual embeddings, learning-to-rank on real outcomes, LLM-drafted clarifying questions with guardrails.

## Author & Project Details
- **Author:** Swarnadeep Das
- **Roll No / ID:** 45C
- **Program:** MBA (International Business)
- **Institution:** Indian Institute of Foreign Trade (IIFT), Delhi
- **Email:** swarnadeepdas@icloud.com
- **GitHub:** [@swarnadeep-das](https://github.com/swarnadeep-das)
- **LinkedIn:** [linkedin.com/in/swarnadeepdas](https://linkedin.com/in/swarnadeepdas)

© 2026 Swarnadeep Das · MIT License
