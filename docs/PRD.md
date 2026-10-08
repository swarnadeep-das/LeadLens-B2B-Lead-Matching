# PRD — LeadLens: Lead Quality, Routing & Supplier Liquidity
**Author:** Swarnadeep Das (IIFT Delhi, Roll No. 45C) · **Status:** Prototype PRD; all numbers from a synthetic simulator

## 1. Problem & why now
Suppliers pay to receive buyer enquiries; their renewal depends on whether leads convert into conversations and orders. Weak enquiries ("price?") and mis-routed leads reduce supplier trust, while new suppliers receive no leads and churn before proving quality. Enquiry volume is growing (IndiaMART reports 29M unique business enquiries in Q1 FY26, +17% YoY), so manual curation does not scale.

## 2. Goals / Non-goals
**Goals:** raise lead→call-connect rate and supplier-perceived lead quality **without** cutting total supplier value; give new suppliers a fair path to a track record.
**Non-goals (v1):** pricing changes, new buyer-facing UI redesign, payments/escrow.

## 3. Personas
| Persona | Need | Pain |
|---|---|---|
| Supplier (SME owner, Hindi-first) | Few, relevant, serious leads | Wasted calls on vague enquiries |
| Buyer (trader / procurement) | Fast, relevant quotes | Never hears back, or gets irrelevant calls |
| New supplier | First leads to prove quality | Cold start |
| Trust & Safety / Ops | Catch spam & low-intent | Manual review does not scale |

## 4. Solution
1. **Understand** the enquiry: category, language, slot completeness.
2. **Score** P(genuine buyer) using verified status, history and completeness.
3. **Route** by tier: Hot → wide & instant; Warm → narrower; Cold → *clarify first* (auto-generated, in the buyer's language), then release.
4. **Match** suppliers with a learned ranker; **explore** new suppliers via Thompson sampling.

## 5. Metric tree
- **North star:** lead → call-connected rate; secondary: lead → order.
- **Supplier health:** lead-quality rating, acceptance rate, renewal rate.
- **Buyer health:** time-to-first-response, % enquiries with ≥1 quote.
- **Marketplace health:** share of suppliers receiving ≥1 lead/week; new-supplier 30-day activation.
- **Guardrails:** total connections (revenue proxy) ≥ −5%; genuine buyers lost at clarification; buyer drop-off; supplier lead volume Gini; fairness by buyer segment & language.

## 6. What the simulator says (assumption-driven)
Default tiering lifts connects per delivered lead +14% and cuts junk share 35.7% → 25.1%, **but total connects fall 18%** because fewer leads are delivered. Constrained policy search finds configs within −5% total connects (+5.6% connects/lead, −5.3 pp junk share). **Decision: ship the guardrailed policy, not the aggressive one.**

## 7. Rollout
1. **Shadow mode** (log scores, change nothing) – validate calibration against real outcomes.
2. **Cold-bucket experiment** on 5% of buyers: clarifying question vs none → measure real clarification response rates (the key unknown).
3. **A/B on routing** (buyer-level randomisation), 5% → 25% → 50% if guardrails hold. Sample size from the app's sizing tool; run ≥ 2 weekly cycles.
4. **Cold-start exploration** limited to a capped share of each category's leads.

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Label noise / proxy labels | Use multiple outcomes (connect, feedback, order); audit samples |
| Genuine buyers held back | Clarification is a delay, not a rejection; monitor lost-genuine rate |
| Feedback loops (down-ranked buyers never get data) | Exploration budget; periodic randomised holdout |
| Hinglish / Indic scripts / voice | Phase 2: multilingual embeddings, transliteration |
| Supplier revenue dip | Hard guardrail on total connects; kill-switch |
| Fairness (new vs repeat buyers) | Segment dashboards; thresholds reviewed monthly |

## 9. Roadmap
v1 (this prototype) → v2 multilingual embeddings & real labels → v3 learning-to-rank on live outcomes → v4 LLM-drafted clarifying questions with templates/guardrails → v5 supplier-facing "lead quality" explainer.

## 10. Open questions
What is the real clarification response rate? How much total-connect loss is acceptable vs. supplier-satisfaction gains? Does faster routing matter more than wider routing?
