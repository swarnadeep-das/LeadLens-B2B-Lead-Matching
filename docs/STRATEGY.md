# Strategy Notes — Where lead quality fits in a B2B marketplace
*Independent student analysis using public information only. Hypotheses are labelled as such.*

## Business context (public, sourced)
From IndiaMART's Q1 FY26 results release ([link](https://corporate.indiamart.com/2025/07/18/indiamart-intermesh-limited-announces-strong-q1fy26-results-with-12-revenue-growth)):
- 218K paying suppliers at quarter-end (net +1.5K QoQ)
- ~29M unique business enquiries (+17% YoY); 8.4M supplier storefronts (+6% YoY)
- Collections ₹430 Cr consolidated (+17%); deferred revenue ₹1,735 Cr (+18%)

**Reading the numbers (my inference):** enquiries are growing faster than supplier storefronts (+17% vs +6% YoY), so the binding constraint is not buyer demand but converting more of it into supplier value – which makes **lead quality, routing and supplier retention** the highest-leverage product levers.

## Hypotheses (to validate with data, not assumed true)
1. A meaningful share of enquiries is low-intent/under-specified, and structured clarification is cheaper than discarding them. *(Test: shadow-mode completeness distribution.)*
2. Supplier renewal correlates with perceived lead relevance more than raw lead count. *(Test: cohort analysis of renewal vs lead-quality rating.)*
3. New suppliers are under-exposed because ranking leans on historical response data. *(Test: exposure vs. quality by supplier age.)*

## Prioritisation (RICE-style, qualitative)
| Initiative | Reach | Impact | Confidence | Effort | Priority |
|---|---|---|---|---|---|
| Clarifying question for vague enquiries | High | Med-High | Med | Low | **1** |
| Tiered routing with volume guardrail | High | Med | Med | Med | **2** |
| Cold-start exploration | Med | Med | High (sim) | Low | **3** |
| Learned ranker | High | Low-Med | Med | Med | 4 |
| LLM question generation | Med | Low-Med | Low | Med | 5 |

## Why I did not start with a bigger model
The ablation showed structured slots beat raw text n-grams. The cheapest, most explainable lever (ask for the missing information) wins before heavier NLP is justified.
