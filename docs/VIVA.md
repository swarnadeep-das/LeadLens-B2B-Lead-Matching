# Viva / Interview Notes — LeadLens (read this before explaining the project)
**One-line pitch:** "A prototype that scores and routes B2B buyer enquiries so suppliers get better leads, while a simulator shows the volume-vs-quality trade-off a PM must manage."

**Be upfront:** data is synthetic; simulator results are assumption-driven; the code was built with AI assistance and I understand and can walk through every part.

## Likely questions
1. **What problem does it solve?** Suppliers pay for leads; vague/mis-routed leads reduce trust and renewals; new suppliers can't get started.
2. **Why linear models?** Explainable to suppliers/ops, cheap to serve, and the ablation showed fancier text features did not help.
3. **What was your most important finding?** Structured information (slot completeness) lifts AUC 0.62→0.76 — more than raw text — so asking buyers the right question is the product lever.
4. **Biggest trade-off?** Tiered routing raises connects per lead (+14%) and cuts junk share but total connects fall 18%; I added a guardrail search to find a policy within −5%.
5. **Why simulate rather than prove?** No access to real marketplace data; the simulator is for sizing trade-offs and designing the real experiment. Its assumptions are visible and adjustable.
6. **How would you validate in production?** Shadow mode → 5% cold-bucket test to measure clarification response → buyer-level A/B with guardrails; sample size via the sizing tool.
7. **What are the failure modes?** Label noise, genuine buyers delayed, feedback loops, Hinglish/Indic coverage, fairness across buyer segments.
8. **What is Thompson sampling in plain words?** Pick each supplier with probability proportional to the chance it's actually the best — uncertain (new) suppliers get tried, proven ones keep most traffic.
9. **What's the north-star metric and why?** Lead→call-connected rate: closest measurable proxy for buyer-supplier value that Product controls; paired with total connects as the revenue guardrail.
10. **What would you build next?** Multilingual embeddings, real outcome labels, LLM-drafted clarifying questions behind templates, supplier-facing lead-quality explanations.
