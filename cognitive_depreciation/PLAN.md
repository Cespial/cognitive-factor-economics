# Paper 2 — Cognitive Capital Depreciation

## *"Cognitive Capital Depreciation in the Age of Generative AI: A Dynamic Human Capital Model with Endogenous Skill Half-Lives"*

**Target:** Journal of Economic Theory (Q1, IF=1.8) / Econometrica (Q1, IF=6.5) / Review of Economic Studies (Q1, IF=5.8)

**Author:** Cristian Espinal Maya (EAFIT)

**Quality standard:** Q1 — formal proofs, novel data, comprehensive robustness

---

## The Core Idea

Standard human capital theory treats the depreciation rate δ_H as a biological constant (~1-3% per year). This paper shows it is **endogenous to the rate of AI capability advancement**:

$$\delta_H^o(t) = \delta_0 + \lambda_o \cdot \dot{\Omega}_{AI}^o(t)$$

where:
- δ₀ = biological/natural depreciation (~1.5%/year)
- λ_o = occupation-specific sensitivity to AI advancement
- Ω̇_AI^o(t) = rate of AI capability improvement in tasks relevant to occupation o

**The key prediction:** when AI improves fast (Ω̇ high), routine cognitive skills depreciate at 10-20%/year instead of 1-3%, giving them a "half-life" of 3-7 years instead of decades. Augmentable skills depreciate slower (or even appreciate) because AI amplifies rather than substitutes them.

---

## Why This Is Q1-Worthy

1. **Resolves a 50-year puzzle:** Why do education returns decline faster for some occupations? Because δ_H is not constant — it depends on technology.

2. **Novel formal object:** The "skill half-life" as a derived quantity from the endogenous depreciation model:
   $$t_{1/2}^o = \frac{\ln 2}{\delta_0 + \lambda_o \cdot \dot{\Omega}_{AI}^o}$$

3. **Novel data source:** AI benchmark time series (MMLU, HumanEval, MATH, etc.) as empirical proxies for Ω̇_AI, mapped to occupations via O*NET tasks.

4. **Optimal control solution:** Workers choose investment in H^A vs H^C optimally under the endogenous depreciation regime → closed-form investment path.

5. **First empirical estimates** of occupation-specific, AI-induced skill depreciation rates for a developing economy (Colombia).

6. **Direct policy implications:** Optimal educational investment horizons, reskilling program design, curriculum half-life.

---

## Theoretical Architecture

### Block 1: The Dynamic Model

**Setup:**
- Worker i at age a has cognitive human capital stock: K_i(a) = K_i^C(a) + K_i^A(a)
- Each component follows an accumulation equation with endogenous depreciation
- The worker chooses investment allocation s(a) ∈ [0,1] between H^C and H^A at each age

**Accumulation equations:**
```
K̇_i^C(a) = s(a) · θ_C · I(a) − [δ₀ + λ_C · Ω̇_AI(t)] · K_i^C(a)
K̇_i^A(a) = (1−s(a)) · θ_A · I(a) − [δ₀ − μ · Ω̇_AI(t)] · K_i^A(a)
```

where:
- I(a) = total investment capacity (decreasing in age, Ben-Porath style)
- θ_C, θ_A = productivity of investment in each type
- μ > 0 captures that AI advancement *reduces* depreciation of H^A (appreciation!)

**Key insight:** H^C depreciates faster when AI advances; H^A depreciates slower (or appreciates). This creates a **time-varying switching point** in the optimal investment path.

### Block 2: Optimal Investment Path

**The worker's problem:**
```
max_{s(·)} ∫₀^T e^{-ρa} · w(K^C(a), K^A(a), D(t+a)) da

subject to:
  K̇^C = s·θ_C·I − (δ₀ + λ_C·Ω̇)·K^C
  K̇^A = (1−s)·θ_A·I − (δ₀ − μ·Ω̇)·K^A
  s(a) ∈ [0,1]
  K^C(0) = K₀^C, K^A(0) = K₀^A given
```

**Propositions to prove:**

1. **Existence of a switching age a*:** For Ω̇ > Ω̇*, the optimal investment shifts entirely to H^A after age a* (bang-bang solution).

2. **Comparative statics of a*:** ∂a*/∂Ω̇ < 0 — faster AI advancement causes earlier switching.

3. **NPV of H^C investment:** There exists Ω̇** such that for Ω̇ > Ω̇**, the NPV of investing one more unit in H^C is negative at any age. This is the "skill death" threshold.

4. **Skill half-life formula:** 
   t_{1/2}^o = ln(2) / (δ₀ + λ_o · Ω̇_o)
   with closed-form expressions for λ_o as a function of task composition.

5. **Welfare analysis:** The welfare loss from using constant-δ models (standard Mincer) is increasing in Ω̇. Quantify the magnitude.

### Block 3: Empirical Measurement of Ω̇_AI

**Innovation:** Map AI benchmark improvements to occupation-level AI advancement rates.

**Step 1:** Collect AI benchmark time series:
- MMLU scores by subject (2020-2026): language, math, science, social science, etc.
- HumanEval pass rates (2021-2026): code generation capability
- MATH benchmark (2021-2026): mathematical reasoning
- Additional: ARC, HellaSwag, GSM8K

**Step 2:** Map benchmark domains to O*NET task categories:
- MMLU Language → occupations with high reading comprehension tasks
- MMLU Math → occupations with mathematical reasoning tasks
- HumanEval → occupations with programming/coding tasks
- MMLU Social Science → occupations with social analysis tasks

**Step 3:** Construct Ω̇_o for each occupation:
```
Ω̇_o(t) = Σ_b w_{ob} · ΔScore_b(t) / Score_b(t-1)
```
where b indexes benchmarks and w_{ob} is the relevance weight of benchmark b to occupation o (derived from O*NET task-to-benchmark mapping).

### Block 4: Empirical Estimation

**Strategy 1 — Cross-sectional estimation of λ:**
Using GEIH data, estimate:
```
ln w_{ioa} = α + β₁·Educ_i + β₂·Exp_i + β₃·Exp²_i 
           + γ₁·AHC_o + γ₂·AHC_o×Exp_i 
           + γ₃·Ω̇_o + γ₄·Ω̇_o×Exp_i
           + controls + ε
```

The coefficient γ₄ on Ω̇_o × Experience estimates λ: in occupations with faster AI advancement, experience returns (which capture depreciation) are lower.

**Strategy 2 — Calibration:**
Use the structural model to match observed:
- Age-earnings profiles by occupation (from GEIH)
- Returns to experience by AI exposure (from Paper 1 regression results)
- Cross-country differences in returns (if data available)

Calibrate {δ₀, λ_C, μ, θ_C, θ_A} to match these moments.

**Strategy 3 — Simulation:**
- Simulate optimal investment paths under 4 scenarios:
  A) Ω̇ = 0 (no AI advancement) — baseline
  B) Ω̇ = historical (2020-2026 trend)
  C) Ω̇ = 2× historical (acceleration)
  D) Ω̇ = exponential (AGI trajectory, Korinek & Suh 2024)
- Report: optimal switching ages, skill half-lives, welfare losses

---

## Backlog

### Sprint 0 — Infrastructure (Day 1-2)
```
□ S0.1  Create project directory + symlinks to Paper 1 data
□ S0.2  Download AI benchmark time series:
        - MMLU scores from PapersWithCode/HELM (2020-2026)
        - HumanEval from PapersWithCode (2021-2026)
        - MATH from PapersWithCode (2021-2026)
        - Composite from Epoch AI
□ S0.3  Build benchmark-to-occupation mapping:
        - MMLU subjects → O*NET knowledge areas → SOC occupations
        - HumanEval → O*NET "Computer and Information Technology" tasks
        - Construct w_{ob} weights
□ S0.4  Compute Ω̇_o for each occupation (annual AI advancement rate)
□ S0.5  Merge Ω̇_o with GEIH estimation sample from Paper 1
□ S0.6  Literature review: skill depreciation, optimal control, AI benchmarks
```

### Sprint 1 — Theoretical Model (Week 1)
```
□ S1.1  Write the dynamic accumulation equations
□ S1.2  Set up the optimal control problem (Hamiltonian)
□ S1.3  Derive first-order conditions (costate equations)
□ S1.4  PROVE Proposition 1: existence of switching age a*
□ S1.5  PROVE Proposition 2: comparative statics ∂a*/∂Ω̇ < 0
□ S1.6  PROVE Proposition 3: NPV negativity threshold Ω̇**
□ S1.7  DERIVE Proposition 4: skill half-life formula
□ S1.8  PROVE Proposition 5: welfare loss from constant-δ misspecification
□ S1.9  Write special cases:
        - If Ω̇ = 0: model reduces to standard Ben-Porath
        - If μ = 0: no appreciation of H^A (pure depreciation model)
        - If λ_C → ∞: H^C becomes worthless instantly
□ S1.10 Verify all proofs analytically (or with Mathematica/SymPy)
□ S1.11 Write Sections 1-2 of the paper (Intro + Theory)
```

### Sprint 2 — AI Benchmark Data (Week 2)
```
□ S2.1  Scrape/download MMLU leaderboard history (model × date × subject score)
□ S2.2  Scrape/download HumanEval leaderboard history
□ S2.3  Scrape/download MATH, GSM8K, ARC leaderboards
□ S2.4  Download Epoch AI compute/capability trends
□ S2.5  Construct clean panel: benchmark × year × score
□ S2.6  Compute annual growth rates: ΔScore/Score by benchmark by year
□ S2.7  Map benchmarks to O*NET knowledge domains:
        - Create crosswalk: MMLU subject → O*NET Knowledge Area → SOC
        - Weight: w_{ob} = importance of knowledge area b in occupation o
□ S2.8  Compute Ω̇_o for each of 207 SOC occupations
□ S2.9  Rank occupations by Ω̇: which face fastest AI advancement?
□ S2.10 Validate: do high-Ω̇ occupations have lower experience returns in GEIH?
□ S2.11 Write Section 3 (Data: AI Benchmarks as Ω̇ Proxies)
```

### Sprint 3 — Empirical Estimation (Week 3)
```
□ S3.1  Estimate cross-sectional depreciation model:
        ln w = α + β₁·Educ + β₂·Exp + β₃·Exp² + γ₁·AHC + γ₂·AHC×Exp
             + γ₃·Ω̇ + γ₄·Ω̇×Exp + controls
        Target: γ₄ < 0 (faster AI → lower experience returns)
□ S3.2  Estimate separately for formal vs informal sectors
□ S3.3  Estimate separately by education level
□ S3.4  Compute implied δ_H(o) for each occupation:
        δ̂_o = −(β₂ + γ₄·Ω̇_o) (from experience return)
□ S3.5  Compute implied skill half-lives: t_{1/2}^o = ln(2)/δ̂_o
□ S3.6  Report top 10 and bottom 10 occupations by half-life
□ S3.7  Robustness: alternative Ω̇ constructions (different benchmarks)
□ S3.8  Robustness: different experience specifications (quartic, spline)
□ S3.9  Write Section 4 (Empirical Results)
```

### Sprint 4 — Calibration & Simulation (Week 4)
```
□ S4.1  Calibrate structural parameters {δ₀, λ_C, μ, θ_C, θ_A, ρ}
        Target moments: age-earnings profiles, returns to experience,
        AHC premium by age (from Paper 1)
□ S4.2  Solve optimal investment paths numerically (shooting method)
□ S4.3  Simulate 4 scenarios (no AI, historical, 2×, exponential)
□ S4.4  Plot: optimal s(a) paths under each scenario
□ S4.5  Plot: K^C(a) and K^A(a) trajectories
□ S4.6  Compute: switching ages a* under each scenario
□ S4.7  Compute: welfare losses from constant-δ assumption
□ S4.8  Sensitivity analysis: vary λ_C, μ, Ω̇ growth rate
□ S4.9  Write Section 5 (Calibration & Simulation)
```

### Sprint 5 — Policy Analysis (Week 4-5)
```
□ S5.1  Optimal education horizon:
        How many years of H^C education is optimal given Ω̇?
        Show: optimal decreases from ~4 years to ~2 years as Ω̇ doubles
□ S5.2  Reskilling ROI:
        For displaced H^C workers, what is the return to retraining into H^A?
        Show: ROI depends critically on age and Ω̇
□ S5.3  Curriculum redesign:
        What fraction of curriculum should shift from H^C to H^A content?
        Derive optimal fraction as function of Ω̇ and student age
□ S5.4  CONPES 4144 evaluation (link to Paper S5.0-D):
        Does Colombia's AI policy account for endogenous depreciation?
□ S5.5  Write Section 6 (Policy Implications)
```

### Sprint 6 — Figures & Tables (Week 5)
```
□ S6.1  Fig 1: AI benchmark advancement curves (MMLU, HumanEval, MATH over time)
□ S6.2  Fig 2: Ω̇_o distribution across occupations
□ S6.3  Fig 3: Skill half-lives by occupation (bar chart, top/bottom 20)
□ S6.4  Fig 4: Optimal investment paths s(a) under 4 scenarios
□ S6.5  Fig 5: K^C(a) and K^A(a) trajectories under 4 scenarios
□ S6.6  Fig 6: Age-earnings profiles — model vs data (calibration fit)
□ S6.7  Fig 7: Welfare loss from constant-δ as function of Ω̇
□ S6.8  Fig 8: Policy diagram — optimal H^C/H^A curriculum mix by Ω̇

□ S6.9  Table 1: AI benchmark data summary
□ S6.10 Table 2: Occupation-level depreciation rates and half-lives
□ S6.11 Table 3: Cross-sectional regression results (γ₄)
□ S6.12 Table 4: Calibrated structural parameters
□ S6.13 Table 5: Simulation results (switching ages, welfare)
```

### Sprint 7 — Paper Writing (Week 5-6)
```
□ S7.1  Section 1: Introduction (the depreciation puzzle)
□ S7.2  Section 2: Dynamic Model with Endogenous Depreciation
□ S7.3  Section 3: Measuring AI Advancement (Ω̇) with Benchmarks
□ S7.4  Section 4: Empirical Estimation of Depreciation Rates
□ S7.5  Section 5: Calibration & Simulation
□ S7.6  Section 6: Policy Implications
□ S7.7  Section 7: Connection to CFE Program (Papers 1, 0)
□ S7.8  Section 8: Conclusion
□ S7.9  Appendix A: Proofs of all propositions
□ S7.10 Appendix B: Data construction details
□ S7.11 Appendix C: Numerical solution method
```

### Sprint 8 — Quality Assurance (Week 6)
```
□ S8.1  Verify all proofs line by line
□ S8.2  Cross-check all empirical numbers against code output
□ S8.3  Run full robustness battery (at least 20 specifications)
□ S8.4  External review (Jiménez-Builes, Restrepo Carmona)
□ S8.5  Scopus literature check: verify no one published this in 2026
□ S8.6  Proofread entire manuscript
□ S8.7  Format for arXiv (flat ZIP, \input{main.bbl})
□ S8.8  Submit to arXiv (econ.TH, cross-list econ.EM + cs.AI)
```

---

## Timeline

```
Week 1 (Apr 7-11):    Sprint 0 (data) + Sprint 1 (theory + proofs)
Week 2 (Apr 14-18):   Sprint 2 (AI benchmarks + Ω̇ construction)
Week 3 (Apr 21-25):   Sprint 3 (empirical estimation)
Week 4 (Apr 28-May 2): Sprint 4 (calibration) + Sprint 5 (policy)
Week 5 (May 5-9):     Sprint 6 (figures) + Sprint 7 (writing)
Week 6 (May 12-16):   Sprint 8 (QA) + submission
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| AI benchmark data not available as time series | Medium | High | Use Epoch AI composite; manual collection from Papers With Code |
| Optimal control problem has no closed-form | Low | Medium | Numerical solution + qualitative propositions |
| γ₄ (Ω̇×Exp) not significant in GEIH | Medium | High | Use calibration as primary strategy; cross-section as validation |
| Existing paper publishes similar model in 2026 | Low | High | Scopus check before submission; emphasize developing-country angle |
| Proofs require stronger assumptions than desired | Medium | Medium | State clearly; provide numerical verification |

---

## What Makes This Q1

1. **5 formal propositions with proofs** (not just claims)
2. **Novel data**: AI benchmarks mapped to occupations (no one has done this)
3. **Novel object**: skill half-life as a derived quantity from structural model
4. **Comprehensive empirical section**: cross-sectional + calibration + simulation
5. **4-scenario simulation** with welfare analysis
6. **Policy analysis** with concrete recommendations
7. **8 publication figures** + 5 tables
8. **Full appendix** with proofs, data, numerical methods
9. **Replication package** on GitHub
10. **Connection to 2 published arXiv papers** (ecosystem value)
