# Paper 2 — Plan Definitivo: De Working Paper a Paper de Talla Mundial

**Target:** Econometrica / Review of Economic Studies / Journal of Economic Theory
**Páginas target:** 25-30 (main) + 15-20 (online appendix)
**Referencias target:** 40-50
**Figuras target:** 8-10
**Tablas target:** 6-8

---

## Gap Analysis: Working Paper (9pp) → Paper Completo (30pp)

| Componente | Working Paper | Paper Completo | Gap |
|-----------|---------------|----------------|-----|
| Proposiciones | 5 enunciadas | 5 con pruebas formales | **Pruebas rigurosas** |
| Related Work | 2 párrafos | Sección dedicada (2pp) | **Positioning vs 15+ papers** |
| Datos | 48 benchmark scores | + Epoch AI ECI + HELM | **Expandir cobertura** |
| Estimación | 1 spec + heterogeneidad | 10+ specs + IV + panel | **Robustness completo** |
| Calibración | Ninguna | Structural match moments | **Sprint completo** |
| Simulación | Ninguna | 4 escenarios + welfare | **Sprint completo** |
| Figuras | 0 | 8-10 publicación | **Sprint completo** |
| Tablas | 2 | 6-8 | **Expandir** |
| Appendix | Ninguno | 15-20pp proofs + data | **Sprint completo** |
| Refs | 15 | 40-50 | **Scopus + 25 nuevas** |
| Páginas | 9 | 25-30 | **~20pp nuevas** |

---

## Sprint 0 — Scopus Literature Check (Day 1)
**Objetivo:** Verificar novedad + encontrar 25 refs faltantes

```
□ S0.1  Buscar en Scopus:
        "human capital depreciation" AND ("artificial intelligence" OR "automation")
        "skill obsolescence" AND ("AI" OR "technology" OR "benchmark")
        "skill half-life" AND ("economic" OR "labor")
        "endogenous depreciation" AND "human capital"
        "Ben-Porath" AND ("AI" OR "automation" OR "technology")
□ S0.2  Verificar: ¿alguien publicó modelo de depreciación endógena + AI en 2024-2026?
        Competidores potenciales: Freund & Mann (2025 CESifo), 
        Walter & Lee (2022 Foresight), Weber (2014)
□ S0.3  Identificar 25 refs adicionales para Related Work:
        - 5 depreciation empírica (Dinerstein 2022, Neuman & Weiss 1995, etc.)
        - 5 optimal control/lifecycle (Ben-Porath 1967, Heckman 1976, Cunha 2007)
        - 5 AI benchmarks/capability (Epoch AI, Tolan 2021, HELM)
        - 5 task-based/automation (Acemoglu-Restrepo, Autor, Felten)
        - 5 policy/education (WEF, OECD, Arbesman 2012)
□ S0.4  Construir tabla de positioning (como Paper 1)
```

## Sprint 1 — Pruebas Formales (Week 1)
**Objetivo:** Demostrar rigurosamente las 5 proposiciones

```
□ S1.1  PROPOSITION 1 (Switching Age):
        Setup: Hamiltonian H = e^{-ρa}·w(K^C,K^A,D) + ψ_C·K̇^C + ψ_A·K̇^A
        Derive costate equations: ψ̇_C, ψ̇_A
        Show: ∂H/∂s = ψ_C·θ_C·I - ψ_A·θ_A·I
        At switching: ψ_C·θ_C = ψ_A·θ_A (marginal value equalization)
        Prove existence of a* via intermediate value theorem
        Show bang-bang at boundary

□ S1.2  PROPOSITION 2 (Comparative Statics):
        Implicit function theorem on FOC at a*
        Show: d(ψ_C·θ_C - ψ_A·θ_A)/dΩ̇ < 0 at a*
        This requires δ^C increasing in Ω̇ and δ^A decreasing
        Sign follows from λ > 0, μ > 0

□ S1.3  PROPOSITION 3 (NPV Threshold):
        Compute NPV(H^C) = ∫₀ᵀ e^{-ρa} · (∂w/∂K^C) · θ_C·I/(ρ+δ^C) da
        Show NPV is continuous and decreasing in Ω̇
        Find Ω̇** where NPV = 0
        Verify: for Ω̇ > Ω̇**, NPV < 0 ∀a

□ S1.4  PROPOSITION 4 (Half-Life):
        Standard exponential decay: K^C(t) = K^C(0)·e^{-δ^C·t}
        At t_{1/2}: K^C = K^C(0)/2
        Solve: e^{-δ^C·t_{1/2}} = 1/2 → t_{1/2} = ln(2)/δ^C
        Substitute δ^C = δ₀ + λ·Ω̇ → closed form

□ S1.5  PROPOSITION 5 (Welfare Loss):
        Define W*(δ) = value function under optimal s* given δ
        Envelope theorem: ∂W*/∂δ < 0 (higher depreciation → lower welfare)
        Worker using δ₀ instead of δ₀+λΩ̇ over-invests in H^C
        Loss = W*(δ₀+λΩ̇) - W*(δ₀) where second uses suboptimal s
        Show ∂ΔW/∂Ω̇ > 0 via chain rule

□ S1.6  Verify all proofs with SymPy/Mathematica numerical examples
□ S1.7  Write Appendix A: Full Proofs (10+ pages)
```

## Sprint 2 — Numerical Calibration (Week 2)
**Objetivo:** Calibrar {δ₀, λ, μ, θ_C, θ_A, ρ, η} para match data moments

```
□ S2.1  Define target moments from GEIH data:
        M1: Mean returns to experience (β₂ = 0.025)
        M2: Returns to experience × AHC interaction (from Paper 1)
        M3: Formal/informal experience return gap
        M4: Age-earnings profile concavity
        M5: AHC wage premium (from Paper 1: +9.1%/SD)
        M6: Formal/informal γ₄ ratio (-0.0013 vs +0.0009)

□ S2.2  External calibration:
        δ₀ = 0.043 (from Dinerstein et al. 2022 AER)
        Ω̇ = benchmark growth rates (from our data)
        ρ = 0.05 (standard discount rate)
        T = 45 (working life 20-65)

□ S2.3  Internal calibration (Method of Simulated Moments):
        Parameters to calibrate: {λ, μ, θ_C, θ_A, η, I₀}
        Minimize: Σ_m [M_m^{model} - M_m^{data}]² / σ²_m
        Use Nelder-Mead or differential evolution

□ S2.4  Report calibrated parameters with standard errors
□ S2.5  Calibration diagnostics:
        - Over-identification test (if more moments than parameters)
        - Sensitivity of each parameter to each moment
        - Contour plots of objective function

□ S2.6  Write Section: Calibration (2-3 pages)
```

## Sprint 3 — Simulation & Scenarios (Week 2-3)
**Objetivo:** Resolver numéricamente y simular 4 escenarios

```
□ S3.1  Solve optimal control numerically:
        Method: backward induction (finite horizon)
        Or: shooting method from boundary conditions
        Discretize age a into 450 periods (monthly)
        For each period, solve for optimal s*(a)

□ S3.2  Scenario A: No AI advancement (Ω̇ = 0)
        - Baseline Ben-Porath solution
        - s*(a) declines smoothly with age
        - Standard concave age-earnings profile

□ S3.3  Scenario B: Historical AI advancement (Ω̇ = observed 2020-2025)
        - Use benchmark-specific growth rates
        - Show: switching age a* occurs around age 35-40
        - H^C stock declines after a*

□ S3.4  Scenario C: Accelerated AI (Ω̇ = 2× historical)
        - Switching age drops to 25-30
        - H^C becomes negative NPV for many occupations
        - Welfare loss quantified

□ S3.5  Scenario D: Exponential/AGI trajectory (Korinek & Suh 2024)
        - Ω̇ growing exponentially
        - H^C investment collapses early
        - Near-complete shift to H^A by age 25
        - Massive welfare implications

□ S3.6  Compute for each scenario:
        - Optimal s*(a) path
        - K^C(a) and K^A(a) trajectories
        - Age-earnings profiles
        - Lifetime wealth
        - Switching age a*
        - Welfare loss vs scenario A

□ S3.7  Occupation-specific simulations:
        - Software Developer (high Ω̇): half-life ~3 years
        - Accountant (medium Ω̇): half-life ~8 years
        - Physical Therapist (low Ω̇): half-life ~30 years
        
□ S3.8  Write Section: Simulation Results (3-4 pages)
```

## Sprint 4 — Extended Robustness (Week 3)
**Objetivo:** 20+ especificaciones adicionales

```
□ S4.1  Alternative Ω̇ constructions:
        - Using only MMLU (broadest benchmark)
        - Using Epoch AI ECI composite
        - Using Tolan et al. (2021) cognitive ability mapping
        - Binary: pre/post ChatGPT (Nov 2022) as shock

□ S4.2  Alternative depreciation specifications:
        - Linear: δ = δ₀ + λ·Ω̇ (baseline)
        - Quadratic: δ = δ₀ + λ·Ω̇ + λ₂·Ω̇²
        - Threshold: δ = δ₀ if Ω̇ < Ω̇*, else δ₀ + λ·(Ω̇ - Ω̇*)
        - Log: δ = δ₀ + λ·ln(1 + Ω̇)

□ S4.3  Alternative experience specifications:
        - Quartic in experience
        - Experience splines (knots at 5, 15, 25 years)
        - Non-parametric (experience dummies)

□ S4.4  Placebo tests:
        - Shuffle Ω̇ across occupations → γ₄ should be ~0
        - Use physical-task-intensity instead of Ω̇ → γ₄ should be ~0
        - Use pre-AI period (if data available) → γ₄ should be smaller

□ S4.5  Sample robustness:
        - Manufacturing only (best firm-level data)
        - Urban only
        - Exclude Bogotá
        - By education level
        - By sector (20 sectors separately)

□ S4.6  Leave-one-benchmark-out:
        - Remove MMLU, recompute Ω̇, re-estimate
        - Remove HumanEval, etc.
        - Show results stable across benchmark exclusions

□ S4.7  Write Section: Robustness (2-3 pages) + Appendix table
```

## Sprint 5 — Related Work & Positioning (Week 3)
**Objetivo:** Sección dedicada de Related Work (2-3 páginas)

```
□ S5.1  Stream 1: Human Capital Depreciation
        - Mincer (1974) original treatment
        - Heckman, Lochner & Todd (2006) extensions
        - Dinerstein et al. (2022 AER) causal estimate (4.3%)
        - De Grip & Van Loo (2002) skill obsolescence taxonomy
        - Weber (2014) technology-driven depreciation
        - Neuman & Weiss (1995) vintage effects
        - Gathmann & Schönberg (2010) task-specific HC

□ S5.2  Stream 2: Lifecycle Models
        - Ben-Porath (1967) foundational model
        - Heckman (1976) embedded in labor supply
        - Cunha & Heckman (2007) skill formation technology
        - Chari & Hopenhayn (1991) vintage HC + technology diffusion
        - Freund & Mann (2025) job transformation under AI

□ S5.3  Stream 3: AI Measurement
        - Tolan et al. (2021 JAIR) benchmark-to-occupation mapping
        - Epoch AI ECI composite index
        - Felten et al. (2021/2023) AIOE
        - Eloundou et al. (2024) GPT exposure
        - Brynjolfsson, Mitchell & Rock (2018) SML rubric

□ S5.4  Stream 4: AI + Labor Markets
        - Acemoglu (2024) simple macroeconomics
        - Korinek & Suh (2024) AGI scenarios
        - Espinal Maya (2026a) AHC framework
        - Espinal Maya (2026b) LLM instruments
        - Pizzinelli et al. (2023 IMF) cross-country

□ S5.5  Positioning table:
        | Paper | Endogenous δ | AI benchmarks | Lifecycle | Developing country |
        Show: no paper combines all four elements

□ S5.6  Write Section: Related Work (2-3 pages)
```

## Sprint 6 — Publication Figures (Week 4)
**Objetivo:** 8-10 figuras de calidad Nature/Science

```
□ S6.1  Fig 1: AI Capability Advancement Curves
        - 5 benchmarks over time (2020-2025)
        - Frontier envelope
        - Annotate key model releases (GPT-4, Claude, etc.)
        - Log scale option for growth rate visualization

□ S6.2  Fig 2: Ω̇ Distribution Across Occupations
        - Histogram or density plot
        - Annotate key occupations (Actuaries, Software Dev, PT)
        - Color by ISCO major group

□ S6.3  Fig 3: Skill Half-Life Map
        - Bar chart: top 20 shortest + bottom 20 longest half-lives
        - Or: scatter plot t_{1/2} vs mean wage
        - Color by H^A/H^C composition

□ S6.4  Fig 4: Optimal Investment Paths s*(a) Under 4 Scenarios
        - 4 curves on same plot
        - Annotate switching ages a*
        - Shaded uncertainty bands

□ S6.5  Fig 5: Capital Stock Trajectories K^C(a) and K^A(a)
        - 2×4 panel: rows = K^C and K^A, columns = 4 scenarios
        - Or: 4 panels with both K types in each

□ S6.6  Fig 6: Age-Earnings Profiles — Model vs Data
        - Model prediction overlaid on GEIH empirical profile
        - Separate for formal vs informal
        - Calibration fit quality

□ S6.7  Fig 7: Welfare Loss as Function of Ω̇
        - X-axis: AI advancement rate
        - Y-axis: % welfare loss from constant-δ assumption
        - Annotate current Ω̇ and projected Ω̇

□ S6.8  Fig 8: Policy Diagram — Optimal Curriculum Mix
        - X-axis: Ω̇
        - Y-axis: optimal H^A share in curriculum (%)
        - Shade: "current policy" vs "optimal policy" gap

□ S6.9  Fig 9: Formal/Informal Depreciation Asymmetry
        - Experience returns by Ω̇ percentile
        - Two lines: formal (positive slope) vs informal (negative slope)
        - Crossing point highlighted

□ S6.10 Fig 10: Conceptual Diagram
        - Flow: AI Benchmarks → Ω̇ → δ(t) → t_{1/2} → Investment Path → Welfare
        - TikZ or high-quality diagram
```

## Sprint 7 — Expanded Writing (Week 4-5)
**Objetivo:** De 9pp a 25-30pp

```
□ S7.1  Section 1: Introduction (3 pages)
        - The depreciation puzzle
        - 4 contributions
        - Preview of results
        - Connection to CFE program

□ S7.2  Section 2: Related Work (2-3 pages)
        - 4 literature streams
        - Positioning table

□ S7.3  Section 3: Dynamic Model (4-5 pages)
        - Setup + assumptions
        - Accumulation equations
        - Worker's problem (Hamiltonian)
        - 5 propositions with proof sketches
        - Special cases

□ S7.4  Section 4: Measuring AI Advancement (2-3 pages)
        - Benchmark data
        - Benchmark-to-occupation mapping
        - Ω̇ results
        - Validation

□ S7.5  Section 5: Empirical Estimation (3-4 pages)
        - Cross-sectional specification
        - Main results (with Table)
        - Formal/informal asymmetry
        - Age heterogeneity
        - Implied depreciation rates and half-lives

□ S7.6  Section 6: Calibration (2-3 pages)
        - Target moments
        - Calibrated parameters
        - Model fit

□ S7.7  Section 7: Simulation (3-4 pages)
        - 4 scenarios
        - Optimal paths
        - Welfare analysis
        - Occupation-specific results

□ S7.8  Section 8: Robustness (2 pages)
        - 20+ specifications summary
        - Key sensitivity results

□ S7.9  Section 9: Policy Implications (2 pages)
        - Educational horizons
        - Reskilling design
        - Curriculum composition
        - Society 5.0 connection

□ S7.10 Section 10: Conclusion (1 page)

□ S7.11 Appendix A: Proofs (10 pages)
□ S7.12 Appendix B: Data Documentation (3 pages)
□ S7.13 Appendix C: Numerical Methods (3 pages)
□ S7.14 Appendix D: Full Robustness Tables (4 pages)
```

## Sprint 8 — Quality Assurance (Week 5-6)
**Objetivo:** Paper impecable, sin un solo error

```
□ S8.1   Verify ALL proofs line by line (invite coauthor review)
□ S8.2   Cross-check every number in tables vs code output
□ S8.3   Verify every figure is referenced in text
□ S8.4   Run Scopus check: no new competitor published
□ S8.5   Proofread: grammar, spelling, style consistency
□ S8.6   Check all \ref and \cite resolve (0 undefined)
□ S8.7   Verify .bib has all 40+ entries
□ S8.8   Test arXiv compilation (flat ZIP, \input{main.bbl})
□ S8.9   Generate replication package (all scripts + README)
□ S8.10  External review: Jiménez-Builes + Restrepo Carmona
□ S8.11  Final compile: target 0 warnings, 0 overfull boxes
□ S8.12  Submit to arXiv (econ.TH, cross-list econ.GN + cs.AI)
```

---

## Timeline

```
Week 1 (Apr 7-11):    Sprint 0 (Scopus) + Sprint 1 (proofs formales)
Week 2 (Apr 14-18):   Sprint 2 (calibración) + Sprint 3 (simulación)
Week 3 (Apr 21-25):   Sprint 4 (robustness) + Sprint 5 (related work)
Week 4 (Apr 28-May 2): Sprint 6 (figuras) + Sprint 7 (escritura expandida)
Week 5 (May 5-9):     Sprint 7 cont. + Sprint 8 (QA)
Week 6 (May 12-16):   Sprint 8 cont. + submission

Submission target: May 16, 2026
```

---

## Métricas de Calidad (Q1 Checklist)

| Criterio | Target | Cómo verificar |
|----------|--------|---------------|
| Pruebas formales | 5/5 completas | Appendix A reviewed |
| Referencias | ≥ 40 | \bibliography check |
| Figuras | ≥ 8 | All referenced in text |
| Robustness specs | ≥ 20 | Table en appendix |
| Calibration fit | R² > 0.8 moments | Reported in Table |
| Simulation scenarios | 4 | Figures 4-5-7 |
| Pages (main) | 25-30 | Word count |
| Pages (appendix) | 15-20 | Separate doc |
| Undefined refs | 0 | LaTeX log |
| Overfull boxes | ≤ 5 | LaTeX log |
| External review | 2 reviewers | Written feedback |
