# Augmented Human Capital (AHC) — Research Roadmap

**Paper:** *"Augmented Human Capital: A Unified Theory and LLM-Based Measurement Framework for Cognitive Factor Decomposition in AI-Augmented Economies"*

**Target journals:** Journal of Political Economy > Review of Economic Studies > American Economic Review

**Authors:** Cristian Espinal Maya (EAFIT/CIIID/ESUMER)

**Created:** 2026-03-30

---

## Executive Summary

This roadmap operationalizes a 10,000-word paper that:
1. Decomposes human capital H into three orthogonal components: H^P (physical-manual), H^C (cognitive-routine), H^A (cognitive-augmentable)
2. Builds an LLM-based measurement instrument (AHC index) for the augmentable component
3. Estimates an augmented Mincer equation with AI-interaction terms using Colombian GEIH microdata
4. Provides the first developing-country application with IV identification

---

## Phase 0: Infrastructure & Data Acquisition (Sprint 0)
**Duration:** 1-2 days | **Priority:** CRITICAL — blocks everything
**Goal:** Download all raw data, set up environment, validate crosswalks exist

### Sprint 0.1 — Environment Setup [~2 hours]
- [ ] `S0.1.1` Create conda environment `ahc` with Python 3.11, key packages
- [ ] `S0.1.2` Install: pandas, numpy, scipy, statsmodels, linearmodels, scikit-learn, matplotlib, seaborn, duckdb, requests, beautifulsoup4, tqdm, anthropic/openai SDKs
- [ ] `S0.1.3` Verify access to Llama 3 70B (via Anthropic API / local ollama / HuggingFace)
- [ ] `S0.1.4` Set up project-level .env with API keys (ANTHROPIC_API_KEY, ONET_API_KEY)
- [ ] `S0.1.5` Create symlink to automatizacion_colombia data: `ln -s "../Cognitive Factor Economics (CFE)/automatizacion_colombia/data" data/upstream_auto_col`

### Sprint 0.2 — O*NET Database Download [~1 hour, automatable]
- [ ] `S0.2.1` Download O*NET 30.2 database (all files): https://www.onetcenter.org/database.html
  - Tasks.csv (19,000+ task statements across 900+ occupations)
  - Abilities.csv (52 abilities × 900+ occupations with importance/level ratings)
  - Work Activities.csv (41 generalized + 300+ detailed work activities)
  - Knowledge.csv (33 knowledge areas)
  - Skills.csv (35 skills)
  - Technology Skills.csv (tools and technology per occupation)
  - Task Ratings.csv (task importance, frequency, relevance)
  - Occupation Data.csv (SOC codes, titles, descriptions)
- [ ] `S0.2.2` Download SOC-to-ISCO-08 crosswalk from BLS: https://www.bls.gov/soc/isco_soc_crosswalk.xls
- [ ] `S0.2.3` Download ISCO-08-to-CIUO-08-AC mapping from DANE
- [ ] `S0.2.4` Validate chain: SOC → ISCO-08 → CIUO-08 AC (Colombian occupations)

### Sprint 0.3 — ESCO Database Download [~30 min, automatable]
- [ ] `S0.3.1` Download ESCO v1.2 classification (CSV bulk download)
- [ ] `S0.3.2` Download ESCO-O*NET crosswalk: https://esco.ec.europa.eu/en/about-esco/data-science-and-esco/crosswalk-between-esco-and-onet
- [ ] `S0.3.3` Extract skills/competences linked to each ESCO occupation
- [ ] `S0.3.4` Validate ESCO-ISCO mapping consistency

### Sprint 0.4 — GEIH Microdata [~2 hours]
- [ ] `S0.4.1` Reuse existing GEIH 2024 from upstream (111,672 obs with CIUO-08 codes)
- [ ] `S0.4.2` Download GEIH 2022-2023 from DANE microdatos portal (for panel dimension)
- [ ] `S0.4.3` Download GEIH 2025 Q1-Q2 if available (post-ChatGPT window)
- [ ] `S0.4.4` Verify CIUO-08 occupational codes available at 4-digit level
- [ ] `S0.4.5` Extract key variables: OFICIO_C8, education, age, income, formality, sector, hours, firm_size

### Sprint 0.5 — Complementary Data [~3 hours, automatable]
- [ ] `S0.5.1` Download Felten et al. AIOE data from GitHub: https://github.com/AIOE-Data/AIOE
- [ ] `S0.5.2` Download Webb (2020) AI/software/robot patent exposure scores
- [ ] `S0.5.3` Download Eloundou et al. (2023) GPT exposure scores (supplementary data from Science)
- [ ] `S0.5.4` Download EAM 2016-2024 panel (reuse from upstream: 59,908 firm-years)
- [ ] `S0.5.5` Download IFR robot density data (Our World in Data proxy if IFR premium unavailable)
- [ ] `S0.5.6` Download patent data: USPTO PatentsView AI-classified patents by NAICS/technology class
- [ ] `S0.5.7` Download Penn World Table 11.0 (reuse from upstream)
- [ ] `S0.5.8` Download Revelio Labs public statistics (if WRDS access available via EAFIT)

### Sprint 0.6 — Upstream Data Integration [~1 hour]
- [ ] `S0.6.1` Copy/symlink references.bib from automatizacion_colombia (696 citations)
- [ ] `S0.6.2` Copy key processed datasets: automation_analysis_dataset.csv, eam_panel_constructed.csv
- [ ] `S0.6.3` Validate that OFICIO_C8 (CIUO-08 4-digit) maps to GEIH observations
- [ ] `S0.6.4` Produce data inventory report: rows, columns, coverage, missingness

**Deliverable:** All raw data in `data/raw/`, crosswalk files in `data/crosswalks/`, environment ready
**Overnight automation:** Scripts `01_download_onet.py`, `02_download_esco.py`, `03_download_complementary.py`

---

## Phase 1: Occupational Crosswalk & Task Mapping (Sprint 1)
**Duration:** 3-5 days | **Priority:** HIGH — blocks AHC index construction
**Goal:** Build a validated SOC → ISCO-08 → CIUO-08 crosswalk linking O*NET tasks to Colombian occupations

### Sprint 1.1 — SOC-to-ISCO Crosswalk [~4 hours]
- [ ] `S1.1.1` Parse BLS SOC-ISCO crosswalk (many-to-many mapping)
- [ ] `S1.1.2` Resolve many-to-many: create probabilistic weights based on employment distribution
- [ ] `S1.1.3` Handle unmapped codes: flag SOC codes without ISCO equivalent
- [ ] `S1.1.4` Validate: compare occupation counts before/after mapping
- [ ] `S1.1.5` Export: `crosswalk_soc_isco.parquet` with columns [SOC_code, ISCO_code, weight, confidence]

### Sprint 1.2 — ISCO-to-CIUO Crosswalk [~3 hours]
- [ ] `S1.2.1` Parse DANE CIUO-08 AC documentation for ISCO mapping rules
- [ ] `S1.2.2` CIUO-08 AC is a direct adaptation of ISCO-08 → verify 1:1 at 4-digit level
- [ ] `S1.2.3` Handle Colombian-specific additions (codes not in ISCO-08)
- [ ] `S1.2.4` Build CUOC (2023 update) mapping if GEIH 2025 uses new codes
- [ ] `S1.2.5` Export: `crosswalk_isco_ciuo.parquet`

### Sprint 1.3 — Chained Crosswalk & O*NET Task Attribution [~4 hours]
- [ ] `S1.3.1` Chain: SOC → ISCO → CIUO (probabilistic weights)
- [ ] `S1.3.2` Merge O*NET Tasks.csv with chained crosswalk
- [ ] `S1.3.3` For each CIUO occupation, produce a weighted task vector:
  - Task ID, task description, importance, frequency
  - Source O*NET SOC code(s) and crosswalk confidence
- [ ] `S1.3.4` Validate coverage: what % of GEIH workers have mapped tasks?
  - Target: >85% of weighted employment
- [ ] `S1.3.5` Handle unmapped occupations (military, subsistence farmers):
  - Manual task assignment from ESCO equivalents
  - Flag as lower-confidence
- [ ] `S1.3.6` Export: `occupation_task_matrix.parquet` [CIUO_4d × task_id × importance × frequency × confidence]

### Sprint 1.4 — Task Classification into H^P, H^C, H^A [~6 hours]
- [ ] `S1.4.1` Develop theoretical classification rubric:
  - **H^P (Physical-Manual):** Tasks requiring physical strength, dexterity, motor coordination
    - O*NET abilities: Manual Dexterity, Arm-Hand Steadiness, Static/Dynamic Strength, Stamina
    - Examples: "Operate machinery", "Load materials", "Drive vehicles"
  - **H^C (Cognitive-Routine):** Tasks requiring rule-following, data processing, pattern matching
    - O*NET abilities: Mathematical Reasoning, Perceptual Speed, Number Facility, Memory
    - Examples: "Process invoices", "Enter data", "Follow procedures", "Sort records"
  - **H^A (Cognitive-Augmentable):** Tasks requiring judgment, creativity, strategic thinking, communication
    - O*NET abilities: Originality, Fluency of Ideas, Problem Sensitivity, Inductive Reasoning
    - Examples: "Develop strategies", "Negotiate contracts", "Design solutions", "Counsel clients"
- [ ] `S1.4.2` Build rules-based initial classification using O*NET Work Activities + Abilities
- [ ] `S1.4.3` Cross-validate with Autor-Levy-Murnane (2003) routine/non-routine categories
- [ ] `S1.4.4` Cross-validate with Felten et al. AIOE abilities mapping
- [ ] `S1.4.5` Identify borderline tasks for LLM adjudication in Phase 2
- [ ] `S1.4.6` Export: `task_classification_rules.json`, `task_hpc_initial.parquet`

**Deliverable:** Complete crosswalk chain + initial task classification
**Validation criterion:** >85% GEIH employment covered, Cohen's kappa >0.7 vs. ALM categories

---

## Phase 2: AHC Index Construction via LLM Scoring (Sprint 2)
**Duration:** 5-7 days | **Priority:** CRITICAL — core methodological contribution
**Goal:** Use LLMs to score augmentation potential for each occupation-task pair → construct AHC_o index

### Sprint 2.1 — Prompt Engineering & Validation Design [~2 days]
- [ ] `S2.1.1` Design structured prompt for LLM scoring:
  ```
  CONTEXT: You are an expert labor economist evaluating how generative AI
  affects specific occupational tasks.

  OCCUPATION: {occupation_title} ({occupation_description})
  TASK: {task_description}

  QUESTION 1 — AUGMENTATION POTENTIAL (0-100):
  To what extent can a worker performing this task increase their productivity
  by using generative AI as a complementary tool (not a replacement)?
  Consider: Does AI help the worker think better, produce more, or achieve
  higher quality? Score 0 = no augmentation benefit, 100 = massive amplification.

  QUESTION 2 — SUBSTITUTION RISK (0-100):
  To what extent can generative AI fully replace the human worker in this task,
  producing equivalent or better output without human involvement?
  Score 0 = impossible to replace, 100 = fully automatable by AI.

  QUESTION 3 — AUGMENTATION TYPE:
  Classify the primary augmentation channel (if any):
  A) Information synthesis — AI helps process/organize information
  B) Creative amplification — AI helps generate ideas/solutions
  C) Communication enhancement — AI helps draft/translate/explain
  D) Decision support — AI provides analysis for human judgment
  E) Quality assurance — AI helps verify/improve human output
  F) No meaningful augmentation
  G) Pure substitution (AI replaces, doesn't augment)

  Provide confidence (LOW/MEDIUM/HIGH) for each score.
  ```
- [ ] `S2.1.2` Pilot test with 50 random occupation-task pairs using:
  - Claude Sonnet 4.6 (via AI Gateway)
  - Llama 3 70B (via local ollama or HuggingFace)
  - GPT-4o (via AI Gateway)
- [ ] `S2.1.3` Compute inter-rater reliability (Krippendorff's alpha) across LLMs
  - Target: alpha > 0.7 for augmentation scores
- [ ] `S2.1.4` Compare LLM scores with Eloundou et al. human annotations (convergent validity)
- [ ] `S2.1.5` Iterate prompt until reliability targets met
- [ ] `S2.1.6` Document final prompt version in `prompts/ahc_scoring_v_final.md`

### Sprint 2.2 — Full-Scale LLM Scoring Pipeline [~3 days, automatable overnight]
- [ ] `S2.2.1` Build scoring pipeline:
  - Input: occupation_task_matrix.parquet (~19,000 task statements)
  - Process: batch API calls with rate limiting, retry logic, cost tracking
  - Output: scored_tasks.parquet [task_id, CIUO_4d, augmentation_score, substitution_score, type, confidence]
- [ ] `S2.2.2` Run scoring with primary LLM (Llama 3 70B):
  - Estimated: 19,000 tasks × 3 questions = ~57,000 API calls
  - Cost estimate: ~$15-30 via API, free via local ollama
  - Time estimate: 4-8 hours with batching
- [ ] `S2.2.3` Run validation scoring with secondary LLM (Claude Sonnet):
  - Random 20% subsample (3,800 tasks) for reliability check
  - Cost estimate: ~$5-10 via AI Gateway
- [ ] `S2.2.4` Process raw scores:
  - Winsorize at 1st/99th percentile
  - Normalize to [0,1] range
  - Compute confidence-weighted averages across LLMs
- [ ] `S2.2.5` Log all API calls for reproducibility: `output/logs/llm_scoring_log.jsonl`

### Sprint 2.3 — AHC Index Aggregation [~1 day]
- [ ] `S2.3.1` For each CIUO-08 occupation o, compute:
  ```
  AHC_o = Σ_k (importance_k × frequency_k × augmentation_score_k) / Σ_k (importance_k × frequency_k)
  ```
  where k indexes tasks within occupation o
- [ ] `S2.3.2` Similarly compute:
  - `SUB_o`: Substitution index (weighted average of substitution scores)
  - `PHY_o`: Physical-manual index (share of H^P tasks by importance)
  - `ROU_o`: Routine-cognitive index (share of H^C tasks by importance)
- [ ] `S2.3.3` Verify orthogonality:
  - `cor(AHC_o, SUB_o)` should be low (<0.3) — augmentable ≠ automatable
  - `cor(AHC_o, PHY_o)` should be near zero — augmentable ≠ physical
  - `cor(AHC_o, ROU_o)` should be negative — augmentable ≠ routine
- [ ] `S2.3.4` Compare with existing indices:
  - Felten AIOE: `cor(AHC_o, AIOE_o)` — expect moderate positive (0.3-0.6)
  - Webb AI exposure: `cor(AHC_o, Webb_AI_o)` — expect moderate positive
  - Frey & Osborne: `cor(AHC_o, FO_o)` — expect near zero (they measure substitution)
- [ ] `S2.3.5` Export: `data/indices/ahc_index_by_occupation.parquet`
  - Columns: CIUO_4d, ISCO_4d, SOC_6d, occupation_title, AHC_o, SUB_o, PHY_o, ROU_o, n_tasks, confidence

### Sprint 2.4 — Index Validation [~2 days]
- [ ] `S2.4.1` **Convergent validity:**
  - Correlate AHC with Felten AIOE (ability-based AI exposure)
  - Correlate AHC with Eloundou GPT exposure (task-based LLM scoring)
  - Expected: moderate positive correlations (they measure related but distinct constructs)
- [ ] `S2.4.2` **Discriminant validity:**
  - AHC_o should NOT correlate with Frey-Osborne automation probability
  - AHC_o should be orthogonal to H^C (routine cognitive) by construction
  - Test: regress AHC_o on SUB_o + PHY_o + ROU_o → AHC residual should be large
- [ ] `S2.4.3` **Predictive validity (if Revelio data available):**
  - Do occupations with high AHC_o show increased demand for AI-complementary skills in job postings post-2023?
  - Outcome: change in job postings mentioning "AI", "machine learning", "data analysis" by occupation
- [ ] `S2.4.4` **Face validity:**
  - Top 10 AHC occupations: expect strategic managers, consultants, researchers, designers
  - Bottom 10 AHC occupations: expect manual laborers, machine operators, simple clerks
  - Review with domain experts (Santiago Jiménez, EAFIT colleagues)
- [ ] `S2.4.5` **Sensitivity analysis:**
  - Vary task importance weights (equal vs. O*NET importance vs. frequency)
  - Vary LLM (Llama vs. Claude vs. GPT): index stability across models
  - Vary prompt wording: alternative augmentation framings
  - Bootstrap CIs for occupation-level AHC scores
- [ ] `S2.4.6` Document validation results in `output/tables/ahc_validation_results.csv`

**Deliverable:** Validated AHC index for ~900 occupations mapped to CIUO-08
**This is the paper's MAIN CONTRIBUTION — everything else builds on this index**

---

## Phase 3: Theoretical Model Formalization (Sprint 3)
**Duration:** 5-7 days | **Priority:** HIGH — defines the paper's theoretical contribution
**Goal:** Derive the formal AHC model, prove key propositions, write Sections 1-2

### Sprint 3.1 — Model Setup & Definitions [~2 days]
- [ ] `S3.1.1` Formalize decomposition:
  ```
  H_i = H_i^P ⊕ H_i^C ⊕ H_i^A
  ```
  where ⊕ denotes orthogonal composition (not simple addition)
- [ ] `S3.1.2` Define worker type space: worker i is characterized by vector (h_i^P, h_i^C, h_i^A) ∈ R³₊
- [ ] `S3.1.3` Define digital labor stock D_f for firm f:
  - D_f = number of AI "task-equivalents" deployed (measurable via software investment, AI tool subscriptions)
  - Accumulation: D_f(t+1) = (1-δ_D) D_f(t) + I_D(t) where I_D is AI investment
- [ ] `S3.1.4` Define amplification function φ(D_f):
  - Properties: φ(0) = 1, φ'(D) > 0, φ''(D) < 0, lim φ(D→∞) = φ̄
  - Functional form: φ(D) = 1 + (φ̄ - 1)(1 - e^{-λD}) [bounded exponential]
  - Alternative: φ(D) = φ̄ · D^α / (D^α + D₀^α) [Hill function — from biochemistry]
- [ ] `S3.1.5` Write firm production function:
  ```
  Y_f = F(K_f^{HW}, X_f^{HW}, S_f^{SW})
  where:
    X_f^{HW} = G₁(Σᵢ h_i^P, κ K_f^{Rob})     [hardware aggregate: manual labor + robots]
    S_f^{SW} = G₂(Σᵢ h_i^C + φ(D_f) · Σᵢ h_i^A, D_f)  [software aggregate: cognitive + AI]
  ```
- [ ] `S3.1.6` Specify functional forms:
  - F: CES with elasticity σ_F < 1 between hardware and software (complementary)
  - G₁: CES with elasticity σ₁ > 1 (L^P and K^{Rob} are substitutes)
  - G₂: Nested CES with elasticity σ₂ < 1 between (H^C + φ·H^A) and D
- [ ] `S3.1.7` Discuss micro-foundations: why this decomposition is structural, not ad hoc
  - Connect to neuroscience of cognitive functions (dual-process theory: System 1 ≈ H^C, System 2 ≈ H^A)
  - Connect to Growiec's hardware/software distinction
  - Connect to Agrawal-Gans-Goldfarb's prediction/judgment decomposition

### Sprint 3.2 — Equilibrium & Key Propositions [~3 days]
- [ ] `S3.2.1` **Proposition 1 (Differential returns to education):**
  Solve first-order conditions for wages:
  ```
  w_i^A = ∂Y/∂h_i^A = F_S · G₂_H · φ(D_f) > w_i^C = F_S · G₂_H · 1
  ```
  When D_f > 0: returns to H^A exceed returns to H^C by factor φ(D_f) > 1
  **Implication:** Education that builds augmentable skills has higher returns in AI-adopting firms

- [ ] `S3.2.2` **Proposition 2 (Crossing point in wage distribution):**
  Define critical AI stock D* such that for D_f > D*:
  - Workers with high H^A/H^C ratio see wage increases
  - Workers with low H^A/H^C ratio see wage decreases
  Prove existence and uniqueness of D* under CES assumptions

- [ ] `S3.2.3` **Proposition 3 (Augmented Mincer equation):**
  Derive the wage equation from first-order conditions:
  ```
  ln w_{iot} = α + β₁·AHC_o + β₂·AHC_o × ln D_{f(i)t} + β₃·ROU_o × ln D_{f(i)t}
             + γ₁·S_i + γ₂·X_i + γ₃·X_i² + μ_i + δ_t + ε_{iot}
  ```
  Show β₂ > 0 (augmentation premium) and β₃ < 0 (routine displacement)

- [ ] `S3.2.4` **Proposition 4 (Amplification bounds):**
  Show that the aggregate productivity effect of AI adoption is bounded by:
  ```
  ΔY/Y ≤ s_A · (φ̄ - 1)
  ```
  where s_A = share of augmentable tasks in production
  Compare with Acemoglu (2024) Hulten's theorem result: ΔY/Y ≤ Σ s_k · Δz_k

- [ ] `S3.2.5` **Proposition 5 (Developing country comparative advantage):**
  Show that economies with:
  - High informality (large H^P share, small H^C share)
  - Lower AI adoption (small D)
  Have different augmentation dynamics than advanced economies
  The crossing point D* depends on sectoral composition → different path for Colombia vs. US

- [ ] `S3.2.6` Verify all proofs analytically (closed-form CES solutions)
- [ ] `S3.2.7` Numerical calibration of key parameters using existing literature

### Sprint 3.3 — Connection to Existing Theory [~2 days]
- [ ] `S3.3.1` Map AHC model to Acemoglu-Restrepo task-based framework:
  - Show displacement = D substituting H^C tasks
  - Show reinstatement = creation of new H^A tasks enabled by AI
  - AHC provides micro-foundation for WHY some tasks are displaced vs. reinstated
- [ ] `S3.3.2` Map to Growiec hardware/software:
  - Our G₁ ≈ Growiec's hardware composite
  - Our G₂ ≈ Growiec's software composite
  - Innovation: we decompose the human component of software
- [ ] `S3.3.3` Map to Becker/Mincer:
  - Standard Mincer: ln w = α + βS + γX + γ₂X²
  - AHC Mincer: decomposes βS into β_P·S^P + β_C·S^C + β_A·S^A×φ(D)
  - Show standard Mincer is a special case when D=0 or φ=1
- [ ] `S3.3.4` Discuss Dell'Acqua et al. "jagged frontier" through AHC lens:
  - The frontier is jagged because augmentation potential varies by cognitive factor
  - Tasks within the frontier have high AHC; tasks outside have low AHC or high SUB
- [ ] `S3.3.5` Write Sections 1 (Introduction) and 2 (Theoretical Framework) drafts

**Deliverable:** Complete theoretical model with 5 proved propositions + Sections 1-2 drafts

---

## Phase 4: Empirical Estimation (Sprint 4)
**Duration:** 7-10 days | **Priority:** HIGH — provides the empirical evidence
**Goal:** Estimate augmented Mincer equation, test propositions with GEIH data

### Sprint 4.1 — Sample Construction [~2 days]
- [ ] `S4.1.1` Merge AHC index onto GEIH 2024 microdata via CIUO-08 4-digit codes
- [ ] `S4.1.2` Merge AHC index onto GEIH 2022-2023 for time dimension
- [ ] `S4.1.3` Construct AI adoption proxy D_{f(i)t} at sector-year level:
  - **Option A:** Software investment from EAM by CIIU sector (available 2016-2024)
  - **Option B:** AI patent exposure by sector (Webb methodology applied to Colombian CIIU)
  - **Option C:** Interaction of sector × robot density from IFR (country-level)
  - Use all three and compare
- [ ] `S4.1.4` Apply sample restrictions:
  - Age 18-65, employed, non-military, positive income
  - Occupations with valid CIUO-08 → SOC crosswalk (target >85% coverage)
- [ ] `S4.1.5` Construct variables:
  - `AHC_i`: Worker's occupation-level augmentable human capital index
  - `SUB_i`: Worker's occupation-level substitution risk
  - `ROU_i`: Worker's occupation-level routine-cognitive score
  - `PHY_i`: Worker's occupation-level physical-manual score
  - `ln_D_s`: Log of sector-level AI/software adoption proxy
  - `AHC_x_lnD`: Interaction term (the key coefficient β₂)
  - `ROU_x_lnD`: Interaction term (the displacement coefficient β₃)
  - `education_years`: From GEIH education level mapping
  - `experience`: Age - education_years - 6 (Mincer potential experience)
  - `experience_sq`: Quadratic term
  - Controls: female, urban, departamento, firm_size, sector FE, year FE
- [ ] `S4.1.6` Produce descriptive statistics table (Table 1)
- [ ] `S4.1.7` Export: `data/processed/estimation_sample.parquet`

### Sprint 4.2 — OLS Estimation [~2 days]
- [ ] `S4.2.1` Estimate baseline Mincer (without AHC):
  ```
  ln w_i = α + β·S_i + γ₁·X_i + γ₂·X_i² + controls + ε_i
  ```
- [ ] `S4.2.2` Estimate AHC-augmented Mincer (main specification):
  ```
  ln w_i = α + β₁·AHC_i + β₂·AHC_i × ln D_s + β₃·ROU_i × ln D_s
         + β₄·SUB_i + β₅·PHY_i + γ·S_i + δ·X_i + δ₂·X_i²
         + controls + sector_FE + year_FE + ε_i
  ```
- [ ] `S4.2.3` Test Proposition 3: β₂ > 0 (augmentation premium) and β₃ < 0 (routine displacement)
- [ ] `S4.2.4` Progressive specification:
  - Model 1: Mincer only (S, X, X²)
  - Model 2: + AHC, SUB, ROU, PHY (level effects)
  - Model 3: + AHC×lnD, ROU×lnD (interaction effects — THE KEY)
  - Model 4: + full controls (firm size, urban, gender, department)
  - Model 5: + sector FE + year FE
- [ ] `S4.2.5` Compute and report:
  - Standard errors clustered at sector-department level
  - Sample weights from GEIH
  - R² progression across specifications
  - Marginal effects of AHC at different D levels
- [ ] `S4.2.6` Export: Tables 2-3 (regression results)

### Sprint 4.3 — IV Estimation [~3 days]
- [ ] `S4.3.1` Construct instrument for D_s (sector AI adoption):
  - **Primary IV:** Lagged sectoral patent intensity in AI/software (USPTO PatentsView)
    - Logic: US AI patenting in sector s at t-5 predicts Colombian sector s adoption at t
    - Exogeneity: Colombian firms don't drive US patent trends
  - **Alternative IV:** European robot density by sector (Acemoglu-Restrepo style)
    - From IFR data: EU robot installations by manufacturing sub-sector
    - Interacted with Colombian sector employment shares (Bartik instrument)
  - **Placebo IV:** Agricultural commodity prices (should NOT predict AI adoption)
- [ ] `S4.3.2` First stage:
  ```
  ln D_s = π₀ + π₁·Z_s + π₂·controls + sector_FE + year_FE + u_s
  ```
  Report first-stage F-statistic (target F > 10 for strong instrument)
- [ ] `S4.3.3` Second stage (2SLS):
  ```
  ln w_i = α + β₁·AHC_i + β₂·AHC_i × ln D̂_s + β₃·ROU_i × ln D̂_s + controls + ε_i
  ```
- [ ] `S4.3.4` Report:
  - First-stage F-statistics
  - Sargan/Hansen overidentification test (if multiple IVs)
  - Wu-Hausman endogeneity test (compare OLS vs. 2SLS)
  - Anderson-Rubin weak-instrument robust inference
- [ ] `S4.3.5` Export: Table 4 (IV results)

### Sprint 4.4 — Test Additional Propositions [~2 days]
- [ ] `S4.4.1` **Test Proposition 1 (Differential returns):**
  - Split sample by sector AI adoption: high-D vs. low-D sectors
  - Show returns to AHC higher in high-D sectors
  - Formal test: coefficient on AHC×high_D_dummy
- [ ] `S4.4.2` **Test Proposition 2 (Crossing point):**
  - Compute predicted wage change for each worker as function of D
  - Plot wage change vs. H^A/H^C ratio at different D levels
  - Identify empirical D* where low-H^A workers start losing
- [ ] `S4.4.3` **Test Proposition 5 (Developing country specifics):**
  - Interact AHC×lnD with formality status
  - Show augmentation premium is concentrated in formal sector
  - Show informal sector workers face different dynamics (H^P dominance)
- [ ] `S4.4.4` **Distributional analysis:**
  - Quantile regressions at τ = 0.10, 0.25, 0.50, 0.75, 0.90
  - Does AHC premium vary across wage distribution?
  - Gini decomposition with/without AHC interaction terms
- [ ] `S4.4.5` Export: Tables 5-6, Figures 3-5

**Deliverable:** Complete empirical results with OLS + IV + heterogeneity analysis

---

## Phase 5: Robustness & Extensions (Sprint 5)
**Duration:** 5-7 days | **Priority:** MEDIUM — strengthens the paper
**Goal:** Prove results are robust to alternative specifications, measurement, and samples

### Sprint 5.1 — Measurement Robustness [~2 days]
- [ ] `S5.1.1` Alternative AHC construction:
  - Equal-weighted tasks (no importance weights)
  - Binary classification (augmentable yes/no vs. continuous score)
  - PCA-based index from LLM scores
- [ ] `S5.1.2` Alternative LLM:
  - Re-estimate with Claude-scored AHC vs. Llama-scored AHC
  - Show main results hold regardless of scoring LLM
- [ ] `S5.1.3` Alternative crosswalk:
  - Use ESCO instead of O*NET as task source
  - Use 2-digit CIUO instead of 4-digit (coarser mapping, more robust)
- [ ] `S5.1.4` Alternative AI adoption measure:
  - Replace sector-level software investment with:
    - Sector-level broadband penetration
    - Sector-level ICT capital from EAM
    - Binary: post-ChatGPT (2023+) × AI-exposed occupation
- [ ] `S5.1.5` Export: Table 7 (robustness panel)

### Sprint 5.2 — Sample Robustness [~2 days]
- [ ] `S5.2.1` Restrict to manufacturing only (where EAM adoption data is best)
- [ ] `S5.2.2` Restrict to formal sector only
- [ ] `S5.2.3` Restrict to urban areas only
- [ ] `S5.2.4` Exclude Bogotá (dominates service sector)
- [ ] `S5.2.5` Restrict to age 25-55 (prime working age)
- [ ] `S5.2.6` Jackknife: leave-one-sector-out, leave-one-department-out
- [ ] `S5.2.7` Placebo tests:
  - Randomly permute AHC across occupations → β₂ should be zero
  - Use pre-period (2019-2021) data → effects should be weaker
- [ ] `S5.2.8` Export: Table 8 (sample robustness)

### Sprint 5.3 — Extensions [~3 days]
- [ ] `S5.3.1` **Gender heterogeneity:**
  - Interact AHC×lnD with female dummy
  - Are augmentation premiums gender-neutral?
- [ ] `S5.3.2` **Education heterogeneity:**
  - By education level: primary, secondary, technical, university, postgraduate
  - Does the AHC premium require a minimum education threshold?
- [ ] `S5.3.3` **Age heterogeneity:**
  - By age cohort: 18-30, 31-45, 46-65
  - Are younger workers better positioned to capture augmentation rents?
- [ ] `S5.3.4` **Firm size heterogeneity (from EAM):**
  - Large firms adopt AI faster → stronger augmentation effects?
- [ ] `S5.3.5` **Sectoral decomposition:**
  - Run main regression separately by sector
  - Where are augmentation effects strongest? (Expect: professional services, ICT, finance)
- [ ] `S5.3.6` **International comparison (if time permits):**
  - Apply same AHC methodology to US CPS data
  - Compare β₂ (augmentation premium) across countries
  - Test Proposition 5: developing country specifics
- [ ] `S5.3.7` Export: Tables 9-11, Figures 6-8

**Deliverable:** Comprehensive robustness package demonstrating result stability

---

## Phase 6: Paper Writing & Submission (Sprint 6)
**Duration:** 10-15 days | **Priority:** HIGH — the final product
**Goal:** Write, edit, and submit the 10,000-word paper

### Sprint 6.1 — First Draft [~5 days]
- [ ] `S6.1.1` Section 1: Introduction (1,500 words)
  - The measurement gap in human capital theory
  - What we do and why it matters
  - Preview of main results
- [ ] `S6.1.2` Section 2: Theoretical Framework (2,000 words)
  - AHC decomposition model
  - Production function with amplification
  - Key propositions (formal statements with proof sketches)
- [ ] `S6.1.3` Section 3: Measurement with LLMs (1,500 words)
  - Prompt design and validation
  - AHC index construction
  - Comparison with existing indices
- [ ] `S6.1.4` Section 4: Data (800 words)
  - GEIH, EAM, O*NET, ESCO, IFR
  - Sample construction
  - Descriptive statistics
- [ ] `S6.1.5` Section 5: Identification Strategy (800 words)
  - IV construction and validity
  - Threats to identification
- [ ] `S6.1.6` Section 6: Main Results (1,200 words)
  - OLS and IV estimates
  - Test of propositions
- [ ] `S6.1.7` Section 7: Robustness & Heterogeneity (800 words)
  - Alternative measures, samples, specifications
- [ ] `S6.1.8` Section 8: Implications for Theory (500 words)
  - Revised Mincer equation
  - Connection to Acemoglu-Restrepo, Growiec
- [ ] `S6.1.9` Section 9: Policy Implications (500 words)
  - Curriculum design
  - Reskilling programs
  - Society 5.0 for developing countries
- [ ] `S6.1.10` Section 10: Conclusion (400 words)

### Sprint 6.2 — Tables & Figures [~3 days]
- [ ] `S6.2.1` Table 1: Descriptive statistics
- [ ] `S6.2.2` Table 2: AHC index validation (correlations with existing indices)
- [ ] `S6.2.3` Table 3: Top/bottom 10 occupations by AHC score
- [ ] `S6.2.4` Table 4: OLS regression results (progressive specifications)
- [ ] `S6.2.5` Table 5: IV results (first stage + second stage)
- [ ] `S6.2.6` Table 6: Heterogeneity (gender, education, formality, sector)
- [ ] `S6.2.7` Table 7: Robustness checks
- [ ] `S6.2.8` Figure 1: Conceptual diagram of AHC decomposition
- [ ] `S6.2.9` Figure 2: AHC distribution across Colombian occupations
- [ ] `S6.2.10` Figure 3: AHC vs. Felten AIOE vs. Frey-Osborne scatter
- [ ] `S6.2.11` Figure 4: Marginal effect of AHC on wages at different D levels
- [ ] `S6.2.12` Figure 5: Crossing point — predicted wage change by H^A/H^C ratio
- [ ] `S6.2.13` Figure 6: Quantile regression coefficients across wage distribution
- [ ] `S6.2.14` Figure 7: Colombia vs. US augmentation premium comparison (if extension done)

### Sprint 6.3 — Review & Submission [~5 days]
- [ ] `S6.3.1` Internal review with Santiago Jiménez
- [ ] `S6.3.2` External review with CIIID/ESUMER colleagues
- [ ] `S6.3.3` Proofread and edit for clarity, concision, and flow
- [ ] `S6.3.4` Format for JPE submission (LaTeX template)
- [ ] `S6.3.5` Prepare supplementary materials:
  - Online appendix with full proofs
  - Data appendix with crosswalk documentation
  - Replication package (scripts + instructions)
- [ ] `S6.3.6` Write cover letter
- [ ] `S6.3.7` Submit to JPE
- [ ] `S6.3.8` If desk-rejected: revise for REStud or AER within 1 week

**Deliverable:** Submitted manuscript to top-5 journal

---

## Backlog (Future Iterations)

### Iteration 2 — Cross-Country Extension
- [ ] `B2.1` Apply AHC methodology to US CPS/ACS microdata
- [ ] `B2.2` Apply to EU-SILC microdata (European countries)
- [ ] `B2.3` Apply to India PLFS microdata
- [ ] `B2.4` Cross-country panel: AHC premium vs. AI adoption level
- [ ] `B2.5` Target: QJE or Econometrica (general equilibrium extension)

### Iteration 3 — Dynamic/Longitudinal Extension
- [ ] `B3.1` Use GEIH panel dimension (rotating sample) for individual-level dynamics
- [ ] `B3.2` Track AHC premium evolution 2019-2026 (pre/post ChatGPT)
- [ ] `B3.3` Estimate adjustment speed: how fast do augmentation premiums emerge?
- [ ] `B3.4` Target: REStud or JEEA

### Iteration 4 — Firm-Level with Training Data
- [ ] `B4.1` Obtain EDIT XI (2021-2022) from DANE — training investment data
- [ ] `B4.2` Estimate firm-level production function with AHC decomposition
- [ ] `B4.3` Test whether firms investing in AI also invest in H^A training
- [ ] `B4.4` Target: JPE or AER:Insights

### Iteration 5 — Policy Evaluation
- [ ] `B5.1` Simulate reskilling program effects using AHC framework
- [ ] `B5.2` Cost-benefit: redirecting education spending toward H^A skills
- [ ] `B5.3` Colombia CONPES 4144 evaluation using AHC metrics
- [ ] `B5.4` Target: Journal of Labor Economics or Economic Policy

### Iteration 6 — Methodological Paper
- [ ] `B6.1` Standalone paper on "LLMs as Measurement Instruments for Latent Economic Variables"
- [ ] `B6.2` Systematic comparison: LLM scoring vs. expert panels vs. mTurk
- [ ] `B6.3` Reliability, validity, and generalizability assessment
- [ ] `B6.4` Target: Journal of Econometrics or Review of Economics and Statistics

---

## Dependency Graph

```
Phase 0 (Data)
  ├─→ Phase 1 (Crosswalks)
  │     └─→ Phase 2 (AHC Index) ─────→ Phase 4 (Estimation)
  │                                         ├─→ Phase 5 (Robustness)
  │                                         └─→ Phase 6 (Paper)
  └─→ Phase 3 (Theory) ──────────────────────→ Phase 6 (Paper)
```

Phase 3 (Theory) can run in parallel with Phases 1-2.
Phase 5 (Robustness) starts as soon as Phase 4 produces initial results.
Phase 6 (Paper) starts Section 1-2 during Phase 3, adds empirical sections after Phase 4.

---

## Overnight Automation Plan (Tonight: 2026-03-30)

**Script:** `run_overnight.sh` — executes the following in sequence:

### Block 1: Downloads (~2 hours)
```
01_download_onet.py        → data/raw/onet/
02_download_esco.py        → data/raw/esco/
03_download_felten_webb.py → data/raw/felten/, data/raw/webb/
04_download_patents.py     → data/raw/patents/
05_download_ifr.py         → data/raw/ifr/
```

### Block 2: Crosswalks (~1 hour)
```
10_build_soc_isco_crosswalk.py   → data/crosswalks/soc_isco.parquet
11_build_isco_ciuo_crosswalk.py  → data/crosswalks/isco_ciuo.parquet
12_chain_crosswalk.py            → data/crosswalks/soc_ciuo_chained.parquet
```

### Block 3: Task Matrix (~1 hour)
```
13_build_task_matrix.py          → data/processed/occupation_task_matrix.parquet
14_classify_tasks_hpca.py        → data/processed/task_classification.parquet
```

### Block 4: Initial AHC Scoring (~4-8 hours)
```
20_score_tasks_llm.py            → data/indices/raw_llm_scores.jsonl
21_aggregate_ahc_index.py        → data/indices/ahc_index_by_occupation.parquet
22_validate_ahc_index.py         → output/tables/ahc_validation.csv
```

### Block 5: Merge & Descriptives (~1 hour)
```
30_merge_geih_ahc.py             → data/processed/estimation_sample.parquet
31_descriptive_statistics.py     → output/tables/descriptive_stats.csv
32_correlation_matrix.py         → output/figures/correlation_heatmap.png
```

**Total estimated runtime:** 8-12 hours
**Logs:** `output/logs/overnight_YYYYMMDD.log`

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| CIUO-08 → SOC crosswalk has poor coverage | Medium | High | Fall back to 2-digit ISCO aggregation; use ESCO as alternative source |
| LLM scoring unreliable (low inter-rater agreement) | Low | Critical | Use ensemble of 3 LLMs; increase pilot sample; iterate prompts |
| AI adoption proxy (D_f) poorly measured at firm level | High | High | Use sector-level aggregates; use multiple proxies; acknowledge limitation |
| IV too weak (F < 10) | Medium | High | Use Anderson-Rubin robust inference; combine IVs; present OLS as main with IV as robustness |
| GEIH occupational codes too coarse for task mapping | Medium | Medium | Aggregate AHC to 2-digit CIUO level; adjust confidence weights |
| Desk rejection at JPE | High | Medium | Simultaneous preparation for REStud/AER format; quick turnaround |
| Reviewers challenge LLM-as-instrument validity | Medium | Medium | Extensive validation section; compare with human annotations; cite growing econometric literature |

---

## Timeline

| Phase | Start | End | Dependencies |
|-------|-------|-----|-------------|
| Phase 0 | 2026-03-30 (tonight) | 2026-04-01 | None |
| Phase 1 | 2026-04-01 | 2026-04-05 | Phase 0 |
| Phase 2 | 2026-04-05 | 2026-04-12 | Phase 1 |
| Phase 3 | 2026-04-01 | 2026-04-08 | None (parallel) |
| Phase 4 | 2026-04-12 | 2026-04-22 | Phases 2+3 |
| Phase 5 | 2026-04-22 | 2026-04-29 | Phase 4 |
| Phase 6 | 2026-04-08 (Sec 1-2) | 2026-05-15 | Phases 3+4+5 |
| **Submission** | **2026-05-15** | | All phases |

---

## Key Metrics for Success

1. **AHC index reliability:** Krippendorff's alpha > 0.7 across LLMs
2. **Crosswalk coverage:** >85% of GEIH weighted employment mapped to O*NET tasks
3. **Augmentation premium (β₂):** Statistically significant and positive at 5% level
4. **Displacement effect (β₃):** Statistically significant and negative at 5% level
5. **IV strength:** First-stage F > 10 for at least one instrument
6. **Orthogonality:** cor(AHC, SUB) < 0.3, cor(AHC, PHY) < 0.1
7. **Prediction:** AHC outperforms AIOE/Webb in predicting post-2023 wage changes
