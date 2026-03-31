<p align="center">
  <strong>COGNITIVE FACTOR ECONOMICS</strong><br>
  <em>A Research Program on the Economics of Human-AI Complementarity</em>
</p>

<p align="center">
  <a href="#the-program">Program</a> &middot;
  <a href="#paper-map">Papers</a> &middot;
  <a href="#live-status">Status</a> &middot;
  <a href="#roadmap">Roadmap</a> &middot;
  <a href="#society-50-subline">Society 5.0</a> &middot;
  <a href="#data-architecture">Data</a> &middot;
  <a href="#how-to-reproduce">Reproduce</a>
</p>

---

## The Program

**Cognitive Factor Economics (CFE)** is a multi-paper research program that addresses the foundational measurement and theoretical gap in economics for the AI era:

> The Mincer equation is 52 years old. Becker's human capital theory is 62 years old. Both treat cognitive capacity as a scalar. In economies where AI co-produces cognitive output, **human capital is a vector whose internal composition determines whether a worker is amplified, displaced, or unaffected** — and no existing framework decomposes this formally.

CFE constructs that framework across six interconnected papers, a methodological innovation (LLMs as econometric instruments), and an empirical program spanning four Latin American countries with 1M+ microdata observations.

**Core theoretical object:** The *Augmented Human Capital* decomposition:

$$H_i = H_i^P \oplus H_i^C \oplus H_i^A$$

| Component | Symbol | What it captures | Relationship to AI |
|-----------|--------|-----------------|-------------------|
| Physical-Manual | $H^P$ | Motor coordination, physical presence | Unaffected by GenAI |
| Routine-Cognitive | $H^C$ | Rule-following, data processing, pattern matching | **Substituted** by AI |
| Augmentable-Cognitive | $H^A$ | Judgment, creativity, strategic reasoning, complex communication | **Amplified** by AI |

The **signature prediction**: the wage return to $H^A$ *increases* with firm-level AI adoption ($\beta_2 > 0$), while the return to $H^C$ *decreases* ($\beta_3 < 0$) — generating a crossing point in the wage distribution that the standard Mincer equation cannot detect.

> **Full theoretical exposition:** [`README_CFE.md`](README_CFE.md) — the complete academic document (567 lines, covering intellectual genealogy, formal model, measurement methodology, all 6 papers, data architecture, policy implications, and glossary).

---

## Paper Map

The program consists of **6 papers** organized in a dependency graph. Paper 1 is the foundational contribution; Papers 0 and 2-5 extend the framework to methodology, dynamics, geography, justice, and trade.

```
                    Paper 0 (Methodology)
                    LLMs as Econometric Instruments
                    Target: J. Econometrics / Econometrica
                           |
                           v
    Paper 1 ──────> AUGMENTED HUMAN CAPITAL <────── Paper 2
    (Core Theory)   The foundational model          (Dynamics)
    JPE / REStud    H^P + H^C + H^A decomposition   Cognitive Depreciation
                    Augmented Mincer equation        Endogenous Skill Half-Lives
                    LLM-based AHC index              J. Economic Theory
                    Colombia GEIH + EAM
                           |
              ┌────────────┼────────────┐
              v            v            v
         Paper 3      Paper 4      Paper 5
         Geography    Justice      Trade
         Cognitive    Augmentation Cognitive
         Augmentation Inequality   Comparative
         Clusters    & Capabilities Advantage
         J. Urban    J. Public    J. Intl.
         Economics   Economics    Economics
```

### Paper Status

| # | Title | Target Journal | Directory | Status |
|---|-------|---------------|-----------|--------|
| **1** | **Augmented Human Capital: A Unified Theory and LLM-Based Measurement Framework** | JPE / REStud / AER | [`augmented_human_capital/`](augmented_human_capital/) | **Active** — Pipeline running, LLM scoring in progress |
| 0 | LLMs as Econometric Instruments: Measuring What Cannot Be Surveyed | J. Econometrics | `llm_instruments/` (planned) | Planned — derived from Paper 1 methodology |
| 0.5 | Technological Substitution Dynamics in Colombia | Technovation (Q1, IF=10.9) | [`automatizacion_colombia/`](automatizacion_colombia/) | **Submitted** — double-blind review |
| 2 | Cognitive Capital Depreciation in the Age of GenAI | J. Economic Theory | `cognitive_depreciation/` (planned) | Planned |
| 3 | Cognitive Augmentation Clusters: A New Economic Geography | J. Urban Economics | `augmentation_geography/` (planned) | Planned |
| 4 | Distributive Justice in AI-Augmented Economies | J. Public Economics | `augmentation_justice/` (planned) | Planned |
| 5 | Cognitive Comparative Advantage: AI and Knowledge Service Trade | J. Intl. Economics | `cognitive_trade/` (planned) | Planned |

---

## Live Status

### Paper 1 — Augmented Human Capital (Active)

| Phase | Description | Status | Output |
|-------|-------------|--------|--------|
| 0 | Data acquisition (O\*NET, GEIH, EAM, crosswalks) | **Done** | 41 O\*NET files, 111K workers, 60K firm-years |
| 1 | Occupational crosswalk SOC-ISCO-CIUO | **Done** | 66,157 chained mappings |
| 1 | Task-occupation matrix | **Done** | 2,229,863 task-occupation pairs |
| 1 | H^P/H^C/H^A rules-based classification | **Done** | HA 71%, HP 21%, HC 8% |
| 2 | **LLM scoring (Claude Haiku)** | **In progress** | ~5K/18,796 tasks scored, 0 errors |
| 2 | AHC index aggregation | Pending | Blocked by scoring completion |
| 2 | Index validation (Felten, Webb, F&O) | Pending | |
| 3 | Theoretical model formalization | Pending | 5 propositions to prove |
| 4 | Augmented Mincer estimation (OLS + IV) | Pending | |
| 5 | Robustness & heterogeneity | Pending | |
| 6 | Paper writing & submission | Pending | Target: May 2026 |

### Paper 0.5 — Automatizacion Colombia

| Component | Status |
|-----------|--------|
| 15 Python scripts (DiD, logit, simulations, figures) | Complete |
| LaTeX manuscript (ES + EN, 10K words) | Complete |
| Technovation submission (double-blind) | **Under review** |

---

## Roadmap

### 2026 Q2 (April-June) — Paper 1: Augmented Human Capital

```
Week 1-2 (Apr 1-12)
├── Complete LLM scoring (18,796 tasks)
├── Aggregate AHC index by occupation
├── Validate: AHC vs Felten AIOE, Webb, Frey-Osborne
├── Prove orthogonality: cor(AHC, SUB) < 0.3
└── Formalize theoretical model (5 propositions)

Week 3-4 (Apr 13-22)
├── Merge AHC onto GEIH estimation sample (105K workers)
├── Construct AI adoption proxy from EAM (sector-level)
├── Build patent-based IV for AI adoption
├── OLS estimation of augmented Mincer equation
└── IV estimation (2SLS) + first-stage diagnostics

Week 5-6 (Apr 23-May 4)
├── Robustness: alternative AHC measures, samples, specifications
├── Heterogeneity: gender, education, formality, sector, age
├── Quantile regressions across wage distribution
└── Distributional analysis (crossing point estimation)

Week 7-10 (May 5-31)
├── Write full paper (10 sections, 10,000 words)
├── Publication-quality tables (7+) and figures (7+)
├── Internal review (Santiago Jimenez, EAFIT colleagues)
├── LaTeX formatting for JPE
└── Submit to Journal of Political Economy
```

### 2026 Q3 (July-September) — Paper 0: Methodology

```
├── Extract measurement methodology from Paper 1
├── Systematic comparison: LLM vs expert panels vs mTurk
├── Reliability, validity, generalizability framework
├── Application beyond labor economics (contract analysis, policy text)
├── Write standalone methodology paper
└── Submit to Journal of Econometrics
```

### 2026 Q4 (October-December) — Paper 2: Cognitive Depreciation

```
├── Dynamic human capital model with endogenous depreciation
├── Optimal control: investment path in H^A vs H^C under AI acceleration
├── Temporal embeddings of AI capability benchmarks (MMLU, HumanEval)
├── Estimate differential timing of skill depreciation by occupation
├── Write and submit to Journal of Economic Theory
└── Begin data collection for Paper 3 (Geography)
```

### 2027 H1 — Papers 3 & 4: Geography and Justice

```
Paper 3: Geography
├── NEG model with institutional complementarity distance
├── Department-level data for Colombia
├── NUTS-2 data for Europe
├── Multiple equilibria estimation
└── Submit to Journal of Urban Economics

Paper 4: Justice
├── Sen-Nussbaum capabilities extension with augmentation dimension
├── Rawlsian optimal AI access policy derivation
├── Cross-country augmentation inequality measurement (5 LatAm countries)
├── Policy simulation: universal minimum augmentation access
└── Submit to Journal of Public Economics
```

### 2027 H2 — Paper 5: Trade + Program Synthesis

```
Paper 5: Cognitive Comparative Advantage
├── 2-country, 2-sector trade model with H^A/H^C composition
├── Calibrate with WTO/IMF trade-in-services data
├── Test predictions against post-2022 knowledge service trade flows
├── Implications for BPO-dependent economies (India, Philippines, Colombia)
└── Submit to Journal of International Economics

Program Synthesis
├── CFE book proposal (Cambridge University Press / Princeton University Press)
├── Policy white paper for OECD/CEPAL/DNP
└── Conference keynotes and workshops
```

---

## Society 5.0 Subline

CFE provides the **economic foundations** for Society 5.0 — the paradigm articulated by Japan's Cabinet Office (2016) and developed by the European Commission (2021) as the successor to Industry 4.0.

Where Industry 4.0 focuses on automation of physical and routine cognitive processes, Society 5.0 focuses on **human-centricity** in an AI-saturated world.

### Research Agenda: Economics of Society 5.0

This subline develops the institutional and policy implications of CFE for Society 5.0 transitions, with particular focus on Latin America and developing economies.

| Paper | Title | Core Question | Status |
|-------|-------|--------------|--------|
| S5.0-A | **The Economic Definition of Human-Centricity** | What does "human-centric AI" mean in production function terms? CFE proposes: maximize $H^A$ investment, not minimize $H^C$ displacement | Conceptual framework in [README_CFE.md](README_CFE.md) Section 7 |
| S5.0-B | **Institutional Requirements for Society 5.0 Transitions** | What institutional threshold must a region cross to achieve cognitive augmentation clusters? Data governance, educational alignment, labor flexibility, AI access policy | Linked to Paper 3 (Geography) |
| S5.0-C | **Society 5.0 Welfare Criterion** | How to evaluate whether a country is progressing toward Society 5.0? Proposes: augmentation inequality decreasing while average augmentation capability increases | Linked to Paper 4 (Justice) |
| S5.0-D | **Colombia's Path to Society 5.0: CONPES 4144 Evaluation** | Does Colombia's National AI Policy (2025, $479B COP) target $H^A$ development or $H^C$ protection? Use AHC framework to evaluate policy alignment | Planned — after Paper 1 results |
| S5.0-E | **Society 5.0 in the Global South: Informality as Structural Constraint** | How does labor market informality (40-70% in LatAm) alter the transition dynamics to Society 5.0? Informal workers are shielded from displacement but excluded from augmentation | Planned — cross-cuts Papers 1, 3, 4 |
| S5.0-F | **Educational System Redesign for Augmentable Human Capital** | What curriculum changes maximize $H^A$ formation? Shift from routine-cognitive skills (data processing, standard coding) to augmentable skills (contextual judgment, cross-domain synthesis, ethical reasoning) | Policy paper — after Paper 2 (Depreciation) |

### Society 5.0 Target Outputs

- **Academic:** Papers S5.0-A through S5.0-F targeting policy-relevant journals (World Development, Economic Policy, CEPAL Review)
- **Policy:** White papers for DNP (Colombia), CEPAL, OECD, World Bank
- **Institutional:** Input to CONPES 4144 implementation evaluation
- **Education:** Curriculum framework for $H^A$ development in Colombian universities (EAFIT, ESUMER)

### Key Society 5.0 Concepts from CFE

| Concept | Definition | Measurement |
|---------|-----------|-------------|
| **Augmentation Readiness** | A region's capacity to productively deploy AI-human complementarity | Composite: broadband + STEM density + data governance + $H^A$ share |
| **Institutional Complementarity Distance** | Gap between a region's institutional profile and the threshold for productive augmentation | Paper 3 operationalization |
| **Augmentation Inequality** | Distribution of effective access to AI-augmented cognitive production | AHC index + digital access indicators |
| **Cognitive Augmentation Cluster** | Geographic concentration of $H^A$-intensive activity | Paper 3 identification |
| **Skill Half-Life** | Time for a cognitive skill's economic value to reach 50% under AI depreciation | Paper 2 estimation |

---

## Data Architecture

### Layer 1: Occupational Task Data
| Source | Coverage | Purpose |
|--------|----------|---------|
| **O\*NET 30.2** | 900+ US occupations, 19K task statements, 52 abilities | Task decomposition, AHC scoring input |
| **ESCO v1.2** | 3,000+ European occupations, multilingual | Cross-validation, non-US coverage |
| **CIUO-08 AC** | Colombian adaptation of ISCO-08 | Crosswalk to GEIH occupations |

### Layer 2: LLM-Generated Indices
| Index | Method | Validation |
|-------|--------|-----------|
| **AHC (Augmentable Human Capital)** | Claude Haiku scoring of 19K tasks | vs. Felten AIOE, Webb, F&O |
| **SUB (Substitution Risk)** | Same LLM pipeline, substitution dimension | Orthogonal to AHC by design |
| **PHY/ROU (Physical/Routine)** | Rules-based from O\*NET abilities | ALM (2003) cross-validation |

### Layer 3: Household Survey Microdata
| Country | Survey | Sample | Period | Status |
|---------|--------|--------|--------|--------|
| **Colombia** | GEIH (DANE) | 111,672 obs (2024) | 2018-2025 | **Loaded** |
| Mexico | ENOE (INEGI) | ~300K/quarter | 2018-2025 | Planned |
| Chile | CASEN (INE) | ~200K biennial | 2017-2024 | Planned |
| Brazil | PNAD Continua (IBGE) | ~500K/quarter | 2018-2025 | Planned |

### Layer 4: Firm-Level Data
| Source | Variables | Status |
|--------|----------|--------|
| **EAM (DANE)** | 59,908 firm-years, machinery investment, labor costs | **Loaded** |
| **EDIT X (DANE)** | Software investment, R&D, innovation type | Merged |
| Revelio Labs | Job postings, AI skill demand | Planned (WRDS) |
| USPTO PatentsView | AI patents by CPC class | Downloaded |
| IFR (OWID proxy) | Robot density by country | Downloaded |

### Layer 5: Existing Indices
| Index | Authors | Status |
|-------|---------|--------|
| Felten AIOE | Felten, Raj & Seamans (2021) | Pending download (URL fix) |
| Webb AI Exposure | Webb (2020) | Planned |
| Eloundou GPT Exposure | Eloundou et al. (2023) | Planned |
| Penn World Table 11.0 | Feenstra et al. | **Available** |

---

## Repository Structure

```
cognitive-factor-economics/
│
├── README.md                          # This file — repo overview, roadmap, status
├── README_CFE.md                      # Full academic program document (theory, 6 papers)
│
├── augmented_human_capital/           # PAPER 1 — Active development
│   ├── README.md                      # Paper 1 overview and quick start
│   ├── ROADMAP.md                     # Detailed 6-phase sprint plan
│   ├── run_overnight.sh               # Master overnight pipeline
│   ├── scripts/
│   │   ├── 00_setup/                  # Environment setup
│   │   ├── 01_data_acquisition/       # O*NET, Felten, IFR, patent downloads
│   │   ├── 02_crosswalks/             # SOC → ISCO → CIUO mapping chain
│   │   ├── 03_ahc_index/             # LLM scoring + AHC aggregation
│   │   ├── 04_econometrics/          # Estimation sample, Mincer equation
│   │   ├── 05_robustness/            # (planned)
│   │   └── 06_figures/               # (planned)
│   ├── literature/
│   │   └── references.bib            # 50+ core references
│   └── paper/                         # LaTeX manuscript (planned)
│
├── automatizacion_colombia/           # PAPER 0.5 — Under review at Technovation
│   ├── scripts/ (15 scripts)          # Full pipeline: DiD, logit, simulations
│   ├── *.tex                          # Manuscripts (ES + EN)
│   ├── submission_technovation/       # Final submission package
│   ├── literature/ (11 PDFs)          # Key reference papers
│   └── references.bib                 # 696 BibTeX entries
│
├── llm_instruments/                   # PAPER 0 — Methodology (planned)
├── cognitive_depreciation/            # PAPER 2 — Dynamics (planned)
├── augmentation_geography/            # PAPER 3 — Geography (planned)
├── augmentation_justice/              # PAPER 4 — Justice (planned)
├── cognitive_trade/                   # PAPER 5 — Trade (planned)
│
└── society_5_0/                       # SUBLINE — Society 5.0 (planned)
    ├── conpes_4144_evaluation/        # S5.0-D: Colombia AI Policy evaluation
    ├── informality_augmentation/      # S5.0-E: Informality as constraint
    └── curriculum_ha/                 # S5.0-F: Education for H^A
```

---

## Key Hypotheses Across the Program

| ID | Hypothesis | Paper | Testable with |
|----|-----------|-------|---------------|
| H1 | Workers in high-$AHC$ occupations receive a wage premium that **increases** with AI adoption | 1 | GEIH + EAM |
| H2 | Workers in high-$H^C$ occupations receive a wage penalty that **increases** with AI adoption | 1 | GEIH + EAM |
| H3 | The augmentation premium and routine penalty exist **within** education levels | 1 | GEIH |
| H4 | Below threshold $D^*$, AI depresses wages; above it, wages rise for $H^A$-intensive workers | 1 | GEIH + EAM |
| H5 | Routine cognitive skills depreciate faster when AI capability advances faster | 2 | MMLU/HumanEval benchmarks |
| H6 | Regions below institutional complementarity threshold cannot form augmentation clusters | 3 | Department-level Colombia |
| H7 | Augmentation inequality is larger in high-income-inequality, weak-governance countries | 4 | 5-country LatAm panel |
| H8 | Post-2022 knowledge service trade is better predicted by $H^A/H^C$ than by wages or education | 5 | WTO/IMF BOP |

---

## How to Reproduce

### Paper 1 Pipeline

```bash
# Clone
git clone https://github.com/Cespial/cognitive-factor-economics.git
cd cognitive-factor-economics/augmented_human_capital

# Setup environment (Python 3.11+)
python3 scripts/00_setup/setup_environment.py

# Run overnight pipeline (downloads + crosswalks + classification)
./run_overnight.sh --skip-llm

# Run LLM scoring (requires ANTHROPIC_API_KEY, ~$5 with Haiku)
echo "ANTHROPIC_API_KEY=your_key" > .env
python3 scripts/03_ahc_index/20_score_tasks_llm.py

# Aggregate AHC index
python3 scripts/03_ahc_index/21_aggregate_ahc_index.py

# Merge with GEIH and produce descriptives
python3 scripts/04_econometrics/30_merge_geih_ahc.py
python3 scripts/04_econometrics/31_descriptive_statistics.py
```

**Note:** GEIH and EAM microdata require manual download from [DANE](https://microdatos.dane.gov.co). The pipeline expects these in `data/upstream_auto_col/`.

### Paper 0.5 Pipeline

```bash
cd automatizacion_colombia/scripts
python3 00_explore_data.py       # Data profiling
python3 01_international_panel.py # 25-country panel regressions
python3 03_automation_risk_model.py # Frey-Osborne mapping to Colombia
python3 09_did_panel_eam.py      # Difference-in-differences (EAM)
python3 10_did_geih_mw_shock.py  # Minimum wage natural experiment
```

---

## Citation

```bibtex
@techreport{espinal2026cfe,
  title={Cognitive Factor Economics: A Research Program on the Economics
         of Human-AI Complementarity in Society 5.0},
  author={Espinal Maya, Cristian},
  institution={INPLUX / CIIID / ESUMER},
  year={2026},
  address={Medell{\'i}n, Colombia}
}

@techreport{espinal2026ahc,
  title={Augmented Human Capital: A Unified Theory and LLM-Based
         Measurement Framework for Cognitive Factor Decomposition
         in AI-Augmented Economies},
  author={Espinal Maya, Cristian},
  institution={Universidad EAFIT},
  year={2026},
  note={Working paper}
}
```

---

## Author

**Cristian Espinal Maya** ([@Cespial](https://github.com/Cespial))

- M.A. in Economics, Universidad EAFIT
- Doctoral candidate (Engineering)
- Professor, Postgraduate Division, ESUMER
- Research Associate, CIIID (Minciencias-recognized)
- Founder, INPLUX SAS (Fourier.dev)
- Chief AI Architect, REDEK

---

## License

Research program intellectual property of Cristian Espinal / INPLUX SAS.
Code released under MIT License. Papers and datasets under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

<p align="center"><em>Cognitive Factor Economics — INPLUX Research Division</em><br>Medellín, Colombia &middot; 2026</p>
