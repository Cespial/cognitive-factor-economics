# Augmented Human Capital (AHC)

**Paper:** *"Augmented Human Capital: A Unified Theory and LLM-Based Measurement Framework for Cognitive Factor Decomposition in AI-Augmented Economies"*

**Target:** Journal of Political Economy / Review of Economic Studies / American Economic Review

**Author:** Cristian Espinal Maya (EAFIT / CIIID / ESUMER)

## The Problem

The Mincer equation (1974) treats human capital as a scalar — years of education + experience. In the AI era, what matters is the **composition** of cognitive capabilities: which are substituted by AI, which are amplified, and which are unaffected. No existing framework decomposes this formally.

## The Contribution

1. **Theory:** Decomposes H into H^P (physical), H^C (cognitive-routine), H^A (cognitive-augmentable)
2. **Measurement:** Uses LLMs to score 19,000+ O*NET tasks for augmentation potential → AHC index
3. **Estimation:** Augmented Mincer equation with AI-interaction terms, IV identification
4. **Application:** First developing-country evidence using Colombian GEIH microdata (111K workers)

## Project Structure

```
augmented_human_capital/
├── ROADMAP.md              # Full research roadmap (6 phases, sprints, backlog)
├── run_overnight.sh        # Master overnight processing script
├── data/
│   ├── raw/                # O*NET, ESCO, GEIH, Felten, IFR, patents
│   ├── crosswalks/         # SOC → ISCO → CIUO mappings
│   ├── processed/          # Task matrices, estimation samples
│   └── indices/            # AHC index, LLM scores
├── scripts/
│   ├── 00_setup/           # Environment setup
│   ├── 01_data_acquisition/# Download scripts
│   ├── 02_crosswalks/      # Crosswalk construction
│   ├── 03_ahc_index/       # LLM scoring & index aggregation
│   ├── 04_econometrics/    # Estimation & descriptives
│   ├── 05_robustness/      # Robustness checks
│   └── 06_figures/         # Publication figures
├── literature/
│   └── references.bib      # BibTeX (50+ core references)
├── output/
│   ├── tables/
│   ├── figures/
│   └── logs/
└── paper/
    ├── sections/
    └── submission/
```

## Quick Start

```bash
# Run overnight pipeline (skip LLM scoring initially)
./run_overnight.sh --skip-llm

# Or with LLM scoring (requires ANTHROPIC_API_KEY or local Ollama)
export ANTHROPIC_API_KEY=your_key
./run_overnight.sh
```

## Data Sources

| Source | Variables | Coverage |
|--------|-----------|----------|
| GEIH (DANE) | Wages, occupation, education, formality | 111,672 workers, 2024 |
| EAM (DANE) | Firm investment, automation proxy | 59,908 firm-years, 2016-2024 |
| O*NET 30.2 | 19,000+ task statements, abilities, skills | 900+ US occupations |
| ESCO v1.2 | European skills/competences taxonomy | Cross-validation |
| Felten AIOE | AI Occupational Exposure Index | 774 SOC occupations |
| IFR | Industrial robot density | 40+ countries |

## Key Dependencies

- Python 3.11+ (pandas, numpy, scipy, statsmodels, linearmodels)
- LLM access: Anthropic API, OpenAI API, or local Ollama (Llama 3 70B)
- Upstream data: `automatizacion_colombia` project (symlinked)
