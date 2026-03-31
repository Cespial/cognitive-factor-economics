# Dinámica de Sustitución Tecnológica en Colombia

> **Technological Substitution Dynamics in Colombia: The Impact of Labor Costs on the Acceleration of Automation and Artificial Intelligence**
>
> Cristian Espinal Maya — Universidad EAFIT

## Resumen

Investigación académica que analiza si la estructura de costos laborales no salariales de Colombia acelera la adopción de automatización e inteligencia artificial. Utiliza microdatos de la GEIH 2024, EAM 2023, EDIT X 2019-2020, Penn World Table e IFR.

### Hallazgos principales

- **53.8%** de trabajadores colombianos en ocupaciones de alto riesgo de automatización
- Elasticidad del costo laboral: **0.84** — los costos no salariales incentivan la sustitución tecnológica
- Elasticidad de densidad robótica: **3.60** — efecto amplificador en el panel internacional
- **"Paradoja de la formalidad"** — la formalización laboral aumenta los incentivos de automatización

## Pipeline de análisis (Python)

| # | Script | Función |
|---|--------|---------|
| 0 | `00_explore_data.py` | Exploración inicial de datos |
| 1 | `01_international_panel.py` | Panel internacional (Penn World Table + IFR robots) |
| 2 | `02_sectoral_analysis.py` | Análisis sectorial de riesgo de automatización |
| 3 | `03_automation_risk_model.py` | Modelo de riesgo de automatización (Frey & Osborne adapted) |
| 4 | `04_firm_level_analysis.py` | Análisis a nivel de firma (EAM + EDIT) |
| 5 | `05_scenario_simulations.py` | Simulaciones de escenarios de política |
| 6 | `06_regenerate_figures.py` | Regeneración de figuras |
| 7 | `07_improved_figures.py` | Figuras mejoradas para publicación |
| 8 | `08_robustness_checks.py` | Pruebas de robustez |

## Fuentes de datos

| Fuente | Institución | Período |
|--------|------------|---------|
| GEIH | DANE | 2024 |
| EAM | DANE | 2023 |
| EDIT X | DANE | 2019–2020 |
| Penn World Table | Groningen | 10.01 |
| Robot Data | IFR | 2015–2022 |

## Paper

Versiones del paper en LaTeX:
- `main_en.tex` — versión en inglés
- `main_es.tex` — versión en español
- Template: `arxiv.sty`

## Figuras

~40 figuras de calidad publicación incluyendo:
- Comparación internacional de densidad robótica
- Coefficient plots de modelos econométricos
- Simulaciones de escenarios de política
- Heatmaps sectoriales de riesgo
- Curvas ROC del modelo de clasificación

## Stack

- Python 3 (pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn)
- LaTeX (arxiv template)

## Estructura

```
automatizacion_colombia/
├── main_en.tex / main_es.tex    # Paper (EN/ES)
├── arxiv.sty                     # Template
├── 00–08_*.py                    # Pipeline de análisis (9 scripts)
├── images/                       # Figuras (PNG)
├── data/                         # Microdatos DANE + IFR (excluido de git)
├── literature/                   # PDFs de referencia (excluido de git)
└── README.md
```

> **Nota:** Los datos fuente (~900 MB) no se incluyen. Requiere acceso a microdatos DANE.

## Autor

[Cristian Espinal Maya](https://github.com/Cespial) — Universidad EAFIT
