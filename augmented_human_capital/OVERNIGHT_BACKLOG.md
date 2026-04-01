# Overnight Backlog — Paper 1 Final Push

**Fecha:** 2026-03-31 noche → 2026-04-01 mañana
**Objetivo:** Artículo listo para publicar en arXiv al despertar

---

## Lo que corre ESTA NOCHE (automático)

### `run_overnight_final.sh` ejecuta:

| Step | Qué hace | Duración est. | Output |
|------|----------|--------------|--------|
| 1 | Espera a que Sonnet termine (506/3,760 ahora) | ~1.5h | `sonnet_validation_scores.jsonl` |
| 2 | Calcula reliability final (Pearson, Spearman, Krippendorff α) | 10s | `final_interrater_reliability.json` |
| 3 | Regenera 5 figuras PDF | 30s | `figs/*.pdf` |
| 4 | Recompila LaTeX (3 pasadas + bibtex) | 30s | `main.pdf` (13+ páginas) |
| 5 | Crea `arxiv_upload.zip` (tex + bbl + figs) | 5s | `arxiv_upload.zip` |
| 6 | Commit + push a GitHub | 10s | commit automático |
| 7 | Genera resumen matutino en log | — | `overnight_final_*.log` |

**Para lanzar antes de dormir:**
```bash
cd "/Users/cristianespinal/Claude Code/Projects/Research/augmented_human_capital"
nohup bash run_overnight_final.sh > /dev/null 2>&1 &
```

---

## Lo que REVISAS mañana (manual, ~2 horas)

### Prioridad 1 — Lectura y ajustes del manuscrito (~1 hora)

- [ ] **Abrir PDF:** `open paper/arxiv_submission/main.pdf`
- [ ] **Revisar abstract:** ¿Los claims matchean exactamente los coeficientes reportados? Verificar que todos los β y p-values estén correctos
- [ ] **Revisar Table 2 (regresiones):** Los coeficientes están hardcoded — verificar contra `output/tables/table4_augmented_mincer_v2.csv`
- [ ] **Revisar Table 3 (heterogeneity):** Verificar β₂ por subgrupo contra `output/tables/table7_robustness_full.csv`
- [ ] **Revisar figuras:** ¿Las 5 PDFs se ven bien? ¿Legibles en print?
- [ ] **Verificar inter-rater:** Abrir `output/tables/final_interrater_reliability.json` — ¿el alpha mejoró con más datos?
- [ ] **Verificar referencias:** ¿Aparecen los 27 bibitem en la bibliografía?
- [ ] **Leer conclusión:** ¿Las 4 implicaciones son precisas?

### Prioridad 2 — Refinamientos del texto (~30 min)

- [ ] **Título:** ¿"Augmented Human Capital" es el mejor título? Alternativas:
  - "The Augmentation Premium: Cognitive Factor Decomposition in AI-Augmented Labor Markets"
  - "Beyond Automation Risk: LLM-Based Measurement of Human-AI Complementarity"
  - "Who Benefits from AI? Cognitive Augmentation and the Formality Divide"
- [ ] **Agradecimientos:** ¿Agregar otros beyond Jiménez-Builes y Restrepo Carmona?
- [ ] **Keywords:** ¿Agregar "informality" como keyword? ¿"Mincer equation"?
- [ ] **JEL codes:** Verificar que J24 (Human Capital), J31 (Wage Differentials), O33 (Technological Change), O15 (Development), C55 (LLM/computational) son los correctos

### Prioridad 3 — Decisiones de submission (~15 min)

- [ ] **Categoría arXiv:**
  - Primary: `econ.GN` (General Economics)
  - Cross-list: `econ.EM` (Econometrics), `cs.AI` (Artificial Intelligence)
  - Alternativa: `econ.LB` si existe categoría de Labor Economics
- [ ] **License:** Creative Commons CC-BY 4.0 (recomendado para máxima citabilidad)
- [ ] **Comments field:** "Working paper. 13 pages, 3 tables, 5 figures. Comments welcome."
- [ ] **Report number:** Puede dejarse vacío o usar "INPLUX-WP-2026-001"

### Prioridad 4 — Verificación técnica antes de submit (~15 min)

- [ ] **Compilar localmente:**
  ```bash
  cd paper/arxiv_submission
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
  ```
- [ ] **Verificar ZIP contiene todo:**
  ```bash
  unzip -l paper/arxiv_upload.zip
  ```
  Debe tener: `main.tex`, `main.bbl`, `fig2_ahc_by_sector.pdf`, `fig3_ahc_vs_fo.pdf`, `fig5_heterogeneity.pdf`, `fig6_quantile_regression.pdf`, `fig7_validation.pdf`
- [ ] **Test: arXiv acepta el ZIP** — subir como test submission (se puede cancelar antes de publicar)

---

## Backlog FUTURO (post-arXiv, no urgente)

### Mejoras para versión 2 (si hay feedback de arXiv/reviewers)

| Item | Dificultad | Impacto |
|------|-----------|---------|
| Mejorar crosswalk a 4-digit con IBS data (.dta) | Media | Alto — más variación |
| GEIH 2022-2023 para dimensión panel pre/post ChatGPT | Alta | Muy alto — event study |
| Prompt sensitivity: 3 prompts alternativos, reportar variación | Media | Medio — robustness |
| Añadir Eisfeldt GenAI exposure a validación (ya descargado) | Baja | Medio |
| Figuras adicionales: heatmap de correlaciones, mapa de Colombia | Baja | Bajo |
| Online Appendix completo (todas las 47 specs en tabla) | Media | Alto para journal |
| Replicación con CPS/ACS de EE.UU. para comparación cross-country | Alta | Muy alto |

### Para journal submission (post-arXiv, post-feedback)

- [ ] Elegir journal target: JPE (más ambicioso) vs Labour Economics (más probable) vs Technological Forecasting and Social Change (IF=12.9, Q1)
- [ ] Reformatear a template del journal elegido
- [ ] Expandir a ~25 páginas (journals permiten más que arXiv working papers)
- [ ] Agregar Online Appendix formal
- [ ] Cover letter

---

## Resumen de lo logrado hoy (2026-03-31)

```
Completado:
  ✓ 18,796 tasks scored con LLM (0 errores)
  ✓ AHC index v2 con crosswalk mejorado (430 ocupaciones)
  ✓ Augmented Mincer estimation (AHC×D = +0.147***)
  ✓ IV estimation (F=229, β=+0.234**)
  ✓ Oaxaca-Blinder (D explica 21% del gap formal/informal)
  ✓ Quantile regressions (premio 19x mayor en p90 vs p10)
  ✓ 47 especificaciones de robustness (jackknife 20/20)
  ✓ Validación externa vs Felten (+0.86), Eloundou (+0.79), F&O (-0.79)
  ✓ 14 nuevas citas de Scopus (2,070 papers analizados)
  ✓ 5 figuras de publicación en PDF
  ✓ Manuscrito arXiv 13 páginas, compila limpio
  ✓ 13 commits en GitHub

En progreso (overnight):
  ⏳ Sonnet inter-rater scoring (506/3,760)
  ⏳ run_overnight_final.sh (procesa todo automáticamente)
```

---

*Buenas noches. El artículo estará listo mañana.*
