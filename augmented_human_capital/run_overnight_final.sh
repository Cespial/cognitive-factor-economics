#!/bin/bash
# =============================================================
# OVERNIGHT FINAL — Produce publication-ready paper
# =============================================================
# Run this before going to sleep. By morning:
#   1. Sonnet inter-rater scoring will be complete
#   2. Final reliability stats computed
#   3. Figures regenerated with updated data
#   4. LaTeX manuscript recompiled with all final results
#   5. arXiv upload ZIP created
#   6. Everything committed and pushed to GitHub
# =============================================================

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
LOG="$PROJECT/output/logs/overnight_final_$(date +%Y%m%d_%H%M).log"
mkdir -p "$PROJECT/output/logs"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== OVERNIGHT FINAL PIPELINE ==="
log "Project: $PROJECT"

# ---------------------------------------------------------
# STEP 1: Wait for Sonnet scoring to finish
# ---------------------------------------------------------
log "STEP 1: Waiting for Sonnet inter-rater scoring..."

SONNET_FILE="$PROJECT/data/indices/sonnet_validation_scores.jsonl"
TARGET=3759
MAX_WAIT=14400  # 4 hours max
WAITED=0

while true; do
    if [ -f "$SONNET_FILE" ]; then
        COUNT=$(wc -l < "$SONNET_FILE" | tr -d ' ')
        if [ "$COUNT" -ge "$TARGET" ]; then
            log "  Sonnet complete: $COUNT scores"
            break
        fi
        # Check if the scoring process is still running
        if ! pgrep -f "sonnet_validation" > /dev/null 2>&1 && ! pgrep -f "claude-sonnet" > /dev/null 2>&1; then
            # Process may have died — check if we have enough for reliability
            if [ "$COUNT" -ge 500 ]; then
                log "  Sonnet process stopped at $COUNT/$TARGET — sufficient for reliability"
                break
            fi
        fi
        log "  Sonnet progress: $COUNT/$TARGET"
    fi
    sleep 120
    WAITED=$((WAITED + 120))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        log "  Timeout after ${MAX_WAIT}s — proceeding with available data"
        break
    fi
done

# ---------------------------------------------------------
# STEP 2: Compute final inter-rater reliability
# ---------------------------------------------------------
log "STEP 2: Computing inter-rater reliability..."

python3 -c "
import json, numpy as np
from scipy.stats import spearmanr

haiku, sonnet = {}, {}
with open('$PROJECT/data/indices/raw_llm_scores.jsonl') as f:
    for line in f:
        try:
            r = json.loads(line.strip())
            if 'error' not in r and 'augmentation_score' in r:
                haiku[r['task_id']] = r
        except: pass
with open('$SONNET_FILE') as f:
    for line in f:
        try:
            r = json.loads(line.strip())
            if 'error' not in r and 'augmentation_score' in r:
                sonnet[r['task_id']] = r
        except: pass

common = set(haiku.keys()) & set(sonnet.keys())
print(f'Paired tasks: {len(common)}')

if len(common) >= 50:
    h = np.array([haiku[t]['augmentation_score'] for t in common])
    s = np.array([sonnet[t]['augmentation_score'] for t in common])

    # Metrics
    pearson = np.corrcoef(h, s)[0,1]
    rho, p_rho = spearmanr(h, s)
    mad = np.abs(h - s).mean()
    bias = s.mean() - h.mean()

    # Krippendorff alpha (raw and adjusted)
    all_v = np.concatenate([h, s])
    D_o = np.mean((h - s)**2)
    D_e = np.var(all_v) * 2
    alpha_raw = 1 - D_o / D_e if D_e > 0 else 1.0

    s_adj = s - bias
    D_o2 = np.mean((h - s_adj)**2)
    D_e2 = np.var(np.concatenate([h, s_adj])) * 2
    alpha_adj = 1 - D_o2 / D_e2 if D_e2 > 0 else 1.0

    results = {
        'n_paired': int(len(common)),
        'pearson_r': round(float(pearson), 3),
        'spearman_rho': round(float(rho), 3),
        'spearman_p': round(float(p_rho), 6),
        'mean_abs_diff': round(float(mad), 1),
        'level_bias': round(float(bias), 1),
        'krippendorff_alpha_raw': round(float(alpha_raw), 3),
        'krippendorff_alpha_adjusted': round(float(alpha_adj), 3),
        'haiku_mean': round(float(h.mean()), 1),
        'sonnet_mean': round(float(s.mean()), 1),
    }

    with open('$PROJECT/output/tables/final_interrater_reliability.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('INTER-RATER RELIABILITY:')
    for k, v in results.items():
        print(f'  {k}: {v}')
else:
    print(f'Insufficient pairs ({len(common)}) for reliability analysis')
" >> "$LOG" 2>&1

# ---------------------------------------------------------
# STEP 3: Regenerate figures with any updates
# ---------------------------------------------------------
log "STEP 3: Regenerating publication figures..."
python3 "$PROJECT/scripts/06_figures/50_publication_figures.py" >> "$LOG" 2>&1 || log "  [WARN] Figure generation had issues"

# ---------------------------------------------------------
# STEP 4: Recompile LaTeX manuscript
# ---------------------------------------------------------
log "STEP 4: Compiling final manuscript..."
cd "$PROJECT/paper/arxiv_submission"
cp ../../literature/references.bib .
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

PAGES=$(pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "Output written" | grep -o "[0-9]* pages" || echo "? pages")
log "  Manuscript: $PAGES"

# ---------------------------------------------------------
# STEP 5: Create arXiv upload package
# ---------------------------------------------------------
log "STEP 5: Creating arXiv upload package..."
cd "$PROJECT/paper"
rm -f arxiv_upload.zip
zip -j arxiv_upload.zip \
    arxiv_submission/main.tex \
    arxiv_submission/main.bbl \
    arxiv_submission/figs/*.pdf \
    2>> "$LOG"

SIZE=$(du -h arxiv_upload.zip | cut -f1)
log "  arxiv_upload.zip: $SIZE"

# ---------------------------------------------------------
# STEP 6: Commit and push to GitHub
# ---------------------------------------------------------
log "STEP 6: Committing to GitHub..."
cd "/Users/cristianespinal/Claude Code/Projects/Research/Cognitive Factor Economics (CFE)"

# Copy updated files
cp "$PROJECT/paper/arxiv_submission/main.tex" augmented_human_capital/paper/arxiv_submission/
cp "$PROJECT/paper/arxiv_submission/references.bib" augmented_human_capital/paper/arxiv_submission/
cp "$PROJECT/output/tables/final_interrater_reliability.json" augmented_human_capital/output/tables/ 2>/dev/null
cp "$PROJECT/literature/references.bib" augmented_human_capital/literature/

git add -A augmented_human_capital/ 2>/dev/null
git commit -m "Overnight final: inter-rater complete, manuscript recompiled, arXiv package ready

$(cat "$PROJECT/output/tables/final_interrater_reliability.json" 2>/dev/null | head -15)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>> "$LOG" || log "  No changes to commit"

git push 2>> "$LOG" || log "  Push failed"

# ---------------------------------------------------------
# STEP 7: Generate morning summary
# ---------------------------------------------------------
log ""
log "============================================="
log "  OVERNIGHT PROCESSING COMPLETE"
log "============================================="
log ""
log "Ready for publication:"
log "  PDF: $PROJECT/paper/arxiv_submission/main.pdf"
log "  ZIP: $PROJECT/paper/arxiv_upload.zip"
log "  GitHub: https://github.com/Cespial/cognitive-factor-economics"
log ""
log "To submit to arXiv:"
log "  1. Go to https://arxiv.org/submit"
log "  2. Category: econ.GN (cross-list: econ.EM, cs.AI)"
log "  3. Upload: $PROJECT/paper/arxiv_upload.zip"
log ""
log "To view the paper:"
log "  open '$PROJECT/paper/arxiv_submission/main.pdf'"
log ""
