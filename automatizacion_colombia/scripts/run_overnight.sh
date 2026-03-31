#!/bin/bash
# =============================================================================
# OVERNIGHT PROCESSING PIPELINE
# Automatización Colombia — Research Paper Revision
# =============================================================================
# Run this script to execute all heavy-compute tasks sequentially.
# Estimated total runtime: 2-4 hours depending on GEIH processing
#
# Usage: cd scripts/ && chmod +x run_overnight.sh && nohup ./run_overnight.sh > overnight.log 2>&1 &
# =============================================================================

set -e  # Exit on first error
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/overnight.log"

echo "=============================================" | tee "$LOG_FILE"
echo "OVERNIGHT PIPELINE START: $(date)" | tee -a "$LOG_FILE"
echo "Project: $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"

# --- Phase 0: Verify Python environment ---
echo "" | tee -a "$LOG_FILE"
echo "[Phase 0] Verifying Python environment..." | tee -a "$LOG_FILE"
python3 -c "
import pandas, numpy, scipy, sklearn, statsmodels, matplotlib, seaborn
print('All required packages available.')
print(f'  pandas={pandas.__version__}')
print(f'  numpy={numpy.__version__}')
print(f'  statsmodels={statsmodels.__version__}')
print(f'  sklearn={sklearn.__version__}')
" 2>&1 | tee -a "$LOG_FILE"

# Check for linearmodels (needed for panel FE)
python3 -c "import linearmodels; print(f'  linearmodels={linearmodels.__version__}')" 2>&1 | tee -a "$LOG_FILE" || {
    echo "[Phase 0] Installing linearmodels..." | tee -a "$LOG_FILE"
    pip install linearmodels 2>&1 | tee -a "$LOG_FILE"
}

# --- Phase 1: EAM Panel DiD Analysis ---
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "[Phase 1] EAM Panel DiD Analysis (09)" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
if [ -f "$SCRIPT_DIR/09_did_panel_eam.py" ]; then
    python3 "$SCRIPT_DIR/09_did_panel_eam.py" 2>&1 | tee -a "$LOG_FILE"
    echo "[Phase 1] COMPLETED: $(date)" | tee -a "$LOG_FILE"
else
    echo "[Phase 1] SKIPPED: 09_did_panel_eam.py not found yet" | tee -a "$LOG_FILE"
fi

# --- Phase 2: GEIH DiD with Minimum Wage Shock ---
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "[Phase 2] GEIH DiD MW Shock Analysis (10)" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
if [ -f "$SCRIPT_DIR/10_did_geih_mw_shock.py" ]; then
    python3 "$SCRIPT_DIR/10_did_geih_mw_shock.py" 2>&1 | tee -a "$LOG_FILE"
    echo "[Phase 2] COMPLETED: $(date)" | tee -a "$LOG_FILE"
else
    echo "[Phase 2] SKIPPED: 10_did_geih_mw_shock.py not found yet" | tee -a "$LOG_FILE"
fi

# --- Phase 3: IVA Validation ---
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "[Phase 3] IVA/AVI Validation (11)" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
if [ -f "$SCRIPT_DIR/11_iva_validation.py" ]; then
    python3 "$SCRIPT_DIR/11_iva_validation.py" 2>&1 | tee -a "$LOG_FILE"
    echo "[Phase 3] COMPLETED: $(date)" | tee -a "$LOG_FILE"
else
    echo "[Phase 3] SKIPPED: 11_iva_validation.py not found yet" | tee -a "$LOG_FILE"
fi

# --- Phase 4: Re-run existing analysis pipeline ---
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "[Phase 4] Re-running original pipeline (01-08)" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

for script in 01_international_panel.py 02_sectoral_analysis.py 03_automation_risk_model.py 04_firm_level_analysis.py 05_scenario_simulations.py 08_robustness_checks.py; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        echo "  Running $script..." | tee -a "$LOG_FILE"
        python3 "$SCRIPT_DIR/$script" 2>&1 | tee -a "$LOG_FILE" || {
            echo "  WARNING: $script failed, continuing..." | tee -a "$LOG_FILE"
        }
    fi
done
echo "[Phase 4] COMPLETED: $(date)" | tee -a "$LOG_FILE"

# --- Phase 5: Generate publication figures ---
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "[Phase 5] Generating publication figures (06-07)" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

for script in 06_regenerate_figures.py 07_improved_figures.py; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        echo "  Running $script..." | tee -a "$LOG_FILE"
        python3 "$SCRIPT_DIR/$script" 2>&1 | tee -a "$LOG_FILE" || {
            echo "  WARNING: $script failed, continuing..." | tee -a "$LOG_FILE"
        }
    fi
done
echo "[Phase 5] COMPLETED: $(date)" | tee -a "$LOG_FILE"

# --- Phase 6: Compile LaTeX ---
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "[Phase 6] Compiling LaTeX (main_wdp.tex)" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"
if command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode main_wdp.tex 2>&1 | tee -a "$LOG_FILE" || true
    bibtex main_wdp 2>&1 | tee -a "$LOG_FILE" || true
    pdflatex -interaction=nonstopmode main_wdp.tex 2>&1 | tee -a "$LOG_FILE" || true
    pdflatex -interaction=nonstopmode main_wdp.tex 2>&1 | tee -a "$LOG_FILE" || true
    echo "[Phase 6] LaTeX COMPLETED" | tee -a "$LOG_FILE"
else
    echo "[Phase 6] SKIPPED: pdflatex not found" | tee -a "$LOG_FILE"
fi

# --- Summary ---
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "OVERNIGHT PIPELINE COMPLETE: $(date)" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Output files:" | tee -a "$LOG_FILE"
echo "  Data:" | tee -a "$LOG_FILE"
ls -la "$PROJECT_DIR/data/"*.csv 2>/dev/null | tail -20 | tee -a "$LOG_FILE"
echo "  Figures:" | tee -a "$LOG_FILE"
ls "$PROJECT_DIR/images/en/"*.png 2>/dev/null | wc -l | xargs -I{} echo "    {} PNG figures in images/en/" | tee -a "$LOG_FILE"
echo "  PDF:" | tee -a "$LOG_FILE"
ls -la "$PROJECT_DIR/main_wdp.pdf" 2>/dev/null | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Check overnight.log for details." | tee -a "$LOG_FILE"
