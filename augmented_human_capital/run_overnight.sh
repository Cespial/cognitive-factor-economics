#!/bin/bash
# =============================================================
# AHC Research — Overnight Processing Pipeline
# =============================================================
# Usage: ./run_overnight.sh [--skip-llm] [--dry-run]
#
# This script orchestrates all data acquisition, crosswalk
# construction, task classification, and (optionally) LLM scoring.
#
# Estimated runtime: 2-4 hours without LLM, 8-12 hours with LLM
# =============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$PROJECT_ROOT/scripts"
LOG_DIR="$PROJECT_ROOT/output/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/overnight_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Parse arguments
SKIP_LLM=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --skip-llm) SKIP_LLM=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${BLUE}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

success() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1"
    echo -e "${RED}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

run_script() {
    local name="$1"
    local script="$2"
    local start_time=$(date +%s)

    log "Starting: $name"
    if [ "$DRY_RUN" = true ]; then
        log "  [DRY RUN] Would execute: python3 $script"
        return 0
    fi

    if python3 "$script" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        success "$name completed (${duration}s)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "$name FAILED after ${duration}s — check $LOG_FILE"
        return 1
    fi
}

# =============================================================
echo ""
echo "============================================================="
echo " AUGMENTED HUMAN CAPITAL — Overnight Processing Pipeline"
echo " Started: $(date)"
echo " Log: $LOG_FILE"
echo " Skip LLM: $SKIP_LLM"
echo "============================================================="
echo ""

TOTAL_START=$(date +%s)

# =============================================================
# BLOCK 0: Environment Setup
# =============================================================
log "BLOCK 0: Environment Setup"
run_script "Environment validation" "$SCRIPTS/00_setup/setup_environment.py" || true

# =============================================================
# BLOCK 1: Data Acquisition (~2 hours)
# =============================================================
log ""
log "BLOCK 1: Data Acquisition"
log "========================="

run_script "O*NET Database download" "$SCRIPTS/01_data_acquisition/01_download_onet.py" || {
    error "O*NET download failed — this blocks crosswalk construction"
    # Continue anyway; crosswalk scripts will handle missing data gracefully
}

run_script "Complementary data (Felten, IFR, patents)" "$SCRIPTS/01_data_acquisition/02_download_complementary.py" || {
    warn "Some complementary downloads failed — non-blocking"
}

# =============================================================
# BLOCK 2: Crosswalk Construction (~1 hour)
# =============================================================
log ""
log "BLOCK 2: Crosswalk Construction"
log "================================"

run_script "SOC → ISCO crosswalk" "$SCRIPTS/02_crosswalks/10_build_soc_isco_crosswalk.py" || {
    error "SOC-ISCO crosswalk failed — blocks task matrix"
    exit 1
}

run_script "ISCO → CIUO crosswalk" "$SCRIPTS/02_crosswalks/11_build_isco_ciuo_crosswalk.py" || {
    error "ISCO-CIUO crosswalk failed — blocks task matrix"
    exit 1
}

run_script "Chain crosswalk (SOC → ISCO → CIUO)" "$SCRIPTS/02_crosswalks/12_chain_crosswalk.py" || {
    error "Chained crosswalk failed"
    exit 1
}

# =============================================================
# BLOCK 3: Task Matrix & Classification (~1 hour)
# =============================================================
log ""
log "BLOCK 3: Task Matrix & Classification"
log "======================================"

run_script "Build occupation-task matrix" "$SCRIPTS/03_ahc_index/13_build_task_matrix.py" || {
    error "Task matrix construction failed"
    exit 1
}

run_script "Classify tasks (H^P / H^C / H^A)" "$SCRIPTS/03_ahc_index/14_classify_tasks_hpca.py" || {
    error "Task classification failed"
    exit 1
}

# =============================================================
# BLOCK 4: LLM Scoring (~4-8 hours, optional)
# =============================================================
if [ "$SKIP_LLM" = true ]; then
    warn "BLOCK 4: LLM Scoring SKIPPED (--skip-llm flag)"
    warn "  Run manually later: python3 $SCRIPTS/03_ahc_index/20_score_tasks_llm.py"
else
    log ""
    log "BLOCK 4: LLM Scoring (this will take several hours)"
    log "===================================================="

    # Check for API key
    if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
        warn "No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
        warn "Checking for local Ollama..."
        if command -v ollama &> /dev/null; then
            export LLM_PROVIDER=ollama
            log "Using local Ollama for scoring"
        else
            warn "No LLM provider available — skipping scoring"
            SKIP_LLM=true
        fi
    fi

    if [ "$SKIP_LLM" = false ]; then
        run_script "LLM task scoring" "$SCRIPTS/03_ahc_index/20_score_tasks_llm.py" || {
            warn "LLM scoring failed or incomplete — check checkpoint file"
        }

        run_script "AHC index aggregation" "$SCRIPTS/03_ahc_index/21_aggregate_ahc_index.py" || {
            warn "AHC aggregation failed — may need manual intervention"
        }
    fi
fi

# =============================================================
# BLOCK 5: Merge & Descriptives (~30 min)
# =============================================================
log ""
log "BLOCK 5: Sample Construction & Descriptives"
log "============================================="

run_script "Merge GEIH with AHC index" "$SCRIPTS/04_econometrics/30_merge_geih_ahc.py" || {
    warn "GEIH-AHC merge produced warnings (AHC may be placeholder)"
}

run_script "Descriptive statistics & figures" "$SCRIPTS/04_econometrics/31_descriptive_statistics.py" || {
    warn "Descriptive statistics had issues"
}

# =============================================================
# SUMMARY
# =============================================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))

echo ""
echo "============================================================="
echo " OVERNIGHT PROCESSING COMPLETE"
echo " Duration: ${TOTAL_MINUTES} minutes (${TOTAL_DURATION}s)"
echo " Log: $LOG_FILE"
echo "============================================================="
echo ""
echo " Key outputs:"
echo "   data/crosswalks/soc_ciuo_chained.parquet"
echo "   data/processed/occupation_task_matrix.parquet"
echo "   data/processed/task_classification.parquet"

if [ "$SKIP_LLM" = false ]; then
echo "   data/indices/ahc_index_by_occupation.parquet"
fi

echo "   data/processed/estimation_sample.parquet"
echo "   output/tables/descriptive_statistics.csv"
echo "   output/figures/"
echo ""

log "Pipeline finished at $(date). Total: ${TOTAL_MINUTES} min."
