#!/usr/bin/env bash
# Download ImageNet-C dataset (severity=5 only) from Zenodo record 2235448
#
# Downloads 4 tar files (~50 GB total), extracts only severity=5 subdirs,
# then removes tarballs to save disk space.
#
# Expected output structure:
#   <DATA_DIR>/ImageNet-C/<corruption>/5/<class>/*.JPEG
#
# Usage:
#   bash download_imagenet_c.sh [DATA_DIR]
#
# DATA_DIR defaults to:
#   ~/Lab/v2/experiments/baselines/BATCLIP/classification/data
#
# Examples:
#   bash download_imagenet_c.sh
#   bash download_imagenet_c.sh /path/to/data
#   SEV_ONLY=0 bash download_imagenet_c.sh   # extract ALL severities (larger)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_DIR="${SCRIPT_DIR}/../../experiments/baselines/BATCLIP/classification/data"
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
DATA_DIR="$(cd "${DATA_DIR}" 2>/dev/null && pwd || (mkdir -p "${DATA_DIR}" && cd "${DATA_DIR}" && pwd))"
IMAGENET_C_DIR="${DATA_DIR}/ImageNet-C"
DOWNLOAD_CACHE="${DATA_DIR}/.imagenet_c_download_cache"

# Set to 1 to extract only severity=5 (saves ~80% disk space)
SEV_ONLY="${SEV_ONLY:-1}"
# Set to 1 to keep tar files after extraction
KEEP_TAR="${KEEP_TAR:-0}"

# Zenodo record 2235448 — ImageNet-C (Hendrycks & Dietterich 2019)
ZENODO_BASE="https://zenodo.org/api/records/2235448/files"

# 4 tarballs covering the 15 standard corruptions
# (extra.tar has non-standard corruptions: speckle_noise, spatter, gaussian_blur, saturate)
declare -A TAR_SIZES=(
    ["noise.tar"]="24260397768"    # 22.6 GB  — gaussian_noise, shot_noise, impulse_noise
    ["blur.tar"]="7629350912"      # 7.1 GB   — defocus_blur, glass_blur, motion_blur, zoom_blur
    ["weather.tar"]="13748420608"  # 12.8 GB  — brightness, fog, frost, snow
    ["digital.tar"]="8380293120"   # 7.8 GB   — contrast, elastic_transform, jpeg_compression, pixelate
)

# Which corruptions come from which tar (for verification)
declare -A TAR_CORRUPTIONS=(
    ["noise.tar"]="gaussian_noise shot_noise impulse_noise"
    ["blur.tar"]="defocus_blur glass_blur motion_blur zoom_blur"
    ["weather.tar"]="brightness fog frost snow"
    ["digital.tar"]="contrast elastic_transform jpeg_compression pixelate"
)

ALL_15_CORRUPTIONS=(
    gaussian_noise shot_noise impulse_noise
    defocus_blur glass_blur motion_blur zoom_blur
    snow frost fog
    brightness contrast elastic_transform
    pixelate jpeg_compression
)

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; exit 1; }

check_commands() {
    for cmd in wget tar python3; do
        command -v "$cmd" &>/dev/null || die "Required command not found: $cmd"
    done
}

human_bytes() {
    python3 -c "
b = int('$1')
for unit in ['B','KB','MB','GB','TB']:
    if b < 1024: print(f'{b:.1f} {unit}'); break
    b /= 1024
"
}

# ── Already-done check ────────────────────────────────────────────────────────
check_corruption_done() {
    local corr="$1"
    local sev5_dir="${IMAGENET_C_DIR}/${corr}/5"
    [[ -d "${sev5_dir}" ]] && [[ "$(ls -A "${sev5_dir}" 2>/dev/null | wc -l)" -gt 10 ]]
}

count_done() {
    local done=0
    for corr in "${ALL_15_CORRUPTIONS[@]}"; do
        check_corruption_done "${corr}" && ((done++)) || true
    done
    echo "${done}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    check_commands

    log "ImageNet-C Download Script"
    log "  DATA_DIR:        ${DATA_DIR}"
    log "  IMAGENET_C_DIR:  ${IMAGENET_C_DIR}"
    log "  SEV_ONLY=5:      ${SEV_ONLY}"
    log "  KEEP_TAR:        ${KEEP_TAR}"
    echo ""

    mkdir -p "${IMAGENET_C_DIR}" "${DOWNLOAD_CACHE}"

    # ── Pre-flight: count what's already done ─────────────────────────────────
    done_count=$(count_done)
    log "Already extracted: ${done_count}/15 corruptions"

    if [[ "${done_count}" -eq 15 ]]; then
        log "✅ All 15 corruptions already present at sev=5. Nothing to do."
        log "   Verify: ls ${IMAGENET_C_DIR}"
        exit 0
    fi

    # ── Download + Extract loop ───────────────────────────────────────────────
    total_dl_gb=0
    for tarname in noise.tar blur.tar weather.tar digital.tar; do
        corruptions="${TAR_CORRUPTIONS[${tarname}]}"

        # Check if all corruptions from this tar are already extracted
        all_done=1
        for corr in ${corruptions}; do
            check_corruption_done "${corr}" || { all_done=0; break; }
        done
        if [[ "${all_done}" -eq 1 ]]; then
            log "  ${tarname}: all corruptions already present → SKIP"
            continue
        fi

        tarpath="${DOWNLOAD_CACHE}/${tarname}"
        url="${ZENODO_BASE}/${tarname}/content"
        size="${TAR_SIZES[${tarname}]}"
        size_human=$(human_bytes "${size}")

        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "Processing: ${tarname} (${size_human})"
        log "  Corruptions: ${corruptions}"

        # Download if not cached
        if [[ -f "${tarpath}" ]]; then
            actual_size=$(stat -c%s "${tarpath}" 2>/dev/null || stat -f%z "${tarpath}" 2>/dev/null || echo 0)
            if [[ "${actual_size}" -eq "${size}" ]]; then
                log "  Already downloaded (size matches). Using cache."
            else
                warn "  Cached file size mismatch (${actual_size} vs ${size}). Re-downloading."
                rm -f "${tarpath}"
            fi
        fi

        if [[ ! -f "${tarpath}" ]]; then
            log "  Downloading ${tarname} (${size_human}) ..."
            log "  URL: ${url}"
            wget --continue \
                 --progress=bar:force \
                 --show-progress \
                 -O "${tarpath}" \
                 "${url}" 2>&1 || die "Download failed: ${tarname}"
            log "  ✅ Download complete: ${tarpath}"
        fi

        # Extract
        # Internal tar structure: <corruption>/<severity>/<class>/<image>.JPEG
        # e.g. defocus_blur/5/n03884397/ILSVRC2012_val_00018337.JPEG
        log "  Extracting all severities → ${IMAGENET_C_DIR}/ (then pruning if SEV_ONLY=${SEV_ONLY}) ..."
        tar -xf "${tarpath}" -C "${IMAGENET_C_DIR}/" &
        TAR_PID=$!
        while kill -0 "${TAR_PID}" 2>/dev/null; do
            extracted=$(find "${IMAGENET_C_DIR}" -name "*.JPEG" 2>/dev/null | wc -l)
            printf "\r  Extracted so far: %d images ..." "${extracted}"
            sleep 5
        done
        wait "${TAR_PID}"
        echo ""
        log "  ✅ Extraction complete"

        # Remove severity 1-4 to save disk space (keep only sev=5)
        if [[ "${SEV_ONLY}" -eq 1 ]]; then
            log "  Pruning sev=1,2,3,4 (keeping sev=5 only) ..."
            for corr in ${corruptions}; do
                for sev in 1 2 3 4; do
                    sev_dir="${IMAGENET_C_DIR}/${corr}/${sev}"
                    if [[ -d "${sev_dir}" ]]; then
                        rm -rf "${sev_dir}"
                        printf "."
                    fi
                done
            done
            echo " done"
        fi

        # Clean up tar
        if [[ "${KEEP_TAR}" -eq 0 ]]; then
            log "  Removing ${tarpath} (set KEEP_TAR=1 to keep)"
            rm -f "${tarpath}"
        fi
    done

    # ── Verification ─────────────────────────────────────────────────────────
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Verification (checking sev=5 for each corruption):"
    all_ok=1
    for corr in "${ALL_15_CORRUPTIONS[@]}"; do
        sev5_dir="${IMAGENET_C_DIR}/${corr}/5"
        if [[ -d "${sev5_dir}" ]]; then
            n_classes=$(ls "${sev5_dir}" 2>/dev/null | wc -l)
            n_images=$(find "${sev5_dir}" -name "*.JPEG" 2>/dev/null | wc -l)
            if [[ "${n_classes}" -gt 0 ]]; then
                log "  ✅ ${corr}/5/  ${n_classes} classes, ${n_images} images"
            else
                warn "  ⚠️  ${corr}/5/ exists but empty"
                all_ok=0
            fi
        else
            log "  ❌ ${corr}/5/ MISSING"
            all_ok=0
        fi
    done

    if [[ "${all_ok}" -eq 1 ]]; then
        log ""
        log "✅ All 15 corruptions verified at ${IMAGENET_C_DIR}"
        log ""
        log "Usage in experiment:"
        log "  cd experiments/baselines/BATCLIP/classification"
        log "  python ... --cfg cfgs/imagenet_c/ours.yaml DATA_DIR ./data"
    else
        warn "Some corruptions missing. Check logs above."
        exit 1
    fi

    # ── Cleanup download cache if empty ──────────────────────────────────────
    if [[ -d "${DOWNLOAD_CACHE}" ]] && [[ -z "$(ls -A "${DOWNLOAD_CACHE}")" ]]; then
        rmdir "${DOWNLOAD_CACHE}"
    fi
}

main "$@"
