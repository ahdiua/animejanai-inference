# shellcheck shell=bash
# Model manifest helpers shared by deploy.sh and the runtime packager, so a
# source install and the packaged runtime keep the same files and digests.
# Source this file; it defines functions only.

AJI_ASSETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AJI_MODEL_MANIFEST="${AJI_ASSETS_DIR}/runtime-models.tsv"

# Manifest rows "group kind file sha256 url mirror", tab-separated, for the
# given groups (all rows when none are given).
aji_model_rows() {
    awk -F'\t' -v groups=" $* " '
        /^#/ || NF < 6 { next }
        groups == "  " || index(groups, " " $1 " ") { print }
    ' "$AJI_MODEL_MANIFEST"
}

aji_model_groups() {
    awk -F'\t' '!/^#/ && NF >= 6 && !seen[$1]++ { print $1 }' "$AJI_MODEL_MANIFEST"
}

# The pinned digest of a manifest file name ("-" when unpinned).
aji_model_sha() {
    awk -F'\t' -v f="$1" '!/^#/ && $3 == f { print $4; exit }' "$AJI_MODEL_MANIFEST"
}

# True when <file> exists and matches <sha256> (any non-empty file for "-").
aji_sha_ok() {
    local file="$1" expected="$2" actual
    [ -s "$file" ] || return 1
    [ "$expected" = "-" ] && return 0
    actual=$(sha256sum < "$file") || return 1
    [ "${actual%% *}" = "$expected" ]
}

# Extract one fp16 RIFE model from upstream's RIFE release archive (a flat
# 7z of rife_v*.onnx): aji_rife_extract <archive> <file, e.g. rife/rife_v4.26.onnx>
#   <dest file> <python with py7zr when 7z is missing> <empty work dir>.
# The model must match the manifest digest before it replaces <dest file>.
# (Re-converting vs-mlrt's models with tools/convert_rife_fp16.py gives the
# same graph, but the bytes depend on the Python/protobuf toolchain.)
aji_rife_extract() {
    local archive="$1" file="$2" dest="$3" python="$4" work="$5"
    local name expected
    name=$(basename -- "$file")
    mkdir -p "$work" || return 1
    if command -v 7z >/dev/null 2>&1; then
        7z e -y -o"$work" "$archive" "$name" >/dev/null || return 1
    else
        "$python" - "$archive" "$work" "$name" <<'PY' || return 1
import sys, py7zr
with py7zr.SevenZipFile(sys.argv[1]) as archive:
    archive.extract(path=sys.argv[2], targets=[sys.argv[3]])
PY
    fi
    if [ ! -s "$work/$name" ]; then
        echo "${name} was not found in ${archive}." >&2
        return 1
    fi
    expected=$(aji_model_sha "$file")
    if ! aji_sha_ok "$work/$name" "${expected:--}"; then
        echo "${name} from ${archive} does not match the pinned digest." >&2
        return 1
    fi
    mkdir -p "$(dirname -- "$dest")" || return 1
    mv -f -- "$work/$name" "$dest"
}
