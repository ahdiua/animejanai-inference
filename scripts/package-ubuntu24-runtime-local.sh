#!/usr/bin/env bash
# Local entry point. Run directly on Ubuntu 24.04, or use Docker/Podman on
# another Linux distribution while sharing the same packager as CI.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGER="${PROJECT_ROOT}/scripts/package-ubuntu24-runtime.sh"

if [[ -r /etc/os-release ]]; then
    # Do not let os-release variables leak into the packager invocation.
    detected_id="$(. /etc/os-release; printf '%s' "${ID:-}")"
    detected_version="$(. /etc/os-release; printf '%s' "${VERSION_ID:-}")"
    if [[ "${detected_id}" == "ubuntu" && "${detected_version}" == "24.04" ]]; then
        exec "${PACKAGER}" --install-deps "$@"
    fi
fi

container_engine="${AJI_CONTAINER_ENGINE:-}"
if [[ -z "${container_engine}" ]]; then
    if command -v docker >/dev/null 2>&1; then
        container_engine=docker
    elif command -v podman >/dev/null 2>&1; then
        container_engine=podman
    fi
fi

if [[ -z "${container_engine}" ]]; then
    cat >&2 <<'EOF'
Local packaging needs either Ubuntu 24.04 or Docker/Podman.
Install a container engine, then rerun this command. No GPU passthrough is
needed because engines are generated later on the target machine.
EOF
    exit 1
fi

case "${container_engine}" in
    docker|podman) ;;
    *)
        echo "Unsupported AJI_CONTAINER_ENGINE: ${container_engine}" >&2
        exit 2
        ;;
esac

exec "${container_engine}" run --rm \
    --env "AJI_OUTPUT_UID=$(id -u)" \
    --env "AJI_OUTPUT_GID=$(id -g)" \
    --volume "${PROJECT_ROOT}:/workspace" \
    --workdir /workspace \
    ubuntu:24.04 \
    ./scripts/package-ubuntu24-runtime.sh --install-deps "$@"
