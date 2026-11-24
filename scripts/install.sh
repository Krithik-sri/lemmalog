#!/usr/bin/env bash
# lemmalog installer: builds the binaries, registers the MCP server with
# every supported agent CLI it finds, and installs the skill.
#
#   ./scripts/install.sh              # install
#   ./scripts/install.sh --uninstall  # remove registrations + skill
#
# Env: LEMMALOG_SNAPSHOT (default ~/.lemmalog/memory.snap) — where memory
# persists across sessions. LEMMALOG_FEATURES (default mcp,gliner,glinker)
# — cargo features; the default builds in LLM-free extraction of free text,
# loading ONNX Runtime at run time, so the build itself needs no download
# (LEMMALOG_FEATURES=mcp for a protocol-only build; a list with
# gliner-download builds the old statically linked way). LEMMALOG_GLINER_DIR
# (default ~/.lemmalog/models/gliner-multitask-large-v0.5) — GLiNER model.
# ONNX Runtime: ORT_DYLIB_PATH if set; otherwise fetched into ~/.lemmalog/lib.
# On Linux x86_64 that is Intel's onnxruntime-openvino $ORT_OPENVINO_VERSION
# build from PyPI (ONNX Runtime with the OpenVINO execution provider and
# OpenVINO's CPU/GPU/NPU plugins; needs python3), elsewhere Microsoft's plain
# ONNX Runtime $ORT_VERSION. LEMMALOG_ORT_BUILD=cpu fetches the plain build
# everywhere; LEMMALOG_FETCH_ORT=0 skips; LEMMALOG_ORT_URL points at a mirror
# (a .whl is unpacked as the OpenVINO build, a .tgz as the plain one).

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BIN="$ROOT/target/release"
SNAP="${LEMMALOG_SNAPSHOT:-$HOME/.lemmalog/memory.snap}"
SKILL_SRC="$ROOT/skills/lemmalog"
MODE="install"
[ "${1:-}" = "--uninstall" ] && MODE="uninstall"

have() { command -v "$1" >/dev/null 2>&1; }

FEATURES="${LEMMALOG_FEATURES:-mcp,gliner,glinker}"
GLINER_DIR="${LEMMALOG_GLINER_DIR:-$HOME/.lemmalog/models/gliner-multitask-large-v0.5}"
# the ONNX Runtime version ort 2.0.0-rc.9 is built against (1.20.x)
ORT_VERSION="1.20.1"
ORT_OPENVINO_VERSION="1.20.0"
ORT_LIB_DIR="$HOME/.lemmalog/lib"
case "$(uname -s)" in
    Darwin) ORT_LIB=libonnxruntime.dylib ;;
    *) ORT_LIB=libonnxruntime.so ;;
esac

# Put the ONNX Runtime shared library where lemmalog looks for it
# (~/.lemmalog/lib), unless ORT_DYLIB_PATH already names one.
fetch_ort() {
    if [ -n "${ORT_DYLIB_PATH:-}" ]; then
        echo "-- ONNX Runtime: using ORT_DYLIB_PATH=$ORT_DYLIB_PATH"
        return
    fi
    if [ -e "$ORT_LIB_DIR/$ORT_LIB" ]; then
        echo "-- ONNX Runtime: $ORT_LIB_DIR/$ORT_LIB"
        return
    fi
    local url="${LEMMALOG_ORT_URL:-}"
    if [ "${LEMMALOG_FETCH_ORT:-1}" != "0" ] && have curl && have python3 \
        && { case "$url" in *.whl) true ;; "") [ "${LEMMALOG_ORT_BUILD:-openvino}" = "openvino" ] && [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] ;; *) false ;; esac; }; then
        fetch_ort_openvino "$url" && return
        echo "-- falling back to the plain ONNX Runtime build"
        url=""
    fi
    local pkg=""
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) pkg="onnxruntime-linux-x64-$ORT_VERSION" ;;
        Linux-aarch64|Linux-arm64) pkg="onnxruntime-linux-aarch64-$ORT_VERSION" ;;
        Darwin-arm64) pkg="onnxruntime-osx-arm64-$ORT_VERSION" ;;
        Darwin-x86_64) pkg="onnxruntime-osx-x86_64-$ORT_VERSION" ;;
    esac
    [ -z "$url" ] && url="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$pkg.tgz"
    if [ "${LEMMALOG_FETCH_ORT:-1}" = "0" ] || [ -z "$pkg$([ -n "${LEMMALOG_ORT_URL:-}" ] && echo x)" ] || ! have curl; then
        echo "-- ONNX Runtime $ORT_VERSION not installed: free text will not be extracted until it is."
        echo "   Put $ORT_LIB in $ORT_LIB_DIR, or set ORT_DYLIB_PATH, e.g. from"
        echo "     https://github.com/microsoft/onnxruntime/releases/tag/v$ORT_VERSION"
        return
    fi
    echo "==> fetching ONNX Runtime $ORT_VERSION into $ORT_LIB_DIR"
    local tmp
    tmp="$(mktemp -d)"
    if curl -fsSL "$url" -o "$tmp/ort.tgz" && tar -xzf "$tmp/ort.tgz" -C "$tmp"; then
        mkdir -p "$ORT_LIB_DIR"
        # the library and its versioned names (symlinks kept)
        find "$tmp" -path "*/lib/libonnxruntime*" \( -type f -o -type l \) -exec cp -P {} "$ORT_LIB_DIR/" \;
        if [ ! -e "$ORT_LIB_DIR/$ORT_LIB" ]; then
            local real
            real="$(ls "$ORT_LIB_DIR"/libonnxruntime* 2>/dev/null | sort | tail -1)"
            [ -n "$real" ] && ln -sf "$(basename "$real")" "$ORT_LIB_DIR/$ORT_LIB"
        fi
    fi
    rm -rf "$tmp"
    if [ -e "$ORT_LIB_DIR/$ORT_LIB" ]; then
        echo "    -> $ORT_LIB_DIR/$ORT_LIB"
    else
        echo "-- could not fetch ONNX Runtime from $url"
        echo "   Put $ORT_LIB in $ORT_LIB_DIR, or set ORT_DYLIB_PATH."
    fi
}

# Intel's onnxruntime-openvino wheel: unpack its native libraries (ONNX
# Runtime, the OpenVINO provider, OpenVINO and TBB; not the Python module)
# into ~/.lemmalog/lib. $1: wheel URL, or empty to look it up on PyPI.
fetch_ort_openvino() {
    local url="$1" tmp
    tmp="$(mktemp -d)"
    if [ -z "$url" ]; then
        url="$(curl -fsSL "https://pypi.org/pypi/onnxruntime-openvino/$ORT_OPENVINO_VERSION/json" 2>/dev/null | python3 -c '
import json, sys
for f in json.load(sys.stdin)["urls"]:
    if f["filename"].endswith(".whl") and "manylinux" in f["filename"] and "x86_64" in f["filename"]:
        print(f["url"]); break' 2>/dev/null)" || true
    fi
    if [ -z "$url" ]; then
        echo "-- could not find onnxruntime-openvino $ORT_OPENVINO_VERSION on PyPI"
        rm -rf "$tmp"; return 1
    fi
    echo "==> fetching ONNX Runtime with OpenVINO (onnxruntime-openvino $ORT_OPENVINO_VERSION) into $ORT_LIB_DIR"
    if curl -fsSL "$url" -o "$tmp/ort.whl" && mkdir -p "$ORT_LIB_DIR" && python3 - "$tmp/ort.whl" "$ORT_LIB_DIR" <<'PY'
import os, sys, zipfile
whl, dest = sys.argv[1], sys.argv[2]
names = []
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        b = os.path.basename(n)
        if n.startswith("onnxruntime/capi/") and ".so" in b and "pybind11" not in b:
            with open(os.path.join(dest, b), "wb") as f:
                f.write(z.read(n))
            os.chmod(os.path.join(dest, b), 0o755)
            names.append(b)
real = sorted(b for b in names if b.startswith("libonnxruntime.so."))
if not real:
    sys.exit("no libonnxruntime in the wheel")
link = os.path.join(dest, "libonnxruntime.so")
if os.path.lexists(link):
    os.remove(link)
os.symlink(real[-1], link)
PY
    then
        rm -rf "$tmp"
        echo "    -> $ORT_LIB_DIR/$ORT_LIB (OpenVINO execution provider included)"
        return 0
    fi
    rm -rf "$tmp"
    echo "-- could not fetch onnxruntime-openvino from $url"
    return 1
}

build() {
    echo "==> building lemmalog (release, features: $FEATURES)..."
    case ",$FEATURES," in
        # statically linked (downloads at build time): no default load-dynamic
        *,gliner-download,*) cargo build --release --no-default-features --features "$FEATURES" ;;
        *) cargo build --release --features "$FEATURES" ;;
    esac
    case ",$FEATURES," in
        *,gliner-download,*) ;;
        *,gliner,*|*,gliner-load-dynamic,*) fetch_ort ;;
    esac
    case ",$FEATURES," in
        *,gliner,*|*,gliner-load-dynamic,*|*,gliner-download,*)
            if [ ! -f "$GLINER_DIR/onnx/model.onnx" ] || [ ! -f "$GLINER_DIR/tokenizer.json" ]; then
                echo "-- GLiNER model not found in $GLINER_DIR: free text will not be extracted"
                echo "   until it is there (line-protocol facts work either way). Fetch it with:"
                echo "     huggingface-cli download onnx-community/gliner-multitask-large-v0.5 \\"
                echo "       tokenizer.json onnx/model.onnx --local-dir \"$GLINER_DIR\""
            fi ;;
    esac
}

register_claude() {
    if have claude; then
        if [ "$MODE" = "install" ]; then
            echo "==> registering MCP server with Claude Code"
            claude mcp add lemmalog \
                --env LEMMALOG_MCP_PATH="$SNAP" --env LEMMALOG_GLINER_DIR="$GLINER_DIR" \
                -- "$BIN/lemmalog-mcp" 2>/dev/null \
              || claude mcp remove lemmalog >/dev/null 2>&1 && \
                 claude mcp add lemmalog \
                    --env LEMMALOG_MCP_PATH="$SNAP" --env LEMMALOG_GLINER_DIR="$GLINER_DIR" \
                    -- "$BIN/lemmalog-mcp"
            mkdir -p ~/.claude/skills
            rm -rf ~/.claude/skills/lemmalog
            cp -r "$SKILL_SRC" ~/.claude/skills/lemmalog
            echo "    skill -> ~/.claude/skills/lemmalog"
        else
            echo "==> removing lemmalog from Claude Code"
            claude mcp remove lemmalog >/dev/null 2>&1 || true
            rm -rf ~/.claude/skills/lemmalog
        fi
    else
        echo "-- claude CLI not found, skipping"
    fi
}

register_kimi() {
    if have kimi; then
        if [ "$MODE" = "install" ]; then
            echo "==> registering MCP server with Kimi CLI"
            kimi mcp add lemmalog \
                --env LEMMALOG_MCP_PATH="$SNAP" --env LEMMALOG_GLINER_DIR="$GLINER_DIR" \
                -- "$BIN/lemmalog-mcp" 2>/dev/null \
              || { kimi mcp remove lemmalog >/dev/null 2>&1 || true; \
                   kimi mcp add lemmalog \
                    --env LEMMALOG_MCP_PATH="$SNAP" --env LEMMALOG_GLINER_DIR="$GLINER_DIR" \
                    -- "$BIN/lemmalog-mcp"; }
            mkdir -p ~/.kimi/skills 2>/dev/null || true
            if [ -d ~/.kimi ]; then
                rm -rf ~/.kimi/skills/lemmalog
                cp -r "$SKILL_SRC" ~/.kimi/skills/lemmalog
                echo "    skill -> ~/.kimi/skills/lemmalog"
            fi
        else
            echo "==> removing lemmalog from Kimi CLI"
            kimi mcp remove lemmalog >/dev/null 2>&1 || true
            rm -rf ~/.kimi/skills/lemmalog 2>/dev/null || true
        fi
    else
        echo "-- kimi CLI not found, skipping"
    fi
}

if [ "$MODE" = "install" ]; then
    build
    mkdir -p "$(dirname "$SNAP")"
    echo "==> memory persists at $SNAP"
else
    echo "==> uninstalling lemmalog registrations"
fi

register_claude
register_kimi

if [ "$MODE" = "install" ]; then
    cat <<EOF

done. verify with:
  claude mcp list  (or: kimi mcp list)

then in a session, look for the lemmalog_* tools. the snapshot at
$SNAP carries memory across sessions and restarts.
EOF
else
    echo
    echo "done. (binaries and $SNAP left in place — delete manually if wanted)"
fi
