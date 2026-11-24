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
# ONNX Runtime: ORT_DYLIB_PATH if set; otherwise put in ~/.lemmalog/lib, by
# LEMMALOG_ORT_BUILD:
#   auto (default)  vg-ort on a machine with an NVIDIA GPU (nvidia-smi works),
#                   else openvino on Linux x86_64, else cpu; each falls back
#                   to the next if it cannot be fetched
#   vg-ort          ONNX Runtime $ORT_VG_VERSION with the CUDA provider, from
#                   the $ORT_VG_IMAGE image (LEMMALOG_ORT_IMAGE; docker or
#                   podman), or from a saved image tarball given as
#                   LEMMALOG_ORT_URL (docker save / OCI layout, .tar[.gz]).
#                   Runs on the GPU when CUDA 12 and cuDNN 9 are installed.
#   openvino        Intel's onnxruntime-openvino $ORT_OPENVINO_VERSION from
#                   PyPI (OpenVINO provider + CPU/GPU/NPU plugins; python3)
#   cpu             Microsoft's plain ONNX Runtime $ORT_VERSION
# LEMMALOG_FETCH_ORT=0 skips; LEMMALOG_ORT_URL overrides the download
# (a .whl for openvino, a .tgz for cpu, an image tarball for vg-ort).

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
ORT_VG_VERSION="1.24.2"
ORT_VG_IMAGE="${LEMMALOG_ORT_IMAGE:-innoura/vg-ort:$ORT_VG_VERSION}"
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
    if [ "${LEMMALOG_FETCH_ORT:-1}" = "0" ]; then
        ort_hint
        return
    fi
    local want="${LEMMALOG_ORT_BUILD:-auto}" order
    case "$want" in
        auto)
            order=""
            have nvidia-smi && nvidia-smi -L >/dev/null 2>&1 && order="vg-ort"
            [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] && order="$order openvino"
            order="$order cpu" ;;
        vg-ort) order="vg-ort openvino cpu" ;;
        openvino) order="openvino cpu" ;;
        cpu) order="cpu" ;;
        *) echo "-- unknown LEMMALOG_ORT_BUILD=$want (auto, vg-ort, openvino or cpu)"; return ;;
    esac
    local b
    for b in $order; do
        case "$b" in
            vg-ort) fetch_ort_vg && { cuda_hint; return; } ;;
            openvino) have python3 && fetch_ort_openvino "$([ "$want" = openvino ] && echo "${LEMMALOG_ORT_URL:-}")" && return ;;
            cpu) fetch_ort_cpu "$([ "$want" = cpu ] && echo "${LEMMALOG_ORT_URL:-}")" && return ;;
        esac
        echo "-- falling back from the $b build"
    done
    ort_hint
}

ort_hint() {
    echo "-- ONNX Runtime not installed: free text will not be extracted until it is."
    echo "   Put $ORT_LIB in $ORT_LIB_DIR, or set ORT_DYLIB_PATH."
}

# Warn when the CUDA build is installed but CUDA 12 / cuDNN 9 are not.
cuda_hint() {
    local missing="" l
    for l in libcudart.so.12 libcublasLt.so.12 libcublas.so.12 libcufft.so.11 libcudnn.so.9; do
        if ! { ldconfig -p 2>/dev/null | grep -q "$l"; } && ! { [ -n "${LD_LIBRARY_PATH:-}" ] && find ${LD_LIBRARY_PATH//:/ } -maxdepth 1 -name "$l*" 2>/dev/null | grep -q .; }; then
            missing="$missing $l"
        fi
    done
    if [ -n "$missing" ]; then
        echo "-- CUDA libraries not found:$missing"
        echo "   lemmalog runs on CPU until CUDA 12 and cuDNN 9 are installed."
    fi
}

# Microsoft's plain ONNX Runtime release. $1: tarball URL, or empty.
fetch_ort_cpu() {
    local pkg="" url="$1" tmp
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) pkg="onnxruntime-linux-x64-$ORT_VERSION" ;;
        Linux-aarch64|Linux-arm64) pkg="onnxruntime-linux-aarch64-$ORT_VERSION" ;;
        Darwin-arm64) pkg="onnxruntime-osx-arm64-$ORT_VERSION" ;;
        Darwin-x86_64) pkg="onnxruntime-osx-x86_64-$ORT_VERSION" ;;
    esac
    [ -z "$url" ] && [ -n "$pkg" ] && url="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$pkg.tgz"
    [ -z "$url" ] && return 1
    have curl || return 1
    echo "==> fetching ONNX Runtime $ORT_VERSION into $ORT_LIB_DIR"
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
        return 0
    fi
    echo "-- could not fetch ONNX Runtime from $url"
    return 1
}

# ONNX Runtime with the CUDA provider from the vg-ort image: /ort/lib, taken
# from a saved image tarball (LEMMALOG_ORT_URL) or with docker/podman.
fetch_ort_vg() {
    local tmp url="${LEMMALOG_ORT_URL:-}" engine=""
    [ "$(uname -s)" = "Linux" ] || return 1
    tmp="$(mktemp -d)"
    mkdir -p "$ORT_LIB_DIR"
    if [ -n "$url" ]; then
        echo "==> unpacking ONNX Runtime $ORT_VG_VERSION (CUDA) from $url into $ORT_LIB_DIR"
        if ! { curl -fsSL "$url" -o "$tmp/image.tar" && have python3 && python3 - "$tmp/image.tar" "$ORT_LIB_DIR" <<'PY'
import io, json, os, sys, tarfile
image, dest = sys.argv[1], sys.argv[2]
outer = tarfile.open(image)  # docker save or OCI layout, gzip or not
names = set(outer.getnames())
manifest = json.load(outer.extractfile("manifest.json"))
found = 0
for layer in manifest[0]["Layers"]:
    data = outer.extractfile(layer).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for m in t.getmembers():
            path = m.name.lstrip("./")
            if not path.startswith("ort/lib/") or path.endswith("/"):
                continue
            base = os.path.basename(path)
            if not base.startswith("libonnxruntime"):
                continue
            out = os.path.join(dest, base)
            if os.path.lexists(out):
                os.remove(out)
            if m.issym() and "/" not in m.linkname and ".." not in m.linkname:
                os.symlink(m.linkname, out)
            elif m.isfile():
                with open(out, "wb") as f:
                    f.write(t.extractfile(m).read())
                os.chmod(out, 0o755)
            else:
                continue
            found += 1
if not found:
    sys.exit("no /ort/lib/libonnxruntime* in the image")
PY
        }; then
            rm -rf "$tmp"; echo "-- could not unpack the vg-ort image from $url"; return 1
        fi
    else
        if have docker; then engine=docker; elif have podman; then engine=podman; else rm -rf "$tmp"; return 1; fi
        echo "==> copying ONNX Runtime $ORT_VG_VERSION (CUDA) from $ORT_VG_IMAGE into $ORT_LIB_DIR"
        local cid
        if ! cid="$($engine create "$ORT_VG_IMAGE" 2>/dev/null)"; then
            rm -rf "$tmp"; echo "-- could not get the $ORT_VG_IMAGE image ($engine create failed)"; return 1
        fi
        $engine cp "$cid:/ort/lib/." "$tmp/" >/dev/null 2>&1 || true
        $engine rm "$cid" >/dev/null 2>&1 || true
        cp -P "$tmp"/libonnxruntime* "$ORT_LIB_DIR/" 2>/dev/null || true
    fi
    rm -rf "$tmp"
    if [ -e "$ORT_LIB_DIR/$ORT_LIB" ]; then
        echo "    -> $ORT_LIB_DIR/$ORT_LIB (CUDA execution provider included)"
        return 0
    fi
    echo "-- no $ORT_LIB in the vg-ort image"
    return 1
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
