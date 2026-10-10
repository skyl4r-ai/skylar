#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# GnuCOBOL >= 3.2 without root, for COBOLEval (eval/bin.coboleval.py) and the COBOL rewards of post-training.
# The 3.1 of most distributions gives a false 0% (eval/README.md). The environment comes from conda-forge with its
# own C compiler, so it does not depend on the cluster's toolchain.
#
#   bash training/slurm/build_gnucobol.sh online   <prefix>             # one step, needs internet
#   bash training/slurm/build_gnucobol.sh download <pkgdir>             # where there is internet: packages + micromamba
#   bash training/slurm/build_gnucobol.sh install  <prefix> <pkgdir>    # no network needed
#
# Then: export COBC=<prefix>/bin/cobc COB_CC=<prefix>/bin/x86_64-conda-linux-gnu-cc (printed at the end; without
# COB_CC the scripts fall back to the system gcc) and run eval/bin.coboleval.py.
set -euo pipefail
MODE=${1:?usage: build_gnucobol.sh online <prefix> | download <pkgdir> | install <prefix> <pkgdir>}
SPECS=(gnucobol=3.2 gcc_linux-64 sysroot_linux-64)
MAMBA_URL=https://micro.mamba.pm/api/micromamba/linux-64/latest

get_micromamba() {   # $1 = directory that gets bin/micromamba
    [ -x "$1/bin/micromamba" ] && return
    mkdir -p "$1" && curl -fsSL "$MAMBA_URL" | tar -xj -C "$1" bin/micromamba
}

check() {   # compile and run a COBOL program with the new cobc and the environment's own C compiler
    local cobc="$1/bin/cobc" t; t=$(mktemp -d)
    export COB_CC="$1/bin/x86_64-conda-linux-gnu-cc"   # cobc's built-in default points to the build machine
    cat > "$t/hello.cob" <<'EOF'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 N PIC 9(4) VALUE 1234.
       PROCEDURE DIVISION.
           DISPLAY "GNUCOBOL OK " N.
           STOP RUN.
EOF
    (cd "$t" && "$cobc" -x hello.cob -o hello && ./hello) | grep -q "GNUCOBOL OK 1234" \
        || { echo "cobc at $cobc cannot compile and run a program"; exit 1; }
    echo "$("$cobc" --version | head -1): compiles and runs. Use it with:"
    echo "  export COBC=$cobc COB_CC=$COB_CC"
    rm -rf "$t"
}

case "$MODE" in
online)
    P=$(realpath -m "${2:?prefix}"); get_micromamba "$P.mamba"
    MAMBA_ROOT_PREFIX="$P.mamba" "$P.mamba/bin/micromamba" create -y -q -p "$P" -c conda-forge "${SPECS[@]}"
    check "$P" ;;
download)
    D=$(realpath -m "${2:?pkgdir}"); get_micromamba "$D"
    # resolve and download into the package cache, without creating anything
    MAMBA_ROOT_PREFIX="$D/root" "$D/bin/micromamba" create -y -q -p "$D/probe" -c conda-forge --download-only "${SPECS[@]}"
    echo "packages in $D/root/pkgs ($(du -sh "$D/root/pkgs" | cut -f1)); copy $D to the cluster and run: install <prefix> $D" ;;
install)
    P=$(realpath -m "${2:?prefix}"); D=$(realpath "${3:?pkgdir}")
    MAMBA_ROOT_PREFIX="$D/root" "$D/bin/micromamba" create -y -q -p "$P" -c conda-forge --offline "${SPECS[@]}"
    check "$P" ;;
*) echo "unknown mode $MODE"; exit 2 ;;
esac
