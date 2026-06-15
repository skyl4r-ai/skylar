#!/usr/bin/env bash
# Launcher for the Skylar COBOL demo (used by the vhs tape).
# Sets the GnuCOBOL env so `cobc -x` uses the system gcc, then runs the real demo.
set -e
ROOT=/home/mwspace/htdocs/skylar/projects/skylar-cobol
export COBC=$ROOT/tools/gnucobol-env/bin/cobc
export COB_CC=/usr/bin/gcc
export LD_LIBRARY_PATH=$ROOT/tools/gnucobol-env/lib:${LD_LIBRARY_PATH:-}
export SKYLAR_DEMO_GATED=${SKYLAR_DEMO_GATED:-}
clear
exec /home/mwspace/htdocs/skylar/.venv/bin/python \
     /home/mwspace/htdocs/skylar/pip-skylar/assets/cobol_demo_runner.py
