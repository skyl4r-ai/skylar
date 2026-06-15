#!/usr/bin/env bash
# Gated launcher for the Skylar COBOL demo (used by the vhs tape).
# Loads weights OFF-CAMERA, then blocks on stdin before each task so the vhs
# recorder controls exactly what is captured (one Enter = next task).
set -e
ROOT=/home/mwspace/htdocs/skylar/projects/skylar-cobol
export COBC=$ROOT/tools/gnucobol-env/bin/cobc
export COB_CC=/usr/bin/gcc
export LD_LIBRARY_PATH=$ROOT/tools/gnucobol-env/lib:${LD_LIBRARY_PATH:-}
export SKYLAR_DEMO_GATED=1
clear
exec /home/mwspace/htdocs/skylar/.venv/bin/python \
     /home/mwspace/htdocs/skylar/pip-skylar/assets/cobol_demo_runner.py
