#!/bin/sh
# PID-1-safe launcher. xvfb-run as ENTRYPOINT left a shell as PID 1 whose python
# child died silently (measured; the same command exec'd by hand binds in <1 s).
# So: start the virtual display ourselves and exec python as PID 1 -- proper
# signal handling, no wrapper, unbuffered logs.
set -e
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp &
export DISPLAY=:99
cd /opt/dss/surrogate
exec python ../app/server.py --renderer xvfb --host 0.0.0.0 --port 8090 \
    --jobs-dir /data/jobs --ckpt /data/ckpt/best_weights --device cpu "$@"
