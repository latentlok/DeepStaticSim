#!/bin/sh
# PID-1-safe launcher. xvfb-run as ENTRYPOINT left a shell as PID 1 whose python
# child died silently (measured; the same command exec'd by hand binds in <1 s).
# So: start the virtual display ourselves and exec python as PID 1 -- proper
# signal handling, no wrapper, unbuffered logs.
#
# Device: cuda when the container can see a working GPU (run with --gpus all on
# a host with the NVIDIA driver + container toolkit), otherwise cpu. Same image,
# same command, either way. Override with `-e DSS_DEVICE=cpu`.
set -e
if [ -n "$DSS_DEVICE" ]; then
    DEVICE="$DSS_DEVICE"
elif python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda
else
    DEVICE=cpu
fi
echo "deepstaticsim: inference device = $DEVICE"
if [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; print('deepstaticsim: gpu =', torch.cuda.get_device_name(0))"
fi

Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp &
export DISPLAY=:99
mkdir -p /data/jobs
cd /opt/dss/surrogate
exec python ../app/server.py --renderer xvfb --host 0.0.0.0 --port 8090 \
    --jobs-dir /data/jobs --ckpt /opt/model/ckpt/best_weights --device "$DEVICE" "$@"
