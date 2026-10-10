"""Optional ZeroGPU integration; import this before Torch or the model backend."""
from __future__ import annotations

import os


ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "t", "true"}

if ZERO_GPU:
    import spaces

    spaces.disable_gradio_auto_wrap()
    gpu = spaces.GPU
else:
    def gpu(**options):
        """Keep the local CPU/CUDA entry point independent of the Spaces SDK."""
        return lambda function: function


_started = False


def start_zero_gpu():
    """Pack the preloaded models before serving our mounted Gradio application."""
    global _started
    if not ZERO_GPU or _started:
        return
    # spaces==0.51.3 installs this hook on Blocks.launch. The studio mounts
    # Blocks into FastAPI instead, so it must invoke that same hook once.
    from spaces.zero import startup

    startup()
    _started = True
