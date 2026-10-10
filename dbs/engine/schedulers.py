"""Learning-rate schedules that are not in ``torch.optim.lr_scheduler``."""
from __future__ import annotations

from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _warmup_step_decay(step: int, warmup_steps: int, decay: float, decay_every: int) -> float:
    return min(1.0, (step + 1) / max(1, warmup_steps)) * decay ** (step // decay_every)


class WarmupStepDecay(LambdaLR):
    """Linear warmup over ``warmup_steps`` updates, then ``lr *= decay`` every ``decay_every`` updates.

    Step it once per optimizer update (``interval: step``).
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int = 2000, decay: float = 0.9,
                 decay_every: int = 40_000, last_epoch: int = -1):
        self.warmup_steps, self.decay, self.decay_every = int(warmup_steps), float(decay), int(decay_every)
        factor = partial(_warmup_step_decay, warmup_steps=self.warmup_steps, decay=self.decay,
                         decay_every=self.decay_every)
        super().__init__(optimizer, factor, last_epoch)
