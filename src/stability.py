"""Auditable optimizer updates and bounded divergence guards (also usable in CPU tests)."""
import csv
import json
import math
from pathlib import Path
import torch


class DivergenceError(RuntimeError):
    pass


def validation_diverged(value, best, settings):
    if not math.isfinite(value):
        return True
    return (math.isfinite(best) and value > best * settings.get('val_ratio', 1.25)
            and value - best >= settings.get('val_delta', .5))


def optimizer_update(model, optimizer, scheduler, scaler, clip):
    """Unscale before clipping; never step optimizer or scheduler on nonfinite norm."""
    scale_before = scaler.get_scale()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    finite = bool(torch.isfinite(norm))
    proposed_lr = scheduler.rate(scheduler.steps + 1)
    if finite:
        lr_used = scheduler.step()
        scaler.step(optimizer)
        scaler.update()
    else:
        lr_used = None
        # unscale_ may find no elementwise inf when FP32 norm reduction overflows.
        # Explicitly skip optimizer.step even then; scaler.step alone is insufficient.
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.update(new_scale=scale_before * .5)
    return dict(grad_norm=float(norm), grad_norm_clipped=min(float(norm), clip) if finite else None,
                proposed_lr=proposed_lr, lr_used=lr_used, optimizer_step=scheduler.steps,
                updated=finite, scale_before=scale_before, scale_after=scaler.get_scale())


class StepLog:
    fields = ['epoch', 'batch', 'attempt_step', 'optimizer_step', 'loss', 'grad_norm', 'grad_norm_clipped',
              'proposed_lr', 'lr_used', 'updated', 'scale_before', 'scale_after', 'event']

    def __init__(self, path, restored_rows=0):
        self.path = Path(path)
        self.count = restored_rows
        # Truncate a partially replayed epoch, preserving only committed checkpoint rows.
        if self.path.exists():
            with self.path.open() as f:
                rows = list(csv.DictReader(f))[:restored_rows]
            with self.path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.fields); w.writeheader(); w.writerows(rows)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()
        self.file = self.path.open('a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)

    def write(self, epoch, batch, loss, update=None, event=''):
        self.count += 1
        row = dict(epoch=epoch, batch=batch, attempt_step=self.count, loss=loss, event=event)
        if update:
            row.update(update)
        self.writer.writerow(row)
        # Line durability matters for diagnosing overflow/disconnects on Drive.
        self.file.flush()

    def close(self):
        self.file.close()


def save_status(output, status, **details):
    path = Path(output) / 'run_status.json'
    path.write_text(json.dumps(dict(status=status, **details), indent=2))
