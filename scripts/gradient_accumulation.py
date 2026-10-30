"""Sample-weighted gradient accumulation with one update per effective batch."""


def accumulated_step(model, criterion, optimizer, samples, targets, device, microbatch_size,
                     before_microbatch=None, before_optimizer_step=None):
    import torch

    count = len(targets)
    if count < 1 or microbatch_size < 1 or len(samples) != count:
        raise ValueError("Need a nonempty aligned batch and a positive microbatch size")
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    detached_outputs = []
    for start in range(0, count, microbatch_size):
        if before_microbatch is not None:
            before_microbatch()
        images = samples[start:start + microbatch_size].to(device, non_blocking=False)
        labels = targets[start:start + microbatch_size].to(device, non_blocking=False)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss in a microbatch, no optimizer step performed")
        weight = len(labels) / count
        weighted_loss += float(loss.detach().cpu()) * weight
        (loss * weight).backward()
        detached_outputs.append(outputs.detach().cpu())
        del images, labels, outputs, loss
    if before_optimizer_step is not None:
        before_optimizer_step()
    for value in model.parameters():
        if value.grad is not None and not torch.isfinite(value.grad).all():
            raise RuntimeError("Nonfinite accumulated gradient, no optimizer step performed")
    optimizer.step()
    return torch.tensor(weighted_loss, dtype=torch.float64), torch.cat(detached_outputs)
