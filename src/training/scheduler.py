from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR

def get_arctic_tilt_scheduler(optimizer, num_training_steps):
    """
    Arctic-TILT scheduler:
    - 1% warmup: 1e-3
    - 89% linear: 1e-3 → 2e-4
    - 10% cosine: 2e-4 → 5e-5
    """

    warmup_steps = max(1, int(0.01 * num_training_steps))
    linear_steps = max(1, int(0.89 * num_training_steps))
    cosine_steps = num_training_steps - warmup_steps - linear_steps

    # 1. Warmup: Constant 1e-3
    warmup_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=warmup_steps
    )

    # 2. Linear decay: 1e-3 → 2e-4
    linear_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=linear_steps
    )

    # 3. Cosine decay: 2e-4 → 5e-5
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=5e-5
    )
    cosine_scheduler.base_lrs = [2e-4]

    # Combine schedulers
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, linear_scheduler, cosine_scheduler],
        milestones=[warmup_steps, warmup_steps + linear_steps]
    )