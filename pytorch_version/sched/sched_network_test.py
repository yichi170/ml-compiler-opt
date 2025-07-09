
import torch
from sched_network import SchedNetwork

def test_sched_network():
    num_candidates = 256
    batch_size = 4

    # Create dummy observations based on the expected spec
    observations = {
        'mask': torch.randint(0, 2, (batch_size, num_candidates), dtype=torch.int64),
        'is_top': torch.randint(0, 2, (batch_size, num_candidates), dtype=torch.int64),
        'is_bot': torch.randint(0, 2, (batch_size, num_candidates), dtype=torch.int64),
        'pos': torch.randn(batch_size, num_candidates, dtype=torch.float32),
        'excess': torch.randn(batch_size, num_candidates, dtype=torch.float32),
        'current_max': torch.randn(batch_size, num_candidates, dtype=torch.float32),
        'critical_max': torch.randn(batch_size, num_candidates, dtype=torch.float32),
        'su_latency': torch.randn(batch_size, num_candidates, dtype=torch.float32).abs() + 1e-5, # Ensure positive for log
        'su_height': torch.randn(batch_size, num_candidates, dtype=torch.float32).abs() + 1e-5,
        'su_depth': torch.randn(batch_size, num_candidates, dtype=torch.float32).abs() + 1e-5,
        'su_succs_left': torch.randn(batch_size, num_candidates, dtype=torch.float32),
        'su_preds_left': torch.randn(batch_size, num_candidates, dtype=torch.float32),
        'su_succs': torch.randn(batch_size, num_candidates, dtype=torch.float32),
        'su_preds': torch.randn(batch_size, num_candidates, dtype=torch.float32),
        'sgpr_critical_limit': torch.randn(batch_size, dtype=torch.float32),
        'vgpr_critical_limit': torch.randn(batch_size, dtype=torch.float32),
        'sgpr_excess_limit': torch.randn(batch_size, dtype=torch.float32),
        'vgpr_excess_limit': torch.randn(batch_size, dtype=torch.float32),
    }

    model = SchedNetwork(num_candidates=num_candidates)
    logits = model(observations)

    # Check output shape
    assert logits.shape == (batch_size, num_candidates), f"Expected output shape {(batch_size, num_candidates)}, but got {logits.shape}"

    # Check that masked logits are very small negative numbers
    # For each item in the batch, check where mask is 0, logits should be -1e9
    for i in range(batch_size):
        invalid_mask = observations['mask'][i] == 0
        assert torch.all(logits[i, invalid_mask] < -1e8), "Masked logits are not sufficiently negative"

    print("SchedNetwork test passed!")

if __name__ == '__main__':
    test_sched_network()
