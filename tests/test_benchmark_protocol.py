"""Regression cases for signed exchange, prescribed actions, and scoring masks."""

import numpy as np
import torch

from ecoverse.bench.metrics import compute_flux_r2
from ecoverse.core.spec import InterventionSpec
from ecoverse.obs.dataset import encode_intervention
from ecoverse.sim.quality import classify_regime
from ecoverse.sim.sampler import sample_system_spec
from scripts.run_eval import EventStreamBaseline


def test_pulse_conditioning_and_signed_outputs():
    model = EventStreamBaseline(12, 8, 23, d_model=16, num_layers=1).eval()
    interventions = [InterventionSpec(type="bio_pulse", target_state=state, kappa=kappa)
                     for state, kappa in [("P1", 0.25), ("Z1", 3.5)]]
    actions = torch.tensor(np.stack([encode_intervention(action, 365) for action in interventions]))
    assert actions[0, 9:].tolist() == [-0.75, 0.0]
    assert actions[1, 9:].tolist() == [0.0, 2.5]
    batch = dict(obs_times=torch.zeros(2, 4), obs_var_ids=torch.zeros(2, 4, dtype=torch.long),
                 obs_values=torch.zeros(2, 4), obs_is_driver=torch.zeros(2, 4, dtype=torch.bool),
                 obs_attn_mask=torch.ones(2, 4, dtype=torch.bool), query_times=torch.ones(2, 3),
                 intervention=actions)
    with torch.no_grad():
        model.rate_head.weight.zero_()
        model.rate_head.bias.fill_(-2)
        x, rates, _, _ = model(batch)
        assert not torch.allclose(x[0], x[1])
        assert (rates[..., model.signed_channels] == -2).all()
        assert (rates[..., ~model.signed_channels] > 0).all()


def test_macro_scores_ignore_inactive_outputs():
    truth = torch.randn(4, 32, 23, generator=torch.Generator().manual_seed(17))
    mask = torch.ones(4, 23, dtype=torch.bool)
    mask[::2, ::2] = False
    pred = truth.masked_fill(~mask[:, None, :], 50)
    scores = compute_flux_r2(pred, truth, mask)
    assert [scores[k] for k in ("mean_channel_r2", "gpp_r2", "er_r2", "nep_r2")] == [1.] * 4


def test_seasonal_cycle_does_not_create_a_regime_shift():
    forcing = sample_system_spec("river", np.random.default_rng(5)).forcing
    states = np.zeros((2190, 8))
    states[:, 1] = 1 + 0.9 * np.sin(2 * np.pi * np.arange(len(states)) / len(states))
    assert classify_regime(states, states.any(axis=0), forcing) == "forced_periodic"
