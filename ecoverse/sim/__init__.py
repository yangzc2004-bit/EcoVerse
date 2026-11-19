"""Offline simulation and production engine for EcoVerse."""

from ecoverse.sim.pipeline import generate_dataset
from ecoverse.sim.quality import QualityReport, classify_regime, evaluate_quality
from ecoverse.sim.sampler import sample_system_spec
from ecoverse.sim.solver import SimulationResult, solve_system

__all__ = [
    "generate_dataset",
    "QualityReport",
    "classify_regime",
    "evaluate_quality",
    "sample_system_spec",
    "SimulationResult",
    "solve_system",
]
