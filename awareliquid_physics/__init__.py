"""
AwareLiquid-Physic — the physics branch of the AwareLiquid line.

Physics is written INTO the architecture (hard-constraint Hamiltonian + symplectic
integration), sitting on the shared LIQUID (LTC) substrate. The liquid core reads
an observed trajectory prefix to infer a system's hidden parameters and conditions
the Hamiltonian's energy landscape; a symplectic rollout then predicts the future
with energy conserved by construction.

Scope: continuous-state trajectory prediction, validated on physics metrics
(rollout MSE, energy drift). NOT a language model; no hallucination claim.
"""

from .datasets import gen_wave_1d, gen_wave_1d_inhomogeneous
from .hamiltonian import (FiLMHamiltonianHead, HamiltonianHead,
                          MLPFieldHead, OperatorHamiltonianHead)
from .liquid_core import LiquidCore
from .model import GRUSeqModel, LiquidHamiltonianModel, LiquidOperatorHamiltonianModel
from .operator_potential import FiLM, FNOBlock, OperatorPotential, SpectralConv1d
from .parallel_scan import pscan, pscan_constant_A, pscan_sequential
from .train import finetune, train_pretrain, train_semigroup

__all__ = [
    "HamiltonianHead",
    "FiLMHamiltonianHead",
    "MLPFieldHead",
    "OperatorHamiltonianHead",
    "LiquidCore",
    "LiquidHamiltonianModel",
    "LiquidOperatorHamiltonianModel",
    "GRUSeqModel",
    "OperatorPotential",
    "SpectralConv1d",
    "FNOBlock",
    "FiLM",
    "gen_wave_1d",
    "gen_wave_1d_inhomogeneous",
    "train_semigroup",
    "train_pretrain",
    "finetune",
    "pscan",
    "pscan_constant_A",
    "pscan_sequential",
]
