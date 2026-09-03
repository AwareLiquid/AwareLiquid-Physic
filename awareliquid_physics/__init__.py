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

from .datasets import (gen_nbody, gen_wave_1d, gen_wave_1d_inhomogeneous,
                       gen_wave_2d)
from .hamiltonian import (FiLMHamiltonianHead, HamiltonianHead,
                          MLPFieldHead, NonseparableHamiltonianHead,
                          OperatorHamiltonianHead,
                          OperatorHamiltonianHead2d)
from .liquid_core import LiquidCore
from .model import (GRUSeqModel, LiquidHamiltonianModel,
                    LiquidNBodyModel, LiquidOperatorHamiltonianModel)
from .operator_potential import (FiLM, FNOBlock, FNOBlock2d,
                                 OperatorPotential, OperatorPotential2d,
                                 SpectralConv1d, SpectralConv2d)
from .pairwise_potential import NBodyHamiltonianHead, PairwisePotential
from .parallel_scan import pscan, pscan_constant_A, pscan_sequential
from .train import finetune, train_pretrain, train_semigroup

__all__ = [
    "HamiltonianHead",
    "FiLMHamiltonianHead",
    "MLPFieldHead",
    "OperatorHamiltonianHead",
    "OperatorHamiltonianHead2d",
    "NonseparableHamiltonianHead",
    "NBodyHamiltonianHead",
    "LiquidCore",
    "LiquidHamiltonianModel",
    "LiquidOperatorHamiltonianModel",
    "LiquidNBodyModel",
    "GRUSeqModel",
    "OperatorPotential",
    "OperatorPotential2d",
    "SpectralConv1d",
    "SpectralConv2d",
    "FNOBlock",
    "FNOBlock2d",
    "FiLM",
    "PairwisePotential",
    "gen_wave_1d",
    "gen_wave_1d_inhomogeneous",
    "gen_wave_2d",
    "gen_nbody",
    "train_semigroup",
    "train_pretrain",
    "finetune",
    "pscan",
    "pscan_constant_A",
    "pscan_sequential",
]
