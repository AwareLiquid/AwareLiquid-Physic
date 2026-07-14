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

from .hamiltonian import HamiltonianHead, MLPFieldHead
from .liquid_core import LiquidCore
from .model import GRUSeqModel, LiquidHamiltonianModel
from .parallel_scan import pscan, pscan_constant_A, pscan_sequential

__all__ = [
    "HamiltonianHead",
    "MLPFieldHead",
    "LiquidCore",
    "LiquidHamiltonianModel",
    "GRUSeqModel",
    "pscan",
    "pscan_constant_A",
    "pscan_sequential",
]
