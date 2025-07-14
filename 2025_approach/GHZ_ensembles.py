"""
Deterministic GHZ-state preparation using adaptive QND measurements
in the symmetric Dicke basis for ensembles of arbitrary atom number N.
"""

import numpy as np
from qutip import basis, tensor, qeye, Qobj, jmat, expect # type: ignore
from typing import List, Tuple
from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# Utilities for spin-N/2 ensembles
# -----------------------------------------------------------------------------

def dicke_basis_state(N: int, k: int) -> Qobj:
    """Return Dicke basis state |k> for an ensemble of N atoms."""
    return basis(N + 1, k)

def plus_state_spin(N: int, n_ensembles: int) -> Qobj:
    """Return |+>^{⊗ n} in the symmetric Dicke basis."""
    psi = sum([basis(N + 1, k) for k in range(N + 1)]).unit()
    return tensor([psi] * n_ensembles)

def ident(N: int, n: int) -> Qobj:
    """Return identity operator on n ensembles of dimension N+1."""
    return tensor([qeye(N + 1)] * n)

def Sz(N: int) -> Qobj:
    """Collective spin-z operator for spin-N/2 system."""
    return Qobj(np.diag([2 * k - N for k in range(N + 1)]))

def adaptive_rotation_y(N: int, j: int, delta: float, n: int) -> Qobj:
    """Apply adaptive R_y rotation on ensemble j with θ = πΔ/N."""
    theta = np.pi * delta / N if abs(delta) > 1e-6 else np.pi
    S = N / 2
    Ry = (-1j * theta / 2 * jmat(S, 'y')).expm()
    ops = [Ry if idx == j else qeye(N + 1) for idx in range(n)]
    return tensor(ops)

# -----------------------------------------------------------------------------
# QND parity measurement (simplified model)
# -----------------------------------------------------------------------------

@dataclass
class MeasurementResult:
    outcome: int
    post_state: Qobj
    delta: float

def measure_ZiZj(state: Qobj, i: int, j: int, N: int, n: int) -> MeasurementResult:
    """
    Simulate a QND Z_i Z_j parity measurement.
    Returns measurement result (+1/-1), updated state, and population imbalance Δ.
    """
    Sz_i = Sz(N)
    Sz_j = Sz(N)
    ops = [qeye(N + 1)] * n
    ops[i] = Sz_i
    ops[j] = -Sz_j
    delta_op = tensor(ops)
    delta = expect(delta_op, state).real
    outcome = +1 if np.random.rand() < 0.5 else -1
    P = (ident(N, n) + outcome * delta_op.unit()) / 2
    post = (P * state).unit()
    return MeasurementResult(outcome, post, delta)

# -----------------------------------------------------------------------------
# Protocol Driver
# -----------------------------------------------------------------------------

@dataclass
class ProtocolLogEntry:
    step: str
    outcome: int
    delta: float
    state: Qobj = field(repr=False)

def ghz_protocol_largeN(N: int, n_ensembles: int, seed: int = 42) -> Tuple[Qobj, List[ProtocolLogEntry]]:
    """
    Run the GHZ preparation protocol with atom number N per ensemble.
    """
    np.random.seed(seed)
    state = plus_state_spin(N, n_ensembles)
    log: List[ProtocolLogEntry] = [ProtocolLogEntry("Init", 0, 0.0, state)]

    for i in range(n_ensembles - 1):
        meas = measure_ZiZj(state, i, i + 1, N, n_ensembles)
        log.append(ProtocolLogEntry(f"Measure Z_{i}Z_{i+1}", meas.outcome, meas.delta, meas.post_state))
        state = meas.post_state

        if meas.outcome == -1:
            R = adaptive_rotation_y(N, i + 1, meas.delta, n_ensembles)
            state = (R * state).unit()
            log.append(ProtocolLogEntry(f"Adaptive Y-rotation on {i+1}", 0, meas.delta, state))

    return state, log

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    N = 10  # Number of atoms per ensemble
    n_ensembles = 3

    final_state, log = ghz_protocol_largeN(N, n_ensembles, seed=123)

    print(f"Final state dimension: {final_state.shape}")
    print(f"Protocol log ({len(log)} steps):")
    for entry in log:
        print(f"{entry.step:>30} | Outcome: {entry.outcome:2d} | Δ = {entry.delta:6.2f}")
