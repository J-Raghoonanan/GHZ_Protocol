"""
Deterministic GHZ-state preparation using adaptive QND measurements
in the symmetric Dicke basis for ensembles of arbitrary atom number N.
"""

import numpy as np
from qutip import basis, tensor, qeye, Qobj, jmat, expect # type: ignore
from typing import List, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

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


def global_x_rotation(N: int, n: int) -> Qobj:
    S = N / 2
    Ry = (-1j * (-np.pi / 2) / 2 * jmat(S, 'y')).expm()
    return tensor([Ry] * n)

def global_z_parity_measurement(state: Qobj, N: int, n: int) -> Tuple[int, Qobj]:
    Sz_op = Sz(N)
    Z_global = tensor([Sz_op] * n)
    parity_op = Z_global.unit()
    outcome = +1 if np.random.rand() < 0.5 else -1
    P = (ident(N, n) + outcome * parity_op) / 2
    post = (P * state).unit()
    return outcome, post

def apply_z_correction(N: int, n: int, state: Qobj) -> Qobj:
    S = N / 2
    Z = jmat(S, 'z')
    Z_ops = [(-1j * np.pi * Z).expm() if i == 0 else qeye(N + 1) for i in range(n)]
    correction = tensor(Z_ops)
    return (correction * state).unit()

# -----------------------------------------------------------------------------
# Protocol Driver
# -----------------------------------------------------------------------------

@dataclass
class ProtocolLogEntry:
    step: str
    outcome: int
    delta: float
    state: Qobj = field(repr=False)

def ghz_protocol_largeN(N: int, n: int, seed: int = 42) -> Tuple[Qobj, List[ProtocolLogEntry]]:
    """
    Run the deterministic GHZ preparation protocol for large ensembles.

    Parameters
    ----------
    N : int
        Atom number per ensemble.
    n : int
        Number of ensembles.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    final_state : Qobj
        The prepared state.
    log : list[ProtocolLogEntry]
        Step-by-step log of the protocol.
    """
    np.random.seed(seed)
    state = plus_state_spin(N, n)
    log: List[ProtocolLogEntry] = [ProtocolLogEntry("Initial |+>^{⊗ n} state", 0, 0.0, state)]

    for i in range(n - 1):
        meas= measure_ZiZj(state, i, i + 1, N, n)
        outcome = meas.outcome
        delta = meas.delta
        post_state = meas.post_state
        log.append(ProtocolLogEntry(f"Measured Z_{i}Z_{i+1} → {outcome}", outcome, delta, post_state))
        state = post_state

        if outcome == -1:
            R = adaptive_rotation_y(N, i + 1, delta, n)
            state = (R * state).unit()
            log.append(ProtocolLogEntry(f"Applied Y‑rotation on ensemble {i+1} (Δ={delta:.2f})", 0, delta, state))

            # Re-measure
            meas2 = measure_ZiZj(state, i, i + 1, N, n)
            outcome2 = meas2.outcome
            delta2 = meas2.delta
            post_state2 = meas2.post_state
            log.append(ProtocolLogEntry(f"Re-measured Z_{i}Z_{i+1} → {outcome2}", outcome2, delta2, post_state2))
            state = post_state2

            if outcome2 == -1:
                raise RuntimeError("Parity correction failed; check rotation rule.")

    # Global X parity check
    rot = global_x_rotation(N, n)
    rotated = (rot * state).unit()
    xpar_outcome, xpar_state = global_z_parity_measurement(rotated, N, n)
    log.append(ProtocolLogEntry(f"Measured global X parity → {xpar_outcome}", xpar_outcome, 0.0, xpar_state))
    state = xpar_state

    if xpar_outcome == -1:
        state = apply_z_correction(N, n, state)
        log.append(ProtocolLogEntry("Applied Z on ensemble 0 to correct global parity", 0, 0.0, state))

    return state, log

# -----------------------------------------------------------------------------
# Calculate Fidelity and plots
# -----------------------------------------------------------------------------


def ghz_state(N: int, n: int) -> Qobj:
    """
    Return ideal GHZ state in symmetric Dicke basis:
        (|k>^{⊗n} + |N-k>^{⊗n}) / sqrt(2), summed over k
    We'll use the central value k = N//2 as a GHZ-like analog for large-N.
    """
    k = N // 2
    ghz = (tensor([basis(N + 1, k)] * n) + tensor([basis(N + 1, N - k)] * n)).unit()
    return ghz

def fidelity(state: Qobj, target: Qobj) -> float:
    return abs(target.overlap(state)) ** 2

def plot_fidelity(log: List[ProtocolLogEntry], target: Qobj, N: int, n_ensembles: int) -> None:
    """
    Plot fidelity of each step in the protocol log against the target GHZ state.
    """
    fid_vals = [fidelity(entry.state, target) for entry in log]
    steps = [entry.step for entry in log]

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(fid_vals)), fid_vals, marker='o')
    plt.xticks(range(len(fid_vals)), steps, rotation=45, ha='right')
    plt.ylabel("Fidelity with GHZ state")
    plt.xlabel("Protocol Step")
    # plt.title(f"GHZ Fidelity vs. Protocol Step (N={target.shape[0] - 1}, n={len(log)})")
    plt.title(f"GHZ Fidelity vs. Protocol Step (N={N}, n={n_ensembles})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/fidelity_vs_step.pdf")
    return


# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    N = 10  # Number of atoms per ensemble
    n_ensembles = 3

    final_state, log = ghz_protocol_largeN(N, n_ensembles, seed=123)

    # Console output
    print(f"Final state dimension: {final_state.shape}")
    print(f"Protocol log ({len(log)} steps):")
    for entry in log:
        print(f"{entry.step:>30} | Outcome: {entry.outcome:2d} | Δ = {entry.delta:6.2f}")

    # Plot fidelity
    target = ghz_state(N, n_ensembles)
    plot_fidelity(log, target, N, n_ensembles)
