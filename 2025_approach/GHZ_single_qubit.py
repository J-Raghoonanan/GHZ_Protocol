"""
GHZ_State_Prep.py

Deterministic measurement-based protocol to prepare an n-qubit GHZ state
using sequential QND parity measurements and adaptive collective rotations.

Based on the extension of the Bell-state algorithm from
Phys.Rev.A 108, 032420 to n>2 ensembles.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
from qutip import (tensor, basis, Qobj, sigmax, sigmay, sigmaz, qeye, ket2dm, expect) # type: ignore

# -----------------------------------------------------------------------------
# Utility: single‑qubit operators positioned at site k
# -----------------------------------------------------------------------------

def op_on_qubit(op: Qobj, k: int, n: int) -> Qobj:
    """
    Return `op` acting on qubit k (0-indexed) and identity elsewhere.
    """
    op_list = [op if i == k else qeye(2) for i in range(n)]
    return tensor(op_list)

def z(i: int, n: int) -> Qobj:
    return op_on_qubit(sigmaz(), i, n)

def x(i: int, n: int) -> Qobj:
    return op_on_qubit(sigmax(), i, n)

def y(i: int, n: int) -> Qobj:
    return op_on_qubit(sigmay(), i, n)

def ident(n: int) -> Qobj:
    """
    n-qubit identity with dims [[2,2,…],[2,2,…]].
    """
    return tensor([qeye(2)] * n)

# -----------------------------------------------------------------------------
# Initial state |+>^{⊗ n}
# -----------------------------------------------------------------------------

def plus_state(n: int) -> Qobj:
    """
    Return |+⟩^{⊗ n} where |+⟩=(|0⟩+|1⟩)/√2.
    """
    plus = (basis(2, 0) + basis(2, 1)).unit()
    return tensor([plus] * n)

# -----------------------------------------------------------------------------
# Projective QND measurement of Z_i Z_j
# -----------------------------------------------------------------------------

@dataclass
class MeasurementResult:
    outcome: int               # +1 or −1 eigenvalue
    prob: float                # probability of that outcome
    post_state: Qobj           # normalised post‑measurement state
    delta: int = 0             # population imbalance (needed for adaptive θ)

def measure_parity(state: Qobj, i: int, j: int, rng: np.random.Generator) -> MeasurementResult:
    """
    Perform a Z_i Z_j parity measurement on `state`.

    Returns outcome (+1 or -1), probability, post-measurement state,
    and Δ = n_1(i) - n_1(j) (population imbalance) which drives the adaptive rotation.
    """
    n = int(np.log2(state.shape[0]))
    ZiZj = z(i, n) * z(j, n)
    I_n     = ident(n)
    P_plus  = (ZiZj + I_n) / 2
    P_minus = (I_n - ZiZj) / 2

    # p_plus  = (state.dag() * P_plus * state).tr().real
    p_plus  = float(expect(P_plus, state).real)
    p_minus = 1 - p_plus

    outcome = +1 if rng.random() < p_plus else -1
    P = P_plus if outcome == +1 else P_minus
    post = (P * state).unit()

    # Compute Δ from populations of |1> in each qubit (qubit model)
    rho = ket2dm(post)
    # n1_i = ((-z(i, n) + ident(n)) * rho / 2).tr().real
    n1_i = float(expect((ident(n) - z(i, n)) / 2, rho).real)
    # n1_j = ((-z(j, n) + ident(n)) * rho / 2).tr().real
    n1_j = float(expect((ident(n) - z(j, n)) / 2, rho).real)
    delta = int(round(n1_i - n1_j))

    return MeasurementResult(outcome, p_plus if outcome == +1 else p_minus, post, delta)

# -----------------------------------------------------------------------------
# Adaptive collective Y rotation by θ on qubit k
# -----------------------------------------------------------------------------
def adaptive_rotation(state: Qobj, k: int, delta: float, N: int) -> Qobj:
    """
    Collective Ry(θ) on qubit k:
        θ = π Δ / N   (many-atom case)
        θ = π         (qubit case or if |Δ| < ½)
    """
    n = int(np.log2(state.shape[0]))

    # --- choose angle ---------------------------------------------
    if N == 1 or abs(delta) < 0.5:        # handle Δ = 0 safely
        theta = np.pi
    else:
        theta = np.pi * delta / N

    # Ry(θ) = exp(-i θ Y / 2)
    Ry = (-1j * theta / 2 * y(k, n)).expm()

    return (Ry * state).unit()


# -----------------------------------------------------------------------------
# Global X parity measurement
# -----------------------------------------------------------------------------

def measure_global_x(state: Qobj, rng: np.random.Generator) -> MeasurementResult:
    n = int(np.log2(state.shape[0]))
    X_global = tensor([sigmax()] * n) 

    # P_plus  = (X_global + qeye(2 ** n)) / 2
    # P_minus = (qeye(2 ** n) - X_global) / 2
    I_n     = ident(n)
    P_plus  = (X_global + I_n) / 2
    P_minus = (I_n - X_global) / 2

    # p_plus  = (state.dag() * P_plus * state).tr().real
    p_plus  = float(expect(P_plus, state).real)
    p_minus = 1 - p_plus

    outcome = +1 if rng.random() < p_plus else -1
    P = P_plus if outcome == +1 else P_minus
    post = (P * state).unit()

    return MeasurementResult(outcome, p_plus if outcome == +1 else p_minus, post)

# -----------------------------------------------------------------------------
# Protocol driver 
# -----------------------------------------------------------------------------

@dataclass
class ProtocolLogEntry:
    description: str
    state: Qobj = field(repr=False)

def ghz_protocol(n_qubits: int, N: int = 1, seed: int | None = None) -> Tuple[Qobj, List[ProtocolLogEntry]]:
    """
    Run the deterministic GHZ preparation protocol.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / ensembles.
    N : int
        Effective spin length (atomic number per ensemble).  For qubit model set N=1.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    final_state : Qobj
        The prepared state.
    log : list[ProtocolLogEntry]
        Step-by-step log useful for debugging.
    """
    rng = np.random.default_rng(seed)
    state = plus_state(n_qubits)
    log: List[ProtocolLogEntry] = [ProtocolLogEntry("Initial |+>^{⊗ n} state", state)]

    # Pair‑parity checks
    for i in range(n_qubits - 1):
        meas = measure_parity(state, i, i + 1, rng)
        log.append(ProtocolLogEntry(f"Measured Z_{i}Z_{i+1} → {meas.outcome}", meas.post_state))
        state = meas.post_state

        if meas.outcome == -1:
            state = adaptive_rotation(state, i + 1, meas.delta, N)
            log.append(ProtocolLogEntry(f"Applied Y‑rotation on qubit {i+1} (Δ={meas.delta})", state))

            # Re‑measure to confirm parity now +1
            meas2 = measure_parity(state, i, i + 1, rng)
            log.append(ProtocolLogEntry(f"Re‑measured Z_{i}Z_{i+1} → {meas2.outcome}", meas2.post_state))
            state = meas2.post_state
            if meas2.outcome == -1:
                raise RuntimeError("Parity correction failed; check rotation rule.")

    # Global X parity
    gmeas = measure_global_x(state, rng)
    log.append(ProtocolLogEntry(f"Measured global X parity → {gmeas.outcome}", gmeas.post_state))
    state = gmeas.post_state

    if gmeas.outcome == -1:
        # Correct by applying Z on first qubit
        state = (z(0, n_qubits) * state).unit()
        log.append(ProtocolLogEntry("Applied Z on qubit 0 to correct global parity", state))

    return state, log

# -----------------------------------------------------------------------------
# Fidelity helper
# -----------------------------------------------------------------------------

def ghz_state(n: int) -> Qobj:
    """Ideal n‑qubit GHZ state."""
    zero = tensor([basis(2, 0)] * n)
    one  = tensor([basis(2, 1)] * n)
    return (zero + one).unit()

def fidelity(state: Qobj, target: Qobj) -> float:
    """|⟨target|state⟩|²"""
    overlap = target.overlap(state)  # built-in QuTiP overlap
    return abs(overlap) ** 2

# -----------------------------------------------------------------------------
# Demo usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    n = 3
    final, steps = ghz_protocol(n, seed=42)
    F = fidelity(final, ghz_state(n))
    print(f"Fidelity with ideal GHZ_{n}: {F:.6f}")
    print("Protocol log:")
    for step in steps:
        print(" –", step.description)
