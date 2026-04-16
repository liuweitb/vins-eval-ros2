"""Sliding-window marginalization via Schur complement."""
import numpy as np
from typing import Callable, List, Tuple


class MarginalizationInfo:
    """Accumulates residuals and Jacobians before performing Schur complement."""

    def __init__(self):
        self.n = 0           # size of kept variables
        self.m = 0           # size of marginalized variables
        self.linearized_jacobians: np.ndarray | None = None
        self.linearized_residuals: np.ndarray | None = None
        self.valid = False

    def marginalize(self,
                    H: np.ndarray,
                    b: np.ndarray,
                    keep_size: int,
                    marg_size: int) -> None:
        """
        Apply Schur complement to eliminate marg_size variables.

        The ordering is [kept | marginalized].
        H and b must be arranged accordingly before calling.
        """
        self.n = keep_size
        self.m = marg_size

        Hmm = H[keep_size:, keep_size:]
        Hmk = H[keep_size:, :keep_size]
        Hkk = H[:keep_size, :keep_size]
        bm = b[keep_size:]
        bk = b[:keep_size]

        # Regularise Hmm for numerical stability
        Hmm += 1e-8 * np.eye(marg_size)
        Hmm_inv = np.linalg.inv(Hmm)

        H_schur = Hkk - Hmk.T @ Hmm_inv @ Hmk
        b_schur = bk  - Hmk.T @ Hmm_inv @ bm

        # Cholesky decomposition of the Schur complement
        try:
            L = np.linalg.cholesky(H_schur + 1e-8 * np.eye(keep_size))
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(H_schur)
            shift = max(0, -eigvals.min()) + 1e-6
            L = np.linalg.cholesky(H_schur + shift * np.eye(keep_size))

        self.linearized_jacobians = L.T          # upper triangular sqrt of info matrix
        self.linearized_residuals = np.linalg.solve(L, b_schur)
        self.valid = True

    def evaluate(self, x_delta: np.ndarray) -> np.ndarray:
        """Return residual given a perturbation x_delta of the kept variables."""
        if not self.valid:
            return np.zeros(self.n)
        return self.linearized_jacobians @ x_delta - self.linearized_residuals

    def get_jacobian(self) -> np.ndarray:
        return self.linearized_jacobians if self.valid else np.zeros((self.n, self.n))

    def get_residual(self) -> np.ndarray:
        return self.linearized_residuals if self.valid else np.zeros(self.n)
