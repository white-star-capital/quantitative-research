"""Vector Genetic Programming (VGP).

Evolve trading strategies from multi-asset crypto data using DEAP + vectorbt.

Architecture invariants (enforced — do not violate):
  - vgp.evolution must NOT import vectorbt
  - vgp.backtest must NOT import deap
  - Interface between them: numpy array in -> fitness tuple out
  - All GP primitive functions must be module-level (not lambdas)
  - All primitives accept and return np.ndarray — no pandas inside primitives
"""

__version__ = "0.1.0"
