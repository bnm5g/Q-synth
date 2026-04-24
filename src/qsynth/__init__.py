"""
Q-Synth: Quantum Program Synthesis Compiler for Financial Optimization.

A rigorous compiler pipeline that synthesizes quantum circuits from
high-level financial objective functions via:
  Parser → AST → IR (Pauli Hamiltonian) → Synthesizer → Compiler → Evaluator
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("qsynth")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__author__ = "Q-Synth Research Team"
__all__ = ["__version__", "__author__"]
