import itertools
import numpy as np
import pytest
import sympy as sp

from qsynth.parser.hobo_parser import parse_sympy_expression
from qsynth.compiler.hobo_to_qubo import RosenbergReductionPass, _DegreeAnalyzer
from qsynth.parser.qubo_builder import build_qubo


def test_parse_sympy_expression_simple():
    expr = "3 * x0 * x1 * x2 - 2 * x1 * x2 + x0"
    objective = parse_sympy_expression(expr)
    
    assert objective.variable_vec.n_vars == 3
    assert set(objective.variable_vec.names) == {"x0", "x1", "x2"}
    
    analyzer = _DegreeAnalyzer()
    objective.ast.accept(analyzer)
    assert analyzer.max_degree == 3


def test_parse_sympy_expression_binary_reduction():
    # Test that x0**2 is reduced to x0
    expr = "x0**2 * x1 + x0**3"
    objective = parse_sympy_expression(expr)
    
    analyzer = _DegreeAnalyzer()
    objective.ast.accept(analyzer)
    # The max degree should be 2 (x0*x1), because x0**3 reduces to x0 (degree 1)
    assert analyzer.max_degree == 2


def test_rosenberg_reduction_logic():
    expr = "5 * x0 * x1 * x2"
    objective = parse_sympy_expression(expr)
    
    pass_manager = RosenbergReductionPass(penalty=10.0)
    qubo_obj = pass_manager.run(objective)
    
    # An auxiliary variable should be introduced
    assert qubo_obj.variable_vec.n_vars == 4
    assert "aux_0" in qubo_obj.variable_vec.names
    
    analyzer = _DegreeAnalyzer()
    qubo_obj.ast.accept(analyzer)
    assert analyzer.max_degree == 2


def test_rosenberg_reduction_energy_equivalence():
    """
    Test that the minimum energy of the resulting QUBO (over auxiliary variables)
    equals the energy of the original HOBO for all assignments of the original variables.
    """
    expr = "-3 * x0 * x1 * x2 + 2 * x0 * x3 - x1 * x2 * x3"
    objective = parse_sympy_expression(expr)
    
    pass_manager = RosenbergReductionPass(penalty="auto")
    qubo_obj = pass_manager.run(objective)
    
    qubo_problem = build_qubo(qubo_obj)
    
    original_names = ["x0", "x1", "x2", "x3"]
    aux_names = [n for n in qubo_problem.variable_names if n.startswith("aux_")]
    
    assert len(aux_names) >= 2 
    
    x_syms = sp.symbols("x0 x1 x2 x3")
    sympy_expr = sp.sympify(expr)
    
    for orig_vals in itertools.product([0, 1], repeat=4):
        # Evaluate classical HOBO
        hobo_energy = float(sympy_expr.subs(dict(zip(x_syms, orig_vals))))
        
        # Evaluate minimum QUBO energy over all aux assignments
        min_qubo_energy = float('inf')
        for aux_vals in itertools.product([0, 1], repeat=len(aux_names)):
            full_state = {}
            for name, val in zip(original_names, orig_vals):
                full_state[name] = val
            for name, val in zip(aux_names, aux_vals):
                full_state[name] = val
                
            x_vec = np.array([full_state[name] for name in qubo_problem.variable_names])
            energy = qubo_problem.energy(x_vec)
            min_qubo_energy = min(min_qubo_energy, energy)
            
        assert np.isclose(min_qubo_energy, hobo_energy)
