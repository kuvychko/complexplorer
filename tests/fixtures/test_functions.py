"""
Common test functions for complexplorer tests.
"""

import numpy as np


class TestFunctions:
    """Collection of standard complex functions for testing."""
    
    @staticmethod
    def identity(z):
        """f(z) = z"""
        return z
    
    @staticmethod
    def quadratic(z):
        """f(z) = z^2"""
        return z**2
    
    @staticmethod
    def rational(z):
        """f(z) = (z - 1) / (z^2 + z + 1)"""
        return (z - 1) / (z**2 + z + 1)
    
    @staticmethod
    def exponential(z):
        """f(z) = e^z"""
        return np.exp(z)
    
    @staticmethod
    def sine(z):
        """f(z) = sin(z)"""
        return np.sin(z)
    
    @staticmethod
    def reciprocal(z):
        """f(z) = 1/z"""
        with np.errstate(divide='ignore', invalid='ignore'):
            return 1.0 / z
    
    @staticmethod
    def mobius(z):
        """f(z) = (z - i) / (z + i)"""
        return (z - 1j) / (z + 1j)
    
    @staticmethod
    def constant(z):
        """f(z) = 1 + 2i"""
        return np.full_like(z, 1 + 2j)
    
    @staticmethod
    def polynomial(z):
        """f(z) = z^4 - 1"""
        return z**4 - 1
    
    @staticmethod
    def logarithm(z):
        """f(z) = log(z)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(z)
    
    @staticmethod
    def sqrt(z):
        """f(z) = sqrt(z)"""
        return np.sqrt(z)
    
    @staticmethod
    def blaschke(z, a=0.5):
        """Blaschke factor: f(z) = (z - a) / (1 - conj(a)*z)"""
        return (z - a) / (1 - np.conj(a) * z)


def get_test_function(name):
    """Get test function by name."""
    functions = {
        'identity': TestFunctions.identity,
        'quadratic': TestFunctions.quadratic,
        'rational': TestFunctions.rational,
        'exponential': TestFunctions.exponential,
        'sine': TestFunctions.sine,
        'reciprocal': TestFunctions.reciprocal,
        'mobius': TestFunctions.mobius,
        'constant': TestFunctions.constant,
        'polynomial': TestFunctions.polynomial,
        'logarithm': TestFunctions.logarithm,
        'sqrt': TestFunctions.sqrt,
        'blaschke': TestFunctions.blaschke,
    }
    
    if name not in functions:
        raise ValueError(f"Unknown test function: {name}")
    
    return functions[name]


def get_special_points():
    """Get special complex points for testing."""
    return {
        'zeros': [0, 1, -1, 1j, -1j],  # Common zeros
        'poles': [0, 1, -1, 1j, -1j],  # Common poles
        'branch_points': [0, 1, -1],   # Branch points for sqrt, log
        'essential_singularities': [0],  # For exp(1/z)
        'unit_circle': np.exp(2j * np.pi * np.linspace(0, 1, 16, endpoint=False)),
        'real_axis': np.linspace(-2, 2, 9),
        'imaginary_axis': 1j * np.linspace(-2, 2, 9),
    }