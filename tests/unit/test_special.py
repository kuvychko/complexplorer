"""Unit tests for mathematical special functions."""

import pytest
import numpy as np
import warnings
from complexplorer.special import (
    gamma, erf, get_special_function, special, HAS_SCIPY
)


class TestSpecialFunctions:
    """Test special mathematical functions."""
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_gamma_function(self):
        """Test gamma function for known values."""
        # Gamma(1) = 0! = 1
        assert abs(gamma(1) - 1.0) < 1e-10
        
        # Gamma(2) = 1! = 1
        assert abs(gamma(2) - 1.0) < 1e-10
        
        # Gamma(3) = 2! = 2
        assert abs(gamma(3) - 2.0) < 1e-10
        
        # Gamma(4) = 3! = 6
        assert abs(gamma(4) - 6.0) < 1e-10
        
        # Test with complex argument
        z = gamma(1 + 1j)
        assert isinstance(z, complex)
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_gamma_array_input(self):
        """Test gamma function with array input."""
        z = np.array([1, 2, 3, 4])
        result = gamma(z)
        expected = np.array([1, 1, 2, 6])
        
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_erf_function(self):
        """Test error function."""
        # erf(0) = 0
        assert abs(erf(0)) < 1e-10
        
        # erf(∞) → 1
        assert abs(erf(10) - 1.0) < 1e-6
        
        # erf(-x) = -erf(x)
        x = 1.5
        assert abs(erf(-x) + erf(x)) < 1e-10
    
    def test_erf_complex(self):
        """Test error function with complex argument."""
        z = erf(1 + 1j)
        assert isinstance(z, (complex, np.complex128))
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_special_namespace(self):
        """Test the special namespace."""
        # Check that functions are accessible via namespace
        assert hasattr(special, 'gamma')
        assert hasattr(special, 'erf')
        assert hasattr(special, 'digamma')
        assert hasattr(special, 'psi')  # Alias for digamma
        
        # Test that they work
        result = special.gamma(3)
        assert abs(result - 2.0) < 1e-10
    
    def test_get_special_function(self):
        """Test getting functions by name."""
        # Get valid function
        func = get_special_function('erf')
        assert callable(func)
        assert abs(func(0)) < 1e-10
        
        # Test case insensitive
        func = get_special_function('ERF')
        assert callable(func)
        
        # Test invalid function name
        with pytest.raises(ValueError, match="Unknown function"):
            get_special_function('invalid_function')
    
    @pytest.mark.skipif(HAS_SCIPY, reason="scipy is available")
    def test_no_scipy_warnings(self):
        """Test that appropriate warnings/errors are raised without scipy."""
        # Should raise NotImplementedError for functions requiring scipy
        with pytest.raises(NotImplementedError):
            gamma(2.5)
        
        # erf should work but warn about approximation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = erf(1.0)
            assert len(w) > 0
            assert "approximate" in str(w[0].message).lower()


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
class TestSpecialFunctionsWithScipy:
    """Tests that require scipy."""
    
    def test_zeta_function(self):
        """Test Riemann zeta function."""
        from complexplorer.special import zeta
        
        # ζ(2) = π²/6
        result = zeta(2)
        expected = np.pi**2 / 6
        assert abs(result - expected) < 1e-10
        
        # ζ(0) = -1/2
        result = zeta(0)
        assert abs(result - (-0.5)) < 1e-10
    
    def test_bessel_functions(self):
        """Test Bessel functions."""
        from complexplorer.special import bessel_j, bessel_y
        
        # J₀(0) = 1
        assert abs(bessel_j(0, 0) - 1.0) < 1e-10
        
        # J₁(0) = 0
        assert abs(bessel_j(1, 0)) < 1e-10
        
        # Test with complex argument
        z = bessel_j(0, 1 + 1j)
        assert isinstance(z, (complex, np.complex128))
        
        # Y functions should work too
        y = bessel_y(0, 1.0)
        assert isinstance(y, (float, np.float64))
    
    def test_airy_functions(self):
        """Test Airy functions."""
        from complexplorer.special import airy_ai, airy_bi
        
        # Ai(0) ≈ 0.35502805
        ai_0 = airy_ai(0)
        assert abs(ai_0 - 0.35502805) < 1e-6
        
        # Bi(0) ≈ 0.61492663
        bi_0 = airy_bi(0)
        assert abs(bi_0 - 0.61492663) < 1e-6
        
        # Test with complex
        z = airy_ai(1 + 1j)
        assert isinstance(z, (complex, np.complex128))
    
    def test_elliptic_integrals(self):
        """Test elliptic integrals."""
        from complexplorer.special import elliptic_k, elliptic_e
        
        # K(0) = π/2
        k_0 = elliptic_k(0)
        assert abs(k_0 - np.pi/2) < 1e-10
        
        # E(0) = π/2
        e_0 = elliptic_e(0)
        assert abs(e_0 - np.pi/2) < 1e-10
        
        # E(1) = 1
        e_1 = elliptic_e(1)
        assert abs(e_1 - 1.0) < 1e-10
    
    def test_lambert_w(self):
        """Test Lambert W function."""
        from complexplorer.special import lambert_w
        
        # W(0) = 0
        assert abs(lambert_w(0)) < 1e-10
        
        # W(e) = 1
        result = lambert_w(np.e)
        assert abs(result - 1.0) < 1e-10
        
        # Test identity: W(z) * exp(W(z)) = z
        z = 2.0
        w = lambert_w(z)
        assert abs(w * np.exp(w) - z) < 1e-10
        
        # Test with complex
        w_complex = lambert_w(1 + 1j)
        assert isinstance(w_complex, (complex, np.complex128))
    
    def test_jacobi_elliptic(self):
        """Test Jacobi elliptic functions."""
        from complexplorer.special import jacobi_elliptic
        
        # When m=0, should reduce to trig functions
        u = 1.0
        result = jacobi_elliptic(u, 0)
        # scipy.special.ellipj returns (sn, cn, dn, ph) - 4 values
        if len(result) == 4:
            sn, cn, dn, _ = result
        else:
            sn, cn, dn = result
        
        assert abs(sn - np.sin(u)) < 1e-10
        assert abs(cn - np.cos(u)) < 1e-10
        assert abs(dn - 1.0) < 1e-10
        
        # Note: scipy.special.ellipj doesn't support complex u
        # Just test with real u and various m values
        result = jacobi_elliptic(1.5, 0.5)
        if len(result) == 4:
            sn, cn, dn, _ = result
        else:
            sn, cn, dn = result
        # Just check they are reasonable values
        assert isinstance(sn, (float, np.float64))
        assert isinstance(cn, (float, np.float64))
        assert isinstance(dn, (float, np.float64))


class TestSpecialFunctionIntegration:
    """Test integration with complexplorer visualization."""
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_visualize_gamma(self):
        """Test that special functions work with complexplorer visualization."""
        import complexplorer as cp
        
        # This should work without error
        domain = cp.Rectangle(4, 4)
        cmap = cp.Phase(phase_sectors=6)
        
        # Get mesh
        z = domain.mesh(50)
        
        # Apply gamma function
        from complexplorer.special import gamma
        f_z = gamma(z)
        
        # Should be able to get colors
        rgb = cmap.rgb(f_z)
        
        assert rgb.shape == (*z.shape, 3)
        assert np.all(np.isfinite(rgb) | np.isnan(f_z[..., None]))
    
    def test_special_with_show(self):
        """Test using special functions with show() API."""
        import complexplorer as cp
        from unittest.mock import patch
        
        # Mock the actual plotting to avoid display
        with patch('complexplorer.api.plot') as mock_plot:
            if HAS_SCIPY:
                # Should work with scipy functions
                cp.show(gamma, (-2, 2), (-2, 2))
                mock_plot.assert_called_once()
            
            # Should work with erf (has fallback)
            mock_plot.reset_mock()
            cp.show(erf, (-2, 2), (-2, 2))
            mock_plot.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])