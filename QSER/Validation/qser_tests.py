"""
QSER Validation Tests
======================

Four system-independent validation tests for the QSER decomposition:

1. Energy conservation in S
2. Zero initial condition of E
3. Exact reconstruction (R = S - E)
4. Infinite memory of S (tau_u scaling with window size)

Usage:
    from QSER.Validation import validate_qser
    
    results = validate_qser(S, E, R, dt=0.1)
    print(results['all_passed'])
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, Optional, Tuple, List


def validate_qser(
    S: np.ndarray,
    E: np.ndarray,
    R: np.ndarray,
    dt: float = 1.0,
    time_axis: int = 0,
    tol_energy: float = 1e-3,
    tol_initial: float = 1e-6,
    tol_reconstruction: float = 1e-3,
    tol_memory: float = 0.1,
    n_windows: int = 3,
    volume: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Union[bool, float, str, List[float]]]:
    """
    Run all four QSER validation tests.
    
    Args:
        S: Source field (time, ...)
        E: Environment field (time, ...)
        R: Response field (time, ...)
        dt: Time step between samples
        time_axis: Axis along which time varies
        tol_energy: Tolerance for energy conservation test (Test 1)
        tol_initial: Tolerance for zero initial condition test (Test 2)
        tol_reconstruction: Tolerance for reconstruction test (Test 3)
        tol_memory: Relative tolerance for memory scaling (Test 4)
        n_windows: Number of window sizes for Test 4
        volume: Volume element for integration
        verbose: Print results and show histogram
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # ============================================================
    # Test 1: Energy conservation in S
    # ============================================================
    test1_result, test1_error, test1_energy = _test_energy_conservation(
        S, dt, time_axis, volume, tol=tol_energy
    )
    results['test1_energy_conservation'] = test1_result
    results['test1_error'] = test1_error
    results['test1_energy'] = test1_energy
    
    # ============================================================
    # Test 2: Zero initial condition of E
    # ============================================================
    test2_result, test2_error = _test_zero_initial_E(
        E, time_axis, tol=tol_initial
    )
    results['test2_zero_initial_E'] = test2_result
    results['test2_error'] = test2_error
    
    # ============================================================
    # Test 3: Exact reconstruction (R = S - E)
    # ============================================================
    test3_result, test3_error = _test_reconstruction(
        S, E, R, tol=tol_reconstruction
    )
    results['test3_reconstruction'] = test3_result
    results['test3_error'] = test3_error
    
    # ============================================================
    # Test 4: Infinite memory of S
    # ============================================================
    test4_result, test4_ratios, test4_mean, test4_max_dev = _test_infinite_memory(
        S, dt, time_axis, n_windows=n_windows, tol=tol_memory
    )
    results['test4_infinite_memory'] = test4_result
    results['test4_ratios'] = test4_ratios
    results['test4_mean_ratio'] = test4_mean
    results['test4_max_deviation'] = test4_max_dev
    
    # ============================================================
    # Summary
    # ============================================================
    all_passed = all([
        test1_result,
        test2_result,
        test3_result,
        test4_result
    ])
    results['all_passed'] = all_passed
    
    # ============================================================
    # Verbose output
    # ============================================================
    if verbose:
        _print_results(results, n_windows)
        
        # Show histogram if n_windows > 10
        if n_windows > 10:
            _plot_histogram(results['test4_ratios'], results['test4_mean_ratio'])
    
    return results


# ============================================================
# Test 1: Energy conservation in S
# ============================================================

def _test_energy_conservation(
    S: np.ndarray,
    dt: float = 1.0,
    time_axis: int = 0,
    volume: float = 1.0,
    tol: float = 1e-3
) -> Tuple[bool, float, np.ndarray]:
    """
    Test 1: Energy of S must be conserved.
    
    ε_S = 0.5 * ∫ S² dV
    dε_S/dt = 0
    
    Pass if: std(ε_S) < tol
    """
    # Compute energy over time
    spatial_axes = tuple(i for i in range(S.ndim) if i != time_axis)
    energy_S = 0.5 * np.sum(S**2, axis=spatial_axes) * volume
    
    # Check standard deviation
    std_energy = np.std(energy_S)
    passed = std_energy < tol
    
    return passed, std_energy, energy_S


# ============================================================
# Test 2: Zero initial condition of E
# ============================================================

def _test_zero_initial_E(
    E: np.ndarray,
    time_axis: int = 0,
    tol: float = 1e-6
) -> Tuple[bool, float]:
    """
    Test 2: Environment must start from zero.
    
    Pass if: |E(0)| < tol
    """
    # Get initial slice along time axis
    indices = [0] * E.ndim
    indices[time_axis] = 0
    E0 = E[tuple(indices)]
    
    max_E0 = np.max(np.abs(E0))
    passed = max_E0 < tol
    
    return passed, max_E0


# ============================================================
# Test 3: Exact reconstruction
# ============================================================

def _test_reconstruction(
    S: np.ndarray,
    E: np.ndarray,
    R: np.ndarray,
    tol: float = 1e-3
) -> Tuple[bool, float]:
    """
    Test 3: R = S - E must hold.
    
    Pass if: max|R - (S - E)| < tol
    """
    error = np.max(np.abs(R - (S - E)))
    passed = error < tol
    
    return passed, error


# ============================================================
# Test 4: Infinite memory of S
# ============================================================

def _test_infinite_memory(
    S: np.ndarray,
    dt: float = 1.0,
    time_axis: int = 0,
    n_windows: int = 3,
    tol: float = 0.1
) -> Tuple[bool, List[float], float, float]:
    """
    Test 4: Infinite memory of S.
    
    For a conservative system, tau_u^S / T is constant across window sizes.
    
    τ_u^S = ∫ t|dS/dt|dt / ∫ |dS/dt|dt
    
    Pass if: max deviation of τ_u/T < tol
    
    Args:
        S: Source field (time, ...)
        dt: Time step
        time_axis: Axis along which time varies
        n_windows: Number of window sizes to test
        tol: Relative tolerance for constant ratio
        
    Returns:
        (passed, ratios, mean_ratio, max_deviation)
    """
    # Import TimeGradient (lazy import to avoid circular dependency)
    try:
        from QSER.Operators import TimeGradient
        time_grad = TimeGradient(backend='numpy')
    except ImportError:
        time_grad = None
    
    T_total = S.shape[time_axis] * dt
    ratios = []
    
    for i in range(2, n_windows + 2):
        fraction = i / (n_windows + 1)
        n_points = int(S.shape[time_axis] * fraction)
        
        # Slice S along time axis
        indices = [slice(None)] * S.ndim
        indices[time_axis] = slice(0, n_points)
        S_window = S[tuple(indices)]
        
        # Compute time derivative
        if time_grad is not None:
            dS = time_grad.compute(S_window, dt=dt, axis=time_axis)
        else:
            # Manual derivative
            dS = np.diff(S_window, axis=time_axis, prepend=S_window.take([0], axis=time_axis)) / dt
        
        # Absolute derivative
        abs_dS = np.abs(dS)
        
        # Sum over spatial dimensions
        spatial_axes = tuple(j for j in range(S.ndim) if j != time_axis)
        if spatial_axes:
            abs_dS_sum = np.sum(abs_dS, axis=spatial_axes)
        else:
            abs_dS_sum = abs_dS
        
        # Compute centroid
        t = np.arange(len(abs_dS_sum)) * dt
        tau_u = np.sum(t * abs_dS_sum) / (np.sum(abs_dS_sum) + 1e-15)
        
        T_window = n_points * dt
        ratios.append(tau_u / T_window)
    
    # Check if ratios are constant
    ratios = np.array(ratios)
    mean_ratio = np.mean(ratios)
    max_deviation = np.max(np.abs(ratios - mean_ratio) / (mean_ratio + 1e-15))
    
    passed = max_deviation < tol
    
    return passed, ratios.tolist(), float(mean_ratio), float(max_deviation)


# ============================================================
# Verbose output
# ============================================================

def _print_results(results: Dict, n_windows: int) -> None:
    """Print validation results in a formatted table."""
    print("\n" + "=" * 60)
    print("QSER VALIDATION RESULTS")
    print("=" * 60)
    
    # Test 1
    status = "PASS" if results['test1_energy_conservation'] else "FAIL"
    print(f"Test 1 (Energy conservation in S): {status}")
    print(f"  Error: {results['test1_error']:.6e} (tol: 1e-3)")
    
    # Test 2
    status = "PASS" if results['test2_zero_initial_E'] else "FAIL"
    print(f"Test 2 (Zero initial E): {status}")
    print(f"  Error: {results['test2_error']:.6e} (tol: 1e-6)")
    
    # Test 3
    status = "PASS" if results['test3_reconstruction'] else "FAIL"
    print(f"Test 3 (Reconstruction): {status}")
    print(f"  Error: {results['test3_error']:.6e} (tol: 1e-3)")
    
    # Test 4
    status = "PASS" if results['test4_infinite_memory'] else "FAIL"
    ratios = results['test4_ratios']
    mean_ratio = results['test4_mean_ratio']
    max_dev = results['test4_max_deviation']
    
    # Show first few ratios
    if len(ratios) > 10:
        ratios_str = f"[{', '.join([f'{r:.3f}' for r in ratios[:5]])}, ...]"
    else:
        ratios_str = f"[{', '.join([f'{r:.3f}' for r in ratios])}]"
    
    print(f"Test 4 (Infinite memory of S): {status}")
    print(f"  Ratios: {ratios_str}")
    print(f"  Mean: {mean_ratio:.4f}, Max deviation: {max_dev*100:.2f}% (tol: {0.1*100:.0f}%)")
    
    # Summary
    print("-" * 60)
    all_passed = results['all_passed']
    print(f"ALL TESTS PASSED: {'YES' if all_passed else 'NO'}")
    print("=" * 60)


def _plot_histogram(ratios: List[float], mean_ratio: float) -> None:
    """Plot histogram of tau_u/T ratios."""
    ratios = np.array(ratios)
    
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # Histogram
    n_bins = min(len(ratios) // 2, 20)
    ax.hist(ratios, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Mean line
    ax.axvline(x=mean_ratio, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_ratio:.4f}')
    
    # Labels
    ax.set_xlabel('tau_u / T', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of tau_u/T ratios (n={len(ratios)})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
