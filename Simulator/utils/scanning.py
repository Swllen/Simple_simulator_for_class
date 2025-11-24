import torch
import numpy as np
from optiland.analysis.irradiance import IncoherentIrradiance
from .computing import compute_irr_in_mask

# ==================== Signal Detection ====================

def is_signal_detected(lens, detector_surface=6, threshold=1e-6):
    """
    Check if optical signal is detected above threshold.
    
    Args:
        lens: Optical system object
        detector_surface: Surface index for detector
        threshold: Minimum irradiance value to consider as signal
        
    Returns:
        bool: True if signal detected, False otherwise
    """
    irr_map, _ = compute_irr_in_mask(lens,detector_surface=detector_surface,radius=15)
    max_irradiance = irr_map.detach().cpu().numpy().max()
    
    return max_irradiance >= threshold


# ==================== Helper Classes and Functions ====================

class ScanTrajectory:
    """Computes sinusoidal scanning trajectory."""
    
    def __init__(self, L, phase_shift_deg=0.0, device=None, dtype=None):
        """
        Args:
            L: Scan length parameter
            phase_shift_deg: Phase shift in degrees for y-axis trajectory
            device: Torch device
            dtype: Torch dtype
        """
        self.L = torch.as_tensor(L, device=device, dtype=dtype)
        self.A = self.L * torch.deg2rad(torch.tensor(2.0, device=device, dtype=dtype))
        self.phase_shift = torch.deg2rad(
            torch.tensor(phase_shift_deg, device=device, dtype=dtype)
        )
    
    def compute_y(self, theta_x):
        """
        Compute y-axis angle for given x-axis angle.
        
        Args:
            theta_x: X-axis angle in radians
            
        Returns:
            torch.Tensor: Y-axis angle in radians
        """
        return (self.A / self.L) * torch.sin(
            2 * torch.pi / self.A * 2 * self.L * theta_x + self.phase_shift
        )


class MirrorController:
    """Controls mirror pose updates."""
    
    def __init__(self, lens, surface_index):
        """
        Args:
            lens: Optical system object
            surface_index: Surface index of the mirror
        """
        self.cs = lens.surface_group.surfaces[surface_index].geometry.cs
        self.rx_origin = self.cs.rx.clone()
        self.ry_origin = self.cs.ry.clone()
        self.device = self.rx_origin.device
        self.dtype = self.rx_origin.dtype
    
    def set_angles(self, delta_rx, delta_ry):
        """Set mirror angles relative to origin."""
        with torch.no_grad():
            self.cs.rx.copy_(self.rx_origin + delta_rx)
            self.cs.ry.copy_(self.ry_origin + delta_ry)
    
    def reset(self):
        """Reset mirror to original position."""
        self.set_angles(
            torch.tensor(0.0, device=self.device, dtype=self.dtype),
            torch.tensor(0.0, device=self.device, dtype=self.dtype)
        )
    
    def get_angles_deg(self):
        """Get current angles in degrees."""
        return (
            float(torch.rad2deg(self.cs.rx)),
            float(torch.rad2deg(self.cs.ry))
        )


def _scan_direction(
    lens,
    mirror,
    trajectory,
    start_angle,
    end_angle,
    step_size,
    direction_name,
    verbose
):
    """
    Scan in one direction (forward or backward).
    
    Args:
        lens: Optical system object
        mirror: MirrorController instance
        trajectory: ScanTrajectory instance
        start_angle: Starting angle in radians
        end_angle: Ending angle in radians (exclusive bound)
        step_size: Step size in radians
        direction_name: String name for logging (e.g., "forward", "backward")
        verbose: Whether to print progress
        
    Returns:
        bool: True if signal detected, False otherwise
    """
    theta = start_angle.clone()
    step_count = 0
    step_direction = 1 if end_angle > start_angle else -1
    
    while (step_direction > 0 and theta < end_angle - 1e-12) or \
          (step_direction < 0 and theta > end_angle + 1e-12):
        
        theta = theta + step_direction * step_size
        theta_y = trajectory.compute_y(theta)
        mirror.set_angles(theta, theta_y)
        
        step_count += 1
        
        if verbose:
            rx_deg, ry_deg = mirror.get_angles_deg()
            print(f"[{direction_name}] Step {step_count}: "
                  f"theta={torch.rad2deg(theta).item():.3f}°, "
                  f"theta_y={torch.rad2deg(theta_y).item():.3f}°, "
                  f"rx={rx_deg:.3f}°, ry={ry_deg:.3f}°")
        
        if is_signal_detected(lens):
            if verbose:
                rx_deg, ry_deg = mirror.get_angles_deg()
                print(f">>> Signal detected! ({direction_name} scan)")
                print(f"    rx={float(mirror.cs.rx):.6f} rad ({rx_deg:.3f}°)")
                print(f"    ry={float(mirror.cs.ry):.6f} rad ({ry_deg:.3f}°)")
            return True
    
    return False


# ==================== Public Scanning Functions ====================

def sine_scan(
    lens,
    L=0.15,
    surface=2,
    max_angle=None,
    step_size=None,
    verbose=True
):
    """
    Perform sinusoidal scan with single mirror.
    Scans from 0 to +max_angle, then from 0 to -max_angle.
    
    Args:
        lens: Optical system object
        L: Scan length parameter
        surface: Surface index of the mirror
        max_angle: Maximum scan angle in radians (default: 4°)
        step_size: Angular step size in radians (default: 0.2°)
        verbose: Whether to print progress
        
    Returns:
        bool: True if signal detected, False otherwise
    """
    if max_angle is None:
        max_angle = torch.deg2rad(torch.tensor(4.0))
    if step_size is None:
        step_size = torch.deg2rad(torch.tensor(0.2))
    
    mirror = MirrorController(lens, surface)
    trajectory = ScanTrajectory(L, device=mirror.device, dtype=mirror.dtype)
    
    # Convert angles to correct device/dtype
    max_angle = max_angle.to(device=mirror.device, dtype=mirror.dtype)
    step_size = step_size.to(device=mirror.device, dtype=mirror.dtype)
    
    if verbose:
        print(f"[INFO] Starting sine scan: max=±{torch.rad2deg(max_angle).item():.3f}°, "
              f"step={torch.rad2deg(step_size).item():.3f}°")
        print(f"[INFO] Initial position: rx={mirror.get_angles_deg()[0]:.3f}°, "
              f"ry={mirror.get_angles_deg()[1]:.3f}°")
    
    # Forward scan: 0 → +max_angle
    if verbose:
        print("[INFO] Phase 1: Forward scan (0 → +max)")
    
    zero_angle = torch.tensor(0.0, device=mirror.device, dtype=mirror.dtype)
    
    if _scan_direction(lens, mirror, trajectory, zero_angle, max_angle, 
                      step_size, "forward", verbose):
        return True
    
    # Backward scan: 0 → -max_angle
    if verbose:
        print("[INFO] Phase 2: Backward scan (0 → -max)")
    
    mirror.reset()
    
    if _scan_direction(lens, mirror, trajectory, zero_angle, -max_angle, 
                      step_size, "backward", verbose):
        return True
    
    # No signal detected
    mirror.reset()
    if verbose:
        print("[INFO] Scan complete: No signal detected")
    
    return False


def sine_scan_dual(
    lens,
    L=0.15,
    surface1=2,
    surface2=3,
    max_angle=None,
    step_size=None,
    verbose=True,
    print_every=1,
    phase_shift_deg=90.0
):
    """
    Perform dual-mirror sinusoidal scan.
    Mirror 1 steps through its range while Mirror 2 performs full sweep at each step.
    
    Args:
        lens: Optical system object
        L: Scan length parameter
        surface1: Surface index of first mirror (outer loop)
        surface2: Surface index of second mirror (inner loop)
        max_angle: Maximum scan angle in radians (default: 4°)
        step_size: Angular step size in radians (default: 0.5°)
        verbose: Whether to print progress
        print_every: Print frequency for inner loop steps
        phase_shift_deg: Phase shift for y-axis trajectory in degrees
        
    Returns:
        bool: True if signal detected, False otherwise
    """
    if max_angle is None:
        max_angle = torch.deg2rad(torch.tensor(4.0))
    if step_size is None:
        step_size = torch.deg2rad(torch.tensor(0.5))
    
    # Initialize controllers
    mirror1 = MirrorController(lens, surface1)
    mirror2 = MirrorController(lens, surface2)
    trajectory = ScanTrajectory(L, phase_shift_deg, mirror1.device, mirror1.dtype)
    
    # Convert angles to correct device/dtype
    max_angle = max_angle.to(device=mirror1.device, dtype=mirror1.dtype)
    step_size = step_size.to(device=mirror1.device, dtype=mirror1.dtype)
    zero_angle = torch.tensor(0.0, device=mirror1.device, dtype=mirror1.dtype)
    
    # Calculate step counts for logging
    n_steps_outer = int(torch.ceil(max_angle / step_size).item())
    n_steps_inner = int(torch.floor((2 * max_angle) / step_size).item()) + 1
    
    if verbose:
        print(f"[INFO] Dual scan: max=±{torch.rad2deg(max_angle).item():.3f}°, "
              f"step={torch.rad2deg(step_size).item():.3f}°")
        print(f"[INFO] Mirror1 steps≈{n_steps_outer}, Mirror2 sweeps≈{n_steps_inner} steps each")
        print(f"[INFO] Phase shift={phase_shift_deg:.1f}°")
    
    # Scan mirror1 in both directions
    for phase_name, start_angle, end_angle in [
        ("forward", zero_angle, max_angle),
        ("backward", zero_angle, -max_angle)
    ]:
        if verbose:
            print(f"[INFO] Phase: Mirror1 {phase_name} scan")
        
        theta1 = start_angle.clone()
        step_direction = 1 if end_angle > start_angle else -1
        step1_count = 0
        
        while (step_direction > 0 and theta1 < end_angle - 1e-12) or \
              (step_direction < 0 and theta1 > end_angle + 1e-12):
            
            # Step mirror1
            theta1 = theta1 + step_direction * step_size
            theta1_y = trajectory.compute_y(theta1)
            mirror1.set_angles(theta1, theta1_y)
            
            step1_count += 1
            
            if verbose and (step1_count % print_every == 0 or step1_count == 1):
                rx1_deg, ry1_deg = mirror1.get_angles_deg()
                print(f"[M1 {phase_name[0]}] Step {step1_count}/{n_steps_outer}: "
                      f"theta1={torch.rad2deg(theta1).item():.3f}°, "
                      f"rx={rx1_deg:.3f}°, ry={ry1_deg:.3f}°")
            
            # Inner loop: sweep mirror2 fully
            if verbose:
                print(f"  [M2] Sweep: -max → +max")
            
            theta2 = -max_angle
            step2_count = 0
            
            while theta2 <= max_angle + 1e-12:
                theta2_y = trajectory.compute_y(theta2)
                mirror2.set_angles(theta2, theta2_y)
                
                step2_count += 1
                
                if verbose and (step2_count % print_every == 0 or step2_count == 1 or 
                               theta2 >= max_angle - 1e-12):
                    rx2_deg, ry2_deg = mirror2.get_angles_deg()
                    print(f"    [M2] Step {step2_count}/{n_steps_inner}: "
                          f"theta2={torch.rad2deg(theta2).item():.3f}°, "
                          f"rx={rx2_deg:.3f}°, ry={ry2_deg:.3f}°")
                
                # Check signal at each position
                if is_signal_detected(lens):
                    if verbose:
                        rx1_deg, ry1_deg = mirror1.get_angles_deg()
                        rx2_deg, ry2_deg = mirror2.get_angles_deg()
                        print(f">>> Signal detected! (Mirror1 {phase_name})")
                        print(f"    Mirror1: rx={float(mirror1.cs.rx):.6f} rad ({rx1_deg:.3f}°), "
                              f"ry={float(mirror1.cs.ry):.6f} rad ({ry1_deg:.3f}°)")
                        print(f"    Mirror2: rx={float(mirror2.cs.rx):.6f} rad ({rx2_deg:.3f}°), "
                              f"ry={float(mirror2.cs.ry):.6f} rad ({ry2_deg:.3f}°)")
                    return True
                
                theta2 = theta2 + step_size
            
            # Reset mirror2 after each sweep
            mirror2.reset()
            if verbose:
                print("  [M2] Reset complete")
    
    # No signal detected - reset both mirrors
    mirror1.reset()
    mirror2.reset()
    
    if verbose:
        print("[INFO] Scan complete: No signal detected")
    
    return False