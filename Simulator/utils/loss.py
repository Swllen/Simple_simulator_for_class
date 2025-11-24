from optiland.analysis.irradiance import IncoherentIrradiance
from .computing import (
    compute_irradiance,
    compute_irr_in_mask, 
    compute_power, 
    compute_centroid
)
import torch

# ==================== Individual Loss Functions ====================

def mse_loss_in_mask(
    optical_sys,
    ideal_map,
    detector_surface=-1,
    radius=1,
    res=(256, 256),
    num_rays=100000
):
    """
    Compute the normalized mean squared error (MSE) of the irradiance
    map inside a circular mask compared to the ideal map.
    
    Args:
        optical_sys: Optical system object
        ideal_map: Target irradiance distribution
        detector_surface: Surface index for detector
        radius: Mask radius
        res: Resolution tuple (height, width)
        num_rays: Number of rays to trace
        
    Returns:
        float: Normalized MSE loss
    """
    irr_map = compute_irr_in_mask(
        optical_sys,
        detector_surface=detector_surface,
        radius=radius,
        res=res,
        num_rays=num_rays
    )

    squared_error = (irr_map - ideal_map) ** 2
    loss = squared_error.sum()  / squared_error.max()

    print(f"------ Detector {detector_surface}, MSE Loss: {loss} ------")
    return loss
 
def power_loss(
    optical_sys,
    p_ideal,
    detector_surface=-1,
    radius=1,
    res=(256, 256),
    num_rays=100000
):
    """
    Compute normalized power loss between current and ideal total power.
    
    Args:
        optical_sys: Optical system object
        p_ideal: Ideal target power
        detector_surface: Surface index for detector
        radius: Integration radius
        res: Resolution tuple (height, width)
        num_rays: Number of rays to trace
        
    Returns:
        float: Relative power difference
    """
    p_now = compute_power(
        optical_sys,
        detector_surface=detector_surface,
        radius=radius,
        res=res,
        num_rays=num_rays
    )

    loss = (p_ideal - p_now) / p_ideal
    return loss

def position_loss(
    optical_sys,
    detector_surface=None,
    ideal_x=0.0,
    ideal_y=0.0,
    res=(256, 256),
    num_rays=100000,
    consistance=False,
):
    """
    Compute the squared error between the centroid of irradiance
    distribution and an ideal (x, y) target position.
    
    Args:
        optical_sys: Optical system object
        detector_surface: Surface index for detector
        ideal_x: Target x coordinate
        ideal_y: Target y coordinate
        res: Resolution tuple (height, width)
        num_rays: Number of rays to trace
        
    Returns:
        float: Squared positional error
    """
    if detector_surface is not None:
        irr_map, x_edges, y_edges = compute_irradiance(
            optical_sys, detector_surface, res, num_rays
        )
        xc_, yc_, m  = compute_centroid(irr_map, x_edges, y_edges)
        
        loss = (ideal_x - xc_) ** 2 + (ideal_y - yc_) ** 2

        if consistance:
            if m < 1e-6:
                loss = 1.0
            else:
                loss = loss / (x_edges[-1] ** 2 + y_edges[-1] ** 2)
        return loss
    else:
        return 0

def position_loss_mask(
    optical_sys,
    detector_surface=-1,
    radius=3.0,
    ideal_x=0.0,
    ideal_y=0.0,
    res=(256, 256),
    num_rays=10000,
):
    """
    Compute the squared error between the centroid of irradiance
    distribution and an ideal (x, y) target position with circular masking.
    
    Args:
        optical_sys: Optical system object
        detector_surface: Surface index for detector
        radius: Mask radius for centroid calculation
        ideal_x: Target x coordinate
        ideal_y: Target y coordinate
        res: Resolution tuple (height, width)
        num_rays: Number of rays to trace
        
    Returns:
        float: Squared positional error
    """
    irr_map, x_edges, y_edges = compute_irradiance(optical_sys, detector_surface, res, num_rays)
    
    xc, yc, _ = compute_centroid(irr_map, x_edges, y_edges, mask_radius=radius)
    
    loss = (ideal_x - xc) ** 2 + (ideal_y - yc) ** 2
    return loss

# ==================== Combined Loss Functions ====================

def compute_loss_center_irr(
    optical_sys,
    ideal_map,
    ideal_x=0.0,
    ideal_y=0.0,
    detector_surface=-1,
    radius=1,
    res=(256, 256),
    num_rays=100000
):
    """
    Combined loss: weighted irradiance MSE inside mask + centroid position error.
    
    Args:
        optical_sys: Optical system object
        ideal_map: Target irradiance distribution
        ideal_x: Target x coordinate for centroid
        ideal_y: Target y coordinate for centroid
        detector_surface: Surface index for detector
        radius: Mask radius
        res: Resolution tuple (height, width)
        num_rays: Number of rays to trace
        
    Returns:
        float: Combined loss value
    """
    mse = mse_loss_in_mask(
        optical_sys, ideal_map, detector_surface, radius, res, num_rays
    )

    center = position_loss(
        optical_sys, detector_surface, ideal_x, ideal_y, res, num_rays
    )

    loss = mse + center
    return loss

def compute_loss_center_power(
    optical_sys,
    ideal_p,
    ideal_x=0.0,
    ideal_y=0.0,
    detector_surface=-1,
    radius=1,
    res=(256, 256),
    num_rays=100000
):
    """
    Combined loss: normalized power error + centroid position error.
    
    Args:
        optical_sys: Optical system object
        ideal_p: Ideal target power
        ideal_x: Target x coordinate for centroid
        ideal_y: Target y coordinate for centroid
        detector_surface: Surface index for detector
        radius: Integration radius
        res: Resolution tuple (height, width)
        num_rays: Number of rays to trace
        
    Returns:
        float: Combined loss value
    """
    p_loss = power_loss(
        optical_sys, ideal_p, detector_surface, radius, res, num_rays
    )

    center_loss = position_loss(
        optical_sys, detector_surface, ideal_x, ideal_y, res, num_rays
    )

    print(f"***** Detector {detector_surface}, Center Loss: {center_loss} *****")
    print(f"***** Detector {detector_surface}, Power Loss: {p_loss} *****")

    loss = p_loss + center_loss
    return loss

def compute_loss_center_mask_power(
    optical_sys,
    ideal_p,
    ideal_x=0.0,
    ideal_y=0.0,
    detector_surface=-1,
    radius=1,
    mask_radius=5.0,
    res=(256, 256),
    num_rays=100000
):
    """
    Combined loss: normalized power error + centroid position error.
    
    Args:
        optical_sys: Optical system object
        ideal_p: Ideal target power
        ideal_x: Target x coordinate for centroid
        ideal_y: Target y coordinate for centroid
        detector_surface: Surface index for detector
        radius: Integration radius
        res: Resolution tuple (height, width)
        num_rays: Number of rays to trace
        
    Returns:
        float: Combined loss value
    """
    p_loss = power_loss(
        optical_sys, ideal_p, detector_surface, radius, res, num_rays
    )

    center_loss = position_loss(
        optical_sys, detector_surface, ideal_x=ideal_x,ideal_y=ideal_y,num_rays=num_rays
    )

    print(f"***** Detector {detector_surface}, Center Loss: {center_loss} *****")
    print(f"***** Detector {detector_surface}, Power Loss: {p_loss} *****")

    loss = p_loss + center_loss
    return loss