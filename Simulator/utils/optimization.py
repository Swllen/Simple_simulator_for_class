import torch.optim as optim
from .computing import spot_centroid_from_irr
from .scanning import sine_scan, is_signal_detected,sine_scan_dual
from .loss import (
    mse_loss_in_mask,
    compute_loss_center_power,
    compute_loss_center_irr,
    compute_loss_center_mask_power
)
from .params_utils import set_trainable, snapshot_params, load_snapshot


def run_irradiance(
    lens,
    params,
    ideal_map1,
    ideal_map2,
    lr,
    steps,
    weight_decay=0.0,
    debug=False
):
    """
    Optimize parameters to minimize irradiance error between detectors maps.
    """
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_log, best_loss, no_improved_step, count = [], float('inf'), 0, 0

    for k in range(steps):
        optimizer.zero_grad(set_to_none=True)

        # Make selected parameters trainable
        set_trainable([params[0], params[1], params[2], params[3]], params)

        # Compute loss at different detector surfaces
        loss2 = mse_loss_in_mask(lens, ideal_map2, detector_surface=-1)
        loss1 = mse_loss_in_mask(lens, ideal_map1, detector_surface=7)
        loss = 0.5 * loss1 + 0.5 * loss2

        # Backpropagation
        loss.backward()
        optimizer.step()
        count += 1

        # Logging every 10 iterations
        if count % 10 == 0:
            print("----------------------------")
            print(f"Iter {k+1}, Loss: {loss.item():.6f}")
            print(f"rx1: {params[0].item()}, ry1: {params[1].item()}")
            print(f"rx2: {params[2].item()}, ry2: {params[3].item()}")
            if debug:
                print(f"Gradients rx1: {params[0].grad}, ry1: {params[1].grad}")
                print(f"Gradients rx2: {params[2].grad}, ry2: {params[3].grad}")

        loss_log.append(loss.item())

        # Early stopping check
        monitor_val = loss.item()
        if monitor_val < (best_loss - 1e-5) or monitor_val > best_loss:
            best_loss, no_improved_step = monitor_val, 0
        else:
            no_improved_step += 1

        if no_improved_step > 10:
            break

    return loss_log


def run_irradiance_center_irr(
    optical_sys,
    params,
    ideal_map1,
    ideal_map2,
    x,
    y,
    detector_surface1=6,
    detector_surface2=-1,
    radius=1,
    res=(256, 256),
    num_rays=100000,
    lr=0.001,
    steps=100,
    weight_decay=0.0,
    debug=False
):
    """
    Optimize parameters using irradiance centered on the spot centroid.
    """
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_log, best_loss, no_improved_step, count = [], float('inf'), 0, 0

    # Compute reference centroid
    xc, yc, _ = spot_centroid_from_irr(ideal_map1, x, y)

    for k in range(steps):
        optimizer.zero_grad(set_to_none=True)
        set_trainable([params[0], params[1], params[2], params[3]], params)

        # Loss computed with respect to centroid position
        loss2 = compute_loss_center_irr(
            optical_sys, ideal_map2, xc, yc,
            detector_surface=detector_surface2,
            radius=radius, res=res, num_rays=num_rays
        )
        loss1 = compute_loss_center_irr(
            optical_sys, ideal_map1, xc, yc,
            detector_surface=detector_surface1,
            radius=radius, res=res, num_rays=num_rays
        )
        loss = 0.5 * loss1 + 0.5 * loss2

        loss.backward()
        optimizer.step()
        count += 1

        # Logging each iteration
        print("----------------------------")
        print(f"Iter {k+1}, Loss: {loss.item():.6f}")
        print(f"rx1: {params[0].item()}, ry1: {params[1].item()}")
        print(f"rx2: {params[2].item()}, ry2: {params[3].item()}")
        if debug:
            print(f"Gradients rx1: {params[0].grad}, ry1: {params[1].grad}")
            print(f"Gradients rx2: {params[2].grad}, ry2: {params[3].grad}")

        loss_log.append(loss.item())

        # Early stopping
        monitor_val = loss.item()
        if monitor_val < (best_loss - 1e-5) or monitor_val > best_loss:
            best_loss, no_improved_step = monitor_val, 0
        else:
            no_improved_step += 1

        if no_improved_step > 10:
            break

    return loss_log




def run_irradiance_center_power(
    optical_sys,
    params,
    ideal_map1,
    ideal_p1,
    ideal_p2,
    x,
    y,
    detector_surface1=6,
    detector_surface2=-1,
    radius=1,
    res=(256, 256),
    num_rays=100000,
    lr=0.001,
    steps=100,
    weight_decay=0.0,
    debug=False
):
    """
    Optimize parameters by minimizing power loss around centroid position.
    """
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_log, best_loss, no_improved_step, count = [], float('inf'), 0, 0

    # Compute reference centroid
    xc, yc, _ = spot_centroid_from_irr(ideal_map1, x, y)

    for k in range(steps):
        optimizer.zero_grad(set_to_none=True)
        set_trainable([params[0], params[1], params[2], params[3]], params)

        # Compute power-based loss
        loss2 = compute_loss_center_power(
            optical_sys, ideal_p2, xc, yc,
            detector_surface=detector_surface2,
            radius=radius, res=res, num_rays=num_rays
        )
        loss1 = compute_loss_center_power(
            optical_sys, ideal_p1, xc, yc,
            detector_surface=detector_surface1,
            radius=radius, res=res, num_rays=num_rays
        )
        loss = 0.5 * loss1 + 0.5 * loss2

        loss.backward()
        optimizer.step()
        count += 1

        # Logging each iteration
        print("----------------------------")
        print(f"Iter {k+1}, Loss: {loss.item():.6f}")
        print(f"rx1: {params[0].item()}, ry1: {params[1].item()}")
        print(f"rx2: {params[2].item()}, ry2: {params[3].item()}")
        if debug:
            print(f"Gradients rx1: {params[0].grad}, ry1: {params[1].grad}")
            print(f"Gradients rx2: {params[2].grad}, ry2: {params[3].grad}")

        loss_log.append(loss.item())

        # Early stopping
        monitor_val = loss.item()
        if monitor_val < (best_loss - 1e-5) or monitor_val > best_loss:
            best_loss, no_improved_step = monitor_val, 0
        else:
            no_improved_step += 1

        if no_improved_step > 10:
            break

    return loss_log


def run_irradiance_center_d1mask_power(
    optical_sys,
    params,
    ideal_map1,
    ideal_p1,
    ideal_p2,
    x,
    y,
    detector_surface1=6,
    detector_surface2=-1,
    physical_radius=5.0,
    radius=1,
    res=(256, 256),
    num_rays=100000,
    lr=0.001,
    steps=100,
    weight_decay=0.0,
    debug=False
):
    """
    Optimize parameters by minimizing power loss around centroid position.
    """
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_log, best_loss, no_improved_step, count = [], float('inf'), 0, 0

    # Compute reference centroid
    xc, yc, _ = spot_centroid_from_irr(ideal_map1, x, y)

    for k in range(steps):
        optimizer.zero_grad(set_to_none=True)
        set_trainable([params[0], params[1], params[2], params[3]], params)

        # Compute power-based loss
        loss2 = compute_loss_center_power(
            optical_sys, ideal_p2, xc, yc,
            detector_surface=detector_surface2,
            radius=radius, res=res, num_rays=num_rays
        )
        loss1 = compute_loss_center_mask_power(
            optical_sys, ideal_p1, xc, yc,
            detector_surface=detector_surface1,
            radius=radius,mask_radius=physical_radius, res=res, num_rays=num_rays
        )
        loss = 0.5 * loss1 + 0.1 * loss2

        loss.backward()
        optimizer.step()
        count += 1

        # Logging each iteration
        print("----------------------------")
        print(f"Iter {k+1}, Loss: {loss.item():.6f}")
        print(f"rx1: {params[0].item()}, ry1: {params[1].item()}")
        print(f"rx2: {params[2].item()}, ry2: {params[3].item()}")
        if debug:
            print(f"Gradients rx1: {params[0].grad}, ry1: {params[1].grad}")
            print(f"Gradients rx2: {params[2].grad}, ry2: {params[3].grad}")

        loss_log.append(loss.item())

        # Early stopping
        monitor_val = loss.item()
        if monitor_val < (best_loss - 1e-5) or monitor_val > best_loss:
            best_loss, no_improved_step = monitor_val, 0
        else:
            no_improved_step += 1

        if no_improved_step > 10:
            break

    return loss_log


def run_scan_gradient(
    lens,
    params,
    ideal_map1,
    ideal_p1,
    ideal_p2,
    x,
    y,
    detector_surface1=6,
    detector_surface2=-1,
    radius=1,
    res=(256, 256),
    num_rays=1000,
    lr=0.001,
    steps=100,
    weight_decay=0.0,
    L=0.15,
    debug=False
):
    """
    Perform a gradient-based scan to optimize parameters.
    Ensures signal detection before optimization starts.
    """
    # Ensure signal detected before starting optimization
    is_detected = is_signal_detected(lens,detector_surface=detector_surface2)
    while not is_detected:
        is_detected = sine_scan(lens,L=L)

    print(params)

    # Use irradiance + power optimization once detection confirmed
    loss_log = run_irradiance_center_d1mask_power(
        lens, params,
        ideal_map1, ideal_p1, ideal_p2,
        x, y,
        detector_surface1=detector_surface1,
        detector_surface2=detector_surface2,
        radius=radius, res=res, num_rays=num_rays,
        lr=lr, steps=steps, weight_decay=weight_decay, debug=True
    )
    return loss_log




def run_scan_gradient_physical_mask(
    lens,
    params,
    ideal_map1,
    ideal_p1,
    ideal_p2,
    x,
    y,
    detector_surface1=6,
    detector_surface2=-1,
    physical_radius = 5.0,
    radius=1,
    res=(256, 256),
    num_rays=1000,
    lr=0.001,
    steps=100,
    weight_decay=0.0,
    L=0.15,
    debug=False
):
    """
    Perform a gradient-based scan to optimize parameters.
    Ensures signal detection before optimization starts.
    """
    # Ensure signal detected before starting optimization
    is_detected = is_signal_detected(lens,detector_surface=detector_surface2)
    while not is_detected:
        is_detected = sine_scan(lens,L=L)

    print(params)

    # Use irradiance + power optimization once detection confirmed
    loss_log = run_irradiance_center_d1mask_power(
        lens, params,
        ideal_map1, ideal_p1, ideal_p2,
        x, y,
        detector_surface1=detector_surface1,
        detector_surface2=detector_surface2,
        physical_radius=physical_radius,
        radius=radius, res=res, num_rays=num_rays,
        lr=lr, steps=steps, weight_decay=weight_decay, debug=True
    )
    return loss_log



def run_irradiance_center_d1mask_power_log(
    optical_sys,
    params,
    ideal_map1,
    ideal_p1,
    ideal_p2,
    x,
    y,
    detector_surface1=6,
    detector_surface2=-1,
    physical_radius=5.0,
    radius=1,
    res=(256, 256),
    num_rays=100000,
    lr=0.001,
    steps=100,
    weight_decay=0.0,
    debug=False
):
    """
    Optimize parameters by minimizing power loss around centroid position.
    """
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_log, best_loss, no_improved_step, count = [], float('inf'), 0, 0
    params_log = []

    xc, yc, _ = spot_centroid_from_irr(ideal_map1, x, y)

    for k in range(steps):
        optimizer.zero_grad(set_to_none=True)
        set_trainable([params[0], params[1], params[2], params[3]], params)

        loss2 = compute_loss_center_power(
            optical_sys, ideal_p2, xc, yc,
            detector_surface=detector_surface2,
            radius=radius, res=res, num_rays=num_rays
        )
        loss1 = compute_loss_center_mask_power(
            optical_sys, ideal_p1, xc, yc,
            detector_surface=detector_surface1,
            radius=radius, mask_radius=physical_radius, res=res, num_rays=num_rays
        )
        loss = 0.5 * loss1 + 0.5 * loss2

        loss.backward()
        optimizer.step()
        count += 1

        # Record current parameters
        params_log.append({
            'iteration': k + 1,
            'rx1': params[0].item(),
            'ry1': params[1].item(),
            'rx2': params[2].item(),
            'ry2': params[3].item(),
            'loss': loss.item()
        })

        print("----------------------------")
        print(f"Iter {k+1}, Loss: {loss.item():.6f}")
        print(f"rx1: {params[0].item()}, ry1: {params[1].item()}")
        print(f"rx2: {params[2].item()}, ry2: {params[3].item()}")
        if debug:
            print(f"Gradients rx1: {params[0].grad}, ry1: {params[1].grad}")
            print(f"Gradients rx2: {params[2].grad}, ry2: {params[3].grad}")

        loss_log.append(loss.item())

        monitor_val = loss.item()
        if monitor_val < (best_loss - 1e-5) or monitor_val > best_loss:
            best_loss, no_improved_step = monitor_val, 0
        else:
            no_improved_step += 1

        if no_improved_step > 10:
            break

    return loss_log, params_log 

def run_scan_gradient_physical_mask_log(
    lens,
    params,
    ideal_map1,
    ideal_p1,
    ideal_p2,
    x,
    y,
    detector_surface1=6,
    detector_surface2=-1,
    physical_radius = 5.0,
    radius=1,
    res=(256, 256),
    num_rays=1000,
    lr=0.001,
    steps=100,
    weight_decay=0.0,
    L=0.15,
    debug=False
):
    """
    Perform a gradient-based scan to optimize parameters.
    Ensures signal detection before optimization starts.
    """
    # Ensure signal detected before starting optimization
    is_detected = is_signal_detected(lens,detector_surface=detector_surface2)
    while not is_detected:
        is_detected = sine_scan(lens,L=L)

    print(params)

    # Use irradiance + power optimization once detection confirmed
    loss_log,params_log = run_irradiance_center_d1mask_power_log(
        lens, params,
        ideal_map1, ideal_p1, ideal_p2,
        x, y,
        detector_surface1=detector_surface1,
        detector_surface2=detector_surface2,
        physical_radius=physical_radius,
        radius=radius, res=res, num_rays=num_rays,
        lr=lr, steps=steps, weight_decay=weight_decay, debug=True
    )
    return loss_log,params_log