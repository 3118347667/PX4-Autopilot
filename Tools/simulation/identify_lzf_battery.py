#!/usr/bin/env python3

"""Identify the LZF simulator battery model from a PX4 ULog."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog
from scipy.optimize import nnls


RPM_TO_RAD_PER_SECOND = 2.0 * np.pi / 60.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit RPM-dependent voltage sag and recovery parameters from a ULog."
    )
    parser.add_argument("ulog", type=Path, help="input ULog")
    parser.add_argument("--output", type=Path, help="output plot path")
    parser.add_argument("--full-voltage", type=float, default=25.2)
    parser.add_argument("--empty-voltage", type=float, default=18.0)
    parser.add_argument("--reference-rpm", type=float, default=15300.0)
    parser.add_argument("--drain-min", type=float, default=800.0)
    parser.add_argument("--drain-max", type=float, default=1400.0)
    parser.add_argument("--drain-step", type=float, default=10.0)
    parser.add_argument("--tau-min", type=float, default=0.2)
    parser.add_argument("--tau-max", type=float, default=5.0)
    parser.add_argument("--tau-step", type=float, default=0.02)
    parser.add_argument("--minimum-load", type=float, default=0.05)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def get_topic(ulog, name):
    matches = [data for data in ulog.data_list if data.name == name and data.multi_id == 0]

    if not matches:
        raise RuntimeError(f"ULog does not contain {name} instance 0")

    return matches[0].data


def model_polarization(load, armed, dt, tau):
    polarization = np.zeros_like(load)

    for index in range(1, load.size):
        target = load[index] if armed[index] else 0.0
        alpha = -np.expm1(-dt[index] / tau)
        polarization[index] = polarization[index - 1] + alpha * (
            target - polarization[index - 1]
        )

    return polarization


def load_ulog(path):
    ulog = ULog(
        str(path),
        message_name_filter_list=["esc_status", "battery_status", "actuator_armed"],
    )
    esc = get_topic(ulog, "esc_status")
    battery = get_topic(ulog, "battery_status")
    actuator_armed = get_topic(ulog, "actuator_armed")

    esc_time = np.asarray(esc["timestamp"], dtype=float) * 1e-6
    battery_time = np.asarray(battery["timestamp"], dtype=float) * 1e-6
    armed_time = np.asarray(actuator_armed["timestamp"], dtype=float) * 1e-6
    rpm = np.column_stack(
        [np.asarray(esc[f"esc[{index}].esc_rpm"], dtype=float) for index in range(4)]
    )

    omega_normalized = np.maximum(rpm, 0.0) * RPM_TO_RAD_PER_SECOND / 1000.0
    esc_load = np.mean(omega_normalized**3, axis=1)
    load = np.interp(battery_time, esc_time, esc_load)
    armed = np.interp(
        battery_time,
        armed_time,
        np.asarray(actuator_armed["armed"], dtype=float),
    ) > 0.5
    voltage = np.asarray(battery["voltage_v"], dtype=float)
    dt = np.diff(battery_time, prepend=battery_time[0])

    if np.any(dt < 0.0):
        raise RuntimeError("battery_status timestamps are not monotonic")

    return battery_time, voltage, load, armed, dt, esc_time


def identify(args):
    battery_time, voltage, load, armed, dt, esc_time = load_ulog(args.ulog)
    reference_omega = args.reference_rpm * RPM_TO_RAD_PER_SECOND
    reference_load = (reference_omega / 1000.0) ** 3
    normalized_energy = np.cumsum(np.where(armed, load, 0.0) * dt) / reference_load
    fit_mask = armed & np.isfinite(voltage) & np.isfinite(load) & (load >= args.minimum_load)

    if np.count_nonzero(fit_mask) < 3:
        raise RuntimeError("not enough armed battery samples with valid RPM feedback")

    drain_values = np.arange(
        args.drain_min, args.drain_max + 0.5 * args.drain_step, args.drain_step
    )
    tau_values = np.arange(args.tau_min, args.tau_max + 0.5 * args.tau_step, args.tau_step)
    best = None

    for tau in tau_values:
        polarization = model_polarization(load, armed, dt, tau)
        design_matrix = np.column_stack((load[fit_mask], polarization[fit_mask]))

        for drain in drain_values:
            soc = np.clip(1.0 - normalized_energy / drain, 0.0, 1.0)
            open_circuit_voltage = args.empty_voltage + (
                args.full_voltage - args.empty_voltage
            ) * soc
            coefficients, _ = nnls(
                design_matrix, open_circuit_voltage[fit_mask] - voltage[fit_mask]
            )
            prediction = (
                open_circuit_voltage
                - coefficients[0] * load
                - coefficients[1] * polarization
            )
            residual = voltage[fit_mask] - prediction[fit_mask]
            rmse = np.sqrt(np.mean(residual**2))

            if best is None or rmse < best["rmse"]:
                total_variance = np.sum(
                    (voltage[fit_mask] - np.mean(voltage[fit_mask])) ** 2
                )
                best = {
                    "drain": drain,
                    "tau": tau,
                    "sag_i": coefficients[0],
                    "sag_p": coefficients[1],
                    "rmse": rmse,
                    "r_squared": 1.0 - np.sum(residual**2) / total_variance,
                    "soc": soc,
                    "open_circuit_voltage": open_circuit_voltage,
                    "prediction": prediction,
                    "polarization": polarization,
                }

    battery_rate = (battery_time.size - 1) / (battery_time[-1] - battery_time[0])
    esc_rate = (esc_time.size - 1) / (esc_time[-1] - esc_time[0])
    print(f"battery samples: {battery_time.size} ({battery_rate:.2f} Hz)")
    print(f"ESC samples:     {esc_time.size} ({esc_rate:.2f} Hz)")
    print(f"fit samples:     {np.count_nonzero(fit_mask)}")
    print(f"SIM_BAT_L_REF:   {reference_load:.6f}")
    print(f"SIM_BAT_DRAIN:   {best['drain']:.1f} s")
    print(f"SIM_BAT_TAU:     {best['tau']:.3f} s")
    print(f"SIM_BAT_SAG_I:   {best['sag_i']:.6f} V/load")
    print(f"SIM_BAT_SAG_P:   {best['sag_p']:.6f} V/load")
    print(f"RMSE:            {best['rmse']:.4f} V")
    print(f"R^2:             {best['r_squared']:.5f}")

    if not args.no_plot:
        output = args.output or args.ulog.with_name(f"{args.ulog.stem}_battery_fit.png")
        elapsed = battery_time - battery_time[0]
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(elapsed, voltage, label="measured", linewidth=1.0)
        axes[0].plot(elapsed, best["prediction"], label="model", linewidth=1.0)
        axes[0].plot(
            elapsed,
            best["open_circuit_voltage"],
            label="open circuit",
            linewidth=0.8,
        )
        axes[0].set_ylabel("Voltage (V)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].plot(elapsed, load, label="instantaneous load", linewidth=0.8)
        axes[1].plot(
            elapsed,
            best["polarization"],
            label="polarization state",
            linewidth=0.8,
        )
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Load")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        figure.suptitle(
            f"LZF battery fit: RMSE {best['rmse']:.3f} V, R^2 {best['r_squared']:.3f}"
        )
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        print(f"plot:            {output}")

    return best


def main():
    args = parse_args()

    try:
        identify(args)
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
