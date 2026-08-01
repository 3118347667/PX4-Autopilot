#!/usr/bin/env python3

"""Analyze an enabled/disabled pair from run_lzf_attitude_sweep.py."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog
from scipy.spatial.transform import Rotation


ARMED_STATE = 2
OFFBOARD_NAV_STATE = 14
AXES = ("roll", "pitch")
FREQUENCIES_HZ = (0.75, 1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 4.0, 4.75)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled-dir", type=Path, required=True)
    parser.add_argument("--disabled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def topic(ulog, name):
    matches = [
        item.data for item in ulog.data_list
        if item.name == name and item.multi_id == 0
    ]
    if not matches:
        raise RuntimeError(f"ULog has no {name} instance 0")
    return matches[0]


def seconds(data):
    return np.asarray(data["timestamp"], dtype=float) * 1e-6


def longest_offboard_start(status):
    timestamp = seconds(status)
    selected = (
        (np.asarray(status["arming_state"], dtype=int) == ARMED_STATE)
        & (np.asarray(status["nav_state"], dtype=int) == OFFBOARD_NAV_STATE)
    )
    indices = np.flatnonzero(selected)
    if not indices.size:
        raise RuntimeError("ULog has no armed OFFBOARD interval")
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split)
    longest = max(groups, key=lambda group: timestamp[group[-1]] - timestamp[group[0]])
    return float(timestamp[longest[0]])


def quaternions(data, prefix):
    return np.column_stack([
        data[f"{prefix}[1]"],
        data[f"{prefix}[2]"],
        data[f"{prefix}[3]"],
        data[f"{prefix}[0]"],
    ])


def wrap_phase_deg(value):
    return (value + 180.0) % 360.0 - 180.0


def chirp_fit(time_s, values, local_center, duration, start_hz, end_hz):
    half_window = 0.75
    selected = (time_s >= local_center - half_window) & (time_s <= local_center + half_window)
    local_time = time_s[selected]
    signal = values[selected]
    rate_hz_s = (end_hz - start_hz) / duration
    phase = 2.0 * math.pi * (
        start_hz * local_time + 0.5 * rate_hz_s * local_time * local_time
    )
    design = np.column_stack([np.sin(phase), np.cos(phase), np.ones_like(phase)])
    coefficients, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)
    sine, cosine = coefficients[:2]
    return {
        "amplitude": float(math.hypot(sine, cosine)),
        "phase_deg": float(math.degrees(math.atan2(cosine, sine))),
        "samples": int(signal.size),
    }


def response_points(time_s, command, actual, duration, start_hz, end_hz):
    rate_hz_s = (end_hz - start_hz) / duration
    points = []
    for frequency in FREQUENCIES_HZ:
        center = (frequency - start_hz) / rate_hz_s
        command_fit = chirp_fit(
            time_s, command, center, duration, start_hz, end_hz
        )
        actual_fit = chirp_fit(
            time_s, actual, center, duration, start_hz, end_hz
        )
        gain = (
            actual_fit["amplitude"] / command_fit["amplitude"]
            if command_fit["amplitude"] > 1e-9
            else float("nan")
        )
        points.append({
            "frequency_hz": frequency,
            "gain": gain,
            "phase_deg": wrap_phase_deg(
                actual_fit["phase_deg"] - command_fit["phase_deg"]
            ),
            "command_amplitude_deg": math.degrees(command_fit["amplitude"]),
            "actual_amplitude_deg": math.degrees(actual_fit["amplitude"]),
            "samples": min(command_fit["samples"], actual_fit["samples"]),
        })
    return points


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def analyze(directory):
    trial = json.loads((directory / "trial.json").read_text(encoding="utf-8"))
    if trial["status"] != "completed":
        raise RuntimeError(f"incomplete trial: {directory}")
    ulog = ULog(str(directory / "flight.ulg"))
    offboard_start = longest_offboard_start(topic(ulog, "vehicle_status"))
    sweep_start = offboard_start + trial["sweep_start_after_offboard_s"]
    duration = float(trial["axis_duration_s"])
    settle = float(trial["settle_duration_s"])
    start_hz, end_hz = trial["frequency_hz"]

    attitude = topic(ulog, "vehicle_attitude")
    attitude_time = seconds(attitude)
    attitude_euler = Rotation.from_quat(
        quaternions(attitude, "q")
    ).as_euler("xyz")
    setpoint = topic(ulog, "vehicle_attitude_setpoint")
    setpoint_time = seconds(setpoint)
    setpoint_euler = Rotation.from_quat(
        quaternions(setpoint, "q_d")
    ).as_euler("xyz")

    angular_velocity = topic(ulog, "vehicle_angular_velocity")
    rate_time = seconds(angular_velocity)
    rate_values = np.column_stack([
        angular_velocity[f"xyz[{index}]"] for index in range(3)
    ])
    rate_setpoint = topic(ulog, "vehicle_rates_setpoint")
    rate_setpoint_time = seconds(rate_setpoint)
    rate_setpoint_values = np.column_stack([
        rate_setpoint["roll"], rate_setpoint["pitch"], rate_setpoint["yaw"]
    ])

    axis_metrics = {}
    for index, axis in enumerate(AXES):
        axis_start = sweep_start + index * (duration + settle)
        axis_end = axis_start + duration
        attitude_selected = (attitude_time >= axis_start) & (attitude_time <= axis_end)
        selected_time = attitude_time[attitude_selected]
        local_time = selected_time - axis_start
        actual = np.unwrap(attitude_euler[attitude_selected, index])
        command = np.interp(
            selected_time, setpoint_time, np.unwrap(setpoint_euler[:, index])
        )

        rate_selected = (rate_time >= axis_start) & (rate_time <= axis_end)
        selected_rate_time = rate_time[rate_selected]
        actual_rate = rate_values[rate_selected, index]
        desired_rate = np.interp(
            selected_rate_time,
            rate_setpoint_time,
            rate_setpoint_values[:, index],
        )
        axis_metrics[axis] = {
            "window_s": [float(axis_start), float(axis_end)],
            "attitude_error_rms_deg": math.degrees(rms(actual - command)),
            "actual_max_abs_deg": math.degrees(float(np.max(np.abs(actual)))),
            "rate_error_rms_rad_s": rms(actual_rate - desired_rate),
            "frequency_response": response_points(
                local_time,
                command,
                actual,
                duration,
                start_hz,
                end_hz,
            ),
        }

    actuator = topic(ulog, "actuator_motors")
    actuator_time = seconds(actuator)
    sweep_end = sweep_start + 2.0 * duration + 2.0 * settle
    actuator_selected = (actuator_time >= sweep_start) & (actuator_time <= sweep_end)
    motors = np.column_stack([
        actuator[f"control[{index}]"] for index in range(4)
    ])[actuator_selected]
    motors = motors[np.all(np.isfinite(motors), axis=1)]

    return {
        "source": str(directory),
        "ulog_sweep_window_s": [float(sweep_start), float(sweep_end)],
        "axis": axis_metrics,
        "motor_command": {
            "minimum": float(np.min(motors)),
            "maximum": float(np.max(motors)),
            "standard_deviation": float(np.std(motors)),
            "saturation_fraction": float(
                np.mean((motors <= 0.01) | (motors >= 0.99))
            ),
        },
    }


def point_at(metrics, axis, frequency):
    return next(
        point for point in metrics["axis"][axis]["frequency_response"]
        if math.isclose(point["frequency_hz"], frequency)
    )


def write_report(result, path):
    enabled = result["enabled"]
    disabled = result["disabled"]
    lines = [
        "# LZF direct attitude-sweep A/B",
        "",
        "Input: 4 deg linear chirp from 0.5 to 5 Hz, 20 s per roll/pitch axis.",
        "",
        "| Metric | Enabled | Disabled |",
        "| --- | ---: | ---: |",
    ]
    for axis in AXES:
        enabled_axis = enabled["axis"][axis]
        disabled_axis = disabled["axis"][axis]
        enabled_22 = point_at(enabled, axis, 2.2)
        disabled_22 = point_at(disabled, axis, 2.2)
        lines.extend([
            f"| {axis} attitude error RMS (deg) | {enabled_axis['attitude_error_rms_deg']:.4f} | {disabled_axis['attitude_error_rms_deg']:.4f} |",
            f"| {axis} rate error RMS (rad/s) | {enabled_axis['rate_error_rms_rad_s']:.4f} | {disabled_axis['rate_error_rms_rad_s']:.4f} |",
            f"| {axis} gain at 2.2 Hz | {enabled_22['gain']:.4f} | {disabled_22['gain']:.4f} |",
            f"| {axis} phase at 2.2 Hz (deg) | {enabled_22['phase_deg']:.2f} | {disabled_22['phase_deg']:.2f} |",
        ])
    lines.extend([
        f"| Motor command standard deviation | {enabled['motor_command']['standard_deviation']:.6f} | {disabled['motor_command']['standard_deviation']:.6f} |",
        f"| Motor saturation fraction | {enabled['motor_command']['saturation_fraction']:.6f} | {disabled['motor_command']['saturation_fraction']:.6f} |",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(result, path):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for column, axis in enumerate(AXES):
        for label, style in (("enabled", "o-"), ("disabled", "s--")):
            points = result[label]["axis"][axis]["frequency_response"]
            frequency = [point["frequency_hz"] for point in points]
            gain = [point["gain"] for point in points]
            phase = np.degrees(np.unwrap(np.radians([
                point["phase_deg"] for point in points
            ])))
            axes[0, column].plot(frequency, gain, style, label=label)
            axes[1, column].plot(frequency, phase, style, label=label)
        axes[0, column].set_title(axis.capitalize())
        axes[0, column].set_ylabel("Gain")
        axes[1, column].set_ylabel("Phase (deg)")
        axes[1, column].set_xlabel("Frequency (Hz)")
        axes[0, column].grid(True, alpha=0.3)
        axes[1, column].grid(True, alpha=0.3)
        axes[0, column].axvline(2.2, color="black", alpha=0.25)
        axes[1, column].axvline(2.2, color="black", alpha=0.25)
        axes[0, column].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    result = {
        "enabled": analyze(args.enabled_dir),
        "disabled": analyze(args.disabled_dir),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "comparison.json"
    report_path = args.output_dir / "comparison.md"
    plot_path = args.output_dir / "frequency_response.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, report_path)
    write_plot(result, plot_path)
    print(json_path)
    print(report_path)
    print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
