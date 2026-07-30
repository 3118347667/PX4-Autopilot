#!/usr/bin/env python3

"""Evaluate one LZF SITL stationary-hover ULog."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from pyulog import ULog
from scipy.signal import welch
from scipy.spatial.transform import Rotation


LOITER_NAV_STATE = 4
ARMED_STATE = 2
AXES = ("roll", "pitch", "yaw")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score the longest armed LZF LOITER segment in a PX4 ULog."
    )
    parser.add_argument("ulog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trim-seconds", type=float, default=2.0)
    parser.add_argument("--required-duration", type=float, default=60.0)
    return parser.parse_args()


def topic(ulog, name):
    matches = [
        item.data
        for item in ulog.data_list
        if item.name == name and item.multi_id == 0
    ]

    if not matches:
        raise RuntimeError(f"ULog does not contain {name} instance 0")

    return matches[0]


def time_seconds(data):
    return np.asarray(data["timestamp"], dtype=float) * 1e-6


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not values.size:
        return {name: None for name in ("mean", "rms", "p95", "max")}

    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def longest_loiter_segment(status):
    timestamp = time_seconds(status)
    nav_state = np.asarray(status["nav_state"], dtype=int)
    armed = np.asarray(status["arming_state"], dtype=int) == ARMED_STATE
    segments = []
    start = 0

    for index in range(1, len(timestamp) + 1):
        changed = (
            index == len(timestamp)
            or nav_state[index] != nav_state[index - 1]
            or armed[index] != armed[index - 1]
        )

        if not changed:
            continue

        if armed[start] and nav_state[start] == LOITER_NAV_STATE:
            segments.append((timestamp[start], timestamp[index - 1]))

        start = index

    if not segments:
        raise RuntimeError("ULog has no armed LOITER segment")

    return max(segments, key=lambda value: value[1] - value[0])


def linear_slope(timestamp, values):
    timestamp = np.asarray(timestamp, dtype=float)
    values = np.asarray(values, dtype=float)

    if timestamp.size < 2:
        return float("nan")

    centered = timestamp - np.mean(timestamp)
    denominator = float(np.dot(centered, centered))
    return (
        float(np.dot(centered, values - np.mean(values)) / denominator)
        if denominator > 0.0
        else float("nan")
    )


def dominant_frequency(timestamp, values):
    timestamp = np.asarray(timestamp, dtype=float)
    values = np.asarray(values, dtype=float)

    if timestamp.size < 16 or timestamp[-1] <= timestamp[0]:
        return float("nan")

    sample_rate = (timestamp.size - 1) / (timestamp[-1] - timestamp[0])
    frequency, density = welch(
        values - np.mean(values),
        fs=sample_rate,
        nperseg=min(4096, values.size),
    )

    if frequency.size < 2:
        return float("nan")

    return float(frequency[np.argmax(density[1:]) + 1])


def interpolate_columns(target_time, source_time, values):
    return np.column_stack(
        [
            np.interp(target_time, source_time, values[:, index])
            for index in range(values.shape[1])
        ]
    )


def parameter_at(ulog, name, timestamp_s):
    value = ulog.initial_parameters.get(name)
    timestamp_us = timestamp_s * 1e6

    for changed_timestamp, changed_name, changed_value in ulog.changed_parameters:
        if changed_timestamp > timestamp_us:
            break

        if changed_name == name:
            value = changed_value

    return value


def analyze(args):
    ulog = ULog(str(args.ulog))
    status = topic(ulog, "vehicle_status")
    raw_start, raw_end = longest_loiter_segment(status)
    start = raw_start + args.trim_seconds
    end = raw_end - args.trim_seconds

    if end <= start:
        raise RuntimeError("LOITER segment is shorter than the requested trim")

    attitude = topic(ulog, "vehicle_attitude")
    attitude_time = time_seconds(attitude)
    attitude_mask = (attitude_time >= start) & (attitude_time <= end)
    quaternion = np.column_stack(
        [
            attitude["q[1]"],
            attitude["q[2]"],
            attitude["q[3]"],
            attitude["q[0]"],
        ]
    )[attitude_mask]
    rotation = Rotation.from_quat(quaternion).as_matrix()
    tilt = np.degrees(np.arccos(np.clip(rotation[:, 2, 2], -1.0, 1.0)))
    selected_attitude_time = attitude_time[attitude_mask]

    angular_velocity = topic(ulog, "vehicle_angular_velocity")
    rate_time = time_seconds(angular_velocity)
    rate_mask = (rate_time >= start) & (rate_time <= end)
    selected_rate_time = rate_time[rate_mask]
    rates = np.column_stack(
        [
            angular_velocity["xyz[0]"],
            angular_velocity["xyz[1]"],
            angular_velocity["xyz[2]"],
        ]
    )[rate_mask]

    rate_setpoint = topic(ulog, "vehicle_rates_setpoint")
    setpoint_time = time_seconds(rate_setpoint)
    setpoint_mask = (setpoint_time >= start) & (setpoint_time <= end)
    selected_setpoint_time = setpoint_time[setpoint_mask]
    setpoints = np.column_stack(
        [
            rate_setpoint["roll"],
            rate_setpoint["pitch"],
            rate_setpoint["yaw"],
        ]
    )[setpoint_mask]
    interpolated_setpoints = interpolate_columns(
        selected_rate_time, selected_setpoint_time, setpoints
    )

    local_position = topic(ulog, "vehicle_local_position")
    local_time = time_seconds(local_position)
    local_mask = (local_time >= start) & (local_time <= end)
    selected_local_time = local_time[local_mask]
    position = np.column_stack(
        [local_position["x"], local_position["y"], local_position["z"]]
    )[local_mask]
    velocity = np.column_stack(
        [local_position["vx"], local_position["vy"], local_position["vz"]]
    )[local_mask]

    actuator = topic(ulog, "actuator_motors")
    actuator_time = time_seconds(actuator)
    actuator_mask = (actuator_time >= start) & (actuator_time <= end)
    motor_command = np.column_stack(
        [actuator[f"control[{index}]"] for index in range(4)]
    )[actuator_mask]
    motor_command = motor_command[np.all(np.isfinite(motor_command), axis=1)]

    esc = topic(ulog, "esc_status")
    esc_time = time_seconds(esc)
    esc_mask = (esc_time >= start) & (esc_time <= end)
    rpm = np.column_stack(
        [esc[f"esc[{index}].esc_rpm"] for index in range(4)]
    )[esc_mask]

    battery = topic(ulog, "battery_status")
    battery_time = time_seconds(battery)
    battery_mask = (battery_time >= start) & (battery_time <= end)
    voltage = np.asarray(battery["voltage_v"], dtype=float)[battery_mask]

    duration = end - start
    rate_metrics = {}

    for index, name in enumerate(AXES):
        error = rates[:, index] - interpolated_setpoints[:, index]
        rate_metrics[name] = {
            "actual_rad_s": summary(np.abs(rates[:, index])),
            "setpoint_rms_rad_s": float(
                np.sqrt(np.mean(interpolated_setpoints[:, index] ** 2))
            ),
            "error_rms_rad_s": float(np.sqrt(np.mean(error * error))),
            "dominant_frequency_hz": dominant_frequency(
                selected_rate_time, rates[:, index]
            ),
        }

    horizontal_offset = position[:, :2] - position[0, :2]
    tilt_slope = linear_slope(selected_attitude_time, tilt)
    rate_rms_max = max(
        rate_metrics[name]["actual_rad_s"]["rms"] for name in AXES
    )
    saturation_fraction = float(
        np.mean((motor_command <= 0.01) | (motor_command >= 0.99))
    )
    checks = {
        "duration": duration >= args.required_duration,
        "maximum_tilt": float(np.max(tilt)) < 15.0,
        "tilt_trend": abs(tilt_slope) < 0.02,
        "angular_rate_rms": rate_rms_max < 0.2,
        "motor_saturation": saturation_fraction < 0.01,
        "horizontal_station_keeping": float(
            np.max(np.linalg.norm(horizontal_offset, axis=1))
        )
        < 1.0,
        "vertical_station_keeping": float(np.ptp(position[:, 2])) < 0.5,
    }
    checks = {name: bool(value) for name, value in checks.items()}

    parameter_names = [
        "MC_ROLLRATE_P",
        "MC_ROLLRATE_I",
        "MC_ROLLRATE_D",
        "MC_PITCHRATE_P",
        "MC_PITCHRATE_I",
        "MC_PITCHRATE_D",
        "MC_YAWRATE_P",
        "MC_YAWRATE_I",
        "MC_YAWRATE_D",
    ]
    parameters = {
        name: parameter_at(ulog, name, start) for name in parameter_names
    }

    return {
        "ulog": str(args.ulog),
        "parameters": parameters,
        "loiter": {
            "raw_duration_s": float(raw_end - raw_start),
            "analyzed_duration_s": float(duration),
            "trim_seconds_each_end": args.trim_seconds,
        },
        "attitude": {
            "tilt_deg": summary(tilt),
            "tilt_linear_slope_deg_s": tilt_slope,
            "first_10_s_mean_tilt_deg": float(
                np.mean(tilt[selected_attitude_time <= start + 10.0])
            ),
            "last_10_s_mean_tilt_deg": float(
                np.mean(tilt[selected_attitude_time >= end - 10.0])
            ),
        },
        "angular_rate": rate_metrics,
        "position": {
            "standard_deviation_m": np.std(position, axis=0).tolist(),
            "peak_to_peak_m": np.ptp(position, axis=0).tolist(),
            "linear_slope_m_s": [
                linear_slope(selected_local_time, position[:, index])
                for index in range(3)
            ],
            "maximum_horizontal_offset_m": float(
                np.max(np.linalg.norm(horizontal_offset, axis=1))
            ),
            "velocity_rms_m_s": float(
                np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))
            ),
        },
        "motor_command": {
            "mean_per_motor": np.mean(motor_command, axis=0).tolist(),
            "standard_deviation_per_motor": np.std(
                motor_command, axis=0
            ).tolist(),
            "minimum": float(np.min(motor_command)),
            "maximum": float(np.max(motor_command)),
            "saturation_fraction": saturation_fraction,
        },
        "esc": {
            "mean_rpm_per_motor": np.mean(rpm, axis=0).tolist(),
            "standard_deviation_rpm_per_motor": np.std(rpm, axis=0).tolist(),
            "publication_rate_hz": float(
                (np.count_nonzero(esc_mask) - 1)
                / (esc_time[esc_mask][-1] - esc_time[esc_mask][0])
            ),
        },
        "battery": {
            "start_voltage_v": float(voltage[0]),
            "end_voltage_v": float(voltage[-1]),
            "minimum_voltage_v": float(np.min(voltage)),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def write_report(result, output):
    rate = result["angular_rate"]
    lines = [
        "# LZF inner-loop hover validation",
        "",
        f"- ULog: `{result['ulog']}`",
        f"- Analyzed LOITER duration: `{result['loiter']['analyzed_duration_s']:.3f} s`",
        f"- Result: `{'PASS' if result['passed'] else 'FAIL'}`",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{name}` | `{value}` |"
        for name, value in result["parameters"].items()
    )
    lines.extend(
        [
            "",
            "## Stability metrics",
            "",
            "| Metric | Roll | Pitch | Yaw |",
            "| --- | ---: | ---: | ---: |",
            "| Angular-rate RMS (rad/s) | "
            + " | ".join(
                f"{rate[name]['actual_rad_s']['rms']:.5f}" for name in AXES
            )
            + " |",
            "| Angular-rate error RMS (rad/s) | "
            + " | ".join(
                f"{rate[name]['error_rms_rad_s']:.5f}" for name in AXES
            )
            + " |",
            "| Dominant frequency (Hz) | "
            + " | ".join(
                f"{rate[name]['dominant_frequency_hz']:.3f}" for name in AXES
            )
            + " |",
            "",
            f"- Maximum tilt: `{result['attitude']['tilt_deg']['max']:.3f} deg`",
            f"- Tilt trend: `{result['attitude']['tilt_linear_slope_deg_s']:.6f} deg/s`",
            f"- Maximum horizontal offset: `{result['position']['maximum_horizontal_offset_m']:.3f} m`",
            f"- Vertical peak-to-peak: `{result['position']['peak_to_peak_m'][2]:.3f} m`",
            f"- Motor command range: `{result['motor_command']['minimum']:.4f}` to "
            f"`{result['motor_command']['maximum']:.4f}`",
            f"- Motor saturation fraction: `{result['motor_command']['saturation_fraction']:.6f}`",
            f"- ESC status rate: `{result['esc']['publication_rate_hz']:.2f} Hz`",
            "",
            "## Acceptance checks",
            "",
        ]
    )
    lines.extend(
        f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}"
        for name, passed in result["checks"].items()
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    if not args.ulog.is_file():
        print(f"error: missing ULog: {args.ulog}", file=sys.stderr)
        return 2

    try:
        result = analyze(args)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    report_path = args.output_dir / "report.md"
    metrics_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(result, report_path)
    print(f"metrics: {metrics_path}")
    print(f"report:  {report_path}")
    print(f"result:  {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
