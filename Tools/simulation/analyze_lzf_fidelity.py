#!/usr/bin/env python3

"""Compare the LZF Gazebo model with PX4 ULogs and paired ROS bags.

The tool deliberately separates calibration data from holdout and simulation
data.  It writes machine-readable metrics, diagnostic plots, and a concise
Markdown report so that model changes can be compared without repeating ad-hoc
notebook work.
"""

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog


RPM_TO_RAD_S = 2.0 * math.pi / 60.0
GRAVITY = 9.80665


@dataclass
class ModelParameters:
    mass: float = 1.326
    propeller_diameter: float = 0.127
    air_density: float = 1.225
    static_ct: float = 0.149294
    motor_idle_rpm: float = 5917.0
    motor_loaded_kv: float = 1682.0
    motor_no_load_kv: float = 2400.0
    rotor_drag_coefficient: float = 2.14e-5
    battery_full_voltage: float = 25.2
    battery_empty_voltage: float = 18.0
    battery_floor_voltage: float = 15.0
    battery_reference_load: float = 4.113
    battery_drain_time: float = 1140.0
    battery_sag_instant: float = 0.1485
    battery_sag_polarization: float = 0.1890
    battery_tau: float = 2.02


@dataclass
class DatasetSpec:
    name: str
    ulog: Path
    bag: Optional[Path]
    role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LZF motor, propeller, battery, drag, and tracking fidelity."
    )
    parser.add_argument("--real-ulog", required=True, type=Path)
    parser.add_argument("--real-bag", type=Path)
    parser.add_argument(
        "--holdout",
        action="append",
        default=[],
        metavar="NAME:ULOG[:BAG]",
        help="independent real-flight validation dataset",
    )
    parser.add_argument(
        "--sim",
        action="append",
        default=[],
        metavar="NAME:ULOG[:BAG]",
        help="SITL dataset to compare against the real flight",
    )
    parser.add_argument(
        "--bag-odom-topic",
        action="append",
        default=[],
        metavar="NAME=TOPIC",
        help=(
            "override the odometry topic for one named ROS-bag dataset; "
            "repeat for multiple datasets"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/lzf_fidelity"),
    )
    parser.add_argument("--mass", type=float, default=1.326)
    parser.add_argument("--rho", type=float, default=1.225)
    parser.add_argument("--static-ct", type=float, default=0.149294)
    parser.add_argument("--drag-coefficient", type=float, default=2.14e-5)
    return parser.parse_args()


def parse_dataset(value: str, role: str) -> DatasetSpec:
    parts = value.split(":", 2)

    if len(parts) < 2:
        raise ValueError(f"invalid dataset '{value}', expected NAME:ULOG[:BAG]")

    return DatasetSpec(
        name=parts[0],
        ulog=Path(parts[1]),
        bag=Path(parts[2]) if len(parts) == 3 and parts[2] else None,
        role=role,
    )


def parse_bag_odom_topics(values: Sequence[str]) -> Dict[str, str]:
    topics: Dict[str, str] = {}

    for value in values:
        name, separator, topic = value.partition("=")

        if not separator or not name or not topic.startswith("/"):
            raise ValueError(
                f"invalid bag odometry override '{value}', expected NAME=/topic"
            )

        if name in topics:
            raise ValueError(f"duplicate bag odometry override for '{name}'")

        topics[name] = topic

    return topics


def get_topic(ulog: ULog, name: str, required: bool = True) -> Optional[Dict[str, np.ndarray]]:
    matches = [
        item.data for item in ulog.data_list if item.name == name and item.multi_id == 0
    ]

    if matches:
        return matches[0]

    if required:
        raise RuntimeError(f"ULog does not contain {name} instance 0")

    return None


def topic_time(data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(data["timestamp"], dtype=float) * 1e-6


def sample_rate(time: np.ndarray) -> float:
    if time.size < 2 or time[-1] <= time[0]:
        return float("nan")

    return (time.size - 1) / (time[-1] - time[0])


def interp_columns(
    target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        return np.interp(target_time, source_time, values)

    return np.column_stack(
        [np.interp(target_time, source_time, values[:, index]) for index in range(values.shape[1])]
    )


def interp_finite_columns(
    target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        valid = np.isfinite(values)
        if not np.any(valid):
            return np.full(target_time.shape, np.nan, dtype=float)
        return np.interp(target_time, source_time[valid], values[valid])

    columns = []

    for index in range(values.shape[1]):
        valid = np.isfinite(values[:, index])
        columns.append(
            np.interp(target_time, source_time[valid], values[valid, index])
            if np.any(valid)
            else np.full(target_time.shape, np.nan, dtype=float)
        )

    return np.column_stack(columns)


def finite_metrics(measured: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    measured = np.asarray(measured, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    valid = np.isfinite(measured) & np.isfinite(predicted)

    if np.count_nonzero(valid) < 2:
        return {
            "samples": int(np.count_nonzero(valid)),
            "rmse": float("nan"),
            "bias_removed_rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "relative_rmse_percent": float("nan"),
            "r_squared": float("nan"),
            "bias_removed_r_squared": float("nan"),
            "correlation": float("nan"),
        }

    measured = measured[valid]
    predicted = predicted[valid]
    residual = predicted - measured
    bias = np.mean(residual)
    centered_residual = residual - bias
    variance = np.sum((measured - np.mean(measured)) ** 2)
    correlation = (
        np.corrcoef(measured, predicted)[0, 1]
        if np.std(measured) > 0.0 and np.std(predicted) > 0.0
        else float("nan")
    )
    return {
        "samples": int(measured.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias_removed_rmse": float(np.sqrt(np.mean(centered_residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(bias),
        "relative_rmse_percent": float(
            100.0 * np.sqrt(np.mean(residual**2)) / max(abs(np.mean(measured)), 1e-9)
        ),
        "r_squared": float(1.0 - np.sum(residual**2) / variance)
        if variance > 0.0
        else float("nan"),
        "bias_removed_r_squared": float(
            1.0 - np.sum(centered_residual**2) / variance
        )
        if variance > 0.0
        else float("nan"),
        "correlation": float(correlation),
    }


def percentile_summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {key: float("nan") for key in ("min", "p05", "median", "mean", "p95", "max")}

    percentiles = np.percentile(values, [0.0, 5.0, 50.0, 95.0, 100.0])
    return {
        "min": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "median": float(percentiles[2]),
        "mean": float(np.mean(values)),
        "p95": float(percentiles[3]),
        "max": float(percentiles[4]),
    }


def quaternion_to_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = quaternion / np.maximum(norm, 1e-12)
    x, y, z, w = quaternion.T
    matrix = np.empty((quaternion.shape[0], 3, 3), dtype=float)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - z * w)
    matrix[:, 0, 2] = 2.0 * (x * z + y * w)
    matrix[:, 1, 0] = 2.0 * (x * y + z * w)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - x * w)
    matrix[:, 2, 0] = 2.0 * (x * z - y * w)
    matrix[:, 2, 1] = 2.0 * (y * z + x * w)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def motor_slot_order(esc: Dict[str, np.ndarray]) -> Tuple[List[int], List[int]]:
    functions = []

    for slot in range(8):
        key = f"esc[{slot}].actuator_function"

        if key not in esc:
            functions.append(0)
            continue

        values = np.asarray(esc[key], dtype=int)
        nonzero = values[values > 0]
        functions.append(int(np.median(nonzero)) if nonzero.size else 0)

    order = []

    for motor_function in range(101, 105):
        matching = [slot for slot, function in enumerate(functions) if function == motor_function]
        order.append(matching[0] if matching else motor_function - 101)

    return order, functions


def battery_polarization(
    load: np.ndarray, armed: np.ndarray, time: np.ndarray, tau: float
) -> np.ndarray:
    state = np.zeros_like(load, dtype=float)

    for index in range(1, load.size):
        dt = max(0.0, time[index] - time[index - 1])
        target = load[index] if armed[index] else 0.0
        alpha = -math.expm1(-dt / tau)
        state[index] = state[index - 1] + alpha * (target - state[index - 1])

    return state


def analyze_ulog(
    spec: DatasetSpec, parameters: ModelParameters, output_dir: Path
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    topic_names = [
        "esc_status",
        "actuator_motors",
        "battery_status",
        "actuator_armed",
        "vehicle_land_detected",
        "vehicle_acceleration",
        "vehicle_local_position",
        "vehicle_attitude",
        "vehicle_angular_velocity",
        "vehicle_rates_setpoint",
        "vehicle_air_data",
    ]
    ulog = ULog(str(spec.ulog), message_name_filter_list=topic_names)
    esc = get_topic(ulog, "esc_status")
    actuator = get_topic(ulog, "actuator_motors")
    battery = get_topic(ulog, "battery_status")
    armed_topic = get_topic(ulog, "actuator_armed")
    acceleration = get_topic(ulog, "vehicle_acceleration")
    local_position = get_topic(ulog, "vehicle_local_position", required=False)
    attitude = get_topic(ulog, "vehicle_attitude", required=False)
    angular_velocity = get_topic(
        ulog, "vehicle_angular_velocity", required=False
    )
    rates_setpoint = get_topic(ulog, "vehicle_rates_setpoint", required=False)
    land = get_topic(ulog, "vehicle_land_detected", required=False)
    air_data = get_topic(ulog, "vehicle_air_data", required=False)

    esc_time = topic_time(esc)
    actuator_time = topic_time(actuator)
    battery_time = topic_time(battery)
    armed_time = topic_time(armed_topic)
    acceleration_time = topic_time(acceleration)
    start_time = min(esc_time[0], actuator_time[0], battery_time[0])
    motor_order, raw_functions = motor_slot_order(esc)
    rpm = np.column_stack(
        [
            np.asarray(esc[f"esc[{slot}].esc_rpm"], dtype=float)
            for slot in motor_order
        ]
    )
    maximum_physical_rpm = (
        parameters.motor_no_load_kv * parameters.battery_full_voltage * 1.05
    )
    rpm_is_physical = (
        np.isfinite(rpm)
        & (rpm >= 0.0)
        & (rpm <= maximum_physical_rpm)
    )
    physical_row = np.all(rpm_is_physical, axis=1)
    physical_rpm = np.where(rpm_is_physical, rpm, np.nan)
    mean_physical_rpm = np.full(esc_time.shape, np.nan, dtype=float)
    mean_physical_rpm[physical_row] = np.mean(
        physical_rpm[physical_row],
        axis=1,
    )
    controls = np.column_stack(
        [
            np.asarray(actuator[f"control[{index}]"], dtype=float)
            for index in range(4)
        ]
    )
    controls_at_esc = interp_columns(esc_time, actuator_time, controls)
    voltage = np.asarray(battery["voltage_v"], dtype=float)
    voltage_at_esc = np.interp(esc_time, battery_time, voltage)
    armed_values = np.asarray(armed_topic["armed"], dtype=float)
    armed_at_esc = np.interp(esc_time, armed_time, armed_values) > 0.5

    if land is not None:
        landed_at_esc = (
            np.interp(
                esc_time,
                topic_time(land),
                np.asarray(land["landed"], dtype=float),
            )
            > 0.5
        )
    else:
        landed_at_esc = ~armed_at_esc

    flight_mask = (
        armed_at_esc
        & ~landed_at_esc
        & physical_row
        & (mean_physical_rpm > 1000.0)
    )
    clipped_control = np.clip(controls_at_esc, 0.0, 1.0)
    rpm_prediction = (
        parameters.motor_idle_rpm
        + parameters.motor_loaded_kv * clipped_control * voltage_at_esc[:, None]
    )
    rpm_prediction = np.minimum(
        rpm_prediction,
        parameters.motor_no_load_kv * voltage_at_esc[:, None],
    )

    per_motor_metrics = []

    for motor in range(4):
        per_motor_metrics.append(
            finite_metrics(
                physical_rpm[flight_mask, motor],
                rpm_prediction[flight_mask, motor],
            )
        )

    pooled_motor_metrics = finite_metrics(
        physical_rpm[flight_mask, :], rpm_prediction[flight_mask, :]
    )

    if local_position is not None:
        local_time = topic_time(local_position)
        local_velocity = np.column_stack(
            [
                np.asarray(local_position[field], dtype=float)
                for field in ("vx", "vy", "vz")
            ]
        )
        speed_at_esc = np.linalg.norm(
            interp_columns(esc_time, local_time, local_velocity), axis=1
        )
        hover_mask = flight_mask & (speed_at_esc < 0.35)
    else:
        hover_mask = flight_mask

    measured_specific_thrust = -np.asarray(acceleration["xyz[2]"], dtype=float)
    rpm_at_acceleration = interp_finite_columns(
        acceleration_time, esc_time, physical_rpm
    )
    physical_rpm_at_acceleration = (
        np.interp(
            acceleration_time,
            esc_time,
            physical_row.astype(float),
        )
        > 0.999
    )
    n_squared_sum = np.sum((np.maximum(rpm_at_acceleration, 0.0) / 60.0) ** 2, axis=1)
    predicted_specific_thrust = (
        parameters.static_ct
        * parameters.air_density
        * parameters.propeller_diameter**4
        * n_squared_sum
        / parameters.mass
    )
    armed_at_acceleration = (
        np.interp(acceleration_time, armed_time, armed_values) > 0.5
    )

    if land is not None:
        landed_at_acceleration = (
            np.interp(
                acceleration_time,
                topic_time(land),
                np.asarray(land["landed"], dtype=float),
            )
            > 0.5
        )
    else:
        landed_at_acceleration = ~armed_at_acceleration

    thrust_mask = (
        armed_at_acceleration
        & ~landed_at_acceleration
        & physical_rpm_at_acceleration
        & (np.mean(rpm_at_acceleration, axis=1) > 5000.0)
        & np.isfinite(measured_specific_thrust)
    )
    thrust_metrics = finite_metrics(
        measured_specific_thrust[thrust_mask],
        predicted_specific_thrust[thrust_mask],
    )
    ct_denominator = (
        parameters.air_density
        * parameters.propeller_diameter**4
        * n_squared_sum[thrust_mask]
    )
    effective_ct = (
        measured_specific_thrust[thrust_mask] * parameters.mass
        / np.maximum(ct_denominator, 1e-12)
    )

    esc_load = np.full(esc_time.shape, np.nan, dtype=float)
    esc_load[physical_row] = np.mean(
        (
            np.maximum(physical_rpm[physical_row], 0.0)
            * RPM_TO_RAD_S
            / 1000.0
        )
        ** 3,
        axis=1,
    )
    load_at_battery = interp_finite_columns(battery_time, esc_time, esc_load)
    armed_at_battery = (
        np.interp(battery_time, armed_time, armed_values) > 0.5
    )
    polarization = battery_polarization(
        load_at_battery,
        armed_at_battery,
        battery_time,
        parameters.battery_tau,
    )
    dt = np.diff(battery_time, prepend=battery_time[0])
    normalized_energy = np.cumsum(
        np.where(armed_at_battery, load_at_battery, 0.0) * np.maximum(dt, 0.0)
    ) / parameters.battery_reference_load
    soc = np.clip(
        1.0 - normalized_energy / parameters.battery_drain_time,
        0.0,
        1.0,
    )
    remaining = np.asarray(battery["remaining"], dtype=float)
    valid_remaining = remaining[np.isfinite(remaining)]
    logged_initial_soc = float(
        np.clip(valid_remaining[0], 0.0, 1.0)
    ) if valid_remaining.size else 1.0
    soc_from_logged_initial = np.clip(
        logged_initial_soc
        - normalized_energy / parameters.battery_drain_time,
        0.0,
        1.0,
    )
    open_circuit_voltage = parameters.battery_empty_voltage + (
        parameters.battery_full_voltage - parameters.battery_empty_voltage
    ) * soc
    battery_prediction = np.clip(
        open_circuit_voltage
        - parameters.battery_sag_instant * load_at_battery
        - parameters.battery_sag_polarization * polarization,
        parameters.battery_floor_voltage,
        parameters.battery_full_voltage,
    )
    open_circuit_voltage_from_logged_initial = (
        parameters.battery_empty_voltage
        + (
            parameters.battery_full_voltage
            - parameters.battery_empty_voltage
        )
        * soc_from_logged_initial
    )
    battery_prediction_from_logged_initial = np.clip(
        open_circuit_voltage_from_logged_initial
        - parameters.battery_sag_instant * load_at_battery
        - parameters.battery_sag_polarization * polarization,
        parameters.battery_floor_voltage,
        parameters.battery_full_voltage,
    )
    battery_fit_mask = (
        armed_at_battery
        & np.isfinite(voltage)
        & np.isfinite(battery_prediction)
        & (load_at_battery > 0.05)
    )
    battery_metrics = finite_metrics(
        voltage[battery_fit_mask], battery_prediction[battery_fit_mask]
    )
    battery_logged_initial_metrics = finite_metrics(
        voltage[battery_fit_mask],
        battery_prediction_from_logged_initial[battery_fit_mask],
    )

    current = np.asarray(battery["current_a"], dtype=float)
    valid_current = current[np.isfinite(current) & (current >= 0.0)]
    power = (
        voltage[np.isfinite(current) & (current >= 0.0)]
        * valid_current
        if valid_current.size
        else np.array([], dtype=float)
    )
    density = (
        np.asarray(air_data["rho"], dtype=float)
        if air_data is not None
        else np.array([], dtype=float)
    )
    stability_metrics: Dict[str, object] = {}

    if attitude is not None:
        attitude_time = topic_time(attitude)
        quaternion_xyzw = np.column_stack(
            [
                np.asarray(attitude["q[1]"], dtype=float),
                np.asarray(attitude["q[2]"], dtype=float),
                np.asarray(attitude["q[3]"], dtype=float),
                np.asarray(attitude["q[0]"], dtype=float),
            ]
        )
        rotation = quaternion_to_rotation_matrix(quaternion_xyzw)
        tilt_deg = np.degrees(
            np.arccos(np.clip(rotation[:, 2, 2], -1.0, 1.0))
        )
        armed_at_attitude = (
            np.interp(attitude_time, armed_time, armed_values) > 0.5
        )
        armed_tilt = tilt_deg[armed_at_attitude]
        stability_metrics["tilt_deg_while_armed"] = percentile_summary(
            armed_tilt
        )

        over_tilt = np.flatnonzero(armed_at_attitude & (tilt_deg > 60.0))
        armed_indices = np.flatnonzero(armed_at_attitude)
        stability_metrics["first_tilt_over_60_s_after_arm"] = (
            float(attitude_time[over_tilt[0]] - attitude_time[armed_indices[0]])
            if over_tilt.size and armed_indices.size
            else float("nan")
        )

    if local_position is not None:
        local_time = topic_time(local_position)
        local_xyz = np.column_stack(
            [
                np.asarray(local_position[field], dtype=float)
                for field in ("x", "y", "z")
            ]
        )
        local_velocity = np.column_stack(
            [
                np.asarray(local_position[field], dtype=float)
                for field in ("vx", "vy", "vz")
            ]
        )
        armed_at_local = (
            np.interp(local_time, armed_time, armed_values) > 0.5
        )
        armed_local_indices = np.flatnonzero(armed_at_local)

        if armed_local_indices.size:
            start_position = local_xyz[armed_local_indices[0]]
            displacement = local_xyz[armed_at_local] - start_position
            stability_metrics.update(
                {
                    "armed_duration_s": float(
                        local_time[armed_local_indices[-1]]
                        - local_time[armed_local_indices[0]]
                    ),
                    "maximum_horizontal_displacement_m": float(
                        np.max(np.linalg.norm(displacement[:, :2], axis=1))
                    ),
                    "maximum_speed_m_s": float(
                        np.max(
                            np.linalg.norm(
                                local_velocity[armed_at_local],
                                axis=1,
                            )
                        )
                    ),
                    "altitude_range_m": [
                        float(np.min(local_xyz[armed_at_local, 2])),
                        float(np.max(local_xyz[armed_at_local, 2])),
                    ],
                    "disarmed_after_flight": bool(
                        armed_values[-1] < 0.5
                    ),
                }
            )

    rate_tracking: Dict[str, object] = {}

    if angular_velocity is not None and rates_setpoint is not None:
        rate_time = topic_time(rates_setpoint)
        desired_rates = np.column_stack(
            [
                np.asarray(rates_setpoint[field], dtype=float)
                for field in ("roll", "pitch", "yaw")
            ]
        )
        actual_rate_time = topic_time(angular_velocity)
        actual_rates = interp_columns(
            rate_time,
            actual_rate_time,
            np.column_stack(
                [
                    np.asarray(angular_velocity[f"xyz[{axis}]"], dtype=float)
                    for axis in range(3)
                ]
            ),
        )
        armed_at_rate = (
            np.interp(rate_time, armed_time, armed_values) > 0.5
        )

        if land is not None:
            landed_at_rate = (
                np.interp(
                    rate_time,
                    topic_time(land),
                    np.asarray(land["landed"], dtype=float),
                )
                > 0.5
            )
        else:
            landed_at_rate = ~armed_at_rate

        rate_mask = (
            armed_at_rate
            & ~landed_at_rate
            & np.all(np.isfinite(desired_rates), axis=1)
            & np.all(np.isfinite(actual_rates), axis=1)
            & (np.linalg.norm(desired_rates, axis=1) < 20.0)
        )
        axis_names = ("roll", "pitch", "yaw")
        per_axis = {
            axis_names[axis]: finite_metrics(
                actual_rates[rate_mask, axis],
                desired_rates[rate_mask, axis],
            )
            for axis in range(3)
        }
        rate_tracking = {
            "samples": int(np.count_nonzero(rate_mask)),
            "setpoint_rate_hz": sample_rate(rate_time),
            "measurement_rate_hz": sample_rate(actual_rate_time),
            "pooled": finite_metrics(
                actual_rates[rate_mask],
                desired_rates[rate_mask],
            ),
            "per_axis": per_axis,
            "desired_rate_rad_s": percentile_summary(
                np.linalg.norm(desired_rates[rate_mask], axis=1)
            ),
            "tracking_error_rad_s": percentile_summary(
                np.linalg.norm(
                    actual_rates[rate_mask] - desired_rates[rate_mask],
                    axis=1,
                )
            ),
        }

    duration = max(
        esc_time[-1], actuator_time[-1], battery_time[-1]
    ) - start_time
    metrics: Dict[str, object] = {
        "name": spec.name,
        "role": spec.role,
        "ulog": str(spec.ulog),
        "duration_s": float(duration),
        "rates_hz": {
            "esc_status": sample_rate(esc_time),
            "actuator_motors": sample_rate(actuator_time),
            "battery_status": sample_rate(battery_time),
            "vehicle_acceleration": sample_rate(acceleration_time),
        },
        "esc": {
            "slot_to_actuator_function": raw_functions,
            "motor_order_slots": motor_order,
            "online_fraction": float(
                np.mean(np.asarray(esc["esc_online_flags"], dtype=int) & 0xF == 0xF)
            ),
            "maximum_physical_rpm_filter": float(maximum_physical_rpm),
            "rejected_nonphysical_samples": int(
                np.count_nonzero(~physical_row)
            ),
            "flight_samples": int(np.count_nonzero(flight_mask)),
            "mean_rpm_per_motor": [
                float(value)
                for value in np.mean(physical_rpm[flight_mask], axis=0)
            ]
            if np.any(flight_mask)
            else [float("nan")] * 4,
            "hover_mean_rpm_per_motor": [
                float(value)
                for value in np.mean(physical_rpm[hover_mask], axis=0)
            ]
            if np.any(hover_mask)
            else [float("nan")] * 4,
            "mean_control_per_motor": [
                float(value)
                for value in np.mean(controls_at_esc[flight_mask], axis=0)
            ]
            if np.any(flight_mask)
            else [float("nan")] * 4,
            "motor_speed_model_pooled": pooled_motor_metrics,
            "motor_speed_model_per_motor": per_motor_metrics,
        },
        "propeller": {
            "specific_thrust_model": thrust_metrics,
            "effective_static_ct": percentile_summary(effective_ct),
            "configured_static_ct": parameters.static_ct,
        },
        "battery": {
            "voltage": percentile_summary(voltage),
            "current_a": percentile_summary(valid_current),
            "power_w": percentile_summary(power),
            "model": battery_metrics,
            "model_with_logged_initial_soc": battery_logged_initial_metrics,
            "remaining_start": float(remaining[0]),
            "remaining_end": float(remaining[-1]),
        },
        "air": {
            "logged_density_kg_m3": percentile_summary(density),
            "configured_density_kg_m3": parameters.air_density,
        },
        "rate_tracking": rate_tracking,
        "stability": stability_metrics,
    }

    series = {
        "esc_elapsed": esc_time - start_time,
        "rpm": physical_rpm,
        "rpm_prediction": rpm_prediction,
        "flight_mask": flight_mask,
        "battery_elapsed": battery_time - start_time,
        "voltage": voltage,
        "battery_prediction": battery_prediction,
        "soc": soc,
        "acceleration_elapsed": acceleration_time - start_time,
        "measured_specific_thrust": measured_specific_thrust,
        "predicted_specific_thrust": predicted_specific_thrust,
        "thrust_mask": thrust_mask,
    }
    plot_ulog(spec, metrics, series, output_dir)
    return metrics, series


def plot_ulog(
    spec: DatasetSpec,
    metrics: Dict[str, object],
    series: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    elapsed = series["esc_elapsed"]

    for motor in range(4):
        axes[0].plot(
            elapsed,
            series["rpm"][:, motor] / 1000.0,
            linewidth=0.55,
            label=f"M{motor + 1}",
        )

    axes[0].set_ylabel("Mechanical RPM (krpm)")
    axes[0].legend(ncol=4)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(
        series["battery_elapsed"],
        series["voltage"],
        linewidth=0.9,
        label="measured",
    )
    axes[1].plot(
        series["battery_elapsed"],
        series["battery_prediction"],
        linewidth=0.9,
        label="current LZF model",
    )
    axes[1].set_ylabel("Battery voltage (V)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    mask = series["thrust_mask"]
    stride = max(1, int(np.count_nonzero(mask) / 12000))
    axes[2].scatter(
        series["measured_specific_thrust"][mask][::stride],
        series["predicted_specific_thrust"][mask][::stride],
        s=2,
        alpha=0.25,
    )
    limits = [
        min(
            np.nanmin(series["measured_specific_thrust"][mask]),
            np.nanmin(series["predicted_specific_thrust"][mask]),
        ),
        max(
            np.nanmax(series["measured_specific_thrust"][mask]),
            np.nanmax(series["predicted_specific_thrust"][mask]),
        ),
    ]
    axes[2].plot(limits, limits, "k--", linewidth=0.8)
    axes[2].set_xlabel("Measured body-z specific force (m/s^2)")
    axes[2].set_ylabel("Static prop model (m/s^2)")
    axes[2].grid(True, alpha=0.25)
    motor_rmse = metrics["esc"]["motor_speed_model_pooled"]["relative_rmse_percent"]
    thrust_rmse = metrics["propeller"]["specific_thrust_model"]["rmse"]
    figure.suptitle(
        f"{spec.name}: motor model RMSE {motor_rmse:.2f}%, "
        f"body-z RMSE {thrust_rmse:.3f} m/s^2"
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"{spec.name}_ulog.png", dpi=160)
    plt.close(figure)


def split_lap_windows(times: np.ndarray, gap_s: float = 0.5) -> List[Tuple[float, float]]:
    if times.size == 0:
        return []

    starts = np.concatenate(([0], np.flatnonzero(np.diff(times) > gap_s) + 1))
    ends = np.concatenate((starts[1:] - 1, [times.size - 1]))
    return [(float(times[start]), float(times[end])) for start, end in zip(starts, ends)]


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2 or np.dot(x, x) <= 0.0:
        return float("nan"), finite_metrics(y, np.full_like(y, np.nan))

    coefficient = float(np.dot(x, y) / np.dot(x, x))
    return coefficient, finite_metrics(y, coefficient * x)


def analyze_bag(
    spec: DatasetSpec,
    parameters: ModelParameters,
    output_dir: Path,
    odom_topic: Optional[str] = None,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    try:
        import rosbag
    except ImportError as error:
        raise RuntimeError(
            "rosbag is unavailable; source /opt/ros/noetic/setup.zsh and the "
            "comet_ws devel setup before running this tool"
        ) from error

    debug_time: List[float] = []
    debug_desired_position: List[List[float]] = []
    debug_desired_velocity: List[List[float]] = []
    odom_time: List[float] = []
    odom_position: List[List[float]] = []
    odom_velocity: List[List[float]] = []
    odom_quaternion: List[List[float]] = []
    imu_time: List[float] = []
    imu_acceleration: List[List[float]] = []
    esc_time: List[float] = []
    esc_rpm_sum: List[float] = []
    command_time: List[float] = []

    with rosbag.Bag(str(spec.bag)) as bag:
        if odom_topic is not None:
            if bag.get_message_count([odom_topic]) <= 0:
                raise RuntimeError(
                    f"ROS bag for {spec.name} does not contain requested "
                    f"odometry topic {odom_topic}"
                )
            odom_source = odom_topic
        else:
            odom_source = (
                "/odom_converter/converted_odom0"
                if bag.get_message_count(["/odom_converter/converted_odom0"]) > 0
                else "/mavros/local_position/odom"
            )
        for topic, message, record_time in bag.read_messages(
            topics=[
                "/debugPx4ctrl",
                odom_source,
                "/mavros/imu/data",
                "/mavros/esc_status",
                "/setpoints_cmd",
            ]
        ):
            time_s = record_time.to_sec()

            if topic == "/debugPx4ctrl":
                debug_time.append(time_s)
                debug_desired_position.append(
                    [message.des_p_x, message.des_p_y, message.des_p_z]
                )
                debug_desired_velocity.append(
                    [message.des_v_x, message.des_v_y, message.des_v_z]
                )

            elif topic == odom_source:
                odom_time.append(time_s)
                odom_position.append(
                    [
                        message.pose.pose.position.x,
                        message.pose.pose.position.y,
                        message.pose.pose.position.z,
                    ]
                )
                odom_velocity.append(
                    [
                        message.twist.twist.linear.x,
                        message.twist.twist.linear.y,
                        message.twist.twist.linear.z,
                    ]
                )
                odom_quaternion.append(
                    [
                        message.pose.pose.orientation.x,
                        message.pose.pose.orientation.y,
                        message.pose.pose.orientation.z,
                        message.pose.pose.orientation.w,
                    ]
                )

            elif topic == "/mavros/imu/data":
                imu_time.append(time_s)
                imu_acceleration.append(
                    [
                        message.linear_acceleration.x,
                        message.linear_acceleration.y,
                        message.linear_acceleration.z,
                    ]
                )

            elif topic == "/mavros/esc_status":
                rpms = np.asarray(
                    [status.rpm for status in message.esc_status[:4]],
                    dtype=float,
                )
                maximum_physical_rpm = (
                    parameters.motor_no_load_kv
                    * parameters.battery_full_voltage
                    * 1.05
                )

                if (
                    rpms.size == 4
                    and np.all(np.isfinite(rpms))
                    and np.all((rpms >= 0.0) & (rpms <= maximum_physical_rpm))
                ):
                    esc_time.append(time_s)
                    esc_rpm_sum.append(float(np.sum(rpms)))

            elif topic == "/setpoints_cmd":
                command_time.append(time_s)

    debug_time_array = np.asarray(debug_time, dtype=float)
    desired_position = np.asarray(debug_desired_position, dtype=float)
    desired_velocity = np.asarray(debug_desired_velocity, dtype=float)
    odom_time_array = np.asarray(odom_time, dtype=float)
    position = np.asarray(odom_position, dtype=float)
    velocity = np.asarray(odom_velocity, dtype=float)
    quaternion = np.asarray(odom_quaternion, dtype=float)
    imu_time_array = np.asarray(imu_time, dtype=float)
    imu_acceleration_array = np.asarray(imu_acceleration, dtype=float)
    esc_time_array = np.asarray(esc_time, dtype=float)
    rpm_sum = np.asarray(esc_rpm_sum, dtype=float)
    command_time_array = np.asarray(command_time, dtype=float)

    required = {
        "debugPx4ctrl": debug_time_array,
        "local_position/odom": odom_time_array,
        "imu/data": imu_time_array,
        "esc_status": esc_time_array,
        "setpoints_cmd": command_time_array,
    }

    for name, values in required.items():
        if values.size == 0:
            raise RuntimeError(f"ROS bag does not contain usable /{name} samples")

    actual_position = interp_columns(debug_time_array, odom_time_array, position)
    actual_velocity = interp_columns(debug_time_array, odom_time_array, velocity)
    lap_windows = split_lap_windows(command_time_array)
    lap_metrics = []
    active_mask = np.zeros(debug_time_array.shape, dtype=bool)
    active_imu_mask = np.zeros(imu_time_array.shape, dtype=bool)

    for lap_index, (start, end) in enumerate(lap_windows):
        mask = (debug_time_array >= start) & (debug_time_array <= end)
        active_imu_mask |= (imu_time_array >= start) & (imu_time_array <= end)

        if np.count_nonzero(mask) < 2:
            continue

        active_mask |= mask
        position_error = actual_position[mask] - desired_position[mask]
        velocity_error = actual_velocity[mask] - desired_velocity[mask]
        lap_metrics.append(
            {
                "lap": lap_index + 1,
                "start_s": start,
                "duration_s": end - start,
                "position_rmse_xyz_m": [
                    float(value)
                    for value in np.sqrt(np.mean(position_error**2, axis=0))
                ],
                "position_rmse_norm_m": float(
                    np.sqrt(np.mean(np.sum(position_error**2, axis=1)))
                ),
                "position_error_p95_norm_m": float(
                    np.percentile(np.linalg.norm(position_error, axis=1), 95.0)
                ),
                "velocity_rmse_xyz_m_s": [
                    float(value)
                    for value in np.sqrt(np.mean(velocity_error**2, axis=0))
                ],
            }
        )

    position_error = actual_position[active_mask] - desired_position[active_mask]
    velocity_error = actual_velocity[active_mask] - desired_velocity[active_mask]
    active_debug_time = debug_time_array[active_mask]
    position_error_norm = np.linalg.norm(position_error, axis=1)
    quaternion_at_debug = interp_columns(
        debug_time_array,
        odom_time_array,
        quaternion,
    )
    rotation_at_debug = quaternion_to_rotation_matrix(quaternion_at_debug)
    tilt_at_debug_deg = np.degrees(
        np.arccos(np.clip(rotation_at_debug[:, 2, 2], -1.0, 1.0))
    )
    active_tilt_deg = tilt_at_debug_deg[active_mask]
    command_start = min(window[0] for window in lap_windows)

    def first_threshold_time(values: np.ndarray, threshold: float) -> float:
        indices = np.flatnonzero(values > threshold)
        return (
            float(active_debug_time[indices[0]] - command_start)
            if indices.size
            else float("nan")
        )

    quaternion_at_imu = interp_columns(imu_time_array, odom_time_array, quaternion)
    rotation_body_to_world = quaternion_to_rotation_matrix(quaternion_at_imu)
    velocity_at_imu = interp_columns(imu_time_array, odom_time_array, velocity)
    body_velocity = np.einsum(
        "nij,nj->ni", np.transpose(rotation_body_to_world, (0, 2, 1)), velocity_at_imu
    )
    omega_sum = (
        np.interp(imu_time_array, esc_time_array, rpm_sum)
        * RPM_TO_RAD_S
    )
    drag_base = -omega_sum[:, None] * body_velocity[:, :2] / parameters.mass
    dynamic_mask = (
        active_imu_mask
        & np.isfinite(omega_sum)
        & (omega_sum > 4.0 * 5000.0 * RPM_TO_RAD_S)
        & (np.linalg.norm(body_velocity[:, :2], axis=1) > 0.3)
    )
    fitted_drag = []

    for axis in range(2):
        coefficient, fit_metrics = fit_through_origin(
            drag_base[dynamic_mask, axis],
            imu_acceleration_array[dynamic_mask, axis],
        )
        current_metrics = finite_metrics(
            imu_acceleration_array[dynamic_mask, axis],
            parameters.rotor_drag_coefficient * drag_base[dynamic_mask, axis],
        )
        fitted_drag.append(
            {
                "axis": "x" if axis == 0 else "y",
                "effective_coefficient": coefficient,
                "effective_fit": fit_metrics,
                "configured_coefficient": parameters.rotor_drag_coefficient,
                "configured_model": current_metrics,
                "effective_to_configured_ratio": float(
                    coefficient / parameters.rotor_drag_coefficient
                ),
            }
        )

    mean_revolutions_per_second = (
        np.interp(imu_time_array, esc_time_array, rpm_sum) / 4.0 / 60.0
    )
    positive_axial_speed = np.maximum(body_velocity[:, 2], 0.0)
    advance_ratio = positive_axial_speed / np.maximum(
        mean_revolutions_per_second * parameters.propeller_diameter,
        1e-9,
    )
    advance_ratio = advance_ratio[dynamic_mask]

    metrics: Dict[str, object] = {
        "bag": str(spec.bag),
        "odometry_source": odom_source,
        "rates_hz": {
            "debugPx4ctrl": sample_rate(debug_time_array),
            "local_position_odom": sample_rate(odom_time_array),
            "imu": sample_rate(imu_time_array),
            "esc_status": sample_rate(esc_time_array),
        },
        "trajectory": {
            "lap_count": len(lap_metrics),
            "position_rmse_xyz_m": [
                float(value)
                for value in np.sqrt(np.mean(position_error**2, axis=0))
            ],
            "position_rmse_norm_m": float(
                np.sqrt(np.mean(np.sum(position_error**2, axis=1)))
            ),
            "position_error_p95_norm_m": float(
                np.percentile(position_error_norm, 95.0)
            ),
            "position_error_max_norm_m": float(np.max(position_error_norm)),
            "position_error_at_end_norm_m": float(position_error_norm[-1]),
            "time_to_position_error_1m_s": first_threshold_time(
                position_error_norm, 1.0
            ),
            "time_to_position_error_2m_s": first_threshold_time(
                position_error_norm, 2.0
            ),
            "time_to_position_error_5m_s": first_threshold_time(
                position_error_norm, 5.0
            ),
            "velocity_rmse_xyz_m_s": [
                float(value)
                for value in np.sqrt(np.mean(velocity_error**2, axis=0))
            ],
            "actual_tilt_deg": percentile_summary(active_tilt_deg),
            "time_to_tilt_over_60_deg_s": first_threshold_time(
                active_tilt_deg, 60.0
            ),
            "desired_speed": percentile_summary(
                np.linalg.norm(desired_velocity[active_mask], axis=1)
            ),
            "laps": lap_metrics,
        },
        "aerodynamics": {
            "body_lateral_speed_m_s": percentile_summary(
                np.linalg.norm(body_velocity[dynamic_mask, :2], axis=1)
            ),
            "advance_ratio_positive": percentile_summary(advance_ratio),
            "rotor_drag": fitted_drag,
            "note": (
                "The fitted coefficient is an effective rotor-plus-airframe value; "
                "the maneuver does not separately identify body drag."
            ),
        },
    }
    series = {
        "debug_elapsed": debug_time_array - debug_time_array[0],
        "desired_position": desired_position,
        "actual_position": actual_position,
        "active_mask": active_mask,
        "drag_base": drag_base,
        "imu_acceleration": imu_acceleration_array,
        "dynamic_mask": dynamic_mask,
    }
    plot_bag(spec, metrics, series, output_dir)
    return metrics, series


def plot_bag(
    spec: DatasetSpec,
    metrics: Dict[str, object],
    series: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    mask = series["active_mask"]
    elapsed = series["debug_elapsed"][mask]
    labels = ("x", "y", "z")

    for axis, label in enumerate(labels):
        axes[0, 0].plot(
            elapsed,
            series["desired_position"][mask, axis],
            linewidth=0.75,
            label=f"{label} desired",
        )
        axes[0, 0].plot(
            elapsed,
            series["actual_position"][mask, axis],
            linewidth=0.55,
            alpha=0.8,
            label=f"{label} actual",
        )

    axes[0, 0].set_xlabel("Time in recorded command windows (s)")
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 1].plot(
        series["desired_position"][mask, 0],
        series["desired_position"][mask, 1],
        linewidth=0.9,
        label="desired",
    )
    axes[0, 1].plot(
        series["actual_position"][mask, 0],
        series["actual_position"][mask, 1],
        linewidth=0.65,
        alpha=0.8,
        label="actual",
    )
    axes[0, 1].axis("equal")
    axes[0, 1].set_xlabel("x (m)")
    axes[0, 1].set_ylabel("y (m)")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()
    dynamic_mask = series["dynamic_mask"]
    stride = max(1, int(np.count_nonzero(dynamic_mask) / 15000))

    for axis in range(2):
        target_axis = axes[1, axis]
        x = series["drag_base"][dynamic_mask, axis][::stride]
        y = series["imu_acceleration"][dynamic_mask, axis][::stride]
        coefficient = metrics["aerodynamics"]["rotor_drag"][axis][
            "effective_coefficient"
        ]
        target_axis.scatter(x, y, s=2, alpha=0.18)
        limits = np.percentile(x[np.isfinite(x)], [1.0, 99.0])
        target_axis.plot(
            limits,
            coefficient * limits,
            "k--",
            linewidth=0.9,
            label=f"effective lambda={coefficient:.3g}",
        )
        target_axis.set_xlabel(r"$-\sum\omega V_\perp/m$")
        target_axis.set_ylabel(f"IMU {labels[axis]} specific force (m/s^2)")
        target_axis.grid(True, alpha=0.25)
        target_axis.legend()

    position_rmse = metrics["trajectory"]["position_rmse_norm_m"]
    figure.suptitle(
        f"{spec.name}: {metrics['trajectory']['lap_count']} laps, "
        f"position vector RMSE {position_rmse:.3f} m"
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"{spec.name}_bag.png", dpi=160)
    plt.close(figure)


def rating(value: float, high: float, medium: float, lower_is_better: bool = True) -> str:
    if not np.isfinite(value):
        return "不可评估"

    if lower_is_better:
        return "高" if value <= high else "中" if value <= medium else "低"

    return "高" if value >= high else "中" if value >= medium else "低"


def write_report(
    datasets: Sequence[DatasetSpec],
    metrics: Dict[str, Dict[str, object]],
    parameters: ModelParameters,
    output_dir: Path,
) -> None:
    calibration = metrics[datasets[0].name]
    lines = [
        "# LZF 仿真真实性评估",
        "",
        "本报告把当前实机主日志视为**标定集**。标定集误差只能说明模型能否复现用于建模的数据，"
        "不能单独证明外推真实性；`holdout` 数据才用于独立验证。",
        "",
        "## 当前模型参数",
        "",
        f"- 质量：`{parameters.mass:.3f} kg`",
        f"- 桨径：`{parameters.propeller_diameter:.3f} m`，静态 `CT={parameters.static_ct:.6f}`",
        f"- 固定空气密度：`{parameters.air_density:.3f} kg/m^3`",
        f"- 电机稳态模型：`RPM=min({parameters.motor_idle_rpm:.0f} + "
        f"{parameters.motor_loaded_kv:.0f} u V, {parameters.motor_no_load_kv:.0f} V)`",
        f"- 旋翼横向阻力系数：`{parameters.rotor_drag_coefficient:.3g}`",
        "",
        "## 数据概况",
        "",
        "| 数据集 | 角色 | 时长 | ESC 频率 | 电池频率 | 轨迹圈数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for spec in datasets:
        item = metrics[spec.name]
        bag = item.get("bag", {})
        lines.append(
            f"| {spec.name} | {spec.role} | {item['ulog']['duration_s']:.1f} s | "
            f"{item['ulog']['rates_hz']['esc_status']:.1f} Hz | "
            f"{item['ulog']['rates_hz']['battery_status']:.1f} Hz | "
            f"{bag.get('trajectory', {}).get('lap_count', '-')} |"
        )

    lines.extend(
        [
            "",
            "## 分系统结果",
            "",
            "| 数据集 | RPM 相对 RMSE | RPM 偏差 | z 轴比力 RMSE | 有效 CT 中位数 | 电压 RMSE | 电压 R² |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for spec in datasets:
        ulog_metrics = metrics[spec.name]["ulog"]
        motor = ulog_metrics["esc"]["motor_speed_model_pooled"]
        thrust = ulog_metrics["propeller"]["specific_thrust_model"]
        battery = ulog_metrics["battery"]["model"]
        effective_ct = ulog_metrics["propeller"]["effective_static_ct"]["median"]
        lines.append(
            f"| {spec.name} | {motor['relative_rmse_percent']:.2f}% | "
            f"{motor['bias']:.1f} rpm | {thrust['rmse']:.3f} m/s² | "
            f"{effective_ct:.6f} | {battery['rmse']:.3f} V | "
            f"{battery['r_squared']:.3f} |"
        )

    calibration_ulog = calibration["ulog"]
    motor_error = calibration_ulog["esc"]["motor_speed_model_pooled"][
        "relative_rmse_percent"
    ]
    thrust_error = calibration_ulog["propeller"]["specific_thrust_model"]["rmse"]
    battery_error = calibration_ulog["battery"]["model"]["rmse"]
    lines.extend(
        [
            "",
            "## 标定集判断",
            "",
            "| 子系统 | 当前可信度 | 依据与限制 |",
            "|---|---:|---|",
            f"| 电机命令/电压到稳态 RPM | {rating(motor_error, 3.0, 5.0)} | "
            f"标定集相对 RMSE `{motor_error:.2f}%`；该 RPM 模型来自同一日志，属于样本内结果。 |",
            f"| 低进动比轴向推力 | {rating(thrust_error, 0.5, 1.0)} | "
            f"机体系 z 轴比力 RMSE `{thrust_error:.3f} m/s²`；仍混有传感器、密度和机动耦合误差。 |",
            f"| 电池端电压 | {rating(battery_error, 0.15, 0.25)} | "
            f"RMSE `{battery_error:.3f} V`；参数由该日志辨识，必须看留出飞行。 |",
            "| 电流、功率、热 | 低 | 当前 `battery_status.current_a=-1`，没有电流、效率和温升模型。 |",
            "| 高频电机/ESC动态 | 低 | SITL 回传 250 Hz 且几乎无延迟/丢包，实机 ESC 仅约 50 Hz。 |",
            "|惯量与结构振动 | 不可评估 | 普通轨迹不能可靠解耦惯量、控制器和气动力；需要逐轴 chirp/阶跃。 |",
        ]
    )

    if "bag" in calibration:
        bag = calibration["bag"]
        drag = bag["aerodynamics"]["rotor_drag"]
        trajectory = bag["trajectory"]
        lines.extend(
            [
                "",
                "## 实机轨迹与横向气动",
                "",
                f"- 记录包含 `{trajectory['lap_count']}` 圈相同轨迹；位置向量 RMSE "
                f"`{trajectory['position_rmse_norm_m']:.3f} m`，95% 误差 "
                f"`{trajectory['position_error_p95_norm_m']:.3f} m`。",
                f"- 正进动比 95% 为 `{bag['aerodynamics']['advance_ratio_positive']['p95']:.3f}`，"
                f"最大 `{bag['aerodynamics']['advance_ratio_positive']['max']:.3f}`；"
                "这份数据只能验证低进动比范围。",
                f"- x/y 方向辨识出的有效阻力系数分别为 "
                f"`{drag[0]['effective_coefficient']:.3g}` 和 "
                f"`{drag[1]['effective_coefficient']:.3g}`，是当前配置的 "
                f"`{drag[0]['effective_to_configured_ratio']:.2f}x` / "
                f"`{drag[1]['effective_to_configured_ratio']:.2f}x`。",
                "- 上述系数包含机身阻力，当前轨迹无法把桨盘阻力与机身阻力独立辨识；"
                "不能直接把有效系数全部写回 `rotorDragCoefficient`。",
            ]
        )

    lines.extend(
        [
            "",
            "## 下一轮验收",
            "",
            "- RPM：留出飞行相对 RMSE `<=3%`，且四电机偏差分别 `<=300 rpm`。",
            "- 推力：留出飞行机体系 z 轴比力偏差 `<=3%`、RMSE `<=0.5 m/s²`。",
            "- 电池：留出飞行电压 RMSE `<=0.25 V`、R² `>=0.95`。",
            "- 气动：分别加入机身阻力与旋翼阻力后，在未参与拟合的高速轨迹上验证。",
            "- 动态：用 roll/pitch/yaw 单轴 chirp 辨识惯量和延迟，用高频油门阶跃辨识 ESC/电机动态。",
            "",
            "## 输出文件",
            "",
            "- `metrics.json`：全部机器可读指标。",
            "- `*_ulog.png`：RPM、电压和轴向推力诊断。",
            "- `*_bag.png`：逐圈轨迹与横向阻力诊断。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def main() -> int:
    args = parse_args()
    parameters = ModelParameters(
        mass=args.mass,
        air_density=args.rho,
        static_ct=args.static_ct,
        rotor_drag_coefficient=args.drag_coefficient,
    )
    datasets = [
        DatasetSpec(
            name="real_calibration",
            ulog=args.real_ulog,
            bag=args.real_bag,
            role="calibration",
        )
    ]

    try:
        datasets.extend(parse_dataset(value, "holdout") for value in args.holdout)
        datasets.extend(parse_dataset(value, "simulation") for value in args.sim)
        bag_odom_topics = parse_bag_odom_topics(args.bag_odom_topic)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    dataset_names = {spec.name for spec in datasets}
    unknown_odom_overrides = sorted(set(bag_odom_topics) - dataset_names)

    if unknown_odom_overrides:
        print(
            "error: bag odometry override references unknown dataset(s): "
            + ", ".join(unknown_odom_overrides),
            file=sys.stderr,
        )
        return 2

    for spec in datasets:
        if not spec.ulog.is_file():
            print(f"error: missing ULog: {spec.ulog}", file=sys.stderr)
            return 2

        if spec.bag is not None and not spec.bag.is_file():
            print(f"error: missing ROS bag: {spec.bag}", file=sys.stderr)
            return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, Dict[str, object]] = {}

    try:
        for spec in datasets:
            print(f"[{spec.name}] analyzing ULog {spec.ulog}")
            ulog_metrics, ulog_series = analyze_ulog(
                spec, parameters, args.output_dir
            )
            dataset_metrics: Dict[str, object] = {"ulog": ulog_metrics}
            del ulog_series
            gc.collect()

            if spec.bag is not None:
                print(f"[{spec.name}] analyzing ROS bag {spec.bag}")
                bag_metrics, bag_series = analyze_bag(
                    spec,
                    parameters,
                    args.output_dir,
                    bag_odom_topics.get(spec.name),
                )
                dataset_metrics["bag"] = bag_metrics
                del bag_series
                gc.collect()

            all_metrics[spec.name] = dataset_metrics

        payload = {
            "model_parameters": asdict(parameters),
            "datasets": all_metrics,
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_report(datasets, all_metrics, parameters, args.output_dir)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"metrics: {args.output_dir / 'metrics.json'}")
    print(f"report:  {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
