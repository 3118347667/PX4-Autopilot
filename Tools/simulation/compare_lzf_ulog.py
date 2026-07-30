#!/usr/bin/env python3

"""Compare the Gazebo Classic LZF model against one or more real PX4 ULogs."""

import argparse
import json
import math
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog
from scipy.spatial.transform import Rotation


GRAVITY = 9.80665
RPM_TO_RAD_PER_SECOND = 2.0 * math.pi / 60.0
TOPICS = [
    "actuator_armed",
    "actuator_motors",
    "battery_status",
    "esc_status",
    "vehicle_acceleration",
    "vehicle_angular_velocity",
    "vehicle_attitude",
    "vehicle_land_detected",
    "vehicle_local_position",
    "vehicle_status",
    "wind",
]
CONTROL_PARAMETERS = [
    "MC_ROLLRATE_P",
    "MC_ROLLRATE_I",
    "MC_ROLLRATE_D",
    "MC_PITCHRATE_P",
    "MC_PITCHRATE_I",
    "MC_PITCHRATE_D",
    "MC_YAWRATE_P",
    "MC_YAWRATE_I",
    "MC_YAWRATE_D",
    "MC_ROLL_P",
    "MC_PITCH_P",
    "MC_YAW_P",
    "MPC_THR_HOVER",
]
SPEED_BINS = [
    (0.0, 5.0, "0-5"),
    (5.0, 15.0, "5-15"),
    (15.0, 30.0, "15-30"),
    (30.0, math.inf, "30+"),
]
ADVANCE_RATIO_BINS = [
    (0.0, 0.1, "0-0.1"),
    (0.1, 0.2, "0.1-0.2"),
    (0.2, 0.3, "0.2-0.3"),
    (0.3, 0.5, "0.3-0.5"),
    (0.5, math.inf, "0.5+"),
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current LZF battery, motor-speed, hover, propeller, "
            "and controller configuration against real ULogs."
        )
    )
    parser.add_argument(
        "ulog",
        type=Path,
        nargs="+",
        help="ULogs to compare; the first is treated as the model-source log",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "build" / "lzf_ulog_validation",
    )
    parser.add_argument(
        "--airframe",
        type=Path,
        default=(
            repo_root
            / "ROMFS/px4fmu_common/init.d-posix/airframes/"
            "10020_gazebo-classic_lzf"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            repo_root
            / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/"
            "models/lzf/lzf.sdf.jinja"
        ),
    )
    parser.add_argument(
        "--parameter-metadata",
        type=Path,
        default=repo_root / "build/px4_sitl_default/parameters.json",
    )
    parser.add_argument("--mass", type=float, default=1.326, help="vehicle mass in kg")
    parser.add_argument(
        "--command-lag-ms",
        type=float,
        default=40.0,
        help="fixed actuator-command alignment used by the quasi-steady RPM check",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=30000,
        help="maximum scatter points per log and plot",
    )
    parser.add_argument(
        "--bag-metrics",
        action="append",
        type=Path,
        default=[],
        help=(
            "optional analyze_lzf_fidelity.py bag-metrics JSON, repeated once "
            "per ULog in the same order"
        ),
    )
    return parser.parse_args()


def parse_number(value):
    value = value.strip().strip("\"'")

    try:
        return float(value)
    except ValueError:
        return value


def parse_airframe(path):
    parameters = {}

    for line in path.read_text().splitlines():
        match = re.match(r"\s*param\s+set-default\s+(\S+)\s+([^#]+)", line)

        if match:
            parameters[match.group(1)] = parse_number(match.group(2))

    return parameters


def parse_tag(text, tag, default=None):
    match = re.search(rf"<{tag}>\s*([^<]+?)\s*</{tag}>", text)

    if match:
        return parse_number(match.group(1))

    if default is not None:
        return default

    raise RuntimeError(f"missing <{tag}> in LZF model")


def parse_jinja_list(text, variable):
    match = re.search(
        rf"{{%\s*set\s+{re.escape(variable)}\s*=\s*['\"]([^'\"]+)['\"]\s*%}}",
        text,
    )

    if not match:
        raise RuntimeError(f"missing Jinja list {variable}")

    return np.fromstring(match.group(1), sep=" ", dtype=float)


def parse_model(path, mass):
    text = path.read_text()
    rotor_positions_flu = []

    for index in range(4):
        match = re.search(
            rf"<link\s+name=['\"]rotor_{index}['\"]>\s*"
            r"<pose>\s*([^<]+?)\s*</pose>",
            text,
            flags=re.DOTALL,
        )

        if not match:
            raise RuntimeError(f"missing rotor_{index} pose")

        pose = np.fromstring(match.group(1), sep=" ", dtype=float)
        rotor_positions_flu.append(pose[:3])

    base_match = re.search(
        r"<link\s+name=['\"]base_link['\"]>.*?<inertia>\s*"
        r"<ixx>([^<]+)</ixx>\s*<ixy>([^<]+)</ixy>\s*"
        r"<ixz>([^<]+)</ixz>\s*<iyy>([^<]+)</iyy>\s*"
        r"<iyz>([^<]+)</iyz>\s*<izz>([^<]+)</izz>",
        text,
        flags=re.DOTALL,
    )
    inertia = None

    if base_match:
        values = [float(value) for value in base_match.groups()]
        inertia = {
            "ixx": values[0],
            "ixy": values[1],
            "ixz": values[2],
            "iyy": values[3],
            "iyz": values[4],
            "izz": values[5],
        }

    # Gazebo uses FLU while PX4 and the ULog use FRD.
    rotor_positions_frd = np.asarray(rotor_positions_flu, dtype=float)
    rotor_positions_frd[:, 1:] *= -1.0
    advance_ratio = parse_jinja_list(text, "da4052_advance_ratio")
    thrust_coefficient = parse_jinja_list(text, "da4052_thrust_coefficient")
    power_coefficient = parse_jinja_list(text, "da4052_power_coefficient")

    if not (
        advance_ratio.size
        == thrust_coefficient.size
        == power_coefficient.size
    ):
        raise RuntimeError("LZF propeller coefficient tables have different lengths")

    return {
        "mass_kg": mass,
        "inertia_kg_m2": inertia,
        "rotor_positions_frd_m": rotor_positions_frd,
        "diameter_m": float(parse_tag(text, "propellerDiameter")),
        "air_density_kg_m3": float(parse_tag(text, "airDensity")),
        "advance_ratio": advance_ratio,
        "thrust_coefficient": thrust_coefficient,
        "power_coefficient": power_coefficient,
        "rotor_drag_coefficient": float(parse_tag(text, "rotorDragCoefficient")),
        "rolling_moment_coefficient": float(
            parse_tag(text, "rollingMomentCoefficient")
        ),
        "time_constant_up_s": float(parse_tag(text, "timeConstantUp")),
        "time_constant_down_s": float(parse_tag(text, "timeConstantDown")),
        "max_rot_velocity_rad_s": float(parse_tag(text, "maxRotVelocity")),
        "motor_idle_rpm": float(parse_tag(text, "motorIdleRpm")),
        "motor_loaded_kv": float(parse_tag(text, "motorLoadedKv")),
        "motor_kv": float(parse_tag(text, "motorKv")),
        "motor_voltage_min_v": float(parse_tag(text, "motorSpeedVoltageMin")),
        "motor_voltage_max_v": float(parse_tag(text, "motorSpeedVoltageMax")),
    }


def load_parameter_defaults(path):
    if not path.exists():
        return {}

    metadata = json.loads(path.read_text())
    defaults = {}

    for parameter in metadata.get("parameters", []):
        value = parameter.get("default")

        if isinstance(value, (int, float)):
            defaults[parameter["name"]] = float(value)

    return defaults


def get_topic(ulog, name, required=True):
    matches = [
        item for item in ulog.data_list if item.name == name and item.multi_id == 0
    ]

    if matches:
        return matches[0].data

    if required:
        raise RuntimeError(f"ULog does not contain {name} instance 0")

    return None


def topic_time(topic):
    return np.asarray(topic["timestamp"], dtype=float) * 1e-6


def topic_rate_hz(topic):
    time = topic_time(topic)

    if time.size < 2 or time[-1] <= time[0]:
        return math.nan

    return (time.size - 1) / (time[-1] - time[0])


def interpolate(time, source_time, source_value):
    source_value = np.asarray(source_value, dtype=float)
    finite = np.isfinite(source_time) & np.isfinite(source_value)

    if np.count_nonzero(finite) < 2:
        return np.full(time.shape, np.nan)

    return np.interp(
        time,
        np.asarray(source_time)[finite],
        source_value[finite],
        left=np.nan,
        right=np.nan,
    )


def interpolate_columns(time, source_time, columns):
    return np.column_stack(
        [interpolate(time, source_time, np.asarray(column)) for column in columns]
    )


def nearest_sample_age(time, source_time):
    source_time = np.asarray(source_time)
    indices = np.searchsorted(source_time, time)
    lower = np.clip(indices - 1, 0, source_time.size - 1)
    upper = np.clip(indices, 0, source_time.size - 1)
    return np.minimum(
        np.abs(time - source_time[lower]), np.abs(time - source_time[upper])
    )


def interpolate_quaternion(time, attitude):
    attitude_time = topic_time(attitude)
    quaternion = interpolate_columns(
        time,
        attitude_time,
        [attitude[f"q[{index}]"] for index in range(4)],
    )
    norm = np.linalg.norm(quaternion, axis=1)
    valid = np.isfinite(norm) & (norm > 1e-6)
    quaternion[valid] /= norm[valid, None]
    quaternion[~valid] = np.array([1.0, 0.0, 0.0, 0.0])
    # scipy expects x, y, z, w. PX4 stores w, x, y, z and maps body to NED.
    return Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]), valid


def dominant_actuator_function(esc, slot):
    values = np.asarray(esc[f"esc[{slot}].actuator_function"], dtype=int)
    values = values[(values >= 101) & (values <= 112)]

    if values.size == 0:
        return None

    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def semantic_esc_rpm(esc):
    esc_time = topic_time(esc)
    rpm = np.full((esc_time.size, 4), np.nan)
    mapping = {}

    for slot in range(8):
        field = f"esc[{slot}].esc_rpm"

        if field not in esc:
            continue

        actuator_function = dominant_actuator_function(esc, slot)

        if actuator_function is None or not 101 <= actuator_function <= 104:
            continue

        motor_index = actuator_function - 101

        if np.any(np.isfinite(rpm[:, motor_index])):
            raise RuntimeError(
                f"duplicate ESC actuator function {actuator_function} in ULog"
            )

        rpm[:, motor_index] = np.asarray(esc[field], dtype=float)
        mapping[slot] = {
            "actuator_function": actuator_function,
            "motor_index": motor_index,
        }

    missing = np.flatnonzero(~np.any(np.isfinite(rpm), axis=0))

    if missing.size:
        raise RuntimeError(
            "ULog is missing RPM for canonical motor indices "
            + ", ".join(str(index) for index in missing)
        )

    return esc_time, rpm, mapping


def metric(measured, predicted, mask=None):
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(measured) & np.isfinite(predicted)

    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    if np.count_nonzero(valid) == 0:
        return {
            "samples": 0,
            "rmse": math.nan,
            "mae": math.nan,
            "bias_pred_minus_meas": math.nan,
            "p95_abs_error": math.nan,
            "r_squared": math.nan,
        }

    residual = predicted[valid] - measured[valid]
    total_variance = np.sum((measured[valid] - np.mean(measured[valid])) ** 2)
    r_squared = math.nan

    if total_variance > 0.0:
        r_squared = 1.0 - np.sum(residual**2) / total_variance

    return {
        "samples": int(np.count_nonzero(valid)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_meas": float(np.mean(residual)),
        "p95_abs_error": float(np.percentile(np.abs(residual), 95.0)),
        "r_squared": float(r_squared),
    }


def interpolate_state_flags(time, armed, landed):
    armed_value = interpolate(
        time, topic_time(armed), np.asarray(armed["armed"], dtype=float)
    )
    landed_value = interpolate(
        time, topic_time(landed), np.asarray(landed["landed"], dtype=float)
    )
    return armed_value > 0.5, landed_value > 0.5


def analyze_battery(data, model, airframe):
    battery = data["topics"]["battery_status"]
    armed = data["topics"]["actuator_armed"]
    battery_time = topic_time(battery)
    esc_time = data["esc_time"]
    rpm_at_battery = interpolate_columns(
        battery_time, esc_time, data["rpm"].T
    )
    fresh = nearest_sample_age(battery_time, esc_time) <= 0.1
    armed_at_battery = interpolate(
        battery_time, topic_time(armed), np.asarray(armed["armed"], dtype=float)
    ) > 0.5
    omega_normalized = (
        np.maximum(rpm_at_battery, 0.0) * RPM_TO_RAD_PER_SECOND / 1000.0
    )
    load = np.mean(omega_normalized**3, axis=1)
    valid_load = fresh & np.all(np.isfinite(rpm_at_battery), axis=1)
    effective_load = np.where(armed_at_battery & valid_load, load, 0.0)
    dt = np.diff(battery_time, prepend=battery_time[0])
    dt = np.maximum(dt, 0.0)

    full_voltage = float(airframe["BAT1_N_CELLS"]) * float(
        airframe["BAT1_V_CHARGED"]
    )
    empty_voltage = float(airframe["BAT1_N_CELLS"]) * float(
        airframe["BAT1_V_EMPTY"]
    )
    drain = float(airframe["SIM_BAT_DRAIN"])
    reference_load = float(airframe["SIM_BAT_L_REF"])
    sag_instant = float(airframe["SIM_BAT_SAG_I"])
    sag_polarization = float(airframe["SIM_BAT_SAG_P"])
    tau = float(airframe["SIM_BAT_TAU"])
    voltage_floor = float(airframe["SIM_BAT_V_FLOOR"])

    soc = np.ones(battery_time.size)
    polarization = np.zeros(battery_time.size)

    for index in range(1, battery_time.size):
        alpha = -math.expm1(-dt[index] / tau)
        polarization[index] = polarization[index - 1] + alpha * (
            effective_load[index] - polarization[index - 1]
        )
        soc[index] = max(
            0.0,
            soc[index - 1]
            - effective_load[index] / reference_load * dt[index] / drain,
        )

    open_circuit_voltage = empty_voltage + (full_voltage - empty_voltage) * soc
    prediction = np.clip(
        open_circuit_voltage
        - sag_instant * effective_load
        - sag_polarization * polarization,
        voltage_floor,
        full_voltage,
    )
    measured = np.asarray(battery["voltage_v"], dtype=float)
    evaluation_mask = (
        armed_at_battery
        & valid_load
        & np.isfinite(measured)
        & (load >= 0.05)
    )
    result_metric = metric(measured, prediction, evaluation_mask)

    if np.any(evaluation_mask):
        indices = np.flatnonzero(evaluation_mask)
        endpoints = {
            "first_measured_v": float(measured[indices[0]]),
            "first_predicted_v": float(prediction[indices[0]]),
            "last_measured_v": float(measured[indices[-1]]),
            "last_predicted_v": float(prediction[indices[-1]]),
        }

    else:
        endpoints = {}

    return {
        "metrics": result_metric,
        "endpoints": endpoints,
        "final_soc": float(soc[-1]),
        "_series": {
            "time": battery_time,
            "measured": measured,
            "predicted": prediction,
            "open_circuit": open_circuit_voltage,
            "load": load,
            "polarization": polarization,
            "mask": evaluation_mask,
        },
    }


def analyze_motor_speed(data, model, command_lag_ms):
    actuator_motors = data["topics"]["actuator_motors"]
    battery = data["topics"]["battery_status"]
    armed = data["topics"]["actuator_armed"]
    esc_time = data["esc_time"]
    control_time = topic_time(actuator_motors)
    control = interpolate_columns(
        esc_time - command_lag_ms * 1e-3,
        control_time,
        [actuator_motors[f"control[{index}]"] for index in range(4)],
    )
    voltage = interpolate(
        esc_time, topic_time(battery), np.asarray(battery["voltage_v"], dtype=float)
    )
    voltage = np.clip(
        voltage, model["motor_voltage_min_v"], model["motor_voltage_max_v"]
    )
    target_rpm = (
        model["motor_idle_rpm"]
        + model["motor_loaded_kv"] * np.clip(control, 0.0, 1.0) * voltage[:, None]
    )
    target_rpm = np.minimum(target_rpm, model["motor_kv"] * voltage[:, None])
    target_rpm = np.maximum(target_rpm, 0.0)
    armed_at_esc = interpolate(
        esc_time, topic_time(armed), np.asarray(armed["armed"], dtype=float)
    ) > 0.5
    valid = (
        armed_at_esc[:, None]
        & np.isfinite(control)
        & np.isfinite(voltage[:, None])
        & np.isfinite(data["rpm"])
        & (data["rpm"] > 1000.0)
    )
    pooled = metric(data["rpm"], target_rpm, valid)
    per_motor = []

    for index in range(4):
        per_motor.append(metric(data["rpm"][:, index], target_rpm[:, index], valid[:, index]))

    return {
        "command_alignment_ms": command_lag_ms,
        "metrics_pooled": pooled,
        "metrics_per_motor": per_motor,
        "actuator_motors_rate_hz": topic_rate_hz(actuator_motors),
        "esc_status_rate_hz": topic_rate_hz(data["topics"]["esc_status"]),
        "dynamic_time_constants_validated": False,
        "_series": {
            "time": esc_time,
            "measured": data["rpm"],
            "predicted": target_rpm,
            "control": control,
            "voltage": voltage,
            "mask": valid,
        },
    }


def common_flight_state(data, model):
    acceleration = data["topics"]["vehicle_acceleration"]
    local_position = data["topics"]["vehicle_local_position"]
    angular_velocity = data["topics"]["vehicle_angular_velocity"]
    time = topic_time(acceleration)
    rotation, quaternion_valid = interpolate_quaternion(
        time, data["topics"]["vehicle_attitude"]
    )
    local_time = topic_time(local_position)
    velocity_ned = interpolate_columns(
        time,
        local_time,
        [
            local_position["vx"],
            local_position["vy"],
            local_position["vz"],
        ],
    )
    position_z = interpolate(time, local_time, local_position["z"])
    velocity_body = rotation.inv().apply(velocity_ned)
    rate_body = interpolate_columns(
        time,
        topic_time(angular_velocity),
        [angular_velocity[f"xyz[{index}]"] for index in range(3)],
    )
    rpm = interpolate_columns(time, data["esc_time"], data["rpm"].T)
    rpm_fresh = nearest_sample_age(time, data["esc_time"]) <= 0.1
    armed, landed = interpolate_state_flags(
        time, data["topics"]["actuator_armed"], data["topics"]["vehicle_land_detected"]
    )
    acceleration_body = np.column_stack(
        [acceleration[f"xyz[{index}]"] for index in range(3)]
    )
    body_z_ned = rotation.apply(np.tile([0.0, 0.0, 1.0], (time.size, 1)))
    tilt_rad = np.arccos(np.clip(body_z_ned[:, 2], -1.0, 1.0))
    finite = (
        quaternion_valid
        & rpm_fresh
        & np.all(np.isfinite(rpm), axis=1)
        & np.all(np.isfinite(velocity_body), axis=1)
        & np.all(np.isfinite(rate_body), axis=1)
        & np.all(np.isfinite(acceleration_body), axis=1)
    )

    return {
        "time": time,
        "velocity_ned": velocity_ned,
        "velocity_body": velocity_body,
        "position_z": position_z,
        "rate_body": rate_body,
        "rpm": rpm,
        "armed": armed,
        "landed": landed,
        "acceleration_body": acceleration_body,
        "tilt_rad": tilt_rad,
        "finite": finite,
        "mass_kg": model["mass_kg"],
    }


def analyze_hover(state, model):
    horizontal_speed = np.linalg.norm(state["velocity_ned"][:, :2], axis=1)
    hover_mask = (
        state["finite"]
        & state["armed"]
        & ~state["landed"]
        & (state["position_z"] < -0.5)
        & (horizontal_speed < 1.0)
        & (np.abs(state["velocity_ned"][:, 2]) < 0.4)
        & (state["tilt_rad"] < math.radians(15.0))
    )
    mean_rpm = np.mean(state["rpm"], axis=1)
    revolutions_per_second = state["rpm"] / 60.0
    sum_n_squared = np.sum(revolutions_per_second**2, axis=1)
    denominator = (
        model["air_density_kg_m3"]
        * model["diameter_m"] ** 4
        * sum_n_squared
    )
    effective_ct = np.full(state["time"].shape, np.nan)
    positive = denominator > 0.0
    effective_ct[positive] = model["mass_kg"] * GRAVITY / denominator[positive]
    model_thrust = (
        model["thrust_coefficient"][0]
        * model["air_density_kg_m3"]
        * model["diameter_m"] ** 4
        * sum_n_squared
    )
    predicted_acceleration_z = -model_thrust / model["mass_kg"]
    z_metric = metric(
        state["acceleration_body"][:, 2], predicted_acceleration_z, hover_mask
    )
    static_hover_n = math.sqrt(
        model["mass_kg"]
        * GRAVITY
        / (
            4.0
            * model["thrust_coefficient"][0]
            * model["air_density_kg_m3"]
            * model["diameter_m"] ** 4
        )
    )
    model_hover_rpm = static_hover_n * 60.0
    count = np.count_nonzero(hover_mask)
    result = {
        "samples": int(count),
        "duration_s_approx": float(
            count / max(1e-6, 1.0 / np.median(np.diff(state["time"])))
        ),
        "model_static_hover_rpm": float(model_hover_rpm),
        "model_static_ct": float(model["thrust_coefficient"][0]),
        "body_z_acceleration_metrics": z_metric,
    }

    if count:
        measured_hover_rpm = mean_rpm[hover_mask]
        ct = effective_ct[hover_mask]
        result.update(
            {
                "measured_rpm_median": float(np.median(measured_hover_rpm)),
                "measured_rpm_p05": float(np.percentile(measured_hover_rpm, 5.0)),
                "measured_rpm_p95": float(np.percentile(measured_hover_rpm, 95.0)),
                "model_hover_rpm_error_percent": float(
                    100.0
                    * (model_hover_rpm - np.median(measured_hover_rpm))
                    / np.median(measured_hover_rpm)
                ),
                "effective_ct_median": float(np.median(ct)),
                "effective_ct_p05": float(np.percentile(ct, 5.0)),
                "effective_ct_p95": float(np.percentile(ct, 95.0)),
                "model_ct_error_percent": float(
                    100.0
                    * (model["thrust_coefficient"][0] - np.median(ct))
                    / np.median(ct)
                ),
            }
        )

    return result


def binned_axis_metrics(measured, predicted, value, mask, bins):
    output = []

    for lower, upper, label in bins:
        bin_mask = mask & (value >= lower) & (value < upper)
        row = {"bin": label, "lower": lower, "upper": upper}
        row.update(metric(measured, predicted, bin_mask))
        output.append(row)

    return output


def analyze_force_model(state, model):
    sample_count = state["time"].size
    total_force = np.zeros((sample_count, 3))
    advance_ratio = np.full((sample_count, 4), np.nan)
    thrust = np.zeros((sample_count, 4))
    drag_basis = np.zeros((sample_count, 4, 3))
    axis = np.array([0.0, 0.0, -1.0])

    for motor_index in range(4):
        omega = np.maximum(state["rpm"][:, motor_index], 0.0) * RPM_TO_RAD_PER_SECOND
        revolutions_per_second = omega / (2.0 * math.pi)
        local_velocity = state["velocity_body"] + np.cross(
            state["rate_body"],
            np.tile(model["rotor_positions_frd_m"][motor_index], (sample_count, 1)),
        )
        axial_airspeed = np.maximum(0.0, local_velocity @ axis)
        valid_speed = revolutions_per_second > 1e-6
        advance_ratio[valid_speed, motor_index] = axial_airspeed[valid_speed] / (
            revolutions_per_second[valid_speed] * model["diameter_m"]
        )
        coefficient = np.interp(
            np.nan_to_num(advance_ratio[:, motor_index], nan=0.0),
            model["advance_ratio"],
            model["thrust_coefficient"],
        )
        thrust[:, motor_index] = (
            coefficient
            * model["air_density_kg_m3"]
            * revolutions_per_second**2
            * model["diameter_m"] ** 4
        )
        total_force[:, 2] -= thrust[:, motor_index]
        velocity_perpendicular = local_velocity.copy()
        velocity_perpendicular[:, 2] = 0.0
        drag_basis[:, motor_index] = -omega[:, None] * velocity_perpendicular
        total_force += (
            model["rotor_drag_coefficient"] * drag_basis[:, motor_index]
        )

    predicted_acceleration = total_force / model["mass_kg"]
    speed = np.linalg.norm(state["velocity_ned"], axis=1)
    valid_advance_ratio_count = np.sum(np.isfinite(advance_ratio), axis=1)
    mean_advance_ratio = np.divide(
        np.nansum(advance_ratio, axis=1),
        valid_advance_ratio_count,
        out=np.full(sample_count, np.nan),
        where=valid_advance_ratio_count > 0,
    )
    flight_mask = (
        state["finite"]
        & state["armed"]
        & ~state["landed"]
        & (np.mean(state["rpm"], axis=1) > 5000.0)
    )
    axis_metrics = {
        axis_name: metric(
            state["acceleration_body"][:, index],
            predicted_acceleration[:, index],
            flight_mask,
        )
        for index, axis_name in enumerate(["x", "y", "z"])
    }
    speed_bins = binned_axis_metrics(
        state["acceleration_body"][:, 2],
        predicted_acceleration[:, 2],
        speed,
        flight_mask,
        SPEED_BINS,
    )
    advance_ratio_bins = binned_axis_metrics(
        state["acceleration_body"][:, 2],
        predicted_acceleration[:, 2],
        mean_advance_ratio,
        flight_mask,
        ADVANCE_RATIO_BINS,
    )

    drag_x = np.sum(drag_basis[:, :, 0], axis=1) / model["mass_kg"]
    drag_y = np.sum(drag_basis[:, :, 1], axis=1) / model["mass_kg"]
    drag_fit_mask = flight_mask & (speed > 2.0)
    design = np.concatenate([drag_x[drag_fit_mask], drag_y[drag_fit_mask]])
    measured = np.concatenate(
        [
            state["acceleration_body"][drag_fit_mask, 0],
            state["acceleration_body"][drag_fit_mask, 1],
        ]
    )
    finite_drag = np.isfinite(design) & np.isfinite(measured)
    denominator = float(np.dot(design[finite_drag], design[finite_drag]))
    effective_drag = math.nan

    if denominator > 0.0:
        effective_drag = max(
            0.0,
            float(np.dot(design[finite_drag], measured[finite_drag]) / denominator),
        )

    table_limit = model["advance_ratio"][-1]
    beyond_table = flight_mask & (mean_advance_ratio >= table_limit)
    negative_thrust = flight_mask & (np.sum(thrust, axis=1) < 0.0)
    return {
        "axis_metrics": axis_metrics,
        "body_z_metrics_by_ground_speed_m_s": speed_bins,
        "body_z_metrics_by_mean_advance_ratio": advance_ratio_bins,
        "mean_advance_ratio_max": float(np.nanmax(mean_advance_ratio[flight_mask])),
        "samples_at_or_beyond_prop_table_percent": float(
            100.0 * np.count_nonzero(beyond_table) / np.count_nonzero(flight_mask)
        ),
        "samples_with_negative_total_prop_thrust_percent": float(
            100.0 * np.count_nonzero(negative_thrust) / np.count_nonzero(flight_mask)
        ),
        "configured_rotor_drag_coefficient": model["rotor_drag_coefficient"],
        "lumped_zero_wind_drag_fit": effective_drag,
        "wind_measurement_used": False,
        "_series": {
            "time": state["time"],
            "speed": speed,
            "mean_advance_ratio": mean_advance_ratio,
            "measured": state["acceleration_body"],
            "predicted": predicted_acceleration,
            "mask": flight_mask,
        },
    }


def normalized_quad_mixer(parameters):
    effectiveness = np.zeros((4, 4))

    for index in range(4):
        position = np.array(
            [
                float(parameters[f"CA_ROTOR{index}_PX"]),
                float(parameters[f"CA_ROTOR{index}_PY"]),
                float(parameters[f"CA_ROTOR{index}_PZ"]),
            ]
        )
        axis = np.array(
            [
                float(parameters.get(f"CA_ROTOR{index}_AX", 0.0)),
                float(parameters.get(f"CA_ROTOR{index}_AY", 0.0)),
                float(parameters.get(f"CA_ROTOR{index}_AZ", -1.0)),
            ]
        )
        ct = float(parameters.get(f"CA_ROTOR{index}_CT", 6.5))
        km = float(parameters[f"CA_ROTOR{index}_KM"])
        thrust = ct * axis
        moment = ct * np.cross(position, axis) - ct * km * axis
        effectiveness[:3, index] = moment
        effectiveness[3, index] = thrust[2]

    mixer = np.linalg.pinv(effectiveness)
    roll_nonzero = max(1, np.count_nonzero(np.abs(mixer[:, 0]) > 1e-3))
    pitch_nonzero = max(1, np.count_nonzero(np.abs(mixer[:, 1]) > 1e-3))
    roll_scale = math.sqrt(
        np.dot(mixer[:, 0], mixer[:, 0]) / (roll_nonzero / 2.0)
    )
    pitch_scale = math.sqrt(
        np.dot(mixer[:, 1], mixer[:, 1]) / (pitch_nonzero / 2.0)
    )
    rp_scale = max(roll_scale, pitch_scale)

    if rp_scale > 0.0:
        mixer[:, 0] /= rp_scale
        mixer[:, 1] /= rp_scale

    yaw_scale = np.max(mixer[:, 2])

    if yaw_scale > 0.0:
        mixer[:, 2] /= yaw_scale

    thrust_scale = np.mean(np.abs(mixer[:, 3]))

    if thrust_scale > 0.0:
        mixer[:, 3] /= thrust_scale

    mixer[np.abs(mixer) < 1e-3] = 0.0
    return mixer


def analyze_configuration(data, current_parameters):
    real_parameters = data["initial_parameters"]
    controller_rows = []

    for name in CONTROL_PARAMETERS:
        if name not in real_parameters or name not in current_parameters:
            continue

        real_value = float(real_parameters[name])
        current_value = float(current_parameters[name])
        relative_difference = math.nan

        if abs(real_value) > 1e-9:
            relative_difference = 100.0 * (current_value - real_value) / real_value

        controller_rows.append(
            {
                "parameter": name,
                "real_ulog": real_value,
                "current_lzf": current_value,
                "relative_difference_percent": relative_difference,
                "matches": math.isclose(
                    real_value, current_value, rel_tol=1e-5, abs_tol=1e-6
                ),
            }
        )

    real_geometry = dict(current_parameters)

    for index in range(4):
        for suffix in ["PX", "PY", "PZ", "KM", "AX", "AY", "AZ", "CT"]:
            name = f"CA_ROTOR{index}_{suffix}"

            if name in real_parameters:
                real_geometry[name] = float(real_parameters[name])

    current_mixer = normalized_quad_mixer(current_parameters)
    real_mixer = normalized_quad_mixer(real_geometry)
    current_x = max(
        abs(float(current_parameters[f"CA_ROTOR{index}_PX"])) for index in range(4)
    )
    current_y = max(
        abs(float(current_parameters[f"CA_ROTOR{index}_PY"])) for index in range(4)
    )
    real_x = max(
        abs(float(real_geometry[f"CA_ROTOR{index}_PX"])) for index in range(4)
    )
    real_y = max(
        abs(float(real_geometry[f"CA_ROTOR{index}_PY"])) for index in range(4)
    )
    return {
        "controller_parameters": controller_rows,
        "controller_parameter_mismatch_count": sum(
            not row["matches"] for row in controller_rows
        ),
        "allocation": {
            "current_lzf_xy_arm_ratio": current_x / current_y,
            "real_ulog_xy_arm_ratio": real_x / real_y,
            "current_lzf_normalized_mixer": current_mixer,
            "real_ulog_normalized_mixer": real_mixer,
            "normalized_mixer_max_abs_difference": float(
                np.max(np.abs(current_mixer - real_mixer))
            ),
        },
    }


def analyze_coverage(data):
    local_position = data["topics"]["vehicle_local_position"]
    local_time = topic_time(local_position)
    armed, landed = interpolate_state_flags(
        local_time,
        data["topics"]["actuator_armed"],
        data["topics"]["vehicle_land_detected"],
    )
    velocity = np.column_stack(
        [
            local_position["vx"],
            local_position["vy"],
            local_position["vz"],
        ]
    )
    speed = np.linalg.norm(velocity, axis=1)
    flight = armed & ~landed & np.all(np.isfinite(velocity), axis=1)
    attitude = data["topics"]["vehicle_attitude"]
    attitude_time = topic_time(attitude)
    rotation, valid = interpolate_quaternion(attitude_time, attitude)
    body_z_ned = rotation.apply(
        np.tile([0.0, 0.0, 1.0], (attitude_time.size, 1))
    )
    tilt = np.degrees(np.arccos(np.clip(body_z_ned[:, 2], -1.0, 1.0)))
    attitude_armed, attitude_landed = interpolate_state_flags(
        attitude_time,
        data["topics"]["actuator_armed"],
        data["topics"]["vehicle_land_detected"],
    )
    attitude_flight = valid & attitude_armed & ~attitude_landed
    angular_velocity = data["topics"]["vehicle_angular_velocity"]
    angular_time = topic_time(angular_velocity)
    body_rate = np.column_stack(
        [angular_velocity[f"xyz[{index}]"] for index in range(3)]
    )
    angular_armed, angular_landed = interpolate_state_flags(
        angular_time,
        data["topics"]["actuator_armed"],
        data["topics"]["vehicle_land_detected"],
    )
    rate_flight = angular_armed & ~angular_landed & np.all(
        np.isfinite(body_rate), axis=1
    )
    rates = {}

    for name, topic in data["topics"].items():
        rates[name] = {
            "samples": int(len(topic["timestamp"])),
            "rate_hz": topic_rate_hz(topic),
        }

    messages = [
        str(message.message)
        for message in data["ulog"].logged_messages
        if "chirp" in str(message.message).lower()
    ]
    return {
        "duration_s": float(
            (data["ulog"].last_timestamp - data["ulog"].start_timestamp) * 1e-6
        ),
        "topic_rates": rates,
        "max_ground_speed_m_s": float(np.max(speed[flight])),
        "max_tilt_deg": float(np.max(tilt[attitude_flight])),
        "max_abs_body_rate_rad_s": float(np.max(np.abs(body_rate[rate_flight]))),
        "chirp_messages": messages,
        "has_dedicated_chirp_excitation": bool(messages),
    }


def load_ulog(path):
    ulog = ULog(str(path), message_name_filter_list=TOPICS)
    required = [
        "actuator_armed",
        "actuator_motors",
        "battery_status",
        "esc_status",
        "vehicle_acceleration",
        "vehicle_angular_velocity",
        "vehicle_attitude",
        "vehicle_land_detected",
        "vehicle_local_position",
    ]
    topics = {name: get_topic(ulog, name) for name in required}

    for optional in ["vehicle_status", "wind"]:
        topic = get_topic(ulog, optional, required=False)

        if topic is not None:
            topics[optional] = topic

    esc_time, rpm, esc_mapping = semantic_esc_rpm(topics["esc_status"])
    return {
        "path": path,
        "ulog": ulog,
        "topics": topics,
        "esc_time": esc_time,
        "rpm": rpm,
        "esc_mapping": esc_mapping,
        "initial_parameters": ulog.initial_parameters,
    }


def load_bag_metrics(path):
    payload = json.loads(path.read_text())

    if "trajectory" in payload and "aerodynamics" in payload:
        return payload

    candidates = []

    for dataset in payload.get("datasets", {}).values():
        if "bag" in dataset:
            candidates.append(dataset["bag"])

    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one bag result in metrics file: {path}"
        )

    return candidates[0]


def analyze_log(
    path,
    role,
    model,
    airframe,
    current_parameters,
    command_lag_ms,
    bag_metrics=None,
):
    print(f"loading {path}")
    data = load_ulog(path)
    print(f"analyzing {path.name}")
    state = common_flight_state(data, model)
    result = {
        "name": path.name,
        "path": str(path.resolve()),
        "role": role,
        "firmware_version": data["ulog"].msg_info_dict.get("ver_sw", "unknown"),
        "esc_slot_mapping": data["esc_mapping"],
        "coverage": analyze_coverage(data),
        "battery": analyze_battery(data, model, airframe),
        "motor_speed": analyze_motor_speed(data, model, command_lag_ms),
        "hover": analyze_hover(state, model),
        "force_model": analyze_force_model(state, model),
        "configuration": analyze_configuration(data, current_parameters),
    }

    if bag_metrics is not None:
        result["motion_capture_bag"] = bag_metrics

    return result


def public_value(value):
    if isinstance(value, dict):
        return {
            key: public_value(item)
            for key, item in value.items()
            if not isinstance(key, str) or not key.startswith("_")
        }

    if isinstance(value, (list, tuple)):
        return [public_value(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def evenly_spaced_indices(mask, maximum):
    indices = np.flatnonzero(mask)

    if indices.size <= maximum:
        return indices

    selected = np.linspace(0, indices.size - 1, maximum, dtype=int)
    return indices[selected]


def plot_battery(results, output):
    figure, axes = plt.subplots(
        len(results), 2, figsize=(13, 4.0 * len(results)), squeeze=False
    )

    for row, result in enumerate(results):
        series = result["battery"]["_series"]
        elapsed = series["time"] - series["time"][0]
        axes[row, 0].plot(elapsed, series["measured"], label="measured", linewidth=0.9)
        axes[row, 0].plot(elapsed, series["predicted"], label="LZF model", linewidth=0.9)
        axes[row, 0].plot(
            elapsed,
            series["open_circuit"],
            label="open circuit",
            linewidth=0.7,
            alpha=0.8,
        )
        axes[row, 0].set_ylabel("Voltage (V)")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 0].legend()
        axes[row, 0].set_title(f"{result['name']} ({result['role']})")

        mask = series["mask"]
        axes[row, 1].scatter(
            series["measured"][mask],
            series["predicted"][mask],
            s=5,
            alpha=0.35,
        )
        low = min(
            np.min(series["measured"][mask]), np.min(series["predicted"][mask])
        )
        high = max(
            np.max(series["measured"][mask]), np.max(series["predicted"][mask])
        )
        axes[row, 1].plot([low, high], [low, high], color="black", linewidth=0.8)
        axes[row, 1].set_xlabel("Measured voltage (V)")
        axes[row, 1].set_ylabel("Predicted voltage (V)")
        axes[row, 1].grid(True, alpha=0.25)
        metrics = result["battery"]["metrics"]
        axes[row, 1].set_title(
            f"RMSE {metrics['rmse']:.3f} V, R2 {metrics['r_squared']:.3f}"
        )

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_motor(results, output, max_points):
    figure, axes = plt.subplots(
        len(results), 2, figsize=(13, 4.2 * len(results)), squeeze=False
    )

    for row, result in enumerate(results):
        series = result["motor_speed"]["_series"]
        valid = series["mask"].ravel()
        measured = series["measured"].ravel()
        predicted = series["predicted"].ravel()
        selected = evenly_spaced_indices(valid, max_points)
        axes[row, 0].scatter(
            measured[selected], predicted[selected], s=3, alpha=0.2
        )
        low = min(np.min(measured[selected]), np.min(predicted[selected]))
        high = max(np.max(measured[selected]), np.max(predicted[selected]))
        axes[row, 0].plot([low, high], [low, high], color="black", linewidth=0.8)
        axes[row, 0].set_xlabel("Measured mechanical RPM")
        axes[row, 0].set_ylabel("Quasi-steady model RPM")
        axes[row, 0].grid(True, alpha=0.25)
        metrics = result["motor_speed"]["metrics_pooled"]
        axes[row, 0].set_title(
            f"{result['name']}: RMSE {metrics['rmse']:.0f} RPM, "
            f"R2 {metrics['r_squared']:.3f}"
        )

        motor_metrics = result["motor_speed"]["metrics_per_motor"]
        rmse = [item["rmse"] for item in motor_metrics]
        bias = [item["bias_pred_minus_meas"] for item in motor_metrics]
        x = np.arange(4)
        axes[row, 1].bar(x - 0.18, rmse, width=0.36, label="RMSE")
        axes[row, 1].bar(x + 0.18, bias, width=0.36, label="bias")
        axes[row, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 1].set_xticks(x)
        axes[row, 1].set_xticklabels(
            [f"Motor {index + 1}" for index in range(4)]
        )
        axes[row, 1].set_ylabel("RPM")
        axes[row, 1].grid(True, axis="y", alpha=0.25)
        axes[row, 1].legend()
        axes[row, 1].set_title("Canonical motor order (actuator function 101-104)")

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_force(results, output, max_points):
    figure, axes = plt.subplots(
        len(results), 3, figsize=(16, 4.0 * len(results)), squeeze=False
    )

    for row, result in enumerate(results):
        series = result["force_model"]["_series"]
        selected = evenly_spaced_indices(series["mask"], max_points)
        residual_z = series["predicted"][:, 2] - series["measured"][:, 2]
        axes[row, 0].scatter(
            series["speed"][selected], residual_z[selected], s=3, alpha=0.25
        )
        axes[row, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 0].set_xlabel("Ground speed (m/s)")
        axes[row, 0].set_ylabel("Predicted - measured body Z (m/s2)")
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 0].set_title(result["name"])

        axes[row, 1].scatter(
            series["mean_advance_ratio"][selected],
            residual_z[selected],
            s=3,
            alpha=0.25,
        )
        axes[row, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 1].set_xlabel("Mean advance ratio J (zero-wind estimate)")
        axes[row, 1].set_ylabel("Predicted - measured body Z (m/s2)")
        axes[row, 1].grid(True, alpha=0.25)
        finite_j = series["mean_advance_ratio"][selected]
        finite_j = finite_j[np.isfinite(finite_j)]

        if finite_j.size:
            axes[row, 1].set_xlim(
                0.0, max(0.1, float(np.percentile(finite_j, 99.5)))
            )

        axes[row, 2].scatter(
            series["measured"][selected, 2],
            series["predicted"][selected, 2],
            s=3,
            alpha=0.25,
        )
        low = min(
            np.min(series["measured"][selected, 2]),
            np.min(series["predicted"][selected, 2]),
        )
        high = max(
            np.max(series["measured"][selected, 2]),
            np.max(series["predicted"][selected, 2]),
        )
        axes[row, 2].plot([low, high], [low, high], color="black", linewidth=0.8)
        axes[row, 2].set_xlabel("Measured body Z acceleration (m/s2)")
        axes[row, 2].set_ylabel("Predicted body Z acceleration (m/s2)")
        axes[row, 2].grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def format_value(value, digits=3):
    if value is None or not np.isfinite(value):
        return "n/a"

    return f"{value:.{digits}f}"


def score_battery(result):
    metrics = result["battery"]["metrics"]

    if metrics["r_squared"] >= 0.95 and metrics["rmse"] <= 0.2:
        return 8

    if metrics["r_squared"] >= 0.9 and metrics["rmse"] <= 0.3:
        return 7

    if metrics["r_squared"] >= 0.8 and metrics["rmse"] <= 0.5:
        return 6

    return 4


def score_motor(result):
    metrics = result["motor_speed"]["metrics_pooled"]

    if metrics["r_squared"] >= 0.85 and metrics["rmse"] <= 500.0:
        return 7

    if metrics["r_squared"] >= 0.7 and metrics["rmse"] <= 800.0:
        return 6

    if metrics["r_squared"] >= 0.5 and metrics["rmse"] <= 1200.0:
        return 5

    return 3


def score_hover(result):
    hover = result["hover"]

    if hover["samples"] < 10:
        return 4

    rpm_error = abs(hover["model_hover_rpm_error_percent"])
    ct_error = abs(hover["model_ct_error_percent"])

    if rpm_error <= 2.0 and ct_error <= 3.0:
        return 8

    if rpm_error <= 3.0 and ct_error <= 5.0:
        return 7

    if rpm_error <= 5.0 and ct_error <= 10.0:
        return 6

    return 4


def score_aerodynamics(result):
    bag = result.get("motion_capture_bag")

    if bag is None:
        return 3

    drag_ratios = [
        abs(item["effective_to_configured_ratio"])
        for item in bag["aerodynamics"]["rotor_drag"]
    ]
    worst_drag_ratio = max(drag_ratios)

    if worst_drag_ratio <= 1.25:
        return 6

    if worst_drag_ratio <= 2.0:
        return 5

    return 3


def write_report(results, model, airframe, output):
    holdouts = [result for result in results if result["role"] == "holdout"]
    validation = holdouts[-1] if holdouts else results[0]
    primary = results[0]
    scores = {
        "Static hover": score_hover(primary),
        "Battery terminal voltage": score_battery(validation),
        "Quasi-steady RPM": score_motor(validation),
        "Forward-flight body force": score_aerodynamics(validation),
        "Closed-loop attitude dynamics": 3,
    }
    overall = sum(scores.values()) / len(scores)
    lines = [
        "# LZF ULog Realism Assessment",
        "",
        "This is an offline component-level comparison. It is not a full closed-loop "
        "replay of the real trajectory in Gazebo.",
        "",
        "## Executive Verdict",
        "",
        "| Area | Heuristic score | Evidence |",
        "|---|---:|---|",
    ]
    hover = primary["hover"]
    battery_metrics = validation["battery"]["metrics"]
    motor_metrics = validation["motor_speed"]["metrics_pooled"]
    validation_bag = validation.get("motion_capture_bag")

    if validation_bag is not None:
        bag_j = validation_bag["aerodynamics"]["advance_ratio_positive"]
        bag_drag = validation_bag["aerodynamics"]["rotor_drag"]
        aerodynamic_evidence = (
            f"real bag max J {bag_j['max']:.3f}; lumped X/Y drag is "
            f"{bag_drag[0]['effective_to_configured_ratio']:.2f}x/"
            f"{bag_drag[1]['effective_to_configured_ratio']:.2f}x configured"
        )

    else:
        aerodynamic_evidence = (
            "no synchronized motion-capture bag; CT(J) and drag remain unvalidated"
        )
    lines.extend(
        [
            (
                f"| Static hover | {scores['Static hover']}/10 | "
                f"model {hover.get('model_static_hover_rpm', math.nan):.0f} RPM vs "
                f"measured median {hover.get('measured_rpm_median', math.nan):.0f} RPM; "
                f"effective CT error "
                f"{hover.get('model_ct_error_percent', math.nan):+.1f}% |"
            ),
            (
                f"| Battery terminal voltage | "
                f"{scores['Battery terminal voltage']}/10 | holdout RMSE "
                f"{battery_metrics['rmse']:.3f} V, R2 "
                f"{battery_metrics['r_squared']:.3f} |"
            ),
            (
                f"| Quasi-steady RPM | {scores['Quasi-steady RPM']}/10 | "
                f"holdout RMSE {motor_metrics['rmse']:.0f} RPM, R2 "
                f"{motor_metrics['r_squared']:.3f}; per-motor offsets remain |"
            ),
            (
                f"| Forward-flight body force | "
                f"{scores['Forward-flight body force']}/10 | "
                f"{aerodynamic_evidence} |"
            ),
            (
                f"| Closed-loop attitude dynamics | "
                f"{scores['Closed-loop attitude dynamics']}/10 | controller gains "
                "do not match the real log; the newly measured inertia and motor "
                "time constant still need closed-loop validation |"
            ),
            "",
            f"**Overall engineering estimate: about {overall:.1f}/10.** The current "
            "LZF is a partially identified grey-box model: useful for hover, battery "
            "trend, and broad RPM behavior, but not yet a digital twin for aggressive "
            "or high-speed flight. Scores are diagnostic summaries, not confidence "
            "intervals.",
            "",
            "## Inputs And Coverage",
            "",
            "| ULog | Role | Duration | ESC rate | Motor command rate | ULog local "
            "speed max | MoCap lateral speed max | MoCap J max | Chirp |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for result in results:
        coverage = result["coverage"]
        bag = result.get("motion_capture_bag")
        bag_speed = (
            bag["aerodynamics"]["body_lateral_speed_m_s"]["max"]
            if bag is not None
            else math.nan
        )
        bag_j_max = (
            bag["aerodynamics"]["advance_ratio_positive"]["max"]
            if bag is not None
            else math.nan
        )
        lines.append(
            f"| `{result['name']}` | {result['role']} | "
            f"{coverage['duration_s']:.1f} s | "
            f"{coverage['topic_rates']['esc_status']['rate_hz']:.2f} Hz | "
            f"{coverage['topic_rates']['actuator_motors']['rate_hz']:.2f} Hz | "
            f"{coverage['max_ground_speed_m_s']:.1f} m/s | "
            f"{format_value(bag_speed, 2)} m/s | "
            f"{format_value(bag_j_max, 3)} | "
            f"{'yes' if coverage['has_dedicated_chirp_excitation'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "The first ULog is the same data family used to derive the present battery "
            "and motor-speed coefficients. Later ULogs are treated as holdout checks.",
            "",
            "Where a motion-capture bag is present, its odometry is authoritative for "
            "vehicle velocity. The PX4 `vehicle_local_position` speed is retained only "
            "as a diagnostic because it is not consistent with the synchronized "
            "motion-capture trajectory.",
            "",
            "## Model Configuration",
            "",
            f"- Vehicle mass: `{model['mass_kg']:.3f} kg`.",
            f"- Propeller: DA4052 5x3.75x3, diameter "
            f"`{model['diameter_m']:.3f} m`, static "
            f"`CT={model['thrust_coefficient'][0]:.6f}`.",
            f"- Motor speed: `RPM = min({model['motor_idle_rpm']:.0f} + "
            f"{model['motor_loaded_kv']:.0f} * u * V, "
            f"{model['motor_kv']:.0f} * V)`.",
            f"- Motor first-order constants: up "
            f"`{model['time_constant_up_s'] * 1000.0:.1f} ms`, down "
            f"`{model['time_constant_down_s'] * 1000.0:.1f} ms`.",
            f"- Battery: drain `{float(airframe['SIM_BAT_DRAIN']):.0f} s`, "
            f"instant sag `{float(airframe['SIM_BAT_SAG_I']):.4f}`, "
            f"polarization sag `{float(airframe['SIM_BAT_SAG_P']):.4f}`, "
            f"tau `{float(airframe['SIM_BAT_TAU']):.2f} s`.",
            f"- Compensated base-link inertia: `{model['inertia_kg_m2']}`; "
            "the assembled target is `Ixx=0.003242418`, `Iyy=0.003092245`, "
            "`Izz=0.005802551 kg m^2`.",
            "",
            "## Battery",
            "",
            "| ULog | RMSE | MAE | Bias (pred-meas) | R2 | Last measured | "
            "Last predicted |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        metrics = result["battery"]["metrics"]
        endpoints = result["battery"]["endpoints"]
        lines.append(
            f"| `{result['name']}` | {metrics['rmse']:.3f} V | "
            f"{metrics['mae']:.3f} V | "
            f"{metrics['bias_pred_minus_meas']:+.3f} V | "
            f"{metrics['r_squared']:.3f} | "
            f"{endpoints.get('last_measured_v', math.nan):.3f} V | "
            f"{endpoints.get('last_predicted_v', math.nan):.3f} V |"
        )

    lines.extend(
        [
            "",
            "The voltage trend generalizes to the holdout log, but end-of-flight "
            "voltage is predicted too high. The model still has no measured current, "
            "cell imbalance, temperature, resistance, or ESC efficiency state.",
            "",
            "![Battery comparison](battery_comparison.png)",
            "",
            "## Motor Speed",
            "",
            "RPM channels are reordered by `esc_status.esc[i].actuator_function`; "
            "the ULog array slot is not assumed to be the PX4 motor index.",
            "",
            "| ULog | Pooled RMSE | MAE | Bias | R2 | P95 abs error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        metrics = result["motor_speed"]["metrics_pooled"]
        lines.append(
            f"| `{result['name']}` | {metrics['rmse']:.0f} RPM | "
            f"{metrics['mae']:.0f} RPM | "
            f"{metrics['bias_pred_minus_meas']:+.0f} RPM | "
            f"{metrics['r_squared']:.3f} | "
            f"{metrics['p95_abs_error']:.0f} RPM |"
        )

    lines.extend(
        [
            "",
            "Per-motor metrics:",
            "",
            "| ULog | Motor | RMSE | Bias | R2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        for index, metrics in enumerate(
            result["motor_speed"]["metrics_per_motor"], start=1
        ):
            lines.append(
                f"| `{result['name']}` | {index} | {metrics['rmse']:.0f} RPM | "
                f"{metrics['bias_pred_minus_meas']:+.0f} RPM | "
                f"{metrics['r_squared']:.3f} |"
            )

    lines.extend(
        [
            "",
            f"The comparison uses a fixed "
            f"`{primary['motor_speed']['command_alignment_ms']:.0f} ms` command "
            "alignment. Real `actuator_motors` is logged at only about 10 Hz, slower "
            "than the configured 24.568 ms motor dynamics, so this validates only "
            "the quasi-steady command-voltage-RPM surface, not the time constants.",
            "",
            "![Motor model comparison](motor_model_comparison.png)",
            "",
            "## Static Hover",
            "",
            "| ULog | Samples | Measured median RPM | Model hover RPM | RPM error | "
            "Effective CT | Model CT error | Body-Z RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        item = result["hover"]
        rpm_error = format_value(item.get("model_hover_rpm_error_percent"), 2)
        rpm_error_text = "n/a" if rpm_error == "n/a" else f"{rpm_error}%"
        ct_error = format_value(item.get("model_ct_error_percent"), 2)
        ct_error_text = "n/a" if ct_error == "n/a" else f"{ct_error}%"
        lines.append(
            f"| `{result['name']}` | {item['samples']} | "
            f"{format_value(item.get('measured_rpm_median'), 0)} | "
            f"{item['model_static_hover_rpm']:.0f} | "
            f"{rpm_error_text} | "
            f"{format_value(item.get('effective_ct_median'), 6)} | "
            f"{ct_error_text} | "
            f"{format_value(item['body_z_acceleration_metrics']['rmse'], 3)} "
            "m/s2 |"
        )

    lines.extend(
        [
            "",
            "The strict hover windows are short in these dynamic flights, so the "
            "result is encouraging but not a substitute for a long stationary hover "
            "test in still air.",
            "",
            "## Motion-Capture Trajectory And Lateral Force",
            "",
            "| ULog | Laps | Position RMSE | Position P95 | Desired speed max | "
            "J P95 | J max | Effective drag X/Y |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        bag = result.get("motion_capture_bag")

        if bag is None:
            continue

        trajectory = bag["trajectory"]
        aerodynamics = bag["aerodynamics"]
        advance_ratio = aerodynamics["advance_ratio_positive"]
        drag = aerodynamics["rotor_drag"]
        lines.append(
            f"| `{result['name']}` | {trajectory['lap_count']} | "
            f"{trajectory['position_rmse_norm_m']:.3f} m | "
            f"{trajectory['position_error_p95_norm_m']:.3f} m | "
            f"{trajectory['desired_speed']['max']:.2f} m/s | "
            f"{advance_ratio['p95']:.3f} | {advance_ratio['max']:.3f} | "
            f"{drag[0]['effective_coefficient']:.3e} / "
            f"{drag[1]['effective_coefficient']:.3e} |"
        )

    lines.extend(
        [
            "",
            f"The UIUC CT(J) table starts its first positive-J measurement at "
            f"`J={model['advance_ratio'][1]:.3f}`. Both real bags remain below "
            "that point even at maximum J, so these flights validate the near-static "
            "region only. They do not validate the forward-flight CT(J) curve.",
            "",
            "The two independent bags reproduce nearly the same lumped X/Y drag "
            "coefficient, roughly 3.1-3.5 times the configured rotor-only value. "
            "This is strong evidence that total lateral damping is under-modelled, "
            "but the fit includes fuselage drag and must be split into rotor and body "
            "terms before changing `rotorDragCoefficient`.",
            "",
            "The position RMSE above describes the real vehicle/controller tracking "
            "performance. A matching SITL bag is still required before simulation "
            "trajectory error can be compared directly.",
            "",
            "## ULog-Only Force Diagnostic (Rejected As Velocity Truth)",
            "",
            "The exact current CT(J) table and rotor-drag formula are evaluated using "
            "measured RPM, attitude, body rates, and ground velocity. No measured wind "
            "or airspeed is available, and the ULog local velocity conflicts with "
            "motion capture by a factor of roughly 7-8 at its maximum. These numbers "
            "are included to expose that inconsistency and are not used as physical "
            "validation metrics.",
            "",
            "| ULog | Axis | RMSE | Bias (pred-meas) | R2 |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for result in results:
        for axis, metrics in result["force_model"]["axis_metrics"].items():
            lines.append(
                f"| `{result['name']}` | {axis.upper()} | "
                f"{metrics['rmse']:.3f} m/s2 | "
                f"{metrics['bias_pred_minus_meas']:+.3f} m/s2 | "
                f"{metrics['r_squared']:.3f} |"
            )

    lines.extend(
        [
            "",
            "Body-Z error by speed:",
            "",
            "| ULog | Speed | Samples | RMSE | Bias | R2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        for row in result["force_model"]["body_z_metrics_by_ground_speed_m_s"]:
            lines.append(
                f"| `{result['name']}` | {row['bin']} m/s | {row['samples']} | "
                f"{row['rmse']:.3f} | {row['bias_pred_minus_meas']:+.3f} | "
                f"{row['r_squared']:.3f} |"
            )

    lines.extend(
        [
            "",
            "Body-Z error by estimated advance ratio:",
            "",
            "| ULog | Mean J | Samples | RMSE | Bias | R2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        for row in result["force_model"]["body_z_metrics_by_mean_advance_ratio"]:
            lines.append(
                f"| `{result['name']}` | {row['bin']} | {row['samples']} | "
                f"{row['rmse']:.3f} | {row['bias_pred_minus_meas']:+.3f} | "
                f"{row['r_squared']:.3f} |"
            )

    lines.extend(["", "Propeller-table and rotor-drag diagnostics:", ""])

    for result in results:
        force_model = result["force_model"]
        lines.append(
            f"- `{result['name']}`: "
            f"{force_model['samples_at_or_beyond_prop_table_percent']:.3f}% "
            "of flight samples are at or beyond the last UIUC J breakpoint; "
            f"{force_model['samples_with_negative_total_prop_thrust_percent']:.3f}% "
            "produce negative total table thrust. The lumped zero-wind lateral "
            f"drag fit is `{force_model['lumped_zero_wind_drag_fit']:.3e}` versus "
            f"the configured `{force_model['configured_rotor_drag_coefficient']:.3e}`."
        )

    lines.extend(
        [
            "",
            "The apparent error growth with ULog speed/J cannot be interpreted as "
            "propeller error because the velocity input is contradicted by motion "
            "capture. The motion-capture result above supersedes this diagnostic.",
            "",
            "![Force model comparison](force_model_comparison.png)",
            "",
            "## Controller And Allocation",
            "",
            "| Parameter | Real ULog | Current LZF | Difference |",
            "|---|---:|---:|---:|",
        ]
    )

    for row in primary["configuration"]["controller_parameters"]:
        difference = format_value(row["relative_difference_percent"], 1)
        difference_text = "n/a" if difference == "n/a" else f"{difference}%"
        lines.append(
            f"| `{row['parameter']}` | {row['real_ulog']:.6g} | "
            f"{row['current_lzf']:.6g} | "
            f"{difference_text} |"
        )

    allocation = primary["configuration"]["allocation"]
    lines.extend(
        [
            "",
            f"- Real ULog CA X/Y arm ratio: "
            f"`{allocation['real_ulog_xy_arm_ratio']:.3f}`.",
            f"- Current LZF physical CA X/Y arm ratio: "
            f"`{allocation['current_lzf_xy_arm_ratio']:.3f}`.",
            f"- Maximum normalized mixer coefficient difference: "
            f"`{allocation['normalized_mixer_max_abs_difference']:.3f}`.",
            "- The real flight used normalized square CA positions (+/-1, +/-1), "
            "while LZF now uses the measured rectangular arms (+/-0.085, +/-0.10 m). "
            "The Gazebo geometry is physically correct, but the PX4 allocation in "
            "this ULog did not use that same geometry.",
            "",
            "## Not Validated By These Logs",
            "",
            "- Motor rise/fall time constants: command logging is about 10 Hz, while "
            "the measured rise/fall constant is 24.568 ms.",
            "- The newly measured inertia and the rolling-moment coefficient still "
            "lack a dedicated roll/pitch/yaw chirp or persistently exciting torque input.",
            "- True CT(J) in flight: no synchronized airspeed or wind measurement is "
            "available, and the UIUC data is at substantially lower RPM.",
            "- Electrical current, ESC efficiency, thermal state, cell imbalance, "
            "and motor torque loading are not represented.",
            "- Closed-loop trajectory tracking: this report does not replay the "
            "offboard attitude setpoints through SITL.",
            "- The current `comet_ws` controller setup is not yet a valid replay "
            "fixture: it launches Nokov/EKF odometry and its active YAML uses "
            "`mass=1.5 kg` and a 4S battery, while LZF is 1.326 kg and 6S.",
            "",
            "## Recommended Next Measurements",
            "",
            "1. Log voltage, current, RPM, and motor command at >=200 Hz during "
            "separate throttle steps.",
            "2. Run dedicated roll, pitch, and yaw chirps with high-rate torque "
            "setpoint, angular acceleration, and RPM logging.",
            "3. Add synchronized airspeed or wind-tunnel data for the complete "
            "vehicle, then identify CT(J), oblique inflow, and fuselage drag "
            "separately.",
            "4. Repeat a long still-air hover and a vertical climb/descent profile "
            "for an independent static/axial validation set.",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def main():
    args = parse_args()

    try:
        for path in [args.airframe, args.model, *args.ulog]:
            if not path.exists():
                raise RuntimeError(f"path does not exist: {path}")

        if args.bag_metrics and len(args.bag_metrics) != len(args.ulog):
            raise RuntimeError(
                "--bag-metrics must be omitted or repeated once per ULog"
            )

        for path in args.bag_metrics:
            if not path.exists():
                raise RuntimeError(f"path does not exist: {path}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        airframe = parse_airframe(args.airframe)
        model = parse_model(args.model, args.mass)
        defaults = load_parameter_defaults(args.parameter_metadata)
        current_parameters = dict(defaults)
        current_parameters.update(airframe)
        bag_metrics = [
            load_bag_metrics(path) for path in args.bag_metrics
        ]
        results = []

        for index, path in enumerate(args.ulog):
            role = "model-source" if index == 0 else "holdout"
            results.append(
                analyze_log(
                    path,
                    role,
                    model,
                    airframe,
                    current_parameters,
                    args.command_lag_ms,
                    bag_metrics[index] if bag_metrics else None,
                )
            )

        plot_battery(results, args.output_dir / "battery_comparison.png")
        plot_motor(
            results,
            args.output_dir / "motor_model_comparison.png",
            args.max_plot_points,
        )
        plot_force(
            results,
            args.output_dir / "force_model_comparison.png",
            args.max_plot_points,
        )
        summary = {
            "model_file": str(args.model.resolve()),
            "airframe_file": str(args.airframe.resolve()),
            "model": model,
            "airframe_parameters": airframe,
            "results": results,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(public_value(summary), indent=2, sort_keys=True) + "\n"
        )
        write_report(
            results, model, airframe, args.output_dir / "report.md"
        )
        print(f"report: {args.output_dir / 'report.md'}")
        print(f"summary: {args.output_dir / 'summary.json'}")
        return 0

    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
