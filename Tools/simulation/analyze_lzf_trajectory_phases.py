#!/usr/bin/env python3

"""Quantify pre-hover, trajectory, and post-hover phases of an LZF trial bag."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.signal import detrend, welch
from scipy.spatial.transform import Rotation


TOPICS = {
    "fsm": "/px4ctrl/fsm_status",
    "command": "/setpoints_cmd",
    "debug": "/debugPx4ctrl",
    "imu": "/mavros/imu/data",
    "attitude_target": "/mavros/setpoint_raw/attitude",
    "esc": "/mavros/esc_status",
    "parameter": "/px4ctrl_param",
}
ODOMETRY_CANDIDATES = (
    "/ekf/ekf_odom",
    "/mavros/local_position/odom",
)
FSM_NAMES = {
    1: "MANUAL_CTRL",
    2: "AUTO_HOVER",
    3: "CMD_CTRL",
    4: "AUTO_TAKEOFF",
    5: "AUTO_LAND",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--odom-topic",
        help=(
            "odometry used by px4ctrl; by default prefer /ekf/ekf_odom when "
            "present, otherwise use /mavros/local_position/odom"
        ),
    )
    parser.add_argument("--oscillation-min-hz", type=float, default=0.2)
    parser.add_argument("--oscillation-max-hz", type=float, default=20.0)
    parser.add_argument("--settling-position-error-m", type=float, default=0.15)
    parser.add_argument("--settling-speed-m-s", type=float, default=0.15)
    parser.add_argument("--settling-tilt-deg", type=float, default=5.0)
    parser.add_argument(
        "--settling-rate-rad-s",
        type=float,
        default=0.5,
        help="body-rate norm limit used by the post-hover settling test",
    )
    parser.add_argument("--settling-dwell-s", type=float, default=2.0)
    parser.add_argument(
        "--post-transient-window-s",
        type=float,
        default=5.0,
        help="post-hover prefix reported separately from the later steady hover",
    )
    parser.add_argument(
        "--hover-comparison-window-s",
        type=float,
        default=60.0,
        help="equal-duration windows used to compare pre- and post-hover",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def vector3(value):
    return [float(value.x), float(value.y), float(value.z)]


def quaternion(value):
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


def rms(values, axis=None):
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(values * values, axis=axis))


def vector_metrics(values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or not len(values):
        return None

    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]

    if not len(values):
        return None

    norm = np.linalg.norm(values, axis=1)
    return {
        "rms_axis": [float(value) for value in rms(values, axis=0)],
        "rms_norm": float(rms(norm)),
        "p95_norm": float(np.quantile(norm, 0.95)),
        "max_norm": float(np.max(norm)),
        "mean_axis": [float(value) for value in np.mean(values, axis=0)],
    }


def scalar_metrics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not len(values):
        return None

    return {
        "mean": float(np.mean(values)),
        "rms": float(rms(values)),
        "peak_abs": float(np.max(np.abs(values))),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
    }


def interpolate_columns(target_time, source_time, values):
    target_time = np.asarray(target_time, dtype=float)
    source_time = np.asarray(source_time, dtype=float)
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        values = values[:, None]

    return np.column_stack(
        [
            np.interp(target_time, source_time, values[:, index])
            for index in range(values.shape[1])
        ]
    )


def topic_has_messages(topic_info, topic):
    return topic in topic_info and topic_info[topic].message_count > 0


def load_bag(path, requested_odom_topic):
    try:
        import rosbag
    except ImportError as error:
        raise RuntimeError(
            "rosbag is unavailable; source /opt/ros/noetic/setup.bash and "
            "/home/user/comet_ws/devel/setup.bash"
        ) from error

    data = {key: [] for key in TOPICS}
    data["odometry"] = []
    odom_velocity_in_body = None

    with rosbag.Bag(str(path)) as bag:
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())
        topic_info = bag.get_type_and_topic_info().topics

        if requested_odom_topic:
            odom_topic = requested_odom_topic

            if not topic_has_messages(topic_info, odom_topic):
                raise RuntimeError(f"bag has no messages on {odom_topic}")
        else:
            odom_topic = next(
                (
                    topic
                    for topic in ODOMETRY_CANDIDATES
                    if topic_has_messages(topic_info, topic)
                ),
                None,
            )

            if odom_topic is None:
                raise RuntimeError(
                    "bag has neither /ekf/ekf_odom nor "
                    "/mavros/local_position/odom"
                )

        selected_topics = [
            topic
            for topic in (*TOPICS.values(), odom_topic)
            if topic_has_messages(topic_info, topic)
        ]
        reverse = {topic: key for key, topic in TOPICS.items()}
        reverse[odom_topic] = "odometry"

        for topic, message, record_time in bag.read_messages(
            topics=selected_topics
        ):
            key = reverse[topic]
            time_s = float(record_time.to_sec())

            if key == "fsm":
                data[key].append((time_s, int(message.state)))
            elif key == "command":
                data[key].append(time_s)
            elif key == "odometry":
                data[key].append(
                    (
                        time_s,
                        vector3(message.pose.pose.position),
                        vector3(message.twist.twist.linear),
                        quaternion(message.pose.pose.orientation),
                    )
                )
            elif key == "debug":
                data[key].append(
                    (
                        time_s,
                        [message.des_p_x, message.des_p_y, message.des_p_z],
                        [message.des_v_x, message.des_v_y, message.des_v_z],
                        float(message.des_thr),
                    )
                )
            elif key == "imu":
                data[key].append(
                    (time_s, vector3(message.angular_velocity))
                )
            elif key == "attitude_target":
                data[key].append((time_s, float(message.thrust)))
            elif key == "esc":
                data[key].append(
                    (time_s, [float(item.rpm) for item in message.esc_status])
                )
            elif key == "parameter":
                odom_velocity_in_body = bool(message.odom_vel_in_body)
                data[key].append(time_s)

        topic_metadata = {
            topic: {
                "message_count": int(info.message_count),
                "message_type": str(info.msg_type),
            }
            for topic, info in topic_info.items()
            if topic in selected_topics
        }

    return {
        "start": bag_start,
        "end": bag_end,
        "duration": bag_end - bag_start,
        "odometry_topic": odom_topic,
        "odom_velocity_in_body": odom_velocity_in_body,
        "topic_metadata": topic_metadata,
        "data": data,
    }


def build_fsm_segments(samples, bag_end):
    if not samples:
        raise RuntimeError(f"bag has no {TOPICS['fsm']} messages")

    segments = []
    segment_start = samples[0][0]
    state = samples[0][1]

    for time_s, next_state in samples[1:]:
        if next_state == state:
            continue

        segments.append(
            {"state": state, "start": segment_start, "end": time_s}
        )
        segment_start = time_s
        state = next_state

    segments.append({"state": state, "start": segment_start, "end": bag_end})

    # Suppress only an isolated, sub-100 ms A-B-A state glitch. Longer state
    # changes are real controller phases and remain visible in the report.
    changed = True

    while changed and len(segments) >= 3:
        changed = False

        for index in range(1, len(segments) - 1):
            previous = segments[index - 1]
            current = segments[index]
            following = segments[index + 1]

            if (
                current["end"] - current["start"] <= 0.1
                and previous["state"] == following["state"]
            ):
                merged = {
                    "state": previous["state"],
                    "start": previous["start"],
                    "end": following["end"],
                }
                segments[index - 1 : index + 2] = [merged]
                changed = True
                break

    return segments


def select_trial_phases(segments, command_times):
    cmd_indices = [
        index for index, segment in enumerate(segments) if segment["state"] == 3
    ]

    if not cmd_indices:
        raise RuntimeError("bag has no CMD_CTRL FSM segment")

    command_times = np.asarray(command_times, dtype=float)

    def command_score(index):
        segment = segments[index]

        if len(command_times):
            count = int(
                np.sum(
                    (command_times >= segment["start"] - 0.02)
                    & (command_times < segment["end"] + 0.02)
                )
            )
        else:
            count = 0

        return count, segment["end"] - segment["start"]

    trajectory_index = max(cmd_indices, key=command_score)
    pre_index = next(
        (
            index
            for index in range(trajectory_index - 1, -1, -1)
            if segments[index]["state"] == 2
        ),
        None,
    )
    post_index = next(
        (
            index
            for index in range(trajectory_index + 1, len(segments))
            if segments[index]["state"] == 2
        ),
        None,
    )

    if pre_index is None or post_index is None:
        raise RuntimeError(
            "CMD_CTRL is not bracketed by pre- and post-trajectory AUTO_HOVER"
        )

    if pre_index != trajectory_index - 1 or post_index != trajectory_index + 1:
        raise RuntimeError(
            "CMD_CTRL is not directly adjacent to both AUTO_HOVER segments"
        )

    return {
        "pre_hover": dict(segments[pre_index]),
        "trajectory": dict(segments[trajectory_index]),
        "post_hover": dict(segments[post_index]),
    }


def phase_mask(time_values, phase):
    time_values = np.asarray(time_values, dtype=float)
    return (time_values >= phase["start"]) & (time_values < phase["end"])


def normalize_odometry(records, velocity_in_body):
    time_s = np.asarray([item[0] for item in records], dtype=float)
    position = np.asarray([item[1] for item in records], dtype=float)
    velocity = np.asarray([item[2] for item in records], dtype=float)
    quaternions = np.asarray([item[3] for item in records], dtype=float)
    quaternion_norm = np.linalg.norm(quaternions, axis=1)
    finite = (
        np.isfinite(time_s)
        & np.all(np.isfinite(position), axis=1)
        & np.all(np.isfinite(velocity), axis=1)
        & np.all(np.isfinite(quaternions), axis=1)
        & (quaternion_norm > 1e-9)
    )
    time_s = time_s[finite]
    position = position[finite]
    velocity = velocity[finite]
    quaternions = quaternions[finite] / quaternion_norm[finite, None]
    rotations = Rotation.from_quat(quaternions)

    if velocity_in_body:
        velocity = rotations.apply(velocity)

    euler_deg = rotations.as_euler("xyz", degrees=True)
    rotation_matrices = rotations.as_matrix()
    tilt_deg = np.degrees(
        np.arccos(np.clip(rotation_matrices[:, 2, 2], -1.0, 1.0))
    )
    return {
        "time": time_s,
        "position": position,
        "velocity": velocity,
        "quaternion": quaternions,
        "euler_deg": euler_deg,
        "tilt_deg": tilt_deg,
    }


def dominant_frequency(time_s, values, axis_names, minimum_hz, maximum_hz):
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        values = values[:, None]

    finite = np.isfinite(time_s) & np.all(np.isfinite(values), axis=1)
    time_s = time_s[finite]
    values = values[finite]

    if len(time_s) < 32:
        return {"available": False, "reason": "fewer than 32 finite samples"}

    minimum_duration = max(2.0, 0.95 / minimum_hz)

    if time_s[-1] - time_s[0] < minimum_duration:
        return {
            "available": False,
            "reason": (
                f"phase is shorter than the {minimum_duration:.2f} s "
                "minimum spectral window"
            ),
        }

    unique_time, unique_indices = np.unique(time_s, return_index=True)
    time_s = unique_time
    values = values[unique_indices]
    delta = np.diff(time_s)
    delta = delta[(delta > 1e-5) & np.isfinite(delta)]

    if not len(delta):
        return {"available": False, "reason": "invalid sample timestamps"}

    sample_rate_hz = 1.0 / float(np.median(delta))
    upper_hz = min(maximum_hz, 0.45 * sample_rate_hz)

    if upper_hz <= minimum_hz:
        return {"available": False, "reason": "sample rate is too low"}

    uniform_time = np.arange(
        time_s[0], time_s[-1] + 0.25 / sample_rate_hz, 1.0 / sample_rate_hz
    )

    if len(uniform_time) < 32:
        return {"available": False, "reason": "phase is too short"}

    uniform_values = interpolate_columns(uniform_time, time_s, values)
    segment_length = min(4096, len(uniform_time))
    axes = {}

    for index, name in enumerate(axis_names):
        signal = detrend(uniform_values[:, index], type="linear")
        frequency, psd = welch(
            signal,
            fs=sample_rate_hz,
            nperseg=segment_length,
            noverlap=segment_length // 2,
            scaling="density",
        )
        selected = (frequency >= minimum_hz) & (frequency <= upper_hz)

        if not np.any(selected):
            continue

        selected_frequency = frequency[selected]
        selected_psd = psd[selected]
        peak_index = int(np.argmax(selected_psd))
        axes[name] = {
            "frequency_hz": float(selected_frequency[peak_index]),
            "peak_psd": float(selected_psd[peak_index]),
            "band_rms": float(
                math.sqrt(max(0.0, np.trapz(selected_psd, selected_frequency)))
            ),
        }

    if not axes:
        return {"available": False, "reason": "frequency band has no bins"}

    strongest_axis = max(axes, key=lambda name: axes[name]["peak_psd"])
    return {
        "available": True,
        "source": TOPICS["imu"] + ".angular_velocity",
        "method": "linear detrend, uniform resampling, Welch PSD",
        "frequency_band_hz": [float(minimum_hz), float(upper_hz)],
        "sample_rate_hz": float(sample_rate_hz),
        "strongest_axis": strongest_axis,
        "frequency_hz": axes[strongest_axis]["frequency_hz"],
        "axes": axes,
    }


def drift_metrics(time_s, position, phase_start):
    if not len(position):
        return None

    initial = time_s <= min(time_s[-1], phase_start + 1.0)
    final = time_s >= max(time_s[0], time_s[-1] - 1.0)
    reference = np.median(position[initial], axis=0)
    displacement = position - reference
    displacement_norm = np.linalg.norm(displacement, axis=1)
    end_position = np.median(position[final], axis=0)
    return {
        "reference": "median measured position during first 1 s",
        "reference_position_m": [float(value) for value in reference],
        "rms_m": float(rms(displacement_norm)),
        "p95_m": float(np.quantile(displacement_norm, 0.95)),
        "max_m": float(np.max(displacement_norm)),
        "end_displacement_m": [
            float(value) for value in end_position - reference
        ],
        "end_displacement_norm_m": float(np.linalg.norm(end_position - reference)),
    }


def esc_metrics(records, phase):
    selected = [
        rpm
        for time_s, rpm in records
        if phase["start"] <= time_s < phase["end"] and rpm
    ]

    if not selected:
        return None

    motor_count = min(len(value) for value in selected)

    if motor_count <= 0:
        return None

    rpm = np.asarray([value[:motor_count] for value in selected], dtype=float)
    finite_rows = np.all(np.isfinite(rpm), axis=1)
    rpm = rpm[finite_rows]

    if not len(rpm):
        return None

    return {
        "source": TOPICS["esc"] + ".esc_status[].rpm",
        "motor_count": int(motor_count),
        "mean": float(np.mean(rpm)),
        "p99": float(np.quantile(rpm, 0.99)),
        "max": float(np.max(rpm)),
        "min": float(np.min(rpm)),
        "per_motor_mean": [
            float(np.mean(rpm[:, index])) for index in range(motor_count)
        ],
        "zero_fraction": float(np.mean(rpm <= 0.0)),
        "saturation_fraction": None,
        "saturation_note": "ESCStatus exposes RPM but no documented RPM limit",
    }


def score_phase(name, phase, bag_data, odometry, oscillation_band):
    raw = bag_data["data"]
    result = {
        "fsm_state": FSM_NAMES[phase["state"]],
        "start_s_from_bag": float(phase["start"] - bag_data["start"]),
        "end_s_from_bag": float(phase["end"] - bag_data["start"]),
        "duration_s": float(phase["end"] - phase["start"]),
    }
    topic_records = {
        "fsm": raw["fsm"],
        "command": [(value,) for value in raw["command"]],
        "odometry": [(value,) for value in odometry["time"]],
        "debug": raw["debug"],
        "imu": raw["imu"],
        "attitude_target": raw["attitude_target"],
        "esc": raw["esc"],
    }
    result["samples"] = {
        key: int(
            sum(
                phase["start"] <= value[0] < phase["end"]
                for value in records
            )
        )
        for key, records in topic_records.items()
    }

    odom_selected = phase_mask(odometry["time"], phase)
    odom_time = odometry["time"][odom_selected]
    position = odometry["position"][odom_selected]
    velocity = odometry["velocity"][odom_selected]
    euler_deg = odometry["euler_deg"][odom_selected]
    tilt_deg = odometry["tilt_deg"][odom_selected]

    if not len(odom_time):
        raise RuntimeError(f"{name} has no finite odometry samples")

    result["position_drift"] = drift_metrics(
        odom_time, position, phase["start"]
    )
    result["velocity"] = vector_metrics(velocity)
    result["attitude"] = {
        "source": bag_data["odometry_topic"] + ".pose.pose.orientation",
        "roll_deg": scalar_metrics(euler_deg[:, 0]),
        "pitch_deg": scalar_metrics(euler_deg[:, 1]),
        "tilt_deg": scalar_metrics(tilt_deg),
    }

    signals = {
        "time": odom_time,
        "position_error_norm": None,
        "speed_norm": np.linalg.norm(velocity, axis=1),
        "tilt_deg": tilt_deg,
        "rate_norm": None,
    }
    debug = raw["debug"]

    if debug:
        debug_time = np.asarray([item[0] for item in debug], dtype=float)
        desired_position = np.asarray([item[1] for item in debug], dtype=float)
        desired_velocity = np.asarray([item[2] for item in debug], dtype=float)
        overlap = (odom_time >= debug_time[0]) & (odom_time <= debug_time[-1])

        if np.any(overlap):
            reference_position = interpolate_columns(
                odom_time[overlap], debug_time, desired_position
            )
            reference_velocity = interpolate_columns(
                odom_time[overlap], debug_time, desired_velocity
            )
            position_error = position[overlap] - reference_position
            velocity_error = velocity[overlap] - reference_velocity
            result["position_error"] = {
                "source": TOPICS["debug"] + ".des_p_*",
                "definition": "measured minus desired position",
                **vector_metrics(position_error),
            }
            result["velocity_error"] = {
                "source": TOPICS["debug"] + ".des_v_*",
                "definition": "measured minus desired velocity",
                **vector_metrics(velocity_error),
            }
            position_error_norm = np.full(len(odom_time), np.nan)
            position_error_norm[overlap] = np.linalg.norm(position_error, axis=1)
            signals["position_error_norm"] = position_error_norm

    imu = [
        item
        for item in raw["imu"]
        if phase["start"] <= item[0] < phase["end"]
    ]

    if imu:
        imu_time = np.asarray([item[0] for item in imu], dtype=float)
        angular_rate = np.asarray([item[1] for item in imu], dtype=float)
        finite = np.isfinite(imu_time) & np.all(np.isfinite(angular_rate), axis=1)
        imu_time = imu_time[finite]
        angular_rate = angular_rate[finite]

        if len(imu_time):
            angular_metrics = vector_metrics(angular_rate)
            result["angular_velocity"] = {
                "source": TOPICS["imu"] + ".angular_velocity",
                "axis_order": ["roll", "pitch", "yaw"],
                **angular_metrics,
                "peak_abs_axis": [
                    float(value)
                    for value in np.max(np.abs(angular_rate), axis=0)
                ],
            }
            result["dominant_oscillation"] = dominant_frequency(
                imu_time,
                angular_rate,
                ("roll_rate", "pitch_rate", "yaw_rate"),
                oscillation_band[0],
                oscillation_band[1],
            )
            imu_overlap = (odom_time >= imu_time[0]) & (odom_time <= imu_time[-1])
            rate_norm = np.full(len(odom_time), np.nan)

            if np.any(imu_overlap):
                interpolated_rate = interpolate_columns(
                    odom_time[imu_overlap], imu_time, angular_rate
                )
                rate_norm[imu_overlap] = np.linalg.norm(
                    interpolated_rate, axis=1
                )

            signals["rate_norm"] = rate_norm

    thrust_records = raw["attitude_target"]
    thrust_source = TOPICS["attitude_target"] + ".thrust"

    if not thrust_records and debug:
        thrust_records = [(item[0], item[3]) for item in debug]
        thrust_source = TOPICS["debug"] + ".des_thr"

    thrust = np.asarray(
        [
            value
            for time_s, value in thrust_records
            if phase["start"] <= time_s < phase["end"] and math.isfinite(value)
        ],
        dtype=float,
    )

    if len(thrust):
        lower_saturation = thrust <= 1e-3
        upper_saturation = thrust >= 1.0 - 1e-3
        result["collective_thrust"] = {
            "source": thrust_source,
            "normalized_range": [0.0, 1.0],
            "mean": float(np.mean(thrust)),
            "p99": float(np.quantile(thrust, 0.99)),
            "max": float(np.max(thrust)),
            "min": float(np.min(thrust)),
            "lower_saturation_fraction": float(np.mean(lower_saturation)),
            "upper_saturation_fraction": float(np.mean(upper_saturation)),
            "saturation_fraction": float(
                np.mean(lower_saturation | upper_saturation)
            ),
        }

    esc = esc_metrics(raw["esc"], phase)

    if esc is not None:
        result["esc_rpm"] = esc

    return result, signals


def first_true_dwell(time_s, condition, dwell_s):
    start = None

    for index, is_true in enumerate(condition):
        if is_true and start is None:
            start = index
        elif not is_true:
            start = None

        if start is not None and time_s[index] - time_s[start] >= dwell_s:
            return start

    return None


def calculate_post_hover_settling(signals, phase, args):
    time_s = signals["time"]
    conditions = {
        "position_error": (
            signals["position_error_norm"], args.settling_position_error_m
        ),
        "speed": (signals["speed_norm"], args.settling_speed_m_s),
        "tilt": (signals["tilt_deg"], args.settling_tilt_deg),
        "angular_rate": (signals["rate_norm"], args.settling_rate_rad_s),
    }
    missing = [name for name, (values, _) in conditions.items() if values is None]

    if missing:
        return {
            "available": False,
            "reason": "missing signals: " + ", ".join(missing),
        }

    finite = np.ones(len(time_s), dtype=bool)

    for values, _ in conditions.values():
        finite &= np.isfinite(values)

    time_s = time_s[finite]

    if len(time_s) < 2:
        return {"available": False, "reason": "insufficient overlapping samples"}

    within = np.ones(len(time_s), dtype=bool)
    condition_results = {}

    for name, (values, threshold) in conditions.items():
        selected = values[finite]
        current = selected <= threshold
        within &= current
        violation = np.flatnonzero(~current)
        condition_results[name] = {
            "threshold": float(threshold),
            "unit": {
                "position_error": "m",
                "speed": "m/s",
                "tilt": "deg",
                "angular_rate": "rad/s",
            }[name],
            "fraction_within": float(np.mean(current)),
            "last_violation_s": (
                float(time_s[violation[-1]] - phase["start"])
                if len(violation)
                else None
            ),
        }

    first_index = first_true_dwell(time_s, within, args.settling_dwell_s)
    violation = np.flatnonzero(~within)
    permanent_index = int(violation[-1] + 1) if len(violation) else 0
    permanent_valid = (
        permanent_index < len(time_s)
        and np.all(within[permanent_index:])
        and time_s[-1] - time_s[permanent_index] >= args.settling_dwell_s
    )
    settling_time = (
        float(time_s[permanent_index] - phase["start"])
        if permanent_valid
        else None
    )
    return {
        "available": True,
        "definition": (
            "time after CMD_CTRL->AUTO_HOVER until all limits remain satisfied "
            "through the end of the recorded hover"
        ),
        "dwell_requirement_s": float(args.settling_dwell_s),
        "settling_time_s": settling_time,
        "settled_by_phase_end": settling_time is not None,
        "first_sustained_window_s": (
            float(time_s[first_index] - phase["start"])
            if first_index is not None
            else None
        ),
        "fraction_all_limits_satisfied": float(np.mean(within)),
        "conditions": condition_results,
    }


def format_number(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def phase_markdown_row(label, phase):
    position = phase.get("position_error") or {}
    drift = phase.get("position_drift") or {}
    velocity = phase.get("velocity") or {}
    attitude = phase.get("attitude") or {}
    roll = attitude.get("roll_deg") or {}
    pitch = attitude.get("pitch_deg") or {}
    rate = phase.get("angular_velocity") or {}
    thrust = phase.get("collective_thrust") or {}
    oscillation = phase.get("dominant_oscillation") or {}
    roll_text = "{} / {}".format(
        format_number(roll.get("rms")),
        format_number(roll.get("peak_abs")),
    )
    pitch_text = "{} / {}".format(
        format_number(pitch.get("rms")),
        format_number(pitch.get("peak_abs")),
    )
    rate_text = "{} / {}".format(
        format_number(rate.get("rms_norm")),
        format_number(rate.get("max_norm")),
    )
    saturation = thrust.get("saturation_fraction")
    saturation_text = (
        f"{100.0 * saturation:.3f}%" if saturation is not None else "n/a"
    )
    frequency = (
        oscillation.get("frequency_hz")
        if oscillation.get("available")
        else None
    )
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        label,
        format_number(phase["duration_s"]),
        phase["samples"]["odometry"],
        format_number(position.get("rms_norm")),
        format_number(drift.get("max_m")),
        format_number(velocity.get("rms_norm")),
        roll_text,
        pitch_text,
        rate_text,
        saturation_text,
        format_number(frequency),
    )


def render_markdown(result):
    lines = [
        "# LZF trajectory phase analysis",
        "",
        f"- Bag: `{result['bag']}`",
        f"- Bag SHA256: `{result['bag_sha256']}`",
        f"- Duration: {result['bag_duration_s']:.3f} s",
        f"- Control odometry: `{result['odometry']['topic']}`",
        (
            "- Phase boundary rule: longest CMD_CTRL segment overlapping "
            "`/setpoints_cmd`, bracketed by the adjacent AUTO_HOVER segments."
        ),
        "",
        "## Phase metrics",
        "",
        (
            "| Phase | Duration (s) | Odom samples | Position RMSE (m) | "
            "Drift max (m) | Velocity RMS (m/s) | Roll RMS / peak (deg) | "
            "Pitch RMS / peak (deg) | Rate RMS / peak (rad/s) | "
            "Thrust saturation | Dominant rate frequency (Hz) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "pre_hover": "Pre AUTO_HOVER",
        "trajectory": "CMD_CTRL trajectory",
        "post_hover": "Post AUTO_HOVER",
    }

    for name in ("pre_hover", "trajectory", "post_hover"):
        lines.append(phase_markdown_row(labels[name], result["phases"][name]))

    diagnostics = result["diagnostic_windows"]
    lines.extend(
        [
            "",
            "## Diagnostic windows",
            "",
            (
                "The active-command interval isolates tracking, the timeout tail "
                "captures the last-command hold, and the post-hover split prevents "
                "the initial recovery transient from being hidden by a 60 s RMS."
            ),
            "",
            (
                "| Window | Duration (s) | Odom samples | Position RMSE (m) | "
                "Drift max (m) | Velocity RMS (m/s) | Roll RMS / peak (deg) | "
                "Pitch RMS / peak (deg) | Rate RMS / peak (rad/s) | "
                "Thrust saturation | Dominant rate frequency (Hz) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    diagnostic_labels = {
        "trajectory_active_command": "Active trajectory commands",
        "trajectory_timeout_tail": "Command-timeout tail",
        "post_hover_initial": "Post hover initial transient",
        "post_hover_steady": "Post hover after transient",
    }

    for name in diagnostic_labels:
        if name in diagnostics:
            lines.append(
                phase_markdown_row(diagnostic_labels[name], diagnostics[name])
            )

    comparison = result["hover_comparison"]
    lines.extend(
        [
            "",
            "## Equal-duration hover comparison",
            "",
            (
                "Both windows are {:.3f} s: the end of pre-hover and the start "
                "of post-hover.".format(comparison["window_duration_s"])
            ),
            "",
            (
                "| Window | Duration (s) | Odom samples | Position RMSE (m) | "
                "Drift max (m) | Velocity RMS (m/s) | Roll RMS / peak (deg) | "
                "Pitch RMS / peak (deg) | Rate RMS / peak (rad/s) | "
                "Thrust saturation | Dominant rate frequency (Hz) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            phase_markdown_row("Pre hover", comparison["pre_hover"]),
            phase_markdown_row("Post hover", comparison["post_hover"]),
        ]
    )

    settling = result["post_hover_settling"]
    lines.extend(["", "## Post-hover settling", ""])

    if settling.get("available"):
        lines.extend(
            [
                (
                    "- Settling time: {} s".format(
                        format_number(settling.get("settling_time_s"))
                    )
                ),
                (
                    "- First sustained {:.1f} s in-limit window: {} s".format(
                        settling["dwell_requirement_s"],
                        format_number(settling.get("first_sustained_window_s")),
                    )
                ),
                (
                    "- Fraction satisfying all limits: {:.2f}%".format(
                        100.0 * settling["fraction_all_limits_satisfied"]
                    )
                ),
                "",
                "| Condition | Limit | Fraction within | Last violation (s) |",
                "|---|---:|---:|---:|",
            ]
        )

        for name, condition in settling["conditions"].items():
            lines.append(
                "| {} | {} {} | {:.2f}% | {} |".format(
                    name,
                    format_number(condition["threshold"]),
                    condition["unit"],
                    100.0 * condition["fraction_within"],
                    format_number(condition["last_violation_s"]),
                )
            )
    else:
        lines.append(f"Unavailable: {settling.get('reason', 'unknown reason')}")

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            (
                "- Position error uses `/debugPx4ctrl.des_p_*`; drift is also "
                "reported against the median measured position in each phase's first second."
            ),
            (
                "- Angular-rate spectra use IMU body rates, linear detrending, "
                "uniform resampling, and Welch PSD. A trajectory peak can include commanded motion."
            ),
            (
                "- Normalized collective-thrust saturation is measured at 0 and 1. "
                "ESC RPM saturation is not inferred because ESCStatus provides no RPM limit."
            ),
            (
                "- CMD_CTRL duration includes the controller's final-command hold "
                "until command timeout returns the FSM to AUTO_HOVER."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args):
    bag = args.bag.resolve()

    if not bag.is_file():
        raise RuntimeError(f"missing bag: {bag}")

    bag_data = load_bag(bag, args.odom_topic)
    raw = bag_data["data"]
    segments = build_fsm_segments(raw["fsm"], bag_data["end"])
    phases = select_trial_phases(segments, raw["command"])

    if not raw["odometry"]:
        raise RuntimeError(
            f"bag has no finite messages on {bag_data['odometry_topic']}"
        )

    velocity_in_body = bool(bag_data["odom_velocity_in_body"])
    odometry = normalize_odometry(raw["odometry"], velocity_in_body)

    if not len(odometry["time"]):
        raise RuntimeError("odometry has no finite samples with valid quaternions")

    scored_phases = {}
    internal_signals = {}

    for name, phase in phases.items():
        scored, signals = score_phase(
            name,
            phase,
            bag_data,
            odometry,
            (args.oscillation_min_hz, args.oscillation_max_hz),
        )
        scored_phases[name] = scored
        internal_signals[name] = signals

    command_time = np.asarray(raw["command"], dtype=float)
    diagnostic_intervals = {}

    if len(command_time):
        active_start = float(command_time[0])
        active_end = float(command_time[-1])

        if active_end > active_start:
            diagnostic_intervals["trajectory_active_command"] = {
                "state": 3,
                "start": active_start,
                "end": active_end,
            }

        if phases["trajectory"]["end"] > active_end:
            diagnostic_intervals["trajectory_timeout_tail"] = {
                "state": 3,
                "start": active_end,
                "end": phases["trajectory"]["end"],
            }

    post_hover = phases["post_hover"]
    transient_end = min(
        post_hover["end"], post_hover["start"] + args.post_transient_window_s
    )
    diagnostic_intervals["post_hover_initial"] = {
        "state": 2,
        "start": post_hover["start"],
        "end": transient_end,
    }

    if post_hover["end"] - transient_end > 0.1:
        diagnostic_intervals["post_hover_steady"] = {
            "state": 2,
            "start": transient_end,
            "end": post_hover["end"],
        }

    diagnostic_windows = {}

    for name, interval in diagnostic_intervals.items():
        scored, _ = score_phase(
            name,
            interval,
            bag_data,
            odometry,
            (args.oscillation_min_hz, args.oscillation_max_hz),
        )
        diagnostic_windows[name] = scored

    comparison_duration = min(
        args.hover_comparison_window_s,
        phases["pre_hover"]["end"] - phases["pre_hover"]["start"],
        phases["post_hover"]["end"] - phases["post_hover"]["start"],
    )
    comparison_intervals = {
        "pre_hover": {
            "state": 2,
            "start": phases["pre_hover"]["end"] - comparison_duration,
            "end": phases["pre_hover"]["end"],
        },
        "post_hover": {
            "state": 2,
            "start": phases["post_hover"]["start"],
            "end": phases["post_hover"]["start"] + comparison_duration,
        },
    }
    hover_comparison = {"window_duration_s": float(comparison_duration)}

    for name, interval in comparison_intervals.items():
        scored, _ = score_phase(
            name,
            interval,
            bag_data,
            odometry,
            (args.oscillation_min_hz, args.oscillation_max_hz),
        )
        hover_comparison[name] = scored

    settling = calculate_post_hover_settling(
        internal_signals["post_hover"], phases["post_hover"], args
    )
    warnings = []

    if bag_data["odom_velocity_in_body"] is None:
        warnings.append(
            "/px4ctrl_param missing; odometry velocity was assumed to be world-frame"
        )

    if not raw["debug"]:
        warnings.append("/debugPx4ctrl missing; position tracking error is unavailable")

    if not raw["imu"]:
        warnings.append("/mavros/imu/data missing; rate and oscillation metrics are unavailable")

    if not raw["attitude_target"] and not raw["debug"]:
        warnings.append("normalized collective thrust is unavailable")

    result = {
        "schema_version": 1,
        "bag": str(bag),
        "bag_sha256": sha256_file(bag),
        "bag_size_bytes": int(bag.stat().st_size),
        "bag_duration_s": float(bag_data["duration"]),
        "bag_start_unix_s": float(bag_data["start"]),
        "odometry": {
            "topic": bag_data["odometry_topic"],
            "velocity_input_frame": (
                "body" if velocity_in_body else "world"
            ),
            "velocity_metrics_frame": "world",
            "px4ctrl_param_odom_vel_in_body": bag_data[
                "odom_velocity_in_body"
            ],
        },
        "phase_detection": {
            "method": (
                "CMD_CTRL FSM segment with the most /setpoints_cmd samples; "
                "nearest preceding and following AUTO_HOVER FSM segments"
            ),
            "fsm_topic": TOPICS["fsm"],
            "command_topic": TOPICS["command"],
            "command_samples": int(len(command_time)),
            "command_start_s_from_bag": (
                float(command_time[0] - bag_data["start"])
                if len(command_time)
                else None
            ),
            "command_end_s_from_bag": (
                float(command_time[-1] - bag_data["start"])
                if len(command_time)
                else None
            ),
            "fsm_segments": [
                {
                    "state": FSM_NAMES.get(segment["state"], str(segment["state"])),
                    "start_s_from_bag": float(
                        segment["start"] - bag_data["start"]
                    ),
                    "end_s_from_bag": float(
                        segment["end"] - bag_data["start"]
                    ),
                    "duration_s": float(segment["end"] - segment["start"]),
                }
                for segment in segments
            ],
        },
        "topics": bag_data["topic_metadata"],
        "phases": scored_phases,
        "diagnostic_windows": diagnostic_windows,
        "hover_comparison": hover_comparison,
        "post_hover_settling": settling,
        "warnings": warnings,
    }
    return result


def main():
    args = parse_args()
    numeric_options = {
        "--oscillation-min-hz": args.oscillation_min_hz,
        "--oscillation-max-hz": args.oscillation_max_hz,
        "--settling-position-error-m": args.settling_position_error_m,
        "--settling-speed-m-s": args.settling_speed_m_s,
        "--settling-tilt-deg": args.settling_tilt_deg,
        "--settling-rate-rad-s": args.settling_rate_rad_s,
        "--settling-dwell-s": args.settling_dwell_s,
        "--post-transient-window-s": args.post_transient_window_s,
        "--hover-comparison-window-s": args.hover_comparison_window_s,
    }

    for option, value in numeric_options.items():
        if not math.isfinite(value) or value <= 0.0:
            print(f"error: {option} must be finite and positive", file=sys.stderr)
            return 2

    if args.oscillation_max_hz <= args.oscillation_min_hz:
        print(
            "error: --oscillation-max-hz must exceed --oscillation-min-hz",
            file=sys.stderr,
        )
        return 2

    try:
        result = analyze(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "trajectory_phases.json"
    markdown_path = output_dir / "trajectory_phases.md"
    json_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
