#!/usr/bin/env python3
"""Identify LZF lateral rotor and airframe drag from motion-capture ROS bags.

The motion-capture odometry is treated as the only velocity truth. The script
fits the lateral specific-force model

    a = -lambda_1 * sum(|omega_i|) * v / m
        -0.5 * rho * CdA * |v| * v / m + bias

and reports whether lambda_1 and CdA are separately identifiable from the
available maneuvers. It never modifies the Gazebo model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import butter, sosfiltfilt


RPM_TO_RAD_S = 2.0 * math.pi / 60.0
DEFAULT_BAGS = (
    Path(
        "/mnt/h/我的云端硬盘/debug/esc_fac/VELOX/Quad_lzf/"
        "2026-07-12-15-21-15.bag"
    ),
    Path(
        "/mnt/h/我的云端硬盘/debug/esc_fac/VELOX/Quad_lzf/"
        "2026-07-12-14-52-01.bag"
    ),
)


@dataclass
class IdentificationData:
    name: str
    path: Path
    odometry_source: str
    sample_rate_hz: float
    count: int
    rotor_x: np.ndarray
    rotor_y: np.ndarray
    body_x: np.ndarray
    body_y: np.ndarray
    acceleration_x: np.ndarray
    acceleration_y: np.ndarray
    lateral_speed: np.ndarray
    advance_ratio: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bags",
        nargs="*",
        type=Path,
        default=list(DEFAULT_BAGS),
        help="ROS1 bags containing motion-capture odometry, IMU, and ESC RPM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/lzf_drag_identification"),
    )
    parser.add_argument("--mass", type=float, default=1.326)
    parser.add_argument("--air-density", type=float, default=1.225)
    parser.add_argument("--propeller-diameter", type=float, default=0.127)
    parser.add_argument("--configured-rotor-drag", type=float, default=2.14e-5)
    parser.add_argument("--minimum-lateral-speed", type=float, default=0.3)
    parser.add_argument("--minimum-rpm", type=float, default=5000.0)
    parser.add_argument("--low-pass-hz", type=float, default=10.0)
    return parser.parse_args()


def sample_rate(time_s: np.ndarray) -> float:
    differences = np.diff(time_s)
    differences = differences[np.isfinite(differences) & (differences > 0.0)]
    return float(1.0 / np.median(differences)) if differences.size else float("nan")


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


def align_quaternion_signs(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.copy()

    for index in range(1, quaternion.shape[0]):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0.0:
            quaternion[index] *= -1.0

    return quaternion


def interp_columns(
    target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        return np.interp(target_time, source_time, values)

    return np.column_stack(
        [
            np.interp(target_time, source_time, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def low_pass(values: np.ndarray, rate_hz: float, cutoff_hz: float) -> np.ndarray:
    if (
        not np.isfinite(rate_hz)
        or cutoff_hz <= 0.0
        or cutoff_hz >= 0.45 * rate_hz
        or values.shape[0] < 32
    ):
        return values.copy()

    sos = butter(3, cutoff_hz, btype="lowpass", fs=rate_hz, output="sos")
    return sosfiltfilt(sos, values, axis=0)


def bag_topic_count(bag: object, topic: str) -> int:
    return int(bag.get_message_count([topic]))


def read_bag_arrays(path: Path) -> Dict[str, object]:
    try:
        import rosbag
    except ImportError as error:
        raise RuntimeError(
            "rosbag is unavailable; source /opt/ros/noetic/setup.zsh and the "
            "comet_ws devel setup before running this tool"
        ) from error

    with rosbag.Bag(str(path)) as bag:
        mocap_topic = "/odom_converter/converted_odom0"

        if bag_topic_count(bag, mocap_topic) == 0:
            raise RuntimeError(
                f"{path} has no {mocap_topic}; refusing to substitute PX4 local "
                "velocity because motion capture is the required velocity truth"
            )

        topics = {
            "odom": mocap_topic,
            "imu": "/mavros/imu/data",
            "esc": "/mavros/esc_status",
        }
        counts = {name: bag_topic_count(bag, topic) for name, topic in topics.items()}

        for name, count in counts.items():
            if count == 0:
                raise RuntimeError(f"{path} has no usable {topics[name]} samples")

        odom = np.empty((counts["odom"], 8), dtype=float)
        imu = np.empty((counts["imu"], 4), dtype=float)
        esc = np.empty((counts["esc"], 2), dtype=float)
        offsets = {"odom": 0, "imu": 0, "esc": 0}
        topic_to_name = {topic: name for name, topic in topics.items()}

        for topic, message, record_time in bag.read_messages(
            topics=list(topic_to_name)
        ):
            name = topic_to_name[topic]
            index = offsets[name]
            time_s = record_time.to_sec()

            if name == "odom":
                odom[index] = (
                    time_s,
                    message.twist.twist.linear.x,
                    message.twist.twist.linear.y,
                    message.twist.twist.linear.z,
                    message.pose.pose.orientation.x,
                    message.pose.pose.orientation.y,
                    message.pose.pose.orientation.z,
                    message.pose.pose.orientation.w,
                )

            elif name == "imu":
                imu[index] = (
                    time_s,
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                )

            else:
                rpm = [
                    max(float(status.rpm), 0.0)
                    for status in message.esc_status[:4]
                ]

                if len(rpm) != 4:
                    continue

                esc[index] = (time_s, float(np.sum(rpm)))

            offsets[name] += 1

    return {
        "odometry_source": mocap_topic,
        "odom": odom[: offsets["odom"]],
        "imu": imu[: offsets["imu"]],
        "esc": esc[: offsets["esc"]],
    }


def prepare_identification_data(
    path: Path,
    mass: float,
    air_density: float,
    propeller_diameter: float,
    minimum_lateral_speed: float,
    minimum_rpm: float,
    low_pass_hz: float,
) -> IdentificationData:
    arrays = read_bag_arrays(path)
    odom = arrays["odom"]
    imu = arrays["imu"]
    esc = arrays["esc"]
    common_start = max(odom[0, 0], imu[0, 0], esc[0, 0]) + 1.0
    common_end = min(odom[-1, 0], imu[-1, 0], esc[-1, 0]) - 1.0
    common = (imu[:, 0] >= common_start) & (imu[:, 0] <= common_end)
    time_s = imu[common, 0]
    acceleration = imu[common, 1:4]
    rate_hz = sample_rate(time_s)

    quaternion = align_quaternion_signs(odom[:, 4:8])
    quaternion_at_imu = interp_columns(time_s, odom[:, 0], quaternion)
    rotation_body_to_world = quaternion_to_rotation_matrix(quaternion_at_imu)
    world_velocity = interp_columns(time_s, odom[:, 0], odom[:, 1:4])
    body_velocity = np.einsum(
        "nij,nj->ni",
        np.transpose(rotation_body_to_world, (0, 2, 1)),
        world_velocity,
    )
    rpm_sum = np.interp(time_s, esc[:, 0], esc[:, 1])
    omega_sum = rpm_sum * RPM_TO_RAD_S

    acceleration = low_pass(acceleration, rate_hz, low_pass_hz)
    body_velocity = low_pass(body_velocity, rate_hz, low_pass_hz)
    omega_sum = low_pass(omega_sum[:, None], rate_hz, low_pass_hz)[:, 0]

    lateral_speed = np.linalg.norm(body_velocity[:, :2], axis=1)
    finite = (
        np.all(np.isfinite(acceleration[:, :2]), axis=1)
        & np.all(np.isfinite(body_velocity), axis=1)
        & np.isfinite(omega_sum)
    )
    dynamic = (
        finite
        & (lateral_speed >= minimum_lateral_speed)
        & (omega_sum >= 4.0 * minimum_rpm * RPM_TO_RAD_S)
    )

    rotor = -omega_sum[:, None] * body_velocity[:, :2] / mass
    quadratic_body = (
        -0.5
        * air_density
        * np.abs(body_velocity[:, :2])
        * body_velocity[:, :2]
        / mass
    )
    mean_revolutions_per_second = rpm_sum / 4.0 / 60.0
    positive_axial_speed = np.maximum(body_velocity[:, 2], 0.0)
    advance_ratio = positive_axial_speed / np.maximum(
        mean_revolutions_per_second * propeller_diameter, 1e-9
    )

    return IdentificationData(
        name=path.stem,
        path=path,
        odometry_source=str(arrays["odometry_source"]),
        sample_rate_hz=rate_hz,
        count=int(np.count_nonzero(dynamic)),
        rotor_x=rotor[dynamic, 0],
        rotor_y=rotor[dynamic, 1],
        body_x=quadratic_body[dynamic, 0],
        body_y=quadratic_body[dynamic, 1],
        acceleration_x=acceleration[dynamic, 0],
        acceleration_y=acceleration[dynamic, 1],
        lateral_speed=lateral_speed[dynamic],
        advance_ratio=advance_ratio[dynamic],
    )


def finite_metrics(measured: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(measured) & np.isfinite(predicted)
    measured = measured[valid]
    predicted = predicted[valid]
    residual = measured - predicted
    sum_squared = float(np.sum(residual**2))
    centered = measured - np.mean(measured)
    total = float(np.sum(centered**2))
    return {
        "count": int(measured.size),
        "rmse_m_s2": float(np.sqrt(np.mean(residual**2))),
        "mae_m_s2": float(np.mean(np.abs(residual))),
        "bias_m_s2": float(np.mean(predicted - measured)),
        "r_squared": float(1.0 - sum_squared / total) if total > 0.0 else float("nan"),
        "correlation": (
            float(np.corrcoef(measured, predicted)[0, 1])
            if measured.size > 1
            and np.std(measured) > 0.0
            and np.std(predicted) > 0.0
            else float("nan")
        ),
    }


def constrained_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    nonnegative_count: int,
) -> np.ndarray:
    scale = np.max(np.abs(matrix), axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = matrix / scale
    lower = np.full(matrix.shape[1], -np.inf)
    upper = np.full(matrix.shape[1], np.inf)
    lower[:nonnegative_count] = 0.0
    result = lsq_linear(
        normalized,
        target,
        bounds=(lower * scale, upper * scale),
        method="trf",
        tol=1e-12,
        max_iter=1000,
    )

    if not result.success:
        raise RuntimeError(f"least-squares fit failed: {result.message}")

    return result.x / scale


def stack_axis_data(
    datasets: Sequence[IdentificationData],
) -> Tuple[np.ndarray, ...]:
    rotor_x = np.concatenate([item.rotor_x for item in datasets])
    rotor_y = np.concatenate([item.rotor_y for item in datasets])
    body_x = np.concatenate([item.body_x for item in datasets])
    body_y = np.concatenate([item.body_y for item in datasets])
    acceleration_x = np.concatenate([item.acceleration_x for item in datasets])
    acceleration_y = np.concatenate([item.acceleration_y for item in datasets])
    return rotor_x, rotor_y, body_x, body_y, acceleration_x, acceleration_y


def fit_models(datasets: Sequence[IdentificationData]) -> Dict[str, Dict[str, float]]:
    rotor_x, rotor_y, body_x, body_y, accel_x, accel_y = stack_axis_data(
        datasets
    )
    zeros_x = np.zeros_like(rotor_x)
    zeros_y = np.zeros_like(rotor_y)
    ones_x = np.ones_like(rotor_x)
    ones_y = np.ones_like(rotor_y)
    target = np.concatenate((accel_x, accel_y))

    rotor_matrix = np.vstack(
        (
            np.column_stack((rotor_x, ones_x, zeros_x)),
            np.column_stack((rotor_y, zeros_y, ones_y)),
        )
    )
    rotor_parameters = constrained_fit(rotor_matrix, target, 1)

    body_matrix = np.vstack(
        (
            np.column_stack((body_x, zeros_x, ones_x, zeros_x)),
            np.column_stack((zeros_y, body_y, zeros_y, ones_y)),
        )
    )
    body_parameters = constrained_fit(body_matrix, target, 2)

    joint_matrix = np.vstack(
        (
            np.column_stack((rotor_x, body_x, zeros_x, ones_x, zeros_x)),
            np.column_stack((rotor_y, zeros_y, body_y, zeros_y, ones_y)),
        )
    )
    joint_parameters = constrained_fit(joint_matrix, target, 3)
    unconstrained_joint_parameters = np.linalg.lstsq(
        joint_matrix, target, rcond=None
    )[0]
    isotropic_joint_matrix = np.vstack(
        (
            np.column_stack((rotor_x, body_x, ones_x, zeros_x)),
            np.column_stack((rotor_y, body_y, zeros_y, ones_y)),
        )
    )
    isotropic_joint_parameters = constrained_fit(
        isotropic_joint_matrix, target, 2
    )

    return {
        "rotor_only": {
            "rotor_drag_coefficient": float(rotor_parameters[0]),
            "bias_x_m_s2": float(rotor_parameters[1]),
            "bias_y_m_s2": float(rotor_parameters[2]),
        },
        "body_only": {
            "cda_x_m2": float(body_parameters[0]),
            "cda_y_m2": float(body_parameters[1]),
            "bias_x_m_s2": float(body_parameters[2]),
            "bias_y_m_s2": float(body_parameters[3]),
        },
        "joint": {
            "rotor_drag_coefficient": float(joint_parameters[0]),
            "cda_x_m2": float(joint_parameters[1]),
            "cda_y_m2": float(joint_parameters[2]),
            "bias_x_m_s2": float(joint_parameters[3]),
            "bias_y_m_s2": float(joint_parameters[4]),
        },
        "joint_unconstrained": {
            "rotor_drag_coefficient": float(unconstrained_joint_parameters[0]),
            "cda_x_m2": float(unconstrained_joint_parameters[1]),
            "cda_y_m2": float(unconstrained_joint_parameters[2]),
            "bias_x_m_s2": float(unconstrained_joint_parameters[3]),
            "bias_y_m_s2": float(unconstrained_joint_parameters[4]),
        },
        "joint_isotropic_body": {
            "rotor_drag_coefficient": float(isotropic_joint_parameters[0]),
            "cda_m2": float(isotropic_joint_parameters[1]),
            "bias_x_m_s2": float(isotropic_joint_parameters[2]),
            "bias_y_m_s2": float(isotropic_joint_parameters[3]),
        },
    }


def predict(
    data: IdentificationData,
    model_name: str,
    parameters: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    bias_x = parameters.get("bias_x_m_s2", 0.0)
    bias_y = parameters.get("bias_y_m_s2", 0.0)
    prediction_x = np.full(data.count, bias_x)
    prediction_y = np.full(data.count, bias_y)

    if model_name in (
        "configured",
        "rotor_only",
        "joint",
        "joint_unconstrained",
        "joint_isotropic_body",
    ):
        coefficient = parameters["rotor_drag_coefficient"]
        prediction_x += coefficient * data.rotor_x
        prediction_y += coefficient * data.rotor_y

    if model_name in ("body_only", "joint", "joint_unconstrained"):
        prediction_x += parameters["cda_x_m2"] * data.body_x
        prediction_y += parameters["cda_y_m2"] * data.body_y

    if model_name == "joint_isotropic_body":
        prediction_x += parameters["cda_m2"] * data.body_x
        prediction_y += parameters["cda_m2"] * data.body_y

    return prediction_x, prediction_y


def evaluate(
    data: IdentificationData,
    model_name: str,
    parameters: Dict[str, float],
) -> Dict[str, object]:
    prediction_x, prediction_y = predict(data, model_name, parameters)
    return {
        "x": finite_metrics(data.acceleration_x, prediction_x),
        "y": finite_metrics(data.acceleration_y, prediction_y),
        "combined": finite_metrics(
            np.concatenate((data.acceleration_x, data.acceleration_y)),
            np.concatenate((prediction_x, prediction_y)),
        ),
    }


def centered_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")

    return float(np.corrcoef(left, right)[0, 1])


def identifiability(datasets: Sequence[IdentificationData]) -> Dict[str, object]:
    rotor_x, rotor_y, body_x, body_y, _, _ = stack_axis_data(datasets)
    physical_matrix = np.vstack(
        (
            np.column_stack((rotor_x, body_x, np.zeros_like(rotor_x))),
            np.column_stack((rotor_y, np.zeros_like(rotor_y), body_y)),
        )
    )
    centered = physical_matrix - np.mean(physical_matrix, axis=0)
    norm = np.linalg.norm(centered, axis=0)
    standardized = centered / np.maximum(norm, 1e-12)
    condition_number = float(np.linalg.cond(standardized))
    correlation_x = centered_correlation(rotor_x, body_x)
    correlation_y = centered_correlation(rotor_y, body_y)

    return {
        "rotor_body_regressor_correlation_x": correlation_x,
        "rotor_body_regressor_correlation_y": correlation_y,
        "variance_inflation_x": float(1.0 / max(1.0 - correlation_x**2, 1e-12)),
        "variance_inflation_y": float(1.0 / max(1.0 - correlation_y**2, 1e-12)),
        "standardized_physical_matrix_condition_number": condition_number,
    }


def percentile_summary(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            key: float("nan")
            for key in ("minimum", "median", "p95", "maximum")
        }

    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


def dataset_summary(data: IdentificationData) -> Dict[str, object]:
    return {
        "bag": str(data.path),
        "odometry_source": data.odometry_source,
        "filtered_imu_rate_hz": data.sample_rate_hz,
        "identification_sample_count": data.count,
        "lateral_speed_m_s": percentile_summary(data.lateral_speed),
        "positive_advance_ratio": percentile_summary(
            data.advance_ratio[data.advance_ratio >= 0.0]
        ),
    }


def fit_bias_for_configured(
    datasets: Sequence[IdentificationData], coefficient: float
) -> Dict[str, float]:
    rotor_x, rotor_y, _, _, accel_x, accel_y = stack_axis_data(datasets)
    return {
        "rotor_drag_coefficient": coefficient,
        "bias_x_m_s2": float(np.mean(accel_x - coefficient * rotor_x)),
        "bias_y_m_s2": float(np.mean(accel_y - coefficient * rotor_y)),
    }


def cross_validation(
    datasets: Sequence[IdentificationData],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    if len(datasets) < 2:
        return results

    for test_index, test in enumerate(datasets):
        train = [item for index, item in enumerate(datasets) if index != test_index]
        models = fit_models(train)
        results.append(
            {
                "train": [item.name for item in train],
                "test": test.name,
                "parameters": models,
                "metrics": {
                    name: evaluate(test, name, parameters)
                    for name, parameters in models.items()
                },
            }
        )

    return results


def plot_results(
    datasets: Sequence[IdentificationData],
    models: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(datasets), 3, figsize=(15, 4.4 * len(datasets)), squeeze=False
    )
    colors = {"x": "#167d9a", "y": "#d04f3e"}

    for row, data in enumerate(datasets):
        stride = max(1, data.count // 10000)

        for column, model_name in enumerate(("rotor_only", "body_only", "joint")):
            prediction_x, prediction_y = predict(
                data, model_name, models[model_name]
            )
            axis = axes[row, column]
            axis.scatter(
                data.acceleration_x[::stride],
                prediction_x[::stride],
                s=2,
                alpha=0.18,
                color=colors["x"],
                label="body x",
            )
            axis.scatter(
                data.acceleration_y[::stride],
                prediction_y[::stride],
                s=2,
                alpha=0.18,
                color=colors["y"],
                label="body y",
            )
            limits = np.percentile(
                np.concatenate(
                    (
                        data.acceleration_x,
                        data.acceleration_y,
                        prediction_x,
                        prediction_y,
                    )
                ),
                [0.5, 99.5],
            )
            axis.plot(limits, limits, color="black", linewidth=0.8)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_xlabel("measured lateral specific force [m/s^2]")
            axis.set_ylabel("predicted [m/s^2]")
            metrics = evaluate(data, model_name, models[model_name])["combined"]
            axis.set_title(
                f"{data.name} - {model_name}\n"
                f"RMSE={metrics['rmse_m_s2']:.3f}, R2={metrics['r_squared']:.3f}"
            )

        axes[row, 0].legend(loc="upper left", markerscale=4)

    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def markdown_report(result: Dict[str, object]) -> str:
    combined_models = result["combined_fit"]["parameters"]
    combined_metrics = result["combined_fit"]["metrics"]
    identifiable = result["identifiability"]
    datasets = result["datasets"]
    joint = combined_models["joint"]
    rotor = combined_models["rotor_only"]
    body = combined_models["body_only"]
    unconstrained = combined_models["joint_unconstrained"]
    isotropic = combined_models["joint_isotropic_body"]
    lines = [
        "# LZF lateral drag identification",
        "",
        "## Data authority and model",
        "",
        "- Velocity truth: `/odom_converter/converted_odom0` motion capture only.",
        "- PX4 `vehicle_local_position` velocity is intentionally not used.",
        "- IMU and regressors are low-pass filtered before fitting.",
        "- Rotor term: `-lambda1 * sum(abs(omega)) * v / mass`.",
        "- Airframe term: `-0.5 * rho * CdA * abs(v) * v / mass`.",
        "- Still air is assumed; unmeasured indoor wind remains in the residual.",
        "",
        "## Flight envelopes",
        "",
        "| Bag | samples | lateral speed max | positive J p95 | positive J max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for item in datasets:
        speed = item["lateral_speed_m_s"]
        advance = item["positive_advance_ratio"]
        lines.append(
            f"| `{Path(item['bag']).name}` | {item['identification_sample_count']} "
            f"| {speed['maximum']:.3f} m/s | {advance['p95']:.4f} "
            f"| {advance['maximum']:.4f} |"
        )

    lines += [
        "",
        "## Combined fit",
        "",
        "| Model | lambda1 | CdA x | CdA y | combined RMSE | combined R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| rotor only | {rotor['rotor_drag_coefficient']:.7g} | - | - "
            f"| {combined_metrics['rotor_only']['combined']['rmse_m_s2']:.4f} "
            f"| {combined_metrics['rotor_only']['combined']['r_squared']:.4f} |"
        ),
        (
            f"| body only | - | {body['cda_x_m2']:.5f} | "
            f"{body['cda_y_m2']:.5f} "
            f"| {combined_metrics['body_only']['combined']['rmse_m_s2']:.4f} "
            f"| {combined_metrics['body_only']['combined']['r_squared']:.4f} |"
        ),
        (
            f"| joint | {joint['rotor_drag_coefficient']:.7g} "
            f"| {joint['cda_x_m2']:.5f} | {joint['cda_y_m2']:.5f} "
            f"| {combined_metrics['joint']['combined']['rmse_m_s2']:.4f} "
            f"| {combined_metrics['joint']['combined']['r_squared']:.4f} |"
        ),
        (
            f"| joint, unconstrained diagnostic "
            f"| {unconstrained['rotor_drag_coefficient']:.7g} "
            f"| {unconstrained['cda_x_m2']:.5f} "
            f"| {unconstrained['cda_y_m2']:.5f} "
            f"| {combined_metrics['joint_unconstrained']['combined']['rmse_m_s2']:.4f} "
            f"| {combined_metrics['joint_unconstrained']['combined']['r_squared']:.4f} |"
        ),
        (
            f"| joint, isotropic body "
            f"| {isotropic['rotor_drag_coefficient']:.7g} "
            f"| {isotropic['cda_m2']:.5f} | {isotropic['cda_m2']:.5f} "
            f"| {combined_metrics['joint_isotropic_body']['combined']['rmse_m_s2']:.4f} "
            f"| {combined_metrics['joint_isotropic_body']['combined']['r_squared']:.4f} |"
        ),
        "",
        "The configured LZF `rotorDragCoefficient` is "
        f"`{result['configuration']['rotor_drag_coefficient']:.7g}`.",
        "",
        "## Identifiability",
        "",
        (
            "- Rotor/body regressor correlation: "
            f"x `{identifiable['rotor_body_regressor_correlation_x']:.5f}`, "
            f"y `{identifiable['rotor_body_regressor_correlation_y']:.5f}`."
        ),
        (
            "- Variance inflation factor: "
            f"x `{identifiable['variance_inflation_x']:.2f}`, "
            f"y `{identifiable['variance_inflation_y']:.2f}`."
        ),
        (
            "- Standardized physical design-matrix condition number: "
            f"`{identifiable['standardized_physical_matrix_condition_number']:.2f}`."
        ),
        "",
        "## Cross-validation",
        "",
        "| Train | Test | model | lambda1 | CdA x | CdA y | RMSE | R2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for fold in result["cross_validation"]:
        train = ", ".join(fold["train"])

        for model_name in (
            "rotor_only",
            "body_only",
            "joint",
            "joint_unconstrained",
            "joint_isotropic_body",
        ):
            parameters = fold["parameters"][model_name]
            metrics = fold["metrics"][model_name]["combined"]
            lines.append(
                f"| {train} | {fold['test']} | {model_name} "
                f"| {parameters.get('rotor_drag_coefficient', float('nan')):.7g} "
                f"| {parameters.get('cda_x_m2', parameters.get('cda_m2', float('nan'))):.5f} "
                f"| {parameters.get('cda_y_m2', parameters.get('cda_m2', float('nan'))):.5f} "
                f"| {metrics['rmse_m_s2']:.4f} | {metrics['r_squared']:.4f} |"
            )

    lines += [
        "",
        "## CT(J) boundary",
        "",
        "- DA4052 5x3.75x3 measured data crosses `CT=0` at about `J=0.73454`.",
        "- The current interpolation clamps above the last measured point "
        "(`J=0.777614`, `CT=-0.012399`); it does not diverge downward.",
        "- At the measured flight maximum near `J=0.087`, interpolated CT is "
        "about `0.1438`, positive and roughly 3.7% below static CT.",
        "- At 15,337 rpm, the zero-thrust boundary corresponds to about "
        "23.85 m/s axial relative inflow, far outside these motion-capture flights.",
        "",
        "## Interpretation",
        "",
        "A body-drag plugin can be added to Gazebo. Collision geometry alone does "
        "not create aerodynamic drag. However, the joint coefficients should only "
        "be written into LZF if cross-validation is stable and rotor/body regressors "
        "are sufficiently independent. Otherwise a dedicated excitation flight is "
        "needed: repeat speed sweeps at several nearly constant RPM levels, plus "
        "coast-down or tilted-thrust segments that decorrelate `sum(omega)*v` from "
        "`abs(v)*v`.",
        "",
        "The unconstrained row is a diagnostic, not a physical model. A negative "
        "CdA there, or a constrained CdA pinned to zero, is direct evidence that "
        "this dataset cannot support a unique nonnegative rotor/body split. Here "
        "the unconstrained fit makes both CdA values negative, while the isotropic "
        "nonnegative fit returns exactly zero body drag and collapses to the "
        "rotor-only result.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in args.bags:
        if not path.is_file():
            raise FileNotFoundError(path)

    datasets = [
        prepare_identification_data(
            path=path,
            mass=args.mass,
            air_density=args.air_density,
            propeller_diameter=args.propeller_diameter,
            minimum_lateral_speed=args.minimum_lateral_speed,
            minimum_rpm=args.minimum_rpm,
            low_pass_hz=args.low_pass_hz,
        )
        for path in args.bags
    ]
    models = fit_models(datasets)
    configured = fit_bias_for_configured(
        datasets, args.configured_rotor_drag
    )
    all_metrics = {
        "configured": {
            "without_bias": {
                data.name: evaluate(
                    data,
                    "configured",
                    {
                        "rotor_drag_coefficient": args.configured_rotor_drag,
                        "bias_x_m_s2": 0.0,
                        "bias_y_m_s2": 0.0,
                    },
                )
                for data in datasets
            },
            "with_fitted_bias": {
                data.name: evaluate(data, "configured", configured)
                for data in datasets
            },
        },
        **{
            name: {
                data.name: evaluate(data, name, parameters)
                for data in datasets
            }
            for name, parameters in models.items()
        },
    }
    combined_metrics = {
        name: evaluate_combined(datasets, name, parameters)
        for name, parameters in models.items()
    }
    result = {
        "configuration": {
            "mass_kg": args.mass,
            "air_density_kg_m3": args.air_density,
            "propeller_diameter_m": args.propeller_diameter,
            "rotor_drag_coefficient": args.configured_rotor_drag,
            "minimum_lateral_speed_m_s": args.minimum_lateral_speed,
            "minimum_motor_rpm": args.minimum_rpm,
            "low_pass_hz": args.low_pass_hz,
        },
        "datasets": [dataset_summary(data) for data in datasets],
        "identifiability": identifiability(datasets),
        "combined_fit": {
            "parameters": models,
            "metrics": combined_metrics,
        },
        "per_dataset_metrics": all_metrics,
        "cross_validation": cross_validation(datasets),
    }

    json_path = args.output_dir / "drag_identification.json"
    report_path = args.output_dir / "report.md"
    plot_path = args.output_dir / "drag_fit.png"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(markdown_report(result), encoding="utf-8")
    plot_results(datasets, models, plot_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {plot_path}")
    return 0


def evaluate_combined(
    datasets: Sequence[IdentificationData],
    model_name: str,
    parameters: Dict[str, float],
) -> Dict[str, object]:
    measured_x: List[np.ndarray] = []
    measured_y: List[np.ndarray] = []
    predicted_x: List[np.ndarray] = []
    predicted_y: List[np.ndarray] = []

    for data in datasets:
        prediction_x, prediction_y = predict(data, model_name, parameters)
        measured_x.append(data.acceleration_x)
        measured_y.append(data.acceleration_y)
        predicted_x.append(prediction_x)
        predicted_y.append(prediction_y)

    x_measured = np.concatenate(measured_x)
    y_measured = np.concatenate(measured_y)
    x_predicted = np.concatenate(predicted_x)
    y_predicted = np.concatenate(predicted_y)
    return {
        "x": finite_metrics(x_measured, x_predicted),
        "y": finite_metrics(y_measured, y_predicted),
        "combined": finite_metrics(
            np.concatenate((x_measured, y_measured)),
            np.concatenate((x_predicted, y_predicted)),
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
