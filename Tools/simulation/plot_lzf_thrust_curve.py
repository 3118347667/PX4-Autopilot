#!/usr/bin/env python3

"""Generate steady, static-air LZF single-rotor thrust curves from its SDF."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


PX4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = (
    PX4_ROOT
    / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/lzf/lzf.sdf"
)


def child_float(element: ET.Element, name: str) -> float:
    child = element.find(name)

    if child is None or child.text is None:
        raise ValueError(f"Missing <{name}> in {element.get('name', element.tag)}")

    return float(child.text)


def load_parameters(sdf_path: Path) -> dict[str, float]:
    root = ET.parse(sdf_path).getroot()
    model = root.find("model")

    if model is None:
        raise ValueError(f"No <model> found in {sdf_path}")

    motors = [
        plugin
        for plugin in model.findall("plugin")
        if plugin.get("filename") == "libgazebo_motor_model.so"
    ]
    mavlink = next(
        (
            plugin
            for plugin in model.findall("plugin")
            if plugin.get("filename") == "libgazebo_mavlink_interface.so"
        ),
        None,
    )

    if not motors or mavlink is None:
        raise ValueError(f"Missing motor or MAVLink plugin in {sdf_path}")

    motor = motors[0]
    propeller_model = motor.findtext("propellerModelEnabled", "false").lower()

    if propeller_model != "true":
        raise ValueError("The LZF UIUC propeller model is not enabled")

    thrust_table = [
        float(value) for value in motor.findtext("thrustCoefficientTable", "").split()
    ]
    power_table = [
        float(value) for value in motor.findtext("powerCoefficientTable", "").split()
    ]

    if not thrust_table or not power_table:
        raise ValueError("The propeller coefficient tables are empty")

    return {
        "air_density": child_float(motor, "airDensity"),
        "diameter": child_float(motor, "propellerDiameter"),
        "max_rot_velocity": child_float(motor, "maxRotVelocity"),
        "moment_constant": child_float(motor, "momentConstant"),
        "static_ct": thrust_table[0],
        "static_cp": power_table[0],
        "voltage_min": child_float(mavlink, "motorSpeedVoltageMin"),
        "voltage_max": child_float(mavlink, "motorSpeedVoltageMax"),
        "idle_rpm": child_float(mavlink, "motorIdleRpm"),
        "loaded_kv": child_float(mavlink, "motorLoadedKv"),
        "motor_kv": child_float(mavlink, "motorKv"),
    }


def evaluate_curve(
    command: np.ndarray, voltage: float, parameters: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voltage = float(
        np.clip(voltage, parameters["voltage_min"], parameters["voltage_max"])
    )
    loaded_rpm = (
        parameters["idle_rpm"] + parameters["loaded_kv"] * voltage * command
    )
    no_load_limit_rpm = parameters["motor_kv"] * voltage
    model_limit_rpm = parameters["max_rot_velocity"] * 60.0 / (2.0 * math.pi)
    rpm = np.minimum(loaded_rpm, min(no_load_limit_rpm, model_limit_rpm))
    revolutions_per_second = rpm / 60.0
    common = (
        parameters["air_density"]
        * np.square(revolutions_per_second)
        * parameters["diameter"] ** 4
    )
    thrust = parameters["static_ct"] * common
    torque = (
        parameters["static_cp"]
        * common
        * parameters["diameter"]
        / (2.0 * math.pi)
    )
    return rpm, thrust, torque


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    parser.add_argument(
        "--output-dir", type=Path, default=PX4_ROOT / "build/lzf_thrust_curve"
    )
    parser.add_argument(
        "--voltages",
        type=float,
        nargs="+",
        default=[15.0, 18.0, 21.0, 22.2, 25.2],
    )
    parser.add_argument("--mass", type=float, default=1.326)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = load_parameters(args.sdf)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    command = np.linspace(0.0, 1.0, 201)
    voltages = sorted(set(args.voltages))
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for voltage in voltages:
        curves[voltage] = evaluate_curve(command, voltage, parameters)

    csv_path = output_dir / "lzf_single_motor_static_thrust.csv"

    with csv_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["normalized_command", "voltage_v", "rpm", "thrust_n", "torque_nm"]
        )

        for voltage in voltages:
            rpm, thrust, torque = curves[voltage]

            for values in zip(command, rpm, thrust, torque):
                writer.writerow([f"{value:.9g}" for value in (values[0], voltage, *values[1:])])

    static_cq = parameters["static_cp"] / (2.0 * math.pi)
    effective_moment_constant = (
        parameters["diameter"] * static_cq / parameters["static_ct"]
    )
    main_voltage = max(voltages)
    main_rpm, main_thrust, main_torque = curves[main_voltage]
    sample_commands = np.linspace(0.0, 1.0, 11)
    sample_indices = np.rint(sample_commands * (command.size - 1)).astype(int)
    summary = {
        "conditions": {
            "armed": True,
            "steady_state": True,
            "axial_airspeed_m_s": 0.0,
            "advance_ratio": 0.0,
        },
        "source_sdf": str(args.sdf),
        "parameters": {
            **parameters,
            "static_cq": static_cq,
            "effective_static_moment_constant_m": effective_moment_constant,
        },
        "main_curve_voltage_v": main_voltage,
        "single_motor_hover_thrust_n": args.mass * 9.80665 / 4.0,
        "main_curve_samples": [
            {
                "normalized_command": float(command[index]),
                "rpm": float(main_rpm[index]),
                "thrust_n": float(main_thrust[index]),
                "torque_nm": float(main_torque[index]),
            }
            for index in sample_indices
        ],
    }
    json_path = output_dir / "lzf_single_motor_static_thrust.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")

    figure, axis = plt.subplots(figsize=(9.0, 5.6))

    for voltage in voltages:
        _, thrust, _ = curves[voltage]
        is_main = voltage == main_voltage
        axis.plot(
            command,
            thrust,
            linewidth=2.6 if is_main else 1.5,
            label=f"{voltage:.1f} V",
            zorder=3 if is_main else 2,
        )

    hover_thrust = args.mass * 9.80665 / 4.0
    axis.axhline(
        hover_thrust,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Hover target ({hover_thrust:.3f} N/motor)",
    )
    axis.set(
        title="LZF single-motor steady static thrust",
        xlabel="Normalized motor command",
        ylabel="Thrust (N)",
        xlim=(0.0, 1.0),
        ylim=(0.0, None),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    png_path = output_dir / "lzf_single_motor_static_thrust.png"
    figure.savefig(png_path, dpi=180)
    plt.close(figure)

    print(f"csv:     {csv_path}")
    print(f"json:    {json_path}")
    print(f"plot:    {png_path}")
    print(
        "static coefficients: "
        f"CT={parameters['static_ct']:.6f}, "
        f"CP={parameters['static_cp']:.6f}, "
        f"CQ={static_cq:.9f}, "
        f"Q/T={effective_moment_constant:.10f} m"
    )


if __name__ == "__main__":
    main()
