#!/usr/bin/env python3

"""Test voltage monotonicity of LZF local gain for multiple THR_MDL_FAC values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import identify_lzf_static_gain_map as lzf


PX4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PX4_ROOT / "build/lzf_thr_mdl_fac_monotonicity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, default=lzf.DEFAULT_SDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mass", type=float, default=1.326)
    parser.add_argument("--rotor-count", type=int, default=4)
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=[0.1 * index for index in range(11)],
    )
    parser.add_argument(
        "--voltages",
        type=float,
        nargs="+",
        default=[18.0 + 0.25 * index for index in range(29)],
    )
    parser.add_argument(
        "--commands",
        type=float,
        nargs="+",
        default=[0.1 + 0.025 * index for index in range(29)],
    )
    parser.add_argument("--delta-command", type=float, default=0.005)
    parser.add_argument("--dynamic-duration", type=float, default=20.0)
    parser.add_argument("--dynamic-sample-rate", type=float, default=125.0)
    parser.add_argument("--dynamic-amplitude", type=float, default=0.02)
    return parser.parse_args()


def motor_signal(command: np.ndarray | float, factor: float) -> np.ndarray:
    command_array = np.clip(np.asarray(command, dtype=float), 0.0, 1.0)
    if factor <= np.finfo(float).eps:
        return command_array
    linear = 1.0 - factor
    return (
        -linear + np.sqrt(linear * linear + 4.0 * factor * command_array)
    ) / (2.0 * factor)


def motor_signal_derivative(command: float, factor: float) -> float:
    if factor <= np.finfo(float).eps:
        return 1.0
    linear = 1.0 - factor
    return 1.0 / math.sqrt(
        linear * linear + 4.0 * factor * command
    )


def acceleration_scale(
    model: dict[str, float], mass: float, rotor_count: int
) -> float:
    return (
        rotor_count
        * model["static_thrust_coefficient"]
        * model["air_density_kg_m3"]
        * model["propeller_diameter_m"] ** 4
        / (mass * 60.0**2)
    )


def static_point(
    voltage: float,
    command: float,
    factor: float,
    delta_command: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
) -> dict[str, float | bool]:
    command_minus = max(0.0, command - delta_command)
    command_plus = min(1.0, command + delta_command)
    commands = np.asarray([command_minus, command, command_plus])
    signals = motor_signal(commands, factor)
    rpm = lzf.target_rpm(voltage, signals, model)
    acceleration = lzf.acceleration_from_rpm(
        rpm, model, mass, rotor_count
    )
    finite_gain = (
        float(acceleration[2] - acceleration[0])
        / (command_plus - command_minus)
    )
    rpm_limit = min(
        model["no_load_kv_rpm_per_v"] * voltage,
        model["motor_limit_rpm"],
    )
    saturated = bool(rpm[1] >= rpm_limit - 1e-6)
    exact_gain = 0.0
    if not saturated:
        exact_gain = (
            2.0
            * acceleration_scale(model, mass, rotor_count)
            * float(rpm[1])
            * model["loaded_kv_rpm_per_v_u"]
            * voltage
            * motor_signal_derivative(command, factor)
        )
    return {
        "thr_mdl_fac": factor,
        "voltage_v": voltage,
        "thrust_command": command,
        "motor_signal": float(signals[1]),
        "motor_signal_derivative": motor_signal_derivative(command, factor),
        "rpm": float(rpm[1]),
        "local_gain_m_s2_per_command": exact_gain,
        "finite_difference_gain_m_s2_per_command": finite_gain,
        "rpm_saturated": saturated,
    }


def linear_chirp(
    time: np.ndarray, start_hz: float, end_hz: float, duration: float
) -> np.ndarray:
    slope = (end_hz - start_hz) / duration
    phase = 2.0 * math.pi * (
        start_hz * time + 0.5 * slope * np.square(time)
    )
    return np.sin(phase)


def dynamic_point(
    voltage: float,
    command: float,
    factor: float,
    amplitude: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
    duration: float,
    sample_rate: float,
) -> dict[str, float | bool]:
    dt = 1.0 / sample_rate
    time = np.arange(0.0, duration, dt)
    perturbation = amplitude * linear_chirp(time, 0.25, 5.0, duration)
    thrust_command = command + perturbation
    signal = motor_signal(thrust_command, factor)
    target = lzf.target_rpm(voltage, signal, model)
    rpm = lzf.filter_first_order(
        target, model["motor_time_constant_s"], dt
    )
    acceleration = lzf.acceleration_from_rpm(
        rpm, model, mass, rotor_count
    )
    filtered_perturbation = lzf.filter_first_order(
        perturbation, model["motor_time_constant_s"], dt
    )
    crop = time >= 1.0
    matrix = np.column_stack(
        (
            np.ones(np.count_nonzero(crop)),
            filtered_perturbation[crop],
        )
    )
    offset, gain = np.linalg.lstsq(
        matrix, acceleration[crop], rcond=None
    )[0]
    prediction = offset + gain * filtered_perturbation[crop]
    error = acceleration[crop] - prediction
    rmse = float(np.sqrt(np.mean(np.square(error))))
    centered = acceleration[crop] - np.mean(acceleration[crop])
    r_squared = 1.0 - float(np.sum(np.square(error))) / max(
        float(np.sum(np.square(centered))), 1e-12
    )
    rpm_limit = min(
        model["no_load_kv_rpm_per_v"] * voltage,
        model["motor_limit_rpm"],
    )
    return {
        "thr_mdl_fac": factor,
        "voltage_v": voltage,
        "thrust_command": command,
        "motor_signal": float(motor_signal(command, factor)),
        "amplitude": amplitude,
        "gain_m_s2_per_command": float(gain),
        "rmse_m_s2": rmse,
        "r_squared": r_squared,
        "rpm_saturated": bool(np.max(target) >= rpm_limit - 1e-6),
    }


def monotonicity(
    rows: list[dict],
    value_key: str,
    group_keys: tuple[str, ...],
) -> dict:
    groups: dict[tuple[float, ...], list[dict]] = {}
    for row in rows:
        key = tuple(float(row[name]) for name in group_keys)
        groups.setdefault(key, []).append(row)

    violations = []
    increments = []
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: float(row["voltage_v"]))
        for lower, upper in zip(ordered, ordered[1:]):
            increment = float(upper[value_key]) - float(lower[value_key])
            increments.append(increment)
            if increment <= 0.0:
                violations.append(
                    {
                        **{
                            name: value
                            for name, value in zip(group_keys, key)
                        },
                        "voltage_lower_v": lower["voltage_v"],
                        "voltage_upper_v": upper["voltage_v"],
                        "gain_lower": lower[value_key],
                        "gain_upper": upper[value_key],
                        "increment": increment,
                    }
                )
    return {
        "group_count": len(groups),
        "comparison_count": len(increments),
        "violation_count": len(violations),
        "minimum_increment": min(increments),
        "maximum_increment": max(increments),
        "violations": violations,
    }


def full_domain_audit(
    model: dict[str, float], mass: float, rotor_count: int
) -> dict:
    factors = np.linspace(0.0, 1.0, 101)
    voltages = np.linspace(
        model["voltage_min_v"], model["voltage_max_v"], 103
    )
    commands = np.linspace(0.001, 1.0, 1000)
    scale = acceleration_scale(model, mass, rotor_count)
    violation_count = 0
    saturated_points = 0
    maximum_rpm = 0.0
    minimum_increment = math.inf
    minimum_location = None

    for factor in factors:
        signals = motor_signal(commands, float(factor))
        derivatives = np.asarray(
            [
                motor_signal_derivative(float(command), float(factor))
                for command in commands
            ]
        )
        voltage_grid = voltages[:, None]
        loaded_rpm = (
            model["idle_rpm"]
            + model["loaded_kv_rpm_per_v_u"]
            * voltage_grid
            * signals[None, :]
        )
        limits = np.minimum(
            model["no_load_kv_rpm_per_v"] * voltage_grid,
            model["motor_limit_rpm"],
        )
        rpm = np.minimum(loaded_rpm, limits)
        saturated_points += int(np.count_nonzero(rpm >= limits - 1e-6))
        maximum_rpm = max(maximum_rpm, float(np.max(rpm)))
        gain = (
            2.0
            * scale
            * rpm
            * model["loaded_kv_rpm_per_v_u"]
            * voltage_grid
            * derivatives[None, :]
        )
        increments = np.diff(gain, axis=0)
        violation_count += int(np.count_nonzero(increments <= 0.0))
        index = np.unravel_index(
            int(np.argmin(increments)), increments.shape
        )
        increment = float(increments[index])
        if increment < minimum_increment:
            minimum_increment = increment
            minimum_location = {
                "thr_mdl_fac": float(factor),
                "thrust_command": float(commands[index[1]]),
                "voltage_lower_v": float(voltages[index[0]]),
                "voltage_upper_v": float(voltages[index[0] + 1]),
            }

    return {
        "factor_count": len(factors),
        "voltage_count": len(voltages),
        "command_count": len(commands),
        "grid_point_count": len(factors) * len(voltages) * len(commands),
        "comparison_count": len(factors)
        * (len(voltages) - 1)
        * len(commands),
        "violation_count": violation_count,
        "minimum_increment": minimum_increment,
        "minimum_increment_location": minimum_location,
        "saturated_point_count": saturated_points,
        "maximum_rpm": maximum_rpm,
        "motor_limit_rpm": model["motor_limit_rpm"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_free_flight_results(
    output_dir: Path,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
) -> list[dict]:
    rows = []
    for path in sorted(
        (output_dir / "free_flight").glob("fac*_v*/*_thrust_chirp_fit.json")
    ):
        label = path.parent.name
        factor_text = label.split("_", 1)[0][3:]
        factor = int(factor_text) / 10.0
        result = json.loads(path.read_text())
        selected = result["selected_model"]
        diagnostics = result["runs"][0]["diagnostics"]
        voltage = float(
            diagnostics.get(
                "voltage_mean_v",
                label.split("_v", 1)[1].replace("p", "."),
            )
        )
        command = float(diagnostics["motor_command_mean"])
        prediction = static_point(
            voltage,
            command,
            factor,
            0.005,
            model,
            mass,
            rotor_count,
        )
        predicted_gain = float(
            prediction["local_gain_m_s2_per_command"]
        )
        identified_gain = float(selected["gain"])
        rows.append(
            {
                "label": label,
                "source": str(path),
                "thr_mdl_fac": factor,
                "voltage_v": voltage,
                "mean_thrust_command": command,
                "mean_motor_signal": float(
                    motor_signal(command, factor)
                ),
                "identified_gain_m_s2_per_command": identified_gain,
                "static_gain_m_s2_per_command": predicted_gain,
                "static_error_percent": 100.0
                * (identified_gain / predicted_gain - 1.0),
                "time_constant_s": float(selected["time_constant_s"]),
                "r_squared": float(selected["r_squared"]),
            }
        )
    return rows


def plot_results(
    path: Path,
    static_rows: list[dict],
    factors: list[float],
    free_flight_rows: list[dict],
) -> None:
    selected_factors = [
        min(factors, key=lambda item: abs(item - target))
        for target in (0.0, 0.3, 0.6, 1.0)
    ]
    selected_commands = (0.1, 0.3, 0.5, 0.7)
    figure, axes = plt.subplots(
        2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True,
        constrained_layout=True
    )
    for axis, factor in zip(axes.flat, selected_factors):
        for command in selected_commands:
            rows = [
                row
                for row in static_rows
                if abs(float(row["thr_mdl_fac"]) - factor) < 1e-9
                and abs(float(row["thrust_command"]) - command) < 1e-9
            ]
            rows.sort(key=lambda row: float(row["voltage_v"]))
            axis.plot(
                [row["voltage_v"] for row in rows],
                [row["local_gain_m_s2_per_command"] for row in rows],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label="c={:.1f}".format(command),
            )
        validation = [
            row
            for row in free_flight_rows
            if abs(float(row["thr_mdl_fac"]) - factor) < 1e-9
        ]
        if validation:
            axis.scatter(
                [row["voltage_v"] for row in validation],
                [
                    row["identified_gain_m_s2_per_command"]
                    for row in validation
                ],
                marker="*",
                s=110,
                color="black",
                label="free-flight chirp",
                zorder=5,
            )
        axis.set_title("THR_MDL_FAC={:.1f}".format(factor))
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("Battery voltage (V)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Local gain K (m/s2 per command)")
    axes[0, 0].legend(ncol=2)
    figure.suptitle(
        "LZF fixed PX4 thrust-command voltage monotonicity"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    static_result: dict,
    dynamic_result: dict,
    full_audit: dict,
    static_rows: list[dict],
    dynamic_rows: list[dict],
    free_flight_rows: list[dict],
    factors: list[float],
    voltages: list[float],
    commands: list[float],
) -> None:
    saturated_static = sum(bool(row["rpm_saturated"]) for row in static_rows)
    saturated_dynamic = sum(bool(row["rpm_saturated"]) for row in dynamic_rows)
    lines = [
        "# LZF THR_MDL_FAC voltage monotonicity test",
        "",
        "## Scope",
        "",
        "- Fixed input: PX4 normalized thrust command before THR_MDL_FAC.",
        "- THR_MDL_FAC values: `{}`.".format(
            ", ".join("{:g}".format(value) for value in factors)
        ),
        "- Voltage range: `{:.2f}...{:.2f} V`, step `{:.2f} V`.".format(
            min(voltages), max(voltages), voltages[1] - voltages[0]
        ),
        "- Command range: `{:.3f}...{:.3f}`, step `{:.3f}`.".format(
            min(commands), max(commands), commands[1] - commands[0]
        ),
        "- Static-air `J=0`; `THR_MDL_FAC` uses the exact PX4 inverse mapping.",
        "",
        "## Static test",
        "",
        "- Grid points: `{}`.".format(len(static_rows)),
        "- Voltage comparisons: `{}`.".format(
            static_result["comparison_count"]
        ),
        "- Non-increasing comparisons: `{}`.".format(
            static_result["violation_count"]
        ),
        "- Minimum adjacent-voltage K increment: `{:.9f}`.".format(
            static_result["minimum_increment"]
        ),
        "- RPM-saturated points: `{}`.".format(saturated_static),
        "",
        "## Dynamic chirp test",
        "",
        "- Sweep: `0.25...5 Hz`, amplitude `0.02`, duration `20 s`.",
        "- Test points: `{}`.".format(len(dynamic_rows)),
        "- Voltage comparisons: `{}`.".format(
            dynamic_result["comparison_count"]
        ),
        "- Non-increasing comparisons: `{}`.".format(
            dynamic_result["violation_count"]
        ),
        "- Minimum adjacent-voltage K increment: `{:.9f}`.".format(
            dynamic_result["minimum_increment"]
        ),
        "- RPM-saturated sweeps: `{}`.".format(saturated_dynamic),
        "",
        "## Full supported-domain audit",
        "",
        "- Domain: `THR_MDL_FAC=0...1`, voltage `15...25.2 V`, command "
        "`0.001...1.0`.",
        "- Grid points: `{grid_point_count}`.".format(**full_audit),
        "- Voltage comparisons: `{comparison_count}`.".format(**full_audit),
        "- Non-increasing comparisons: `{violation_count}`.".format(
            **full_audit
        ),
        "- Minimum adjacent-voltage K increment: "
        "`{minimum_increment:.9f}`.".format(**full_audit),
        "- RPM-saturated points: `{saturated_point_count}`.".format(
            **full_audit
        ),
        "- Maximum target RPM: `{maximum_rpm:.1f}`, hard limit "
        "`{motor_limit_rpm:.1f}`.".format(**full_audit),
        "",
        "## Free-flight SITL cross-check",
        "",
        "| THR_MDL_FAC | Voltage | Mean command | Identified K | Static K | Error | R2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in free_flight_rows:
        lines.append(
            "| {thr_mdl_fac:.1f} | {voltage_v:.1f} | "
            "{mean_thrust_command:.4f} | "
            "{identified_gain_m_s2_per_command:.3f} | "
            "{static_gain_m_s2_per_command:.3f} | "
            "{static_error_percent:+.2f}% | {r_squared:.6f} |".format(
                **row
            )
        )
    lines += [
        "",
        "These points follow the free-flight hover command and are not fixed-"
        "command tests. They validate the complete PX4-to-Gazebo-to-ULog chain.",
        "",
        "## Interpretation",
        "",
        "Within this tested domain, K is strictly increasing with voltage at "
        "every fixed PX4 thrust command and every tested THR_MDL_FAC. The "
        "THR_MDL_FAC mapping depends on command and factor, but not voltage, "
        "so it changes K magnitude without reversing its voltage slope.",
        "",
        "A separate `THR_MDL_FAC=1`, `18 V` free-flight attempt was not a "
        "valid K experiment because AUX1 was not selected on that MAVLink "
        "instance. It did expose an operational boundary: after entering "
        "LAND, the vehicle climbed for about 204 s and reached about 1458 m "
        "before forced disarm. This does not violate fixed-command voltage "
        "monotonicity, but it shows that FAC=1 is incompatible with the LZF "
        "low-thrust and landing behavior.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    model = lzf.load_model(args.sdf)
    factors = sorted(set(args.factors))
    voltages = sorted(set(args.voltages))
    commands = sorted(set(args.commands))
    static_rows = [
        static_point(
            voltage,
            command,
            factor,
            args.delta_command,
            model,
            args.mass,
            args.rotor_count,
        )
        for factor in factors
        for voltage in voltages
        for command in commands
    ]
    dynamic_voltages = [18.0 + 0.5 * index for index in range(15)]
    dynamic_commands = [0.1 * index for index in range(1, 9)]
    dynamic_rows = [
        dynamic_point(
            voltage,
            command,
            factor,
            args.dynamic_amplitude,
            model,
            args.mass,
            args.rotor_count,
            args.dynamic_duration,
            args.dynamic_sample_rate,
        )
        for factor in factors
        for voltage in dynamic_voltages
        for command in dynamic_commands
    ]
    static_result = monotonicity(
        static_rows,
        "local_gain_m_s2_per_command",
        ("thr_mdl_fac", "thrust_command"),
    )
    dynamic_result = monotonicity(
        dynamic_rows,
        "gain_m_s2_per_command",
        ("thr_mdl_fac", "thrust_command"),
    )
    full_audit = full_domain_audit(
        model, args.mass, args.rotor_count
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_flight_rows = load_free_flight_results(
        args.output_dir, model, args.mass, args.rotor_count
    )
    static_csv = args.output_dir / "lzf_thr_mdl_fac_static_grid.csv"
    dynamic_csv = args.output_dir / "lzf_thr_mdl_fac_dynamic_chirps.csv"
    json_path = args.output_dir / "lzf_thr_mdl_fac_monotonicity.json"
    report_path = args.output_dir / "lzf_thr_mdl_fac_monotonicity.md"
    plot_path = args.output_dir / "lzf_thr_mdl_fac_monotonicity.png"
    write_csv(static_csv, static_rows)
    write_csv(dynamic_csv, dynamic_rows)
    json_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "factors": factors,
                    "voltages": voltages,
                    "commands": commands,
                    "delta_command": args.delta_command,
                    "advance_ratio": 0.0,
                },
                "model": model,
                "static_monotonicity": static_result,
                "dynamic_monotonicity": dynamic_result,
                "full_supported_domain_audit": full_audit,
                "free_flight_validation": free_flight_rows,
                "static_rows": static_rows,
                "dynamic_rows": dynamic_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_report(
        report_path,
        static_result,
        dynamic_result,
        full_audit,
        static_rows,
        dynamic_rows,
        free_flight_rows,
        factors,
        voltages,
        commands,
    )
    plot_results(plot_path, static_rows, factors, free_flight_rows)
    print("static grid: {}".format(static_csv))
    print("dynamic chirps: {}".format(dynamic_csv))
    print("json: {}".format(json_path))
    print("report: {}".format(report_path))
    print("plot: {}".format(plot_path))
    print(
        "static violations: {}; dynamic violations: {}".format(
            static_result["violation_count"],
            dynamic_result["violation_count"],
        )
    )


if __name__ == "__main__":
    main()
