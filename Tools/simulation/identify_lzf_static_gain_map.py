#!/usr/bin/env python3

"""Identify the static-air LZF collective-thrust gain over voltage and command."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


PX4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = (
    PX4_ROOT
    / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/lzf/lzf.sdf"
)
DEFAULT_OUTPUT = PX4_ROOT / "build/lzf_static_gain_map"
DEFAULT_HOVER_LABELS = tuple(
    "v{}".format(
        str(int(voltage)) if voltage.is_integer() else str(voltage).replace(".", "p")
    )
    for voltage in (18.0 + 0.5 * index for index in range(15))
)
DEFAULT_HOVER_FITS = (
    *(
        PX4_ROOT
        / "build/lzf_static_gain_map/free_flight"
        / label
        / "time_domain"
        / "{}_thrust_chirp_fit.json".format(label)
        for label in DEFAULT_HOVER_LABELS
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--voltages",
        type=float,
        nargs="+",
        default=[18.0 + 0.5 * index for index in range(15)],
    )
    parser.add_argument(
        "--commands",
        type=float,
        nargs="+",
        default=[0.1 + 0.05 * index for index in range(15)],
    )
    parser.add_argument(
        "--hover-fit",
        type=Path,
        action="append",
        default=None,
        help="Optional free-flight thrust-chirp fit JSON to overlay",
    )
    parser.add_argument("--mass", type=float, default=1.326)
    parser.add_argument("--rotor-count", type=int, default=4)
    parser.add_argument("--dynamic-duration", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=float, default=250.0)
    return parser.parse_args()


def required_float(element: ET.Element, name: str) -> float:
    text = element.findtext(name)
    if text is None:
        raise ValueError("missing <{}> in {}".format(name, element.get("name")))
    return float(text)


def load_model(path: Path) -> dict[str, float]:
    model = ET.parse(path).getroot().find("model")
    if model is None:
        raise ValueError("no model in {}".format(path))

    motors = [
        plugin
        for plugin in model.findall("plugin")
        if plugin.get("filename") == "libgazebo_motor_model.so"
        and plugin.findtext("propellerModelEnabled", "false").lower() == "true"
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
        raise ValueError("LZF motor or MAVLink plugin is missing")

    motor = motors[0]
    ct = np.fromstring(motor.findtext("thrustCoefficientTable", ""), sep=" ")
    if ct.size == 0:
        raise ValueError("empty thrust-coefficient table")

    return {
        "air_density_kg_m3": required_float(motor, "airDensity"),
        "propeller_diameter_m": required_float(motor, "propellerDiameter"),
        "static_thrust_coefficient": float(ct[0]),
        "motor_time_constant_s": required_float(motor, "timeConstantUp"),
        "motor_limit_rpm": required_float(motor, "maxRotVelocity")
        * 60.0
        / (2.0 * math.pi),
        "voltage_min_v": required_float(mavlink, "motorSpeedVoltageMin"),
        "voltage_max_v": required_float(mavlink, "motorSpeedVoltageMax"),
        "idle_rpm": required_float(mavlink, "motorIdleRpm"),
        "loaded_kv_rpm_per_v_u": required_float(mavlink, "motorLoadedKv"),
        "no_load_kv_rpm_per_v": required_float(mavlink, "motorKv"),
    }


def adaptive_amplitude(command: float) -> float:
    return min(0.03, 0.25 * command, 0.25 * (1.0 - command))


def target_rpm(
    voltage: np.ndarray | float,
    command: np.ndarray | float,
    model: dict[str, float],
) -> np.ndarray:
    voltage_array = np.clip(
        np.asarray(voltage, dtype=float),
        model["voltage_min_v"],
        model["voltage_max_v"],
    )
    command_array = np.clip(np.asarray(command, dtype=float), 0.0, 1.0)
    loaded = (
        model["idle_rpm"]
        + model["loaded_kv_rpm_per_v_u"] * voltage_array * command_array
    )
    limit = np.minimum(
        model["no_load_kv_rpm_per_v"] * voltage_array,
        model["motor_limit_rpm"],
    )
    return np.minimum(loaded, limit)


def acceleration_from_rpm(
    rpm: np.ndarray | float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
) -> np.ndarray:
    revolutions_per_second = np.asarray(rpm, dtype=float) / 60.0
    thrust = (
        model["static_thrust_coefficient"]
        * model["air_density_kg_m3"]
        * np.square(revolutions_per_second)
        * model["propeller_diameter_m"] ** 4
    )
    return rotor_count * thrust / mass


def static_point(
    voltage: float,
    command: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
) -> dict[str, float | bool]:
    amplitude = adaptive_amplitude(command)
    command_minus = command - amplitude
    command_plus = command + amplitude
    rpm_minus, rpm_center, rpm_plus = target_rpm(
        voltage,
        np.asarray([command_minus, command, command_plus]),
        model,
    )
    acceleration_minus, acceleration_center, acceleration_plus = (
        acceleration_from_rpm(
            np.asarray([rpm_minus, rpm_center, rpm_plus]),
            model,
            mass,
            rotor_count,
        )
    )
    gain = (acceleration_plus - acceleration_minus) / (2.0 * amplitude)
    rpm_limit = min(
        model["no_load_kv_rpm_per_v"] * voltage,
        model["motor_limit_rpm"],
    )
    return {
        "voltage_v": voltage,
        "command": command,
        "delta_command": amplitude,
        "rpm_minus": float(rpm_minus),
        "rpm_center": float(rpm_center),
        "rpm_plus": float(rpm_plus),
        "acceleration_minus_m_s2": float(acceleration_minus),
        "acceleration_center_m_s2": float(acceleration_center),
        "acceleration_plus_m_s2": float(acceleration_plus),
        "gain_m_s2_per_u": float(gain),
        "rpm_saturated": bool(rpm_plus >= rpm_limit - 1e-6),
        "command_saturated": bool(command_minus <= 0.0 or command_plus >= 1.0),
    }


def logarithmic_chirp(
    time: np.ndarray, start_hz: float, end_hz: float, duration: float
) -> np.ndarray:
    ratio = end_hz / start_hz
    phase = (
        2.0
        * math.pi
        * start_hz
        * duration
        / math.log(ratio)
        * (np.power(ratio, time / duration) - 1.0)
    )
    return np.sin(phase)


def filter_first_order(values: np.ndarray, tau: float, dt: float) -> np.ndarray:
    alpha = math.exp(-dt / tau)
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, values.size):
        filtered[index] = (
            alpha * filtered[index - 1] + (1.0 - alpha) * values[index]
        )
    return filtered


def dynamic_point(
    voltage: float,
    command_center: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
    duration: float,
    sample_rate: float,
) -> dict[str, float]:
    dt = 1.0 / sample_rate
    time = np.arange(0.0, duration, dt)
    amplitude = adaptive_amplitude(command_center)
    perturbation = amplitude * logarithmic_chirp(time, 0.25, 10.0, duration)
    command = command_center + perturbation
    rpm_target = target_rpm(voltage, command, model)
    rpm = filter_first_order(rpm_target, model["motor_time_constant_s"], dt)
    acceleration = acceleration_from_rpm(rpm, model, mass, rotor_count)

    crop = time >= 1.0
    observed = acceleration[crop]

    def regression(tau: float) -> tuple[float, float, float]:
        unit = filter_first_order(perturbation, tau, dt)[crop]
        matrix = np.column_stack((np.ones(unit.size), unit))
        offset, gain = np.linalg.lstsq(matrix, observed, rcond=None)[0]
        prediction = offset + gain * unit
        rmse = float(np.sqrt(np.mean(np.square(observed - prediction))))
        return float(gain), float(offset), rmse

    result = optimize.minimize_scalar(
        lambda tau: regression(tau)[2],
        bounds=(0.002, 0.100),
        method="bounded",
        options={"xatol": 1e-10},
    )
    gain, offset, rmse = regression(float(result.x))
    centered = observed - np.mean(observed)
    r_squared = 1.0 - rmse**2 / max(float(np.mean(np.square(centered))), 1e-12)
    return {
        "voltage_v": voltage,
        "command": command_center,
        "delta_command": amplitude,
        "start_frequency_hz": 0.25,
        "end_frequency_hz": 10.0,
        "duration_s": duration,
        "gain_m_s2_per_u": gain,
        "time_constant_s": float(result.x),
        "offset_m_s2": offset,
        "rmse_m_s2": rmse,
        "r_squared": r_squared,
    }


def hover_command(
    voltage: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
) -> float:
    thrust_per_rotor = mass * 9.80665 / rotor_count
    common = (
        model["static_thrust_coefficient"]
        * model["air_density_kg_m3"]
        * model["propeller_diameter_m"] ** 4
    )
    required_rpm = 60.0 * math.sqrt(thrust_per_rotor / common)
    return (
        (required_rpm - model["idle_rpm"])
        / (model["loaded_kv_rpm_per_v_u"] * voltage)
    )


def load_hover_fits(paths: list[Path]) -> list[dict[str, float | str]]:
    points = []
    for path in paths:
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        selected = result["selected_model"]
        for run in result["runs"]:
            diagnostics = run["diagnostics"]
            points.append(
                {
                    "source": str(path),
                    "voltage_v": diagnostics["voltage_mean_v"],
                    "command": diagnostics["motor_command_mean"],
                    "gain_m_s2_per_u": selected["gain"],
                    "time_constant_s": selected["time_constant_s"],
                }
            )
    return points


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_map(
    path: Path,
    voltages: list[float],
    commands: list[float],
    points: list[dict],
    hover_points: list[dict],
    hover_curve: list[dict],
) -> None:
    gain = np.asarray(
        [
            [point["gain_m_s2_per_u"] for point in points if point["voltage_v"] == v]
            for v in voltages
        ]
    ).T
    voltage_grid, command_grid = np.meshgrid(voltages, commands)
    figure, axis = plt.subplots(figsize=(10.0, 6.4), constrained_layout=True)
    image = axis.pcolormesh(
        voltage_grid,
        command_grid,
        gain,
        shading="nearest",
        cmap="viridis",
    )
    contours = axis.contour(
        voltage_grid,
        command_grid,
        gain,
        colors="white",
        linewidths=0.8,
        alpha=0.75,
    )
    axis.clabel(contours, fmt="%.0f", fontsize=8)
    axis.plot(
        [point["voltage_v"] for point in hover_curve],
        [point["command"] for point in hover_curve],
        color="white",
        linewidth=2.0,
        linestyle="--",
        label="Static J=0 hover line",
    )
    if hover_points:
        scatter = axis.scatter(
            [point["voltage_v"] for point in hover_points],
            [point["command"] for point in hover_points],
            c=[point["gain_m_s2_per_u"] for point in hover_points],
            cmap="viridis",
            edgecolor="red",
            linewidth=1.6,
            s=75,
            vmin=float(np.min(gain)),
            vmax=float(np.max(gain)),
            label="Free-flight chirp",
            zorder=4,
        )
        scatter.set_clim(float(np.min(gain)), float(np.max(gain)))
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Local collective gain K (m/s2 per command)")
    axis.set(
        title="LZF static-air local collective gain, THR_MDL_FAC=0",
        xlabel="Battery voltage (V)",
        ylabel="Normalized motor command",
        xlim=(min(voltages) - 0.25, max(voltages) + 0.25),
        ylim=(min(commands) - 0.025, max(commands) + 0.025),
    )
    axis.set_xticks(voltages)
    axis.set_yticks(commands)
    axis.grid(color="white", alpha=0.15)
    axis.legend(loc="upper left")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    model: dict,
    points: list[dict],
    dynamics: list[dict],
    hover_points: list[dict],
    hover_curve: list[dict],
    mass: float,
    rotor_count: int,
) -> None:
    gains = np.asarray([point["gain_m_s2_per_u"] for point in points])
    derivative_factor = (
        2.0
        * rotor_count
        * model["static_thrust_coefficient"]
        * model["air_density_kg_m3"]
        * model["propeller_diameter_m"] ** 4
        * model["loaded_kv_rpm_per_v_u"]
        / (mass * 60.0**2)
    )
    voltage_coefficient = derivative_factor * model["idle_rpm"]
    voltage_command_coefficient = (
        derivative_factor * model["loaded_kv_rpm_per_v_u"]
    )
    lines = [
        "# LZF voltage-command local gain map",
        "",
        "## Conditions",
        "",
        "- `THR_MDL_FAC=0`",
        "- Static air and advance ratio `J=0`",
        "- Four equal motor commands",
        "- Mass: `{:.3f} kg`".format(mass),
        "- Local gain from symmetric command perturbations",
        "- SDF motor voltage-RPM model and UIUC static thrust coefficient",
        "",
        "## Model parameters",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        "| Idle RPM | {:.3f} |".format(model["idle_rpm"]),
        "| Loaded KV | {:.3f} RPM/(V command) |".format(
            model["loaded_kv_rpm_per_v_u"]
        ),
        "| No-load KV limit | {:.3f} RPM/V |".format(
            model["no_load_kv_rpm_per_v"]
        ),
        "| Motor time constant | {:.3f} ms |".format(
            1000.0 * model["motor_time_constant_s"]
        ),
        "| Static CT | {:.6f} |".format(model["static_thrust_coefficient"]),
        "| Propeller diameter | {:.3f} m |".format(
            model["propeller_diameter_m"]
        ),
        "",
        "## Static map summary",
        "",
        "- Gain range: `{:.3f}...{:.3f} m/s2/u`".format(
            float(np.min(gains)), float(np.max(gains))
        ),
        "- Saturated grid points: `{}`".format(
            sum(bool(point["rpm_saturated"]) for point in points)
        ),
        "- Unsaturated closed form: "
        "`K(V,u) = {:.6f} V + {:.6f} V^2 u`".format(
            voltage_coefficient, voltage_command_coefficient
        ),
        "",
        "## Static K matrix",
        "",
    ]
    voltages = sorted({float(point["voltage_v"]) for point in points})
    commands = sorted({float(point["command"]) for point in points})
    lookup = {
        (float(point["voltage_v"]), float(point["command"])): float(
            point["gain_m_s2_per_u"]
        )
        for point in points
    }
    lines += [
        "| Command / voltage | {} |".format(
            " | ".join("{:g} V".format(voltage) for voltage in voltages)
        ),
        "| ---: | {} |".format(" | ".join("---:" for _ in voltages)),
    ]
    for command in commands:
        lines.append(
            "| {:.2f} | {} |".format(
                command,
                " | ".join(
                    "{:.3f}".format(lookup[(voltage, command)])
                    for voltage in voltages
                ),
            )
        )
    lines += [
        "",
        "## Representative dynamic sweeps",
        "",
        "| Voltage | Command | K | Tau | RMSE | R2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in dynamics:
        lines.append(
            "| {voltage_v:.1f} | {command:.2f} | "
            "{gain_m_s2_per_u:.3f} | {tau_ms:.3f} ms | "
            "{rmse_m_s2:.5f} | {r_squared:.6f} |".format(
                tau_ms=1000.0 * point["time_constant_s"], **point
            )
        )
    lines += [
        "",
        "## Static hover line",
        "",
        "| Voltage | Static hover command | Static K |",
        "| ---: | ---: | ---: |",
    ]
    for point in hover_curve:
        lines.append(
            "| {voltage_v:.1f} | {command:.4f} | "
            "{gain_m_s2_per_u:.3f} |".format(**point)
        )
    lines += [
        "",
        "## Free-flight SITL validation",
        "",
        "| Voltage | Mean command | Identified K | Static K | Error | Tau |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in hover_points:
        lines.append(
            "| {voltage_v:.1f} | {command:.4f} | "
            "{gain_m_s2_per_u:.3f} | {static_gain_m_s2_per_u:.3f} | "
            "{static_error_percent:+.2f}% | {tau_ms:.3f} ms |".format(
                tau_ms=1000.0 * point["time_constant_s"], **point
            )
        )
    lines += [
        "",
        "The static map is the deterministic `J=0` slice of the exact LZF SDF "
        "equations. The free-flight points include axial motion and therefore "
        "should not be expected to lie exactly on that slice.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    model = load_model(args.sdf)
    voltages = sorted(set(args.voltages))
    commands = sorted(set(args.commands))
    points = [
        static_point(
            voltage, command, model, args.mass, args.rotor_count
        )
        for voltage in voltages
        for command in commands
    ]
    dynamic_voltages = [18.0, 21.0, 23.0, 25.0]
    dynamic_commands = [0.15, 0.30, 0.50, 0.70]
    dynamics = [
        dynamic_point(
            voltage,
            command,
            model,
            args.mass,
            args.rotor_count,
            args.dynamic_duration,
            args.sample_rate,
        )
        for voltage in dynamic_voltages
        for command in dynamic_commands
    ]
    hover_curve = []
    for voltage in voltages:
        command = hover_command(
            voltage, model, args.mass, args.rotor_count
        )
        point = static_point(
            voltage, command, model, args.mass, args.rotor_count
        )
        hover_curve.append(
            {
                "voltage_v": voltage,
                "command": command,
                "gain_m_s2_per_u": point["gain_m_s2_per_u"],
            }
        )
    hover_paths = args.hover_fit
    if hover_paths is None:
        hover_paths = list(DEFAULT_HOVER_FITS)
    hover_points = load_hover_fits(hover_paths)
    for point in hover_points:
        prediction = static_point(
            float(point["voltage_v"]),
            float(point["command"]),
            model,
            args.mass,
            args.rotor_count,
        )
        static_gain = float(prediction["gain_m_s2_per_u"])
        point["static_gain_m_s2_per_u"] = static_gain
        point["static_error_percent"] = (
            100.0 * (float(point["gain_m_s2_per_u"]) / static_gain - 1.0)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    static_csv = args.output_dir / "lzf_static_gain_grid.csv"
    dynamic_csv = args.output_dir / "lzf_dynamic_gain_checks.csv"
    json_path = args.output_dir / "lzf_static_gain_map.json"
    report_path = args.output_dir / "lzf_static_gain_map.md"
    plot_path = args.output_dir / "lzf_static_gain_map.png"
    write_csv(static_csv, points)
    write_csv(dynamic_csv, dynamics)
    json_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "thr_mdl_fac": 0.0,
                    "advance_ratio": 0.0,
                    "mass_kg": args.mass,
                    "rotor_count": args.rotor_count,
                },
                "model": model,
                "static_grid": points,
                "dynamic_checks": dynamics,
                "static_hover_curve": hover_curve,
                "free_flight_validation": hover_points,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_report(
        report_path,
        model,
        points,
        dynamics,
        hover_points,
        hover_curve,
        args.mass,
        args.rotor_count,
    )
    plot_map(
        plot_path, voltages, commands, points, hover_points, hover_curve
    )
    print("static grid: {}".format(static_csv))
    print("dynamic checks: {}".format(dynamic_csv))
    print("json: {}".format(json_path))
    print("report: {}".format(report_path))
    print("plot: {}".format(plot_path))


if __name__ == "__main__":
    main()
