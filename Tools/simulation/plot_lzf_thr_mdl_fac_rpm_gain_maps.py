#!/usr/bin/env python3

"""Generate LZF command-to-RPM gain maps across THR_MDL_FAC."""

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
import plot_lzf_thr_mdl_fac_gain_map as thrust_maps
import test_lzf_thr_mdl_fac_monotonicity as fac_test


PX4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PX4_ROOT / "build/lzf_thr_mdl_fac_rpm_gain/gain_maps"
)
THRUST_FIT_ROOT = (
    PX4_ROOT / "build/lzf_thr_mdl_fac_monotonicity/free_flight"
)
CHAIN_FIT_ROOT = PX4_ROOT / "build/lzf_thr_mdl_fac_rpm_gain/free_flight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, default=lzf.DEFAULT_SDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--thr-mdl-fac",
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
        default=[0.025 * index for index in range(1, 33)],
    )
    parser.add_argument("--delta-command", type=float, default=0.0025)
    parser.add_argument("--dynamic-duration", type=float, default=20.0)
    parser.add_argument("--dynamic-sample-rate", type=float, default=125.0)
    parser.add_argument("--dynamic-amplitude", type=float, default=0.02)
    parser.add_argument("--mass", type=float, default=1.326)
    parser.add_argument("--rotor-count", type=int, default=4)
    return parser.parse_args()


def rpm_gain_point(
    voltage: float,
    command: float,
    factor: float,
    delta_command: float,
    model: dict[str, float],
) -> dict[str, float | bool]:
    commands = np.asarray(
        [
            max(0.0, command - delta_command),
            command,
            min(1.0, command + delta_command),
        ]
    )
    signals = fac_test.motor_signal(commands, factor)
    rpm = lzf.target_rpm(voltage, signals, model)
    rpm_limit = min(
        model["no_load_kv_rpm_per_v"] * voltage,
        model["motor_limit_rpm"],
    )
    saturated = bool(rpm[1] >= rpm_limit - 1e-6)
    gain = 0.0
    if not saturated:
        gain = (
            model["loaded_kv_rpm_per_v_u"]
            * voltage
            * fac_test.motor_signal_derivative(command, factor)
        )
    finite_gain = float(rpm[2] - rpm[0]) / float(
        commands[2] - commands[0]
    )
    return {
        "thr_mdl_fac": factor,
        "voltage_v": voltage,
        "thrust_command": command,
        "motor_signal": float(signals[1]),
        "rpm": float(rpm[1]),
        "rpm_gain_per_command": gain,
        "finite_difference_rpm_gain_per_command": finite_gain,
        "rpm_saturated": saturated,
    }


def hover_curve(
    voltages: list[float],
    factor: float,
    delta_command: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
) -> list[dict]:
    rows = []
    for voltage in voltages:
        signal = thrust_maps.hover_motor_signal(
            voltage, model, mass, rotor_count
        )
        command = thrust_maps.thrust_command_from_motor_signal(
            signal, factor
        )
        point = rpm_gain_point(
            voltage, command, factor, delta_command, model
        )
        rows.append(
            {
                "voltage_v": voltage,
                "thrust_command": command,
                "motor_signal": signal,
                "rpm": point["rpm"],
                "rpm_gain_per_command": point["rpm_gain_per_command"],
            }
        )
    return rows


def dynamic_point(
    voltage: float,
    command: float,
    factor: float,
    amplitude: float,
    duration: float,
    sample_rate: float,
    model: dict[str, float],
) -> dict[str, float | bool]:
    dt = 1.0 / sample_rate
    time = np.arange(0.0, duration, dt)
    perturbation = amplitude * fac_test.linear_chirp(
        time, 0.25, 5.0, duration
    )
    thrust_command = command + perturbation
    signal = fac_test.motor_signal(thrust_command, factor)
    target_rpm = lzf.target_rpm(voltage, signal, model)
    rpm = lzf.filter_first_order(
        target_rpm, model["motor_time_constant_s"], dt
    )
    filtered_input = lzf.filter_first_order(
        perturbation, model["motor_time_constant_s"], dt
    )
    crop = time >= 1.0
    matrix = np.column_stack(
        (np.ones(np.count_nonzero(crop)), filtered_input[crop])
    )
    offset, gain = np.linalg.lstsq(
        matrix, rpm[crop], rcond=None
    )[0]
    prediction = offset + gain * filtered_input[crop]
    error = rpm[crop] - prediction
    rmse = float(np.sqrt(np.mean(np.square(error))))
    centered = rpm[crop] - np.mean(rpm[crop])
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
        "amplitude": amplitude,
        "rpm_gain_per_command": float(gain),
        "rmse_rpm": rmse,
        "r_squared": r_squared,
        "rpm_saturated": bool(
            np.max(target_rpm) >= rpm_limit - 1e-6
        ),
    }


def full_domain_audit(model: dict[str, float]) -> dict:
    voltages = np.linspace(
        model["voltage_min_v"], model["voltage_max_v"], 103
    )
    commands = np.linspace(0.001, 1.0, 1000)
    factors = np.linspace(0.0, 1.0, 101)
    violation_count = 0
    saturation_count = 0
    minimum_increment = math.inf
    minimum_case = {}
    for factor in factors:
        derivative = np.asarray(
            [
                fac_test.motor_signal_derivative(command, factor)
                for command in commands
            ]
        )
        gain = (
            model["loaded_kv_rpm_per_v_u"]
            * voltages[:, None]
            * derivative[None, :]
        )
        differences = np.diff(gain, axis=0)
        violation_count += int(np.count_nonzero(differences <= 0.0))
        index = np.unravel_index(
            int(np.argmin(differences)), differences.shape
        )
        if differences[index] < minimum_increment:
            minimum_increment = float(differences[index])
            minimum_case = {
                "thr_mdl_fac": float(factor),
                "thrust_command": float(commands[index[1]]),
                "voltage_v": float(voltages[index[0]]),
            }
        motor_signal = fac_test.motor_signal(commands, factor)
        rpm = lzf.target_rpm(
            voltages[:, None], motor_signal[None, :], model
        )
        rpm_limit = np.minimum(
            model["no_load_kv_rpm_per_v"] * voltages[:, None],
            model["motor_limit_rpm"],
        )
        saturation_count += int(
            np.count_nonzero(rpm >= rpm_limit - 1e-6)
        )
    return {
        "voltage_range_v": [
            float(voltages[0]),
            float(voltages[-1]),
        ],
        "voltage_step_v": float(voltages[1] - voltages[0]),
        "command_range": [
            float(commands[0]),
            float(commands[-1]),
        ],
        "thr_mdl_fac_range": [
            float(factors[0]),
            float(factors[-1]),
        ],
        "point_count": len(voltages) * len(commands) * len(factors),
        "comparison_count": (
            (len(voltages) - 1) * len(commands) * len(factors)
        ),
        "non_increasing_count": violation_count,
        "rpm_saturation_count": saturation_count,
        "minimum_adjacent_voltage_increment": minimum_increment,
        "minimum_increment_case": minimum_case,
    }


def load_free_flight_results(
    model: dict[str, float],
) -> list[dict]:
    rows = []
    for chain_path in sorted(
        CHAIN_FIT_ROOT.glob("fac*_v*/*_motor_prop_chain.json")
    ):
        label = chain_path.parent.name
        factor = int(label.split("_", 1)[0][3:]) / 10.0
        thrust_path = (
            THRUST_FIT_ROOT
            / label
            / "{}_thrust_chirp_fit.json".format(label)
        )
        if not thrust_path.is_file():
            continue
        chain = json.loads(chain_path.read_text())
        thrust = json.loads(thrust_path.read_text())
        diagnostics = thrust["runs"][0]["diagnostics"]
        voltage = float(
            diagnostics.get(
                "voltage_mean_v",
                label.split("_v", 1)[1].replace("p", "."),
            )
        )
        command = float(diagnostics["motor_command_mean"])
        prediction = rpm_gain_point(
            voltage, command, factor, 0.0025, model
        )
        identified_gain = float(chain["motor_chain"]["gain"])
        predicted_gain = float(prediction["rpm_gain_per_command"])
        rows.append(
            {
                "label": label,
                "source": str(chain_path),
                "thr_mdl_fac": factor,
                "voltage_v": voltage,
                "mean_thrust_command": command,
                "mean_motor_signal": float(
                    fac_test.motor_signal(command, factor)
                ),
                "identified_rpm_gain_per_command": identified_gain,
                "static_rpm_gain_per_command": predicted_gain,
                "static_error_percent": 100.0
                * (identified_gain / predicted_gain - 1.0),
                "time_constant_s": float(
                    chain["motor_chain"]["time_constant_s"]
                ),
                "r_squared": float(
                    chain["motor_chain"]["r_squared"]
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grid_values(
    points: list[dict],
    voltages: list[float],
    commands: list[float],
) -> np.ndarray:
    lookup = {
        (float(row["voltage_v"]), float(row["thrust_command"])): float(
            row["rpm_gain_per_command"]
        )
        for row in points
    }
    return np.asarray(
        [
            [lookup[(voltage, command)] for voltage in voltages]
            for command in commands
        ]
    )


def plot_single(
    path: Path,
    dataset: dict,
    voltages: list[float],
    commands: list[float],
) -> None:
    gain = grid_values(dataset["points"], voltages, commands)
    voltage_grid, command_grid = np.meshgrid(voltages, commands)
    figure, axis = plt.subplots(
        figsize=(10.4, 6.6), constrained_layout=True
    )
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
        [row["voltage_v"] for row in dataset["hover"]],
        [row["thrust_command"] for row in dataset["hover"]],
        color="white",
        linestyle="--",
        linewidth=2.2,
        label="Static J=0 hover line",
    )
    validation = dataset["free_flight"]
    if validation:
        axis.scatter(
            [row["voltage_v"] for row in validation],
            [row["mean_thrust_command"] for row in validation],
            c=[
                row["identified_rpm_gain_per_command"]
                for row in validation
            ],
            cmap="viridis",
            vmin=float(np.min(gain)),
            vmax=float(np.max(gain)),
            edgecolor="red",
            linewidth=1.6,
            s=80,
            zorder=5,
            label="Free-flight Z chirp",
        )
    axis.axhline(
        0.1,
        color="white",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        label="Previous map lower bound",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(
        "Local RPM gain K2 (mechanical RPM per PX4 thrust command)"
    )
    axis.set(
        title=(
            "LZF command-to-RPM local gain, "
            "THR_MDL_FAC={:.1f}".format(dataset["thr_mdl_fac"])
        ),
        xlabel="Battery voltage (V)",
        ylabel="PX4 normalized thrust command before THR_MDL_FAC",
        xlim=(min(voltages) - 0.125, max(voltages) + 0.125),
        ylim=(min(commands) - 0.0125, max(commands) + 0.0125),
    )
    axis.set_xticks(voltages[::2])
    axis.set_yticks(commands[::2])
    axis.grid(color="white", alpha=0.14)
    axis.legend(loc="upper left")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_comparison(
    path: Path,
    datasets: list[dict],
    voltages: list[float],
    commands: list[float],
) -> None:
    gains = [
        float(row["rpm_gain_per_command"])
        for dataset in datasets
        for row in dataset["points"]
    ]
    vmin, vmax = min(gains), max(gains)
    voltage_grid, command_grid = np.meshgrid(voltages, commands)
    figure, axes = plt.subplots(
        3,
        4,
        figsize=(15.5, 10.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, dataset in zip(axes.flat, datasets):
        gain = grid_values(dataset["points"], voltages, commands)
        image = axis.pcolormesh(
            voltage_grid,
            command_grid,
            gain,
            shading="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.plot(
            [row["voltage_v"] for row in dataset["hover"]],
            [row["thrust_command"] for row in dataset["hover"]],
            color="white",
            linestyle="--",
            linewidth=1.5,
        )
        axis.axhline(
            0.1,
            color="white",
            linestyle=":",
            linewidth=0.7,
            alpha=0.7,
        )
        axis.set_title(
            "THR_MDL_FAC={:.1f}".format(dataset["thr_mdl_fac"])
        )
        axis.grid(color="white", alpha=0.12)
    for axis in axes.flat[len(datasets):]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Battery voltage (V)")
    for axis in axes[:, 0]:
        axis.set_ylabel("PX4 thrust command")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.94)
        colorbar.set_label(
            "Local RPM gain K2 (mechanical RPM per PX4 command)"
        )
    figure.suptitle(
        "LZF command-to-RPM gain across THR_MDL_FAC"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_factor_report(path: Path, dataset: dict) -> None:
    gains = [
        float(row["rpm_gain_per_command"])
        for row in dataset["points"]
    ]
    lines = [
        "# LZF command-to-RPM gain, THR_MDL_FAC={:.1f}".format(
            dataset["thr_mdl_fac"]
        ),
        "",
        "- K2 range: `{:.1f}...{:.1f} mechanical RPM/command`.".format(
            min(gains), max(gains)
        ),
        "- Fixed-command voltage violations: `{}`.".format(
            dataset["monotonicity"]["violation_count"]
        ),
        "- RPM-saturated grid points: `{}`.".format(
            sum(bool(row["rpm_saturated"]) for row in dataset["points"])
        ),
        "",
        "## Free-flight Z-chirp validation",
        "",
        "| Voltage | Mean command | Identified K2 | Static K2 | Error | Tau | R2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset["free_flight"]:
        lines.append(
            "| {voltage_v:.1f} | {mean_thrust_command:.5f} | "
            "{identified_rpm_gain_per_command:.1f} | "
            "{static_rpm_gain_per_command:.1f} | "
            "{static_error_percent:+.2f}% | {tau_ms:.3f} ms | "
            "{r_squared:.6f} |".format(
                tau_ms=1000.0 * row["time_constant_s"], **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_summary(
    path: Path,
    datasets: list[dict],
    dynamic_result: dict,
    domain_audit: dict,
    free_flight: list[dict],
) -> None:
    lines = [
        "# LZF THR_MDL_FAC command-to-RPM gain-map comparison",
        "",
        "K2 is the local gain from PX4 thrust command before `THR_MDL_FAC` "
        "to mean mechanical motor RPM.",
        "",
        "| FAC | K2 min | K2 max | Hover K2 @18 V | Hover K2 @25 V | "
        "Voltage violations | RPM saturation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        gains = [
            float(row["rpm_gain_per_command"])
            for row in dataset["points"]
        ]
        lines.append(
            "| {factor:.1f} | {minimum:.1f} | {maximum:.1f} | "
            "{hover18:.1f} | {hover25:.1f} | {violations} | "
            "{saturated} |".format(
                factor=dataset["thr_mdl_fac"],
                minimum=min(gains),
                maximum=max(gains),
                hover18=dataset["hover"][0]["rpm_gain_per_command"],
                hover25=dataset["hover"][-1]["rpm_gain_per_command"],
                violations=dataset["monotonicity"]["violation_count"],
                saturated=sum(
                    bool(row["rpm_saturated"])
                    for row in dataset["points"]
                ),
            )
        )
    lines += [
        "",
        "## Constrained motor-chain Z-chirp sweep",
        "",
        "These are deterministic `J=0` motor-chain simulations, not "
        "free-flight SITL runs.",
        "",
        "- Test points: `{}`.".format(dynamic_result["point_count"]),
        "- Voltage comparisons: `{}`.".format(
            dynamic_result["comparison_count"]
        ),
        "- Non-increasing comparisons: `{}`.".format(
            dynamic_result["violation_count"]
        ),
        "- Minimum adjacent-voltage K2 increment: `{:.3f}`.".format(
            dynamic_result["minimum_increment"]
        ),
        "",
        "## Full-domain audit",
        "",
        "- Domain: `{:.1f}...{:.1f} V`, command "
        "`{:.3f}...{:.1f}`, THR_MDL_FAC `{:.1f}...{:.1f}`.".format(
            domain_audit["voltage_range_v"][0],
            domain_audit["voltage_range_v"][1],
            domain_audit["command_range"][0],
            domain_audit["command_range"][1],
            domain_audit["thr_mdl_fac_range"][0],
            domain_audit["thr_mdl_fac_range"][1],
        ),
        "- State points: `{}`; voltage comparisons: `{}`.".format(
            domain_audit["point_count"],
            domain_audit["comparison_count"],
        ),
        "- Non-increasing comparisons: `{}`; RPM saturation points: "
        "`{}`.".format(
            domain_audit["non_increasing_count"],
            domain_audit["rpm_saturation_count"],
        ),
        "",
        "## Free-flight ULog validation",
        "",
        "These six rows come from actual LZF SITL free-flight Z-chirp "
        "ULogs. Other THR_MDL_FAC values were not flown for this K2 map.",
        "",
        "| FAC | Voltage | Mean command | Identified K2 | Static K2 | Error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in free_flight:
        lines.append(
            "| {thr_mdl_fac:.1f} | {voltage_v:.1f} | "
            "{mean_thrust_command:.5f} | "
            "{identified_rpm_gain_per_command:.1f} | "
            "{static_rpm_gain_per_command:.1f} | "
            "{static_error_percent:+.2f}% |".format(**row)
        )
    lines += [
        "",
        "No fixed-command voltage reversal was found. With no RPM "
        "saturation, `K2 = motorLoadedKv * V * ds/dc`, and `ds/dc` is "
        "positive and independent of voltage.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    factors = sorted(set(args.thr_mdl_fac))
    if any(not 0.0 <= factor <= 1.0 for factor in factors):
        raise ValueError("--thr-mdl-fac values must be between 0 and 1")
    model = lzf.load_model(args.sdf)
    voltages = sorted(set(args.voltages))
    commands = sorted(set(args.commands))
    free_flight = load_free_flight_results(model)
    datasets = []
    for factor in factors:
        points = [
            rpm_gain_point(
                voltage,
                command,
                factor,
                args.delta_command,
                model,
            )
            for voltage in voltages
            for command in commands
        ]
        dataset = {
            "thr_mdl_fac": factor,
            "points": points,
            "hover": hover_curve(
                voltages,
                factor,
                args.delta_command,
                model,
                args.mass,
                args.rotor_count,
            ),
            "monotonicity": fac_test.monotonicity(
                points,
                "rpm_gain_per_command",
                ("thr_mdl_fac", "thrust_command"),
            ),
            "free_flight": [
                row
                for row in free_flight
                if abs(row["thr_mdl_fac"] - factor) < 1e-9
            ],
        }
        datasets.append(dataset)

    dynamic_voltages = [18.0 + 0.5 * index for index in range(15)]
    dynamic_commands = [0.1 * index for index in range(1, 9)]
    dynamic_rows = [
        dynamic_point(
            voltage,
            command,
            factor,
            args.dynamic_amplitude,
            args.dynamic_duration,
            args.dynamic_sample_rate,
            model,
        )
        for factor in factors
        for voltage in dynamic_voltages
        for command in dynamic_commands
    ]
    dynamic_monotonicity = fac_test.monotonicity(
        dynamic_rows,
        "rpm_gain_per_command",
        ("thr_mdl_fac", "thrust_command"),
    )
    dynamic_monotonicity["point_count"] = len(dynamic_rows)
    domain_audit = full_domain_audit(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        slug = thrust_maps.factor_slug(dataset["thr_mdl_fac"])
        output_dir = args.output_dir / "fac_{}_rpm_gain_map".format(slug)
        prefix = "lzf_thr_mdl_fac_{}_rpm_gain_map".format(slug)
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_csv = output_dir / "{}_grid.csv".format(prefix)
        hover_csv = output_dir / "{}_hover_curve.csv".format(prefix)
        json_path = output_dir / "{}.json".format(prefix)
        report_path = output_dir / "{}.md".format(prefix)
        plot_path = output_dir / "{}.png".format(prefix)
        write_csv(grid_csv, dataset["points"])
        write_csv(hover_csv, dataset["hover"])
        json_path.write_text(
            json.dumps(
                {
                    "conditions": {
                        "thr_mdl_fac": dataset["thr_mdl_fac"],
                        "voltages": voltages,
                        "commands": commands,
                        "advance_ratio": 0.0,
                    },
                    "model": model,
                    "monotonicity": dataset["monotonicity"],
                    "static_grid": dataset["points"],
                    "hover_curve": dataset["hover"],
                    "free_flight_validation": dataset["free_flight"],
                },
                indent=2,
            )
            + "\n",
            encoding="ascii",
        )
        write_factor_report(report_path, dataset)
        plot_single(plot_path, dataset, voltages, commands)
        dataset["plot"] = str(plot_path)
        dataset["report"] = str(report_path)

    comparison_plot = (
        args.output_dir / "lzf_thr_mdl_fac_rpm_gain_maps.png"
    )
    dynamic_csv = (
        args.output_dir / "lzf_thr_mdl_fac_dynamic_rpm_chirps.csv"
    )
    summary_json = (
        args.output_dir / "lzf_thr_mdl_fac_rpm_gain_maps.json"
    )
    summary_report = (
        args.output_dir / "lzf_thr_mdl_fac_rpm_gain_maps.md"
    )
    plot_comparison(comparison_plot, datasets, voltages, commands)
    write_csv(dynamic_csv, dynamic_rows)
    summary_json.write_text(
        json.dumps(
            {
                "conditions": {
                    "factors": factors,
                    "voltages": voltages,
                    "commands": commands,
                    "advance_ratio": 0.0,
                },
                "model": model,
                "dynamic_monotonicity": dynamic_monotonicity,
                "full_domain_audit": domain_audit,
                "free_flight_validation": free_flight,
                "datasets": [
                    {
                        "thr_mdl_fac": dataset["thr_mdl_fac"],
                        "monotonicity": dataset["monotonicity"],
                        "plot": dataset["plot"],
                        "report": dataset["report"],
                    }
                    for dataset in datasets
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_summary(
        summary_report,
        datasets,
        dynamic_monotonicity,
        domain_audit,
        free_flight,
    )
    for dataset in datasets:
        print(
            "THR_MDL_FAC={:.1f}: {}".format(
                dataset["thr_mdl_fac"], dataset["plot"]
            )
        )
    print("comparison: {}".format(comparison_plot))
    print("dynamic chirps: {}".format(dynamic_csv))
    print("summary: {}".format(summary_report))


if __name__ == "__main__":
    main()
