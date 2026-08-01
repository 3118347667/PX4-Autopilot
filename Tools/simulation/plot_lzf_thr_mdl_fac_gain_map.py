#!/usr/bin/env python3

"""Plot an LZF voltage-command local-gain map for one THR_MDL_FAC value."""

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
import test_lzf_thr_mdl_fac_monotonicity as fac_test


PX4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PX4_ROOT / "build/lzf_thr_mdl_fac_monotonicity/gain_maps"
)


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
    parser.add_argument("--mass", type=float, default=1.326)
    parser.add_argument("--rotor-count", type=int, default=4)
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
    return parser.parse_args()


def hover_motor_signal(
    voltage: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
) -> float:
    thrust_per_rotor = mass * 9.80665 / rotor_count
    thrust_scale = (
        model["static_thrust_coefficient"]
        * model["air_density_kg_m3"]
        * model["propeller_diameter_m"] ** 4
    )
    required_rpm = 60.0 * math.sqrt(thrust_per_rotor / thrust_scale)
    return (
        (required_rpm - model["idle_rpm"])
        / (model["loaded_kv_rpm_per_v_u"] * voltage)
    )


def thrust_command_from_motor_signal(signal: float, factor: float) -> float:
    return factor * signal * signal + (1.0 - factor) * signal


def hover_curve(
    voltages: list[float],
    factor: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
    delta_command: float,
) -> list[dict]:
    rows = []
    for voltage in voltages:
        signal = hover_motor_signal(
            voltage, model, mass, rotor_count
        )
        command = thrust_command_from_motor_signal(signal, factor)
        point = fac_test.static_point(
            voltage,
            command,
            factor,
            delta_command,
            model,
            mass,
            rotor_count,
        )
        rows.append(
            {
                "voltage_v": voltage,
                "thrust_command": command,
                "motor_signal": signal,
                "gain_m_s2_per_command": point[
                    "local_gain_m_s2_per_command"
                ],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_map(
    path: Path,
    factor: float,
    voltages: list[float],
    commands: list[float],
    points: list[dict],
    hover: list[dict],
) -> None:
    lookup = {
        (float(row["voltage_v"]), float(row["thrust_command"])): float(
            row["local_gain_m_s2_per_command"]
        )
        for row in points
    }
    gain = np.asarray(
        [
            [lookup[(voltage, command)] for voltage in voltages]
            for command in commands
        ]
    )
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
        [row["voltage_v"] for row in hover],
        [row["thrust_command"] for row in hover],
        color="white",
        linestyle="--",
        linewidth=2.2,
        label="Static J=0 hover line",
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
        "Local collective gain K (m/s2 per PX4 thrust command)"
    )
    axis.set(
        title=(
            "LZF static-air local collective gain, "
            "THR_MDL_FAC={:.1f}".format(factor)
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


def write_report(
    path: Path,
    factor: float,
    voltages: list[float],
    commands: list[float],
    points: list[dict],
    hover: list[dict],
    monotonicity: dict,
) -> None:
    gains = [float(row["local_gain_m_s2_per_command"]) for row in points]
    lines = [
        "# LZF THR_MDL_FAC={:.1f} local-gain map".format(factor),
        "",
        "## Conditions",
        "",
        "- Fixed input is the PX4 normalized thrust command before "
        "`THR_MDL_FAC`.",
        "- Voltage: `{:.2f}...{:.2f} V`, step `{:.2f} V`.".format(
            min(voltages), max(voltages), voltages[1] - voltages[0]
        ),
        "- Command: `{:.3f}...{:.3f}`, step `{:.3f}`.".format(
            min(commands), max(commands), commands[1] - commands[0]
        ),
        "- Static air, advance ratio `J=0`, four equal motor commands.",
        "",
        "## Results",
        "",
        "- Grid points: `{}`.".format(len(points)),
        "- K range: `{:.3f}...{:.3f} m/s2/command`.".format(
            min(gains), max(gains)
        ),
        "- Non-increasing voltage comparisons: `{}`.".format(
            monotonicity["violation_count"]
        ),
        "- Minimum adjacent-voltage K increment: `{:.6f}`.".format(
            monotonicity["minimum_increment"]
        ),
        "- RPM-saturated points: `{}`.".format(
            sum(bool(row["rpm_saturated"]) for row in points)
        ),
        "- Static hover command range: `{:.4f}...{:.4f}`.".format(
            min(row["thrust_command"] for row in hover),
            max(row["thrust_command"] for row in hover),
        ),
        "",
        "## Hover line",
        "",
        "| Voltage | PX4 thrust command | Motor signal | K |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in hover:
        lines.append(
            "| {voltage_v:.2f} | {thrust_command:.5f} | "
            "{motor_signal:.5f} | {gain_m_s2_per_command:.3f} |".format(
                **row
            )
        )
    lines += [
        "",
        "This is a constrained static-air map, not a free-flight validation. "
        "`THR_MDL_FAC=1` was not re-flown because the previous LAND attempt "
        "showed uncontrollable low-command behavior.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def factor_slug(factor: float) -> str:
    return "{:g}".format(factor).replace(".", "p")


def generate_factor(
    output_root: Path,
    factor: float,
    voltages: list[float],
    commands: list[float],
    delta_command: float,
    model: dict[str, float],
    mass: float,
    rotor_count: int,
) -> dict:
    points = [
        fac_test.static_point(
            voltage,
            command,
            factor,
            delta_command,
            model,
            mass,
            rotor_count,
        )
        for voltage in voltages
        for command in commands
    ]
    hover = hover_curve(
        voltages,
        factor,
        model,
        mass,
        rotor_count,
        delta_command,
    )
    monotonicity = fac_test.monotonicity(
        points,
        "local_gain_m_s2_per_command",
        ("thr_mdl_fac", "thrust_command"),
    )

    slug = factor_slug(factor)
    output_dir = output_root / "fac_{}_gain_map".format(slug)
    prefix = "lzf_thr_mdl_fac_{}_gain_map".format(slug)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "{}_grid.csv".format(prefix)
    hover_csv_path = output_dir / "{}_hover_curve.csv".format(prefix)
    json_path = output_dir / "{}.json".format(prefix)
    report_path = output_dir / "{}.md".format(prefix)
    plot_path = output_dir / "{}.png".format(prefix)
    write_csv(csv_path, points)
    write_csv(hover_csv_path, hover)
    json_path.write_text(
        json.dumps(
            {
                "conditions": {
                    "thr_mdl_fac": factor,
                    "advance_ratio": 0.0,
                    "mass_kg": mass,
                    "rotor_count": rotor_count,
                    "voltages": voltages,
                    "commands": commands,
                },
                "model": model,
                "monotonicity": monotonicity,
                "static_grid": points,
                "hover_curve": hover,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_report(
        report_path,
        factor,
        voltages,
        commands,
        points,
        hover,
        monotonicity,
    )
    plot_map(
        plot_path,
        factor,
        voltages,
        commands,
        points,
        hover,
    )
    return {
        "thr_mdl_fac": factor,
        "points": points,
        "hover": hover,
        "monotonicity": monotonicity,
        "grid_csv": str(csv_path),
        "hover_csv": str(hover_csv_path),
        "json": str(json_path),
        "report": str(report_path),
        "plot": str(plot_path),
    }


def plot_comparison(
    path: Path,
    datasets: list[dict],
    voltages: list[float],
    commands: list[float],
) -> None:
    all_gains = [
        float(row["local_gain_m_s2_per_command"])
        for dataset in datasets
        for row in dataset["points"]
    ]
    vmin = min(all_gains)
    vmax = max(all_gains)
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
        lookup = {
            (float(row["voltage_v"]), float(row["thrust_command"])): float(
                row["local_gain_m_s2_per_command"]
            )
            for row in dataset["points"]
        }
        gain = np.asarray(
            [
                [lookup[(voltage, command)] for voltage in voltages]
                for command in commands
            ]
        )
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
            "Local collective gain K (m/s2 per PX4 thrust command)"
        )
    figure.suptitle(
        "LZF static-air local collective gain across THR_MDL_FAC"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary(
    path: Path,
    datasets: list[dict],
    comparison_plot: Path,
) -> None:
    lines = [
        "# LZF THR_MDL_FAC gain-map comparison",
        "",
        "All maps use `18...25 V`, PX4 thrust command `0.025...0.8`, "
        "static air, and `J=0`.",
        "",
        "| THR_MDL_FAC | K min | K max | Hover command @18 V | "
        "Hover command @25 V | Hover K @18 V | Hover K @25 V | "
        "Voltage violations | RPM saturation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        gains = [
            float(row["local_gain_m_s2_per_command"])
            for row in dataset["points"]
        ]
        hover = dataset["hover"]
        lines.append(
            "| {factor:.1f} | {gain_min:.3f} | {gain_max:.3f} | "
            "{hover_command_18:.5f} | {hover_command_25:.5f} | "
            "{hover_gain_18:.3f} | {hover_gain_25:.3f} | "
            "{violations} | {saturated} |".format(
                factor=dataset["thr_mdl_fac"],
                gain_min=min(gains),
                gain_max=max(gains),
                hover_command_18=hover[0]["thrust_command"],
                hover_command_25=hover[-1]["thrust_command"],
                hover_gain_18=hover[0]["gain_m_s2_per_command"],
                hover_gain_25=hover[-1]["gain_m_s2_per_command"],
                violations=dataset["monotonicity"]["violation_count"],
                saturated=sum(
                    bool(row["rpm_saturated"])
                    for row in dataset["points"]
                ),
            )
        )
    lines += [
        "",
        "Comparison plot: `{}`".format(comparison_plot),
        "",
        "The white dashed curve in each map is the static `J=0` hover line. "
        "The dotted line marks command `0.1`, the lower bound of the original "
        "`THR_MDL_FAC=0` map.",
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
    datasets = [
        generate_factor(
            args.output_dir,
            factor,
            voltages,
            commands,
            args.delta_command,
            model,
            args.mass,
            args.rotor_count,
        )
        for factor in factors
    ]
    comparison_plot = args.output_dir / "lzf_thr_mdl_fac_gain_maps.png"
    summary_json = args.output_dir / "lzf_thr_mdl_fac_gain_maps.json"
    summary_report = args.output_dir / "lzf_thr_mdl_fac_gain_maps.md"
    plot_comparison(comparison_plot, datasets, voltages, commands)
    summary_json.write_text(
        json.dumps(
            {
                "factors": factors,
                "voltages": voltages,
                "commands": commands,
                "datasets": [
                    {
                        key: value
                        for key, value in dataset.items()
                        if key not in ("points", "hover")
                    }
                    for dataset in datasets
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_summary(summary_report, datasets, comparison_plot)
    for dataset in datasets:
        print(
            "THR_MDL_FAC={:.1f}: {}".format(
                dataset["thr_mdl_fac"], dataset["plot"]
            )
        )
    print("comparison: {}".format(comparison_plot))
    print("summary: {}".format(summary_report))


if __name__ == "__main__":
    main()
