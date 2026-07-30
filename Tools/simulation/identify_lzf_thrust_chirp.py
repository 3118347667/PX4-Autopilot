#!/usr/bin/env python3

"""Identify hover throttle-to-body-Z-acceleration dynamics from PX4 ULogs."""

import argparse
import json
import math
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog
from scipy import optimize, signal


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ulog", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/lzf_thrust_chirp_identification"),
    )
    parser.add_argument("--label", default="lzf")
    parser.add_argument("--sample-rate", type=float, default=125.0)
    parser.add_argument("--crop-start", type=float, default=0.4)
    parser.add_argument("--crop-end", type=float, default=0.2)
    parser.add_argument("--max-delay", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def dataset(ulog, name):
    matches = [
        item for item in ulog.data_list if item.name == name and item.multi_id == 0
    ]

    if not matches:
        raise ValueError("ULog has no {} topic".format(name))

    return matches[0].data


def optional_dataset(ulog, name):
    matches = [
        item for item in ulog.data_list if item.name == name and item.multi_id == 0
    ]
    return matches[0].data if matches else None


def split_blocks(timestamps, maximum_gap=0.2):
    if len(timestamps) == 0:
        return []
    splits = np.flatnonzero(np.diff(timestamps) > maximum_gap) + 1
    return [block for block in np.split(np.arange(len(timestamps)), splits) if len(block)]


def interpolate_topic(data, field, start_us, end_us):
    if data is None:
        return np.asarray([], dtype=float)
    timestamp = np.asarray(data["timestamp"], dtype=float)
    mask = (timestamp >= start_us) & (timestamp <= end_us)
    return np.asarray(data[field], dtype=float)[mask]


def load_runs(path, sample_rate):
    topic_names = [
        "thrust_chirp_sweep",
        "actuator_motors",
        "battery_status",
        "esc_status",
    ]
    ulog = ULog(str(path), topic_names)
    chirp = dataset(ulog, "thrust_chirp_sweep")
    motors = optional_dataset(ulog, "actuator_motors")
    battery = optional_dataset(ulog, "battery_status")
    esc = optional_dataset(ulog, "esc_status")
    timestamp = np.asarray(chirp["timestamp_sample"], dtype=float) * 1e-6
    runs = []

    for block_number, indices in enumerate(split_blocks(timestamp)):
        if timestamp[indices[-1]] - timestamp[indices[0]] < 4.0:
            continue

        source_t = timestamp[indices]
        start = source_t[0]
        end = source_t[-1]
        dt = 1.0 / sample_rate
        t = np.arange(start, end + 0.25 * dt, dt)
        command = -np.interp(t, source_t, np.asarray(chirp["u"], dtype=float)[indices])
        acceleration = -np.interp(
            t, source_t, np.asarray(chirp["y"], dtype=float)[indices]
        )
        injected = -np.interp(
            t, source_t, np.asarray(chirp["chirp"], dtype=float)[indices]
        )
        start_us = source_t[0] * 1e6
        end_us = source_t[-1] * 1e6

        motor_values = []

        if motors is not None:
            motor_timestamp = np.asarray(motors["timestamp"], dtype=float)
            motor_mask = (motor_timestamp >= start_us) & (motor_timestamp <= end_us)

            if np.any(motor_mask):
                motor_values = np.column_stack(
                    [
                        np.asarray(motors["control[{}]".format(index)], dtype=float)[
                            motor_mask
                        ]
                        for index in range(4)
                    ]
                )

        voltages = interpolate_topic(battery, "voltage_v", start_us, end_us)
        rpm_values = []

        if esc is not None:
            esc_timestamp = np.asarray(esc["timestamp"], dtype=float)
            esc_mask = (esc_timestamp >= start_us) & (esc_timestamp <= end_us)

            if np.any(esc_mask):
                rpm_values = np.column_stack(
                    [
                        np.asarray(
                            esc["esc[{}].esc_rpm".format(index)], dtype=float
                        )[esc_mask]
                        for index in range(4)
                    ]
                )

        runs.append(
            {
                "name": "{}#{}".format(path.name, block_number + 1),
                "path": str(path),
                "t": t - t[0],
                "command": command,
                "acceleration": acceleration,
                "injected": injected,
                "motors": np.asarray(motor_values, dtype=float),
                "voltage": np.asarray(voltages, dtype=float),
                "rpm": np.asarray(rpm_values, dtype=float),
            }
        )

    return runs


def delayed_input(run, delay):
    t = run["t"]
    command = run["command"]
    return np.interp(t - delay, t, command, left=command[0], right=command[-1])


def first_order_unit(run, tau, delay, dt):
    command = delayed_input(run, delay)
    alpha = math.exp(-dt / tau)
    filtered = np.empty_like(command)
    filtered[0] = command[0]

    for index in range(1, len(filtered)):
        filtered[index] = (
            alpha * filtered[index - 1] + (1.0 - alpha) * command[index]
        )

    return filtered


def second_order_unit(run, natural_frequency, damping_ratio, delay, dt):
    command = delayed_input(run, delay)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numerator, denominator, discrete_dt = signal.cont2discrete(
            (
                [natural_frequency * natural_frequency],
                [
                    1.0,
                    2.0 * damping_ratio * natural_frequency,
                    natural_frequency * natural_frequency,
                ],
            ),
            dt,
            method="zoh",
        )

        return signal.dlsim(
            (numerator, denominator, discrete_dt), command
        )[1].reshape(-1)


def cropped_indices(run, crop_start, crop_end):
    end = run["t"][-1] - crop_end
    return np.flatnonzero((run["t"] >= crop_start) & (run["t"] <= end))


def regression_for_units(runs, units, crop_start, crop_end, fixed_gain=None):
    row_count = sum(len(cropped_indices(run, crop_start, crop_end)) for run in runs)
    nuisance_count = 2 * len(runs)
    column_count = nuisance_count + (0 if fixed_gain is not None else 1)
    matrix = np.zeros((row_count, column_count), dtype=float)
    observed = np.zeros(row_count, dtype=float)
    slices = []
    cursor = 0

    for run_index, (run, unit) in enumerate(zip(runs, units)):
        indices = cropped_indices(run, crop_start, crop_end)
        count = len(indices)
        rows = slice(cursor, cursor + count)
        local_t = run["t"][indices]

        if fixed_gain is None:
            matrix[rows, 0] = unit[indices]
            nuisance_offset = 1
        else:
            nuisance_offset = 0

        matrix[rows, nuisance_offset + 2 * run_index] = 1.0
        matrix[rows, nuisance_offset + 2 * run_index + 1] = (
            local_t - np.mean(local_t)
        )
        observed[rows] = run["acceleration"][indices]

        if fixed_gain is not None:
            observed[rows] -= fixed_gain * unit[indices]

        slices.append((rows, indices))
        cursor += count

    coefficients = np.linalg.lstsq(matrix, observed, rcond=None)[0]

    if fixed_gain is None:
        gain = float(coefficients[0])
        nuisance = coefficients[1:]
    else:
        gain = float(fixed_gain)
        nuisance = coefficients

    predictions = []
    residual_parts = []

    for run_index, (run, unit, (_, indices)) in enumerate(zip(runs, units, slices)):
        offset = nuisance[2 * run_index]
        slope = nuisance[2 * run_index + 1]
        local_t = run["t"][indices]
        prediction = (
            gain * unit[indices]
            + offset
            + slope * (local_t - np.mean(local_t))
        )
        predictions.append((indices, prediction))
        residual_parts.append(run["acceleration"][indices] - prediction)

    residual = np.concatenate(residual_parts)
    target = np.concatenate(
        [
            run["acceleration"][indices]
            for run, (indices, _) in zip(runs, predictions)
        ]
    )
    rmse = float(np.sqrt(np.mean(residual * residual)))
    standard_deviation = float(np.std(target))
    nrmse = float(1.0 - rmse / standard_deviation) if standard_deviation > 0 else 0.0
    total_variance = float(np.sum((target - np.mean(target)) ** 2))
    r_squared = (
        float(1.0 - np.sum(residual * residual) / total_variance)
        if total_variance > 0
        else 0.0
    )
    return {
        "gain": gain,
        "rmse": rmse,
        "nrmse": nrmse,
        "r_squared": r_squared,
        "predictions": predictions,
        "sample_count": len(target),
    }


def fit_model(
    runs,
    model_type,
    dt,
    crop_start,
    crop_end,
    max_delay,
    seed,
    fixed_gain=None,
    fast=False,
):
    if model_type == "first_order":
        bounds = [(math.log(0.002), math.log(0.5)), (0.0, max_delay)]

        def units_from_parameters(parameters):
            tau = math.exp(parameters[0])
            delay = parameters[1]
            return [first_order_unit(run, tau, delay, dt) for run in runs]

    else:
        bounds = [
            (math.log(2.0), math.log(300.0)),
            (math.log(0.1), math.log(4.0)),
            (0.0, max_delay),
        ]

        def units_from_parameters(parameters):
            natural_frequency = math.exp(parameters[0])
            damping_ratio = math.exp(parameters[1])
            delay = parameters[2]
            return [
                second_order_unit(
                    run, natural_frequency, damping_ratio, delay, dt
                )
                for run in runs
            ]

    def objective(parameters):
        fit = regression_for_units(
            runs,
            units_from_parameters(parameters),
            crop_start,
            crop_end,
            fixed_gain=fixed_gain,
        )

        if fit["gain"] <= 0.0:
            return 1e6 + fit["gain"] * fit["gain"]

        return fit["rmse"] * fit["rmse"]

    result = optimize.differential_evolution(
        objective,
        bounds,
        seed=seed,
        popsize=8 if fast else 12,
        maxiter=45 if fast else 120,
        tol=1e-7,
        polish=True,
        workers=1,
    )
    units = units_from_parameters(result.x)
    fit = regression_for_units(
        runs, units, crop_start, crop_end, fixed_gain=fixed_gain
    )
    fit["model_type"] = model_type
    fit["optimization_success"] = bool(result.success)

    if model_type == "first_order":
        fit["time_constant_s"] = float(math.exp(result.x[0]))
        fit["cutoff_frequency_hz"] = float(
            1.0 / (2.0 * math.pi * fit["time_constant_s"])
        )
        fit["delay_s"] = float(result.x[1])
    else:
        fit["natural_frequency_rad_s"] = float(math.exp(result.x[0]))
        fit["damping_ratio"] = float(math.exp(result.x[1]))
        fit["delay_s"] = float(result.x[2])

    parameter_count = 4 if model_type == "first_order" else 5
    residual_sum_squares = fit["rmse"] ** 2 * fit["sample_count"]
    fit["aic"] = float(
        fit["sample_count"]
        * math.log(max(residual_sum_squares / fit["sample_count"], 1e-12))
        + 2.0 * parameter_count
    )
    return fit


def cross_validation(
    runs,
    model_type,
    dt,
    crop_start,
    crop_end,
    max_delay,
    seed,
):
    if len(runs) < 3:
        return None
    scores = []

    for held_out in range(len(runs)):
        training = [run for index, run in enumerate(runs) if index != held_out]
        validation = [runs[held_out]]
        trained = fit_model(
            training,
            model_type,
            dt,
            crop_start,
            crop_end,
            max_delay,
            seed + held_out,
            fast=True,
        )

        if model_type == "first_order":
            units = [
                first_order_unit(
                    validation[0],
                    trained["time_constant_s"],
                    trained["delay_s"],
                    dt,
                )
            ]
        else:
            units = [
                second_order_unit(
                    validation[0],
                    trained["natural_frequency_rad_s"],
                    trained["damping_ratio"],
                    trained["delay_s"],
                    dt,
                )
            ]

        validated = regression_for_units(
            validation,
            units,
            crop_start,
            crop_end,
            fixed_gain=trained["gain"],
        )
        scores.append(validated["nrmse"])

    return {
        "fold_nrmse": [float(value) for value in scores],
        "mean_nrmse": float(np.mean(scores)),
        "minimum_nrmse": float(np.min(scores)),
    }


def run_diagnostics(run):
    diagnostics = {}
    motors = run["motors"]
    voltage = run["voltage"]
    rpm = run["rpm"]

    if motors.size:
        diagnostics["motor_command_mean"] = float(np.mean(motors))
        diagnostics["motor_command_p99"] = float(np.quantile(motors, 0.99))
        diagnostics["motor_command_min"] = float(np.min(motors))
        diagnostics["motor_command_max"] = float(np.max(motors))
        diagnostics["motor_saturation_fraction"] = float(
            np.mean((motors <= 1e-3) | (motors >= 1.0 - 1e-3))
        )

    if voltage.size:
        diagnostics["voltage_mean_v"] = float(np.mean(voltage))
        diagnostics["voltage_min_v"] = float(np.min(voltage))
        diagnostics["voltage_max_v"] = float(np.max(voltage))

    if rpm.size:
        diagnostics["mean_rpm"] = [
            float(np.mean(rpm[:, index])) for index in range(rpm.shape[1])
        ]
        diagnostics["rpm_p99"] = float(np.quantile(rpm, 0.99))

    return diagnostics


def serializable_fit(fit):
    return {key: value for key, value in fit.items() if key != "predictions"}


def plot_results(runs, first, second, selected, output):
    figure, axes = plt.subplots(
        len(runs),
        2,
        figsize=(12, max(4.0, 3.2 * len(runs))),
        squeeze=False,
    )

    for row, run in enumerate(runs):
        axis_input = axes[row, 0]
        axis_output = axes[row, 1]
        axis_input.plot(run["t"], run["command"], label="collective command")
        axis_input.plot(run["t"], run["injected"], label="injected chirp", alpha=0.7)
        axis_input.set_ylabel("normalized")
        axis_input.set_title(run["name"])
        axis_input.grid(True, alpha=0.25)
        axis_input.legend()

        axis_output.plot(run["t"], run["acceleration"], label="measured", linewidth=1.0)

        for fit, label in ((first, "first order"), (second, "second order")):
            indices, prediction = fit["predictions"][row]
            axis_output.plot(
                run["t"][indices],
                prediction,
                label="{}{}".format(label, " selected" if fit is selected else ""),
                linewidth=1.0,
            )

        axis_output.set_ylabel("upward specific acceleration [m/s^2]")
        axis_output.grid(True, alpha=0.25)
        axis_output.legend()

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_report(result, output):
    selected = result["selected_model"]
    lines = [
        "# LZF thrust chirp identification",
        "",
        "## Selected model",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        "| Model | `{}` |".format(selected["model_type"]),
        "| acc_per_unit_throttle | {:.6f} m/s^2/u |".format(selected["gain"]),
        "| Delay | {:.3f} ms |".format(1000.0 * selected["delay_s"]),
        "| RMSE | {:.6f} m/s^2 |".format(selected["rmse"]),
        "| NRMSE fit | {:.2f}% |".format(100.0 * selected["nrmse"]),
        "| R2 | {:.6f} |".format(selected["r_squared"]),
    ]

    if selected["model_type"] == "first_order":
        lines.extend(
            [
                "| Time constant | {:.6f} s |".format(
                    selected["time_constant_s"]
                ),
                "| actuator_dyn.z.cutoff_freq | {:.6f} Hz |".format(
                    selected["cutoff_frequency_hz"]
                ),
            ]
        )
    else:
        lines.extend(
            [
                "| actuator_dyn.z.natural_freq | {:.6f} rad/s |".format(
                    selected["natural_frequency_rad_s"]
                ),
                "| actuator_dyn.z.damping_ratio | {:.6f} |".format(
                    selected["damping_ratio"]
                ),
            ]
        )

    lines.extend(
        [
            "",
            "The gain is the identified DC gain from normalized collective thrust "
            "command to upward body-Z specific acceleration. It is not a trajectory "
            "tuning variable.",
            "",
            "## Model comparison",
            "",
            "| Model | Gain | RMSE | NRMSE | R2 | AIC |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for key in ("first_order", "second_order"):
        fit = result[key]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.2f}% | {:.6f} | {:.2f} |".format(
                key,
                fit["gain"],
                fit["rmse"],
                100.0 * fit["nrmse"],
                fit["r_squared"],
                fit["aic"],
            )
        )

    lines.extend(["", "## Run diagnostics", ""])

    for run in result["runs"]:
        lines.append("### `{}`".format(run["name"]))
        lines.append("")

        for key, value in run["diagnostics"].items():
            lines.append("- `{}`: `{}`".format(key, value))

        lines.append("")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    for path in args.ulog:
        if not path.is_file():
            print("error: missing ULog {}".format(path), file=sys.stderr)
            return 2

    runs = []

    try:
        for path in args.ulog:
            runs.extend(load_runs(path, args.sample_rate))
    except Exception as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    if not runs:
        print("error: no complete thrust chirp sweep found", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / args.sample_rate
    first = fit_model(
        runs,
        "first_order",
        dt,
        args.crop_start,
        args.crop_end,
        args.max_delay,
        args.seed,
    )
    second = fit_model(
        runs,
        "second_order",
        dt,
        args.crop_start,
        args.crop_end,
        args.max_delay,
        args.seed + 100,
    )
    first["cross_validation"] = cross_validation(
        runs,
        "first_order",
        dt,
        args.crop_start,
        args.crop_end,
        args.max_delay,
        args.seed + 200,
    )
    second["cross_validation"] = cross_validation(
        runs,
        "second_order",
        dt,
        args.crop_start,
        args.crop_end,
        args.max_delay,
        args.seed + 300,
    )

    if first["cross_validation"] and second["cross_validation"]:
        validation_improvement = (
            second["cross_validation"]["mean_nrmse"]
            - first["cross_validation"]["mean_nrmse"]
        )
    else:
        validation_improvement = second["nrmse"] - first["nrmse"]

    selected = second if validation_improvement >= 0.05 else first
    selection_reason = (
        "second-order validation NRMSE improved by at least 5 percentage points"
        if selected is second
        else "second-order validation improvement was below 5 percentage points"
    )
    result = {
        "label": args.label,
        "sample_rate_hz": args.sample_rate,
        "crop_start_s": args.crop_start,
        "crop_end_s": args.crop_end,
        "selection_reason": selection_reason,
        "first_order": serializable_fit(first),
        "second_order": serializable_fit(second),
        "selected_model": serializable_fit(selected),
        "runs": [
            {
                "name": run["name"],
                "path": run["path"],
                "duration_s": float(run["t"][-1]),
                "diagnostics": run_diagnostics(run),
            }
            for run in runs
        ],
    }
    json_path = args.output_dir / "{}_thrust_chirp_fit.json".format(args.label)
    report_path = args.output_dir / "{}_thrust_chirp_fit.md".format(args.label)
    plot_path = args.output_dir / "{}_thrust_chirp_fit.png".format(args.label)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, report_path)
    plot_results(runs, first, second, selected, plot_path)
    print(
        "{}: model={}, K={:.6f} m/s^2/u, RMSE={:.4f}, NRMSE={:.1f}%".format(
            args.label,
            selected["model_type"],
            selected["gain"],
            selected["rmse"],
            100.0 * selected["nrmse"],
        )
    )
    print(json_path)
    print(report_path)
    print(plot_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
