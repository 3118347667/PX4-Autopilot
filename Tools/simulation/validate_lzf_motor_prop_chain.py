#!/usr/bin/env python3

"""Validate the LZF motor and propeller subchains from PX4 thrust-chirp ULogs."""

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog
from scipy import optimize


DEFAULT_SDF = Path(
    "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/lzf/lzf.sdf"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ulog", nargs="+", type=Path)
    parser.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/lzf_motor_prop_chain_validation"),
    )
    parser.add_argument("--label", default="lzf_motor_prop_chain")
    parser.add_argument("--sample-rate", type=float, default=125.0)
    parser.add_argument("--crop-start", type=float, default=0.5)
    parser.add_argument("--crop-end", type=float, default=0.5)
    return parser.parse_args()


def dataset(ulog, name):
    matches = [
        item for item in ulog.data_list if item.name == name and item.multi_id == 0
    ]
    if not matches:
        raise ValueError("ULog has no {} topic".format(name))
    return matches[0].data


def split_blocks(timestamps, maximum_gap=0.2):
    splits = np.flatnonzero(np.diff(timestamps) > maximum_gap) + 1
    return [
        block
        for block in np.split(np.arange(len(timestamps)), splits)
        if len(block) > 1
    ]


def sdf_value(element, name):
    child = element.find(name)
    if child is None or child.text is None:
        raise ValueError("missing <{}> in LZF SDF".format(name))
    return float(child.text)


def model_mass(model, models_directory, visited):
    mass = 0.0
    for link in model.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None and inertial.find("mass") is not None:
            mass += sdf_value(inertial, "mass")

    for include in model.findall("include"):
        uri = include.findtext("uri", "").strip()
        if not uri.startswith("model://"):
            continue
        model_name = uri[len("model://") :].split("/", 1)[0]
        included_directory = models_directory / model_name
        included_path = included_directory / "model.sdf"
        model_config = included_directory / "model.config"
        if not included_path.is_file() and model_config.is_file():
            configured_sdf = ET.parse(model_config).getroot().findtext("sdf", "")
            included_path = included_directory / configured_sdf.strip()
        if included_path in visited or not included_path.is_file():
            continue
        visited.add(included_path)
        included_root = ET.parse(included_path).getroot()
        included_model = included_root.find("model")
        if included_model is not None:
            mass += model_mass(included_model, models_directory, visited)
    return mass


def load_model(path):
    root = ET.parse(path).getroot()
    model = root.find("model")
    if model is None:
        raise ValueError("LZF SDF has no <model>")

    mass = model_mass(model, path.parent.parent, {path})
    if mass <= 0.0:
        raise ValueError("LZF SDF has no link inertial masses")

    motor = None
    for plugin in model.findall("plugin"):
        enabled = plugin.find("propellerModelEnabled")
        if enabled is not None and enabled.text.strip().lower() == "true":
            motor = plugin
            break
    if motor is None:
        raise ValueError("LZF SDF has no enabled propeller motor plugin")

    breakpoints = np.fromstring(
        motor.findtext("advanceRatioBreakpoints", ""), sep=" ", dtype=float
    )
    coefficients = np.fromstring(
        motor.findtext("thrustCoefficientTable", ""), sep=" ", dtype=float
    )
    if len(breakpoints) < 2 or len(breakpoints) != len(coefficients):
        raise ValueError("invalid LZF advance-ratio/thrust-coefficient table")

    return {
        "mass_kg": mass,
        "motor_time_constant_s": sdf_value(motor, "timeConstantUp"),
        "propeller_diameter_m": sdf_value(motor, "propellerDiameter"),
        "air_density_kg_m3": sdf_value(motor, "airDensity"),
        "advance_ratio": breakpoints,
        "thrust_coefficient": coefficients,
    }


def body_z_velocity(q, velocity_ned):
    # PX4 vehicle_attitude.q rotates FRD body vectors into NED.
    w, x, y, z = q.T
    r02 = 2.0 * (x * z + w * y)
    r12 = 2.0 * (y * z - w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return r02 * velocity_ned[:, 0] + r12 * velocity_ned[:, 1] + r22 * velocity_ned[:, 2]


def interpolate_columns(timestamp, values, target):
    return np.column_stack(
        [np.interp(target, timestamp, values[:, index]) for index in range(values.shape[1])]
    )


def load_runs(path, model, sample_rate):
    topic_names = [
        "thrust_chirp_sweep",
        "vehicle_attitude",
        "vehicle_local_position",
    ]
    ulog = ULog(str(path), topic_names)
    chirp = dataset(ulog, "thrust_chirp_sweep")
    attitude = dataset(ulog, "vehicle_attitude")
    position = dataset(ulog, "vehicle_local_position")

    chirp_t = np.asarray(chirp["timestamp_sample"], dtype=float) * 1e-6
    attitude_t = np.asarray(attitude["timestamp_sample"], dtype=float) * 1e-6
    position_t = np.asarray(position["timestamp_sample"], dtype=float) * 1e-6
    attitude_q = np.column_stack(
        [np.asarray(attitude["q[{}]".format(i)], dtype=float) for i in range(4)]
    )
    velocity_ned = np.column_stack(
        [np.asarray(position[name], dtype=float) for name in ("vx", "vy", "vz")]
    )

    runs = []
    for block_number, indices in enumerate(split_blocks(chirp_t)):
        source_t = chirp_t[indices]
        if source_t[-1] - source_t[0] < 4.0:
            continue

        dt = 1.0 / sample_rate
        t_absolute = np.arange(source_t[0], source_t[-1] + 0.25 * dt, dt)
        command = -np.interp(
            t_absolute, source_t, np.asarray(chirp["u"], dtype=float)[indices]
        )
        acceleration = -np.interp(
            t_absolute, source_t, np.asarray(chirp["y"], dtype=float)[indices]
        )
        rpm = np.column_stack(
            [
                np.interp(
                    t_absolute,
                    source_t,
                    np.asarray(chirp["esc_rpm[{}]".format(i)], dtype=float)[indices],
                )
                for i in range(4)
            ]
        )
        q = interpolate_columns(attitude_t, attitude_q, t_absolute)
        q /= np.linalg.norm(q, axis=1)[:, None]
        velocity = interpolate_columns(position_t, velocity_ned, t_absolute)
        axial_airspeed = np.maximum(0.0, -body_z_velocity(q, velocity))

        revolutions_per_second = np.abs(rpm) / 60.0
        advance_ratio = np.divide(
            axial_airspeed[:, None],
            revolutions_per_second * model["propeller_diameter_m"],
            out=np.zeros_like(rpm),
            where=revolutions_per_second > 1e-6,
        )
        thrust_coefficient = np.interp(
            advance_ratio,
            model["advance_ratio"],
            model["thrust_coefficient"],
        )
        rotor_thrust = (
            thrust_coefficient
            * model["air_density_kg_m3"]
            * revolutions_per_second**2
            * model["propeller_diameter_m"] ** 4
        )
        predicted_acceleration = np.sum(rotor_thrust, axis=1) / model["mass_kg"]

        valid = (
            np.all(np.isfinite(rpm), axis=1)
            & np.all(rpm > 1000.0, axis=1)
            & np.isfinite(command)
            & np.isfinite(acceleration)
            & np.isfinite(predicted_acceleration)
        )
        valid[: int(round(0.2 * sample_rate))] = False
        if np.count_nonzero(valid) < sample_rate * 4.0:
            continue

        runs.append(
            {
                "name": "{}#{}".format(path.name, block_number + 1),
                "path": str(path),
                "t": t_absolute - t_absolute[0],
                "command": command,
                "acceleration": acceleration,
                "rpm": rpm,
                "mean_rpm": np.mean(rpm, axis=1),
                "axial_airspeed": axial_airspeed,
                "advance_ratio": advance_ratio,
                "predicted_acceleration": predicted_acceleration,
                "valid": valid,
            }
        )
    return runs


def delayed_signal(t, values, delay):
    return np.interp(t - delay, t, values, left=values[0], right=values[-1])


def first_order_signal(t, values, tau, delay):
    source = delayed_signal(t, values, delay)
    result = np.empty_like(source)
    result[0] = source[0]
    for index in range(1, len(source)):
        alpha = math.exp(-(t[index] - t[index - 1]) / tau)
        result[index] = alpha * result[index - 1] + (1.0 - alpha) * source[index]
    return result


def linear_metrics(inputs, outputs):
    design = np.column_stack([np.ones(len(inputs)), inputs])
    parameters, _, _, _ = np.linalg.lstsq(design, outputs, rcond=None)
    prediction = design @ parameters
    residual = outputs - prediction
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((outputs - np.mean(outputs)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator
    return {
        "offset": float(parameters[0]),
        "gain": float(parameters[1]),
        "rmse": rmse,
        "r_squared": r_squared,
        "prediction": prediction,
    }


def crop_mask(run, crop_start, crop_end):
    return (
        run["valid"]
        & (run["t"] >= crop_start)
        & (run["t"] <= run["t"][-1] - crop_end)
    )


def fit_first_order(runs, input_name, output_name, crop_start, crop_end):
    def evaluate(parameters, include_prediction=False):
        tau = math.exp(parameters[0])
        delay = parameters[1]
        inputs = []
        outputs = []
        filtered_by_run = []
        for run in runs:
            filtered = first_order_signal(run["t"], run[input_name], tau, delay)
            mask = crop_mask(run, crop_start, crop_end)
            inputs.append(filtered[mask])
            outputs.append(run[output_name][mask])
            filtered_by_run.append(filtered)
        metrics = linear_metrics(np.concatenate(inputs), np.concatenate(outputs))
        if include_prediction:
            metrics["filtered_by_run"] = filtered_by_run
        return metrics

    result = optimize.differential_evolution(
        lambda p: evaluate(p)["rmse"],
        [(math.log(0.001), math.log(0.15)), (0.0, 0.08)],
        seed=12,
        popsize=12,
        maxiter=100,
        tol=1e-8,
        polish=True,
        workers=1,
    )
    metrics = evaluate(result.x, include_prediction=True)
    metrics.update(
        {
            "time_constant_s": float(math.exp(result.x[0])),
            "cutoff_frequency_hz": float(
                1.0 / (2.0 * math.pi * math.exp(result.x[0]))
            ),
            "delay_s": float(result.x[1]),
        }
    )
    return metrics


def fit_static_delay(runs, crop_start, crop_end):
    def evaluate(delay, include_prediction=False):
        inputs = []
        outputs = []
        delayed_by_run = []
        for run in runs:
            delayed = delayed_signal(
                run["t"], run["predicted_acceleration"], float(delay)
            )
            mask = crop_mask(run, crop_start, crop_end)
            inputs.append(delayed[mask])
            outputs.append(run["acceleration"][mask])
            delayed_by_run.append(delayed)
        metrics = linear_metrics(np.concatenate(inputs), np.concatenate(outputs))
        if include_prediction:
            metrics["delayed_by_run"] = delayed_by_run
        return metrics

    result = optimize.minimize_scalar(
        lambda delay: evaluate(delay)["rmse"],
        bounds=(0.0, 0.08),
        method="bounded",
        options={"xatol": 1e-8},
    )
    metrics = evaluate(result.x, include_prediction=True)
    metrics["delay_s"] = float(result.x)
    return metrics


def serializable_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("prediction", "filtered_by_run", "delayed_by_run")
    }


def write_report(result, path):
    motor = result["motor_chain"]
    prop = result["propeller_chain"]
    lines = [
        "# LZF motor and propeller chain validation",
        "",
        "## Model parameters",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        "| Configured motor time constant | {:.3f} ms |".format(
            1000.0 * result["model"]["motor_time_constant_s"]
        ),
        "| Configured motor cutoff | {:.3f} Hz |".format(
            result["model"]["motor_cutoff_frequency_hz"]
        ),
        "| Mass | {:.6f} kg |".format(result["model"]["mass_kg"]),
        "| Propeller diameter | {:.3f} m |".format(
            result["model"]["propeller_diameter_m"]
        ),
        "",
        "## Command to RPM",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        "| Identified time constant | {:.3f} ms |".format(
            1000.0 * motor["time_constant_s"]
        ),
        "| Identified cutoff | {:.3f} Hz |".format(
            motor["cutoff_frequency_hz"]
        ),
        "| Delay | {:.3f} ms |".format(1000.0 * motor["delay_s"]),
        "| Local gain | {:.3f} RPM/u |".format(motor["gain"]),
        "| RMSE | {:.3f} RPM |".format(motor["rmse"]),
        "| R2 | {:.6f} |".format(motor["r_squared"]),
        "| Time-constant error | {:+.2f}% |".format(
            result["motor_time_constant_error_percent"]
        ),
        "",
        "## RPM and advance ratio to acceleration",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        "| Fitted force scale (ideal 1) | {:.6f} |".format(prop["gain"]),
        "| Acceleration offset | {:.6f} m/s^2 |".format(prop["offset"]),
        "| Timing offset | {:.3f} ms |".format(1000.0 * prop["delay_s"]),
        "| RMSE | {:.6f} m/s^2 |".format(prop["rmse"]),
        "| R2 | {:.6f} |".format(prop["r_squared"]),
        "",
        "The propeller prediction uses the LZF SDF mass, diameter, air density,",
        "the UIUC thrust-coefficient table, all four mechanical RPM values, and",
        "the measured body-axis axial airspeed.",
        "",
        "## Runs",
        "",
        "| Run | Duration | Mean RPM | Axial airspeed range | Advance-ratio range |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        lines.append(
            "| {name} | {duration_s:.2f} s | {mean_rpm:.1f} | "
            "{axial_airspeed_min:.3f}...{axial_airspeed_max:.3f} m/s | "
            "{advance_ratio_min:.4f}...{advance_ratio_max:.4f} |".format(**run)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(runs, motor, prop, path):
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for index, run in enumerate(runs):
        label = run["name"]
        motor_filtered = motor["filtered_by_run"][index]
        motor_prediction = motor["offset"] + motor["gain"] * motor_filtered
        prop_delayed = prop["delayed_by_run"][index]
        prop_prediction = prop["offset"] + prop["gain"] * prop_delayed
        axes[0].plot(run["t"], run["mean_rpm"], linewidth=0.8, label=label)
        axes[0].plot(
            run["t"], motor_prediction, "--", linewidth=0.8, label=label + " fit"
        )
        axes[1].plot(run["t"], run["acceleration"], linewidth=0.8, label=label)
        axes[1].plot(
            run["t"], prop_prediction, "--", linewidth=0.8, label=label + " model"
        )
    axes[0].set_ylabel("Mean motor speed [RPM]")
    axes[1].set_ylabel("Upward body-Z acceleration [m/s^2]")
    axes[1].set_xlabel("Sweep time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    try:
        model = load_model(args.sdf)
        runs = []
        for path in args.ulog:
            runs.extend(load_runs(path, model, args.sample_rate))
        if not runs:
            raise ValueError("no complete valid thrust-chirp blocks")
    except Exception as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    motor = fit_first_order(
        runs,
        "command",
        "mean_rpm",
        args.crop_start,
        args.crop_end,
    )
    prop = fit_static_delay(runs, args.crop_start, args.crop_end)
    configured_tau = model["motor_time_constant_s"]
    result = {
        "label": args.label,
        "sample_rate_hz": args.sample_rate,
        "model": {
            key: value
            for key, value in model.items()
            if key not in ("advance_ratio", "thrust_coefficient")
        },
        "motor_chain": serializable_metrics(motor),
        "propeller_chain": serializable_metrics(prop),
        "motor_time_constant_error_percent": 100.0
        * (motor["time_constant_s"] - configured_tau)
        / configured_tau,
        "runs": [],
    }
    result["model"]["motor_cutoff_frequency_hz"] = 1.0 / (
        2.0 * math.pi * configured_tau
    )
    for run in runs:
        mask = crop_mask(run, args.crop_start, args.crop_end)
        result["runs"].append(
            {
                "name": run["name"],
                "path": run["path"],
                "duration_s": float(run["t"][-1]),
                "mean_rpm": float(np.mean(run["mean_rpm"][mask])),
                "axial_airspeed_min": float(np.min(run["axial_airspeed"][mask])),
                "axial_airspeed_max": float(np.max(run["axial_airspeed"][mask])),
                "advance_ratio_min": float(np.min(run["advance_ratio"][mask])),
                "advance_ratio_max": float(np.max(run["advance_ratio"][mask])),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "{}.json".format(args.label)
    report_path = args.output_dir / "{}.md".format(args.label)
    plot_path = args.output_dir / "{}.png".format(args.label)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, report_path)
    plot_results(runs, motor, prop, plot_path)
    print(
        "motor tau={:.3f} ms ({:+.2f}%), R2={:.6f}; "
        "prop scale={:.5f}, RMSE={:.5f} m/s^2, R2={:.6f}".format(
            1000.0 * motor["time_constant_s"],
            result["motor_time_constant_error_percent"],
            motor["r_squared"],
            prop["gain"],
            prop["rmse"],
            prop["r_squared"],
        )
    )
    print(json_path)
    print(report_path)
    print(plot_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
