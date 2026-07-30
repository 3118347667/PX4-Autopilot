#!/usr/bin/env python3

"""Identify collective-throttle-to-acceleration dynamics from PX4 ULogs."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog
from scipy import optimize, signal


@dataclass
class Sweep:
    name: str
    source: Path
    time_s: np.ndarray
    throttle: np.ndarray
    acceleration: np.ndarray
    voltage_v: float
    mean_rpm: float


@dataclass
class FrequencyResponse:
    name: str
    frequency_hz: np.ndarray
    response: np.ndarray
    coherence: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit first- and second-order collective-throttle-to-body-Z "
            "acceleration models from PX4 thrust chirp ULogs."
        )
    )
    parser.add_argument("ulogs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/lzf_thrust_dynamics"),
    )
    parser.add_argument("--label", default="lzf")
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--start-frequency", type=float, default=0.5)
    parser.add_argument("--end-frequency", type=float, default=3.0)
    parser.add_argument("--minimum-coherence", type=float, default=0.8)
    parser.add_argument("--trim-seconds", type=float, default=1.0)
    return parser.parse_args()


def dataset(ulog: ULog, name: str):
    matches = [
        item
        for item in ulog.data_list
        if item.name == name and item.multi_id == 0
    ]
    if not matches:
        raise RuntimeError(f"{name} is missing from {ulog.file_name}")
    return matches[0]


def optional_dataset(ulog: ULog, name: str):
    matches = [
        item
        for item in ulog.data_list
        if item.name == name and item.multi_id == 0
    ]
    return matches[0] if matches else None


def timestamp_seconds(data, origin_us: float, sample: bool = False) -> np.ndarray:
    key = "timestamp_sample" if sample and "timestamp_sample" in data else "timestamp"
    return (np.asarray(data[key], dtype=float) - origin_us) * 1e-6


def split_contiguous(timestamps: np.ndarray) -> List[slice]:
    if len(timestamps) < 2:
        return []
    median_dt = float(np.median(np.diff(timestamps)))
    gap_limit = max(0.1, 10.0 * median_dt)
    boundaries = np.flatnonzero(np.diff(timestamps) > gap_limit) + 1
    blocks = np.split(np.arange(len(timestamps)), boundaries)
    return [slice(int(block[0]), int(block[-1]) + 1) for block in blocks if len(block)]


def median_interpolated(
    query_t: np.ndarray,
    source_t: np.ndarray,
    source_values: np.ndarray,
) -> float:
    if not len(source_t):
        return math.nan
    values = np.interp(query_t, source_t, source_values)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else math.nan


def load_sweeps(
    path: Path,
    sample_rate: float,
    trim_seconds: float,
) -> List[Sweep]:
    if not path.is_file():
        raise FileNotFoundError(path)

    ulog = ULog(
        str(path),
        ["thrust_chirp_sweep", "battery_status", "esc_status"],
    )
    chirp = dataset(ulog, "thrust_chirp_sweep")
    chirp_t = timestamp_seconds(chirp.data, ulog.start_timestamp, sample=True)

    battery = optional_dataset(ulog, "battery_status")
    if battery is not None:
        battery_t = timestamp_seconds(battery.data, ulog.start_timestamp)
        battery_v = np.asarray(battery.data["voltage_v"], dtype=float)
    else:
        battery_t = np.empty(0)
        battery_v = np.empty(0)

    rpm_fields = [
        f"esc_rpm[{index}]"
        for index in range(4)
        if f"esc_rpm[{index}]" in chirp.data
    ]
    if rpm_fields:
        rpm = np.column_stack(
            [np.asarray(chirp.data[field], dtype=float) for field in rpm_fields]
        )
        mean_rpm_raw = np.nanmean(np.abs(rpm), axis=1)
    else:
        mean_rpm_raw = np.full(len(chirp_t), math.nan)

    throttle_raw = -np.asarray(chirp.data["u"], dtype=float)
    acceleration_raw = -np.asarray(chirp.data["y"], dtype=float)
    sweeps: List[Sweep] = []

    for segment_index, block in enumerate(split_contiguous(chirp_t), start=1):
        source_t = chirp_t[block]
        duration = float(source_t[-1] - source_t[0])
        if duration < 5.0 or duration <= 2.0 * trim_seconds:
            continue

        start = source_t[0] + trim_seconds
        end = source_t[-1] - trim_seconds
        uniform_t = np.arange(start, end, 1.0 / sample_rate)
        if len(uniform_t) < 64:
            continue

        throttle = np.interp(uniform_t, source_t, throttle_raw[block])
        acceleration = np.interp(uniform_t, source_t, acceleration_raw[block])
        voltage = median_interpolated(uniform_t, battery_t, battery_v)
        mean_rpm = float(np.nanmedian(mean_rpm_raw[block]))
        name = path.stem
        if len(split_contiguous(chirp_t)) > 1:
            name = f"{name}_sweep_{segment_index}"

        sweeps.append(
            Sweep(
                name=name,
                source=path,
                time_s=uniform_t - uniform_t[0],
                throttle=throttle,
                acceleration=acceleration,
                voltage_v=voltage,
                mean_rpm=mean_rpm,
            )
        )

    if not sweeps:
        raise RuntimeError(f"no complete thrust chirp sweep found in {path}")
    return sweeps


def estimate_frequency_response(
    sweep: Sweep,
    sample_rate: float,
) -> FrequencyResponse:
    throttle = signal.detrend(sweep.throttle, type="linear")
    acceleration = signal.detrend(sweep.acceleration, type="linear")
    nperseg = min(512, len(throttle))
    if nperseg < 128:
        raise RuntimeError(f"{sweep.name} is too short for frequency analysis")
    noverlap = min(nperseg - 1, int(0.75 * nperseg))

    frequency, p_uu = signal.welch(
        throttle,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    _, p_uy = signal.csd(
        throttle,
        acceleration,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    _, p_yy = signal.welch(
        acceleration,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    response = p_uy / np.maximum(p_uu, np.finfo(float).tiny)
    coherence = np.abs(p_uy) ** 2 / np.maximum(
        p_uu * p_yy,
        np.finfo(float).tiny,
    )
    return FrequencyResponse(sweep.name, frequency, response, coherence)


def first_order_response(parameters: Sequence[float], frequency_hz: np.ndarray):
    gain, time_constant, delay = parameters
    s = 1j * 2.0 * np.pi * frequency_hz
    return gain / (1.0 + s * time_constant) * np.exp(-s * delay)


def second_order_response(parameters: Sequence[float], frequency_hz: np.ndarray):
    gain, natural_frequency, damping_ratio, delay = parameters
    s = 1j * 2.0 * np.pi * frequency_hz
    denominator = (
        s * s
        + 2.0 * damping_ratio * natural_frequency * s
        + natural_frequency * natural_frequency
    )
    return (
        gain
        * natural_frequency
        * natural_frequency
        / denominator
        * np.exp(-s * delay)
    )


def selected_bins(
    response: FrequencyResponse,
    start_frequency: float,
    end_frequency: float,
    minimum_coherence: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        (response.frequency_hz >= start_frequency)
        & (response.frequency_hz <= end_frequency)
        & (response.coherence >= minimum_coherence)
        & np.isfinite(response.response.real)
        & np.isfinite(response.response.imag)
    )
    if np.count_nonzero(mask) < 8:
        raise RuntimeError(
            f"{response.name} has fewer than 8 coherent frequency bins"
        )
    return (
        response.frequency_hz[mask],
        response.response[mask],
        response.coherence[mask],
    )


def frequency_residual(
    parameters: Sequence[float],
    responses: Iterable[FrequencyResponse],
    model,
    start_frequency: float,
    end_frequency: float,
    minimum_coherence: float,
) -> np.ndarray:
    parts = []
    for response in responses:
        frequency, measured, coherence = selected_bins(
            response,
            start_frequency,
            end_frequency,
            minimum_coherence,
        )
        predicted = model(parameters, frequency)
        weight = np.sqrt(coherence)
        magnitude_error = np.log(np.abs(predicted)) - np.log(np.abs(measured))
        phase_error = np.angle(predicted / measured)
        parts.extend((weight * magnitude_error, weight * phase_error))
    return np.concatenate(parts)


def fit_models(
    responses: Sequence[FrequencyResponse],
    start_frequency: float,
    end_frequency: float,
    minimum_coherence: float,
) -> Dict[str, Dict[str, object]]:
    low_frequency_gains = []
    for response in responses:
        frequency, measured, _ = selected_bins(
            response,
            start_frequency,
            end_frequency,
            minimum_coherence,
        )
        count = min(3, len(frequency))
        low_frequency_gains.append(float(np.median(np.abs(measured[:count]))))
    initial_gain = float(np.median(low_frequency_gains))

    first = optimize.least_squares(
        lambda values: frequency_residual(
            values,
            responses,
            first_order_response,
            start_frequency,
            end_frequency,
            minimum_coherence,
        ),
        x0=[initial_gain, 0.04, 0.005],
        bounds=([1.0, 0.001, 0.0], [150.0, 1.0, 0.2]),
        loss="soft_l1",
    )
    second = optimize.least_squares(
        lambda values: frequency_residual(
            values,
            responses,
            second_order_response,
            start_frequency,
            end_frequency,
            minimum_coherence,
        ),
        x0=[initial_gain, 50.0, 1.0, 0.005],
        bounds=([1.0, 1.0, 0.1, 0.0], [150.0, 300.0, 5.0, 0.2]),
        loss="soft_l1",
    )

    return {
        "first_order": {
            "parameters": first.x,
            "cost": float(first.cost),
            "success": bool(first.success),
        },
        "second_order": {
            "parameters": second.x,
            "cost": float(second.cost),
            "success": bool(second.success),
        },
    }


def delayed_input(values: np.ndarray, sample_rate: float, delay: float) -> np.ndarray:
    time_s = np.arange(len(values), dtype=float) / sample_rate
    return np.interp(
        time_s - delay,
        time_s,
        values,
        left=values[0],
        right=values[-1],
    )


def simulate_first_order(
    parameters: Sequence[float],
    values: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    gain, time_constant, delay = parameters
    source = delayed_input(values, sample_rate, delay)
    alpha = math.exp(-1.0 / (sample_rate * time_constant))
    state = np.empty_like(source)
    state[0] = source[0]
    for index in range(1, len(source)):
        state[index] = alpha * state[index - 1] + (1.0 - alpha) * source[index - 1]
    return gain * state


def simulate_second_order(
    parameters: Sequence[float],
    values: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    gain, natural_frequency, damping_ratio, delay = parameters
    source = delayed_input(values, sample_rate, delay)
    time_s = np.arange(len(values), dtype=float) / sample_rate
    system = signal.TransferFunction(
        [gain * natural_frequency * natural_frequency],
        [
            1.0,
            2.0 * damping_ratio * natural_frequency,
            natural_frequency * natural_frequency,
        ],
    )
    return signal.lsim(system, source, time_s)[1]


def validation_metrics(
    sweeps: Sequence[Sweep],
    models: Dict[str, Dict[str, object]],
    sample_rate: float,
    start_frequency: float,
    end_frequency: float,
) -> Dict[str, List[Dict[str, float]]]:
    nyquist = 0.5 * sample_rate
    low = max(0.1, 0.4 * start_frequency)
    high = min(0.95 * nyquist, 1.5 * end_frequency)
    sos = signal.butter(
        3,
        [low / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )
    result: Dict[str, List[Dict[str, float]]] = {
        "first_order": [],
        "second_order": [],
    }

    for sweep in sweeps:
        throttle = signal.detrend(sweep.throttle, type="linear")
        measured = signal.sosfiltfilt(
            sos,
            signal.detrend(sweep.acceleration, type="linear"),
        )
        for name, simulator in (
            ("first_order", simulate_first_order),
            ("second_order", simulate_second_order),
        ):
            predicted = simulator(
                models[name]["parameters"],
                throttle,
                sample_rate,
            )
            predicted = signal.sosfiltfilt(sos, predicted)
            residual = predicted - measured
            rmse = float(np.sqrt(np.mean(residual * residual)))
            standard_deviation = float(np.std(measured))
            nrmse_fit = (
                1.0 - rmse / standard_deviation
                if standard_deviation > 1e-9
                else math.nan
            )
            denominator = float(np.sum((measured - np.mean(measured)) ** 2))
            r_squared = (
                1.0 - float(np.sum(residual * residual)) / denominator
                if denominator > 1e-12
                else math.nan
            )
            result[name].append(
                {
                    "name": sweep.name,
                    "rmse_m_s2": rmse,
                    "nrmse_fit": nrmse_fit,
                    "r_squared": r_squared,
                }
            )
    return result


def choose_model(
    models: Dict[str, Dict[str, object]],
    validation: Dict[str, List[Dict[str, float]]],
) -> str:
    first_fit = float(
        np.median([item["nrmse_fit"] for item in validation["first_order"]])
    )
    second_fit = float(
        np.median([item["nrmse_fit"] for item in validation["second_order"]])
    )
    cost_improved = (
        float(models["second_order"]["cost"])
        <= 0.95 * float(models["first_order"]["cost"])
    )
    return (
        "second_order"
        if second_fit >= first_fit + 0.05 and cost_improved
        else "first_order"
    )


def model_dictionary(name: str, values: Sequence[float]) -> Dict[str, float]:
    if name == "first_order":
        gain, time_constant, delay = [float(value) for value in values]
        return {
            "gain_m_s2_per_u": gain,
            "time_constant_s": time_constant,
            "cutoff_frequency_hz": 1.0 / (2.0 * math.pi * time_constant),
            "delay_s": delay,
        }
    gain, natural_frequency, damping_ratio, delay = [
        float(value) for value in values
    ]
    return {
        "gain_m_s2_per_u": gain,
        "natural_frequency_rad_s": natural_frequency,
        "damping_ratio": damping_ratio,
        "delay_s": delay,
    }


def serializable_results(
    args: argparse.Namespace,
    sweeps: Sequence[Sweep],
    responses: Sequence[FrequencyResponse],
    models: Dict[str, Dict[str, object]],
    validation: Dict[str, List[Dict[str, float]]],
    selected_model: str,
) -> Dict[str, object]:
    return {
        "label": args.label,
        "input_definition": "delta(-vehicle_attitude_setpoint.thrust_body[2])",
        "output_definition": "delta(-vehicle_acceleration.xyz[2])",
        "sample_rate_hz": args.sample_rate,
        "fit_band_hz": [args.start_frequency, args.end_frequency],
        "minimum_coherence": args.minimum_coherence,
        "sweeps": [
            {
                "name": sweep.name,
                "source": str(sweep.source),
                "duration_s": float(sweep.time_s[-1] - sweep.time_s[0]),
                "voltage_v": sweep.voltage_v,
                "mean_rpm": sweep.mean_rpm,
                "throttle_min": float(np.min(sweep.throttle)),
                "throttle_max": float(np.max(sweep.throttle)),
                "acceleration_min_m_s2": float(np.min(sweep.acceleration)),
                "acceleration_max_m_s2": float(np.max(sweep.acceleration)),
                "median_coherence": float(
                    np.median(
                        selected_bins(
                            response,
                            args.start_frequency,
                            args.end_frequency,
                            args.minimum_coherence,
                        )[2]
                    )
                ),
            }
            for sweep, response in zip(sweeps, responses)
        ],
        "models": {
            name: {
                **model_dictionary(name, value["parameters"]),
                "frequency_fit_cost": float(value["cost"]),
                "validation": validation[name],
                "median_validation_nrmse_fit": float(
                    np.median(
                        [item["nrmse_fit"] for item in validation[name]]
                    )
                ),
            }
            for name, value in models.items()
        },
        "selection_rule": (
            "Use second order only when median validation NRMSE improves by "
            "at least 0.05 and frequency-fit cost improves by at least 5%."
        ),
        "selected_model": selected_model,
        "selected": model_dictionary(
            selected_model,
            models[selected_model]["parameters"],
        ),
    }


def plot_bode(
    output: Path,
    responses: Sequence[FrequencyResponse],
    models: Dict[str, Dict[str, object]],
    start_frequency: float,
    end_frequency: float,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for response in responses:
        mask = (
            (response.frequency_hz >= 0.8 * start_frequency)
            & (response.frequency_hz <= 1.2 * end_frequency)
        )
        frequency = response.frequency_hz[mask]
        measured = response.response[mask]
        axes[0].semilogx(
            frequency,
            np.abs(measured),
            ".",
            alpha=0.6,
            label=response.name,
        )
        axes[1].semilogx(
            frequency,
            np.unwrap(np.angle(measured)) * 180.0 / np.pi,
            ".",
            alpha=0.6,
        )

    frequency = np.geomspace(start_frequency, end_frequency, 300)
    for name, model, style in (
        ("first order", first_order_response, "-"),
        ("second order", second_order_response, "--"),
    ):
        parameters = models[name.replace(" ", "_")]["parameters"]
        response = model(parameters, frequency)
        axes[0].semilogx(
            frequency,
            np.abs(response),
            style,
            linewidth=2.0,
            label=name,
        )
        axes[1].semilogx(
            frequency,
            np.angle(response, deg=True),
            style,
            linewidth=2.0,
        )

    axes[0].set_ylabel("Gain [m/s2/u]")
    axes[1].set_ylabel("Phase [deg]")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_time_validation(
    output: Path,
    sweeps: Sequence[Sweep],
    models: Dict[str, Dict[str, object]],
    sample_rate: float,
) -> None:
    figure, axes = plt.subplots(
        len(sweeps),
        1,
        figsize=(11, max(3.5, 3.0 * len(sweeps))),
        squeeze=False,
    )
    for index, sweep in enumerate(sweeps):
        axis = axes[index, 0]
        throttle = signal.detrend(sweep.throttle, type="linear")
        measured = signal.detrend(sweep.acceleration, type="linear")
        first = simulate_first_order(
            models["first_order"]["parameters"],
            throttle,
            sample_rate,
        )
        second = simulate_second_order(
            models["second_order"]["parameters"],
            throttle,
            sample_rate,
        )
        axis.plot(sweep.time_s, measured, color="black", linewidth=0.8, label="measured")
        axis.plot(sweep.time_s, first, linewidth=0.8, label="first order")
        axis.plot(sweep.time_s, second, linewidth=0.8, label="second order")
        axis.set_title(sweep.name)
        axis.set_ylabel("Delta acceleration [m/s2]")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel("Time [s]")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def markdown_report(results: Dict[str, object]) -> str:
    selected = results["selected"]
    lines = [
        f"# {results['label']} collective thrust dynamics",
        "",
        f"- Selected model: `{results['selected_model']}`",
        (
            f"- Identified gain: "
            f"`{selected['gain_m_s2_per_u']:.4f} m/s^2/u`"
        ),
    ]
    if results["selected_model"] == "first_order":
        lines.extend(
            [
                f"- Time constant: `{selected['time_constant_s']:.6f} s`",
                (
                    f"- Equivalent cutoff: "
                    f"`{selected['cutoff_frequency_hz']:.4f} Hz`"
                ),
                f"- Delay: `{selected['delay_s']:.6f} s`",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"- Natural frequency: "
                    f"`{selected['natural_frequency_rad_s']:.4f} rad/s`"
                ),
                f"- Damping ratio: `{selected['damping_ratio']:.4f}`",
                f"- Delay: `{selected['delay_s']:.6f} s`",
            ]
        )
    lines.extend(
        [
            "",
            "## Sweep data",
            "",
            "| Sweep | Voltage [V] | Mean RPM | Throttle range | Coherence |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for sweep in results["sweeps"]:
        lines.append(
            f"| {sweep['name']} | {sweep['voltage_v']:.3f} | "
            f"{sweep['mean_rpm']:.1f} | "
            f"{sweep['throttle_min']:.3f}...{sweep['throttle_max']:.3f} | "
            f"{sweep['median_coherence']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate models",
            "",
            "| Model | Gain | Validation NRMSE fit | Frequency cost |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, model in results["models"].items():
        lines.append(
            f"| {name} | {model['gain_m_s2_per_u']:.4f} | "
            f"{model['median_validation_nrmse_fit']:.4f} | "
            f"{model['frequency_fit_cost']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The gain is the model DC gain and maps directly to "
            "`thrust_model/acc_per_unit_throttle`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.sample_rate <= 2.5 * args.end_frequency:
        raise ValueError("sample rate is too low for the requested fit band")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweeps: List[Sweep] = []
    for path in args.ulogs:
        sweeps.extend(
            load_sweeps(path, args.sample_rate, args.trim_seconds)
        )
    responses = [
        estimate_frequency_response(sweep, args.sample_rate)
        for sweep in sweeps
    ]
    models = fit_models(
        responses,
        args.start_frequency,
        args.end_frequency,
        args.minimum_coherence,
    )
    validation = validation_metrics(
        sweeps,
        models,
        args.sample_rate,
        args.start_frequency,
        args.end_frequency,
    )
    selected_model = choose_model(models, validation)
    results = serializable_results(
        args,
        sweeps,
        responses,
        models,
        validation,
        selected_model,
    )

    stem = args.label.replace(" ", "_")
    json_path = args.output_dir / f"{stem}_thrust_dynamics.json"
    report_path = args.output_dir / f"{stem}_thrust_dynamics.md"
    bode_path = args.output_dir / f"{stem}_thrust_dynamics_bode.png"
    time_path = args.output_dir / f"{stem}_thrust_dynamics_time.png"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(markdown_report(results), encoding="utf-8")
    plot_bode(
        bode_path,
        responses,
        models,
        args.start_frequency,
        args.end_frequency,
    )
    plot_time_validation(time_path, sweeps, models, args.sample_rate)

    print(f"selected model: {selected_model}")
    print(
        "acc_per_unit_throttle: "
        f"{results['selected']['gain_m_s2_per_u']:.6f}"
    )
    if selected_model == "first_order":
        print(
            "actuator_dyn.z.cutoff_freq: "
            f"{results['selected']['cutoff_frequency_hz']:.6f}"
        )
    else:
        print(
            "actuator_dyn.z.natural_freq: "
            f"{results['selected']['natural_frequency_rad_s']:.6f}"
        )
        print(
            "actuator_dyn.z.damping_ratio: "
            f"{results['selected']['damping_ratio']:.6f}"
        )
    print(f"results: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
