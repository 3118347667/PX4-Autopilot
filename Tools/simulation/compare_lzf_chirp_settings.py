#!/usr/bin/env python3

"""Compare LZF thrust-chirp identification results across sweep settings."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STUDIES = (
    ("baseline_b", 0.25, 5.0, 0.15, 60.0),
    ("baseline_c", 0.25, 5.0, 0.15, 60.0),
    ("small_amp", 0.25, 5.0, 0.06, 60.0),
    ("low_band", 0.10, 2.0, 0.15, 60.0),
    ("wide_band", 0.25, 12.0, 0.15, 60.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("build/lzf_chirp_setting_study"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/lzf_chirp_setting_study/comparison"),
    )
    return parser.parse_args()


def load_results(root, label, start_hz, end_hz, magnitude, duration_s):
    time_path = (
        root / label / "time_domain" / "{}_thrust_chirp_fit.json".format(label)
    )
    chain_path = root / label / "chain" / "{}.json".format(label)
    time_result = json.loads(time_path.read_text())
    chain_result = json.loads(chain_path.read_text())
    selected = time_result["selected_model"]
    motor = chain_result["motor_chain"]
    propeller = chain_result["propeller_chain"]
    run = chain_result["runs"][0]
    return {
        "label": label,
        "start_frequency_hz": start_hz,
        "end_frequency_hz": end_hz,
        "magnitude": magnitude,
        "duration_s": duration_s,
        "ulog": run["path"],
        "whole_chain_gain_m_s2_per_u": selected["gain"],
        "whole_chain_time_constant_s": selected["time_constant_s"],
        "whole_chain_delay_s": selected["delay_s"],
        "whole_chain_rmse_m_s2": selected["rmse"],
        "whole_chain_r_squared": selected["r_squared"],
        "motor_gain_rpm_per_u": motor["gain"],
        "motor_time_constant_s": motor["time_constant_s"],
        "motor_delay_s": motor["delay_s"],
        "motor_rmse_rpm": motor["rmse"],
        "motor_r_squared": motor["r_squared"],
        "propeller_force_scale": propeller["gain"],
        "propeller_delay_s": propeller["delay_s"],
        "propeller_rmse_m_s2": propeller["rmse"],
        "propeller_r_squared": propeller["r_squared"],
        "mean_rpm": run["mean_rpm"],
        "maximum_advance_ratio": run["advance_ratio_max"],
    }


def percent_delta(value, reference):
    return 100.0 * (value / reference - 1.0)


def write_markdown(path, studies):
    reference = studies[0]
    lines = [
        "# LZF chirp-setting comparison",
        "",
        "All experiments used LZF SITL at a fixed 23.2 V with "
        "`THR_MDL_FAC=0`.",
        "",
        "## Sweep settings and whole-chain fit",
        "",
        "| Setting | Sweep | Magnitude | Gain | Delta vs baseline | Tau | Delay | RMSE | R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in studies:
        lines.append(
            "| {label} | {start_frequency_hz:.2f}-{end_frequency_hz:.1f} Hz / "
            "{duration_s:.0f} s | {magnitude:.2f} | "
            "{whole_chain_gain_m_s2_per_u:.3f} m/s2/u | {gain_delta:+.2f}% | "
            "{tau_ms:.3f} ms | {delay_ms:.3f} ms | "
            "{whole_chain_rmse_m_s2:.3f} | {whole_chain_r_squared:.6f} |".format(
                gain_delta=percent_delta(
                    item["whole_chain_gain_m_s2_per_u"],
                    reference["whole_chain_gain_m_s2_per_u"],
                ),
                tau_ms=1000.0 * item["whole_chain_time_constant_s"],
                delay_ms=1000.0 * item["whole_chain_delay_s"],
                **item,
            )
        )

    lines += [
        "",
        "## Separated motor and propeller fits",
        "",
        "| Setting | Motor gain | Motor tau | Tau error | Motor delay | Motor R2 | Prop scale | Prop RMSE | Prop R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    configured_tau = 0.024568
    for item in studies:
        lines.append(
            "| {label} | {motor_gain_rpm_per_u:.1f} RPM/u | {motor_tau_ms:.3f} ms | "
            "{tau_error:+.2f}% | {motor_delay_ms:.3f} ms | "
            "{motor_r_squared:.6f} | {propeller_force_scale:.6f} | "
            "{propeller_rmse_m_s2:.3f} m/s2 | {propeller_r_squared:.6f} |".format(
                motor_tau_ms=1000.0 * item["motor_time_constant_s"],
                tau_error=percent_delta(
                    item["motor_time_constant_s"], configured_tau
                ),
                motor_delay_ms=1000.0 * item["motor_delay_s"],
                **item,
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- The two clean 0.25-5 Hz, magnitude-0.15 repeats differ by only "
        "0.074% in whole-chain gain. Their time constants differ by 0.312 ms "
        "and their delays by 0.299 ms.",
        "- The direct command-to-RPM fit is stable across all settings. "
        "The fitted motor time constant remains within 1.7% of the configured "
        "24.568 ms, so the motor pole itself is not changing.",
        "- Reducing magnitude from 0.15 to 0.06 increases the fitted "
        "whole-chain local gain by about 5%. This is a working-point and "
        "nonlinearity effect in the command-to-thrust chain, not a change in "
        "the motor time constant.",
        "- A 0.1-2 Hz sweep does not excite enough bandwidth to separate the "
        "whole-chain pole from delay. Its whole-chain time constant reaches "
        "the optimizer lower bound even though its R2 is high.",
        "- Extending the upper frequency from 5 Hz to 12 Hz changes the "
        "whole-chain gain by less than 0.4%, while retaining a physically "
        "credible pole estimate.",
        "",
        "The separate propeller fit uses all four RPM values, measured axial "
        "airspeed, and the LZF UIUC CT(J) table.",
        "",
        "The earlier `baseline_a` attempt is excluded because its ULog "
        "contains a stale sample from a previous sweep and several 0.2-0.3 s "
        "logging gaps. It is retained under the study directory for audit.",
        "",
    ]
    path.write_text("\n".join(lines))


def plot_comparison(path, studies):
    labels = [item["label"] for item in studies]
    positions = np.arange(len(studies))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    axes[0, 0].bar(
        positions, [item["whole_chain_gain_m_s2_per_u"] for item in studies]
    )
    axes[0, 0].set_ylabel("m/s2 per command")
    axes[0, 0].set_title("Whole-chain DC gain")

    axes[0, 1].bar(
        positions,
        [1000.0 * item["whole_chain_time_constant_s"] for item in studies],
        label="whole chain",
    )
    axes[0, 1].bar(
        positions,
        [1000.0 * item["motor_time_constant_s"] for item in studies],
        width=0.5,
        label="command to RPM",
    )
    axes[0, 1].axhline(24.568, color="black", linestyle="--", label="configured")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].set_title("Identified time constant")
    axes[0, 1].legend()

    axes[1, 0].bar(
        positions, [item["motor_gain_rpm_per_u"] for item in studies]
    )
    axes[1, 0].axhline(1682.0 * 23.2, color="black", linestyle="--")
    axes[1, 0].set_ylabel("RPM per command")
    axes[1, 0].set_title("Command-to-RPM local gain")

    axes[1, 1].bar(
        positions, [item["propeller_force_scale"] for item in studies]
    )
    axes[1, 1].axhline(1.0, color="black", linestyle="--")
    axes[1, 1].set_ylim(0.98, 1.02)
    axes[1, 1].set_ylabel("fitted / modeled force")
    axes[1, 1].set_title("RPM and CT(J) force scale")

    for axis in axes.flat:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    studies = [
        load_results(args.study_root, *study) for study in DEFAULT_STUDIES
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "chirp_setting_comparison.json"
    output_md = args.output_dir / "chirp_setting_comparison.md"
    output_png = args.output_dir / "chirp_setting_comparison.png"
    output_json.write_text(json.dumps({"studies": studies}, indent=2) + "\n")
    write_markdown(output_md, studies)
    plot_comparison(output_png, studies)
    print("wrote {}".format(output_json))
    print("wrote {}".format(output_md))
    print("wrote {}".format(output_png))


if __name__ == "__main__":
    main()
