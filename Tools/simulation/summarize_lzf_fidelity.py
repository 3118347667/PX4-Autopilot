#!/usr/bin/env python3

"""Merge standalone LZF fidelity runs into a final validation report."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_ROLES = (
    "real-calibration",
    "real-holdout-1",
    "real-holdout-2",
    "sim-hover",
    "sim-exact-fac",
    "sim-real-rates",
    "sim-sitl-rates",
    "sim-as-flown-exact-fac",
    "sim-as-flown-no-fac",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LZF fidelity metrics and write the final report."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="ROLE:METRICS_JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/lzf_fidelity"),
    )
    return parser.parse_args()


def load_dataset(value: str) -> Tuple[str, Path, Dict[str, object], Dict[str, object]]:
    role, path_text = value.split(":", 1)
    path = Path(path_text)
    document = json.loads(path.read_text(encoding="utf-8"))
    datasets = document["datasets"]

    if len(datasets) != 1:
        raise ValueError(f"{path} must contain exactly one standalone dataset")

    return role, path, next(iter(datasets.values())), document["model_parameters"]


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"

    return f"{float(value):.{digits}f}"


def write_summary_plot(
    datasets: Dict[str, Dict[str, object]],
    parameters: Dict[str, object],
    output_path: Path,
) -> None:
    real_roles = ("real-calibration", "real-holdout-1", "real-holdout-2")
    real_labels = ("Main", "Holdout 1", "Holdout 2")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    rpm_errors = [
        datasets[role]["ulog"]["esc"]["motor_speed_model_pooled"][
            "relative_rmse_percent"
        ]
        for role in real_roles
    ]
    axes[0, 0].bar(real_labels, rpm_errors, color="#287271")
    axes[0, 0].axhline(3.0, color="#d1495b", linestyle="--", label="3% target")
    axes[0, 0].set_ylabel("Relative RMSE (%)")
    axes[0, 0].set_title("Motor command/voltage to RPM")
    axes[0, 0].legend()

    effective_ct = [
        datasets[role]["ulog"]["propeller"]["effective_static_ct"]["median"]
        for role in real_roles
    ]
    axes[0, 1].bar(real_labels, effective_ct, color="#f4a261")
    axes[0, 1].axhline(
        parameters["static_ct"],
        color="#264653",
        linestyle="--",
        label="configured CT",
    )
    axes[0, 1].set_ylabel("Effective CT")
    axes[0, 1].set_title("Body-z force inferred CT")
    axes[0, 1].legend()

    drag_x = [
        datasets[role]["bag"]["aerodynamics"]["rotor_drag"][0][
            "effective_to_configured_ratio"
        ]
        for role in real_roles
    ]
    drag_y = [
        datasets[role]["bag"]["aerodynamics"]["rotor_drag"][1][
            "effective_to_configured_ratio"
        ]
        for role in real_roles
    ]
    locations = np.arange(len(real_labels))
    axes[1, 0].bar(locations - 0.18, drag_x, 0.36, label="body x")
    axes[1, 0].bar(locations + 0.18, drag_y, 0.36, label="body y")
    axes[1, 0].axhline(1.0, color="#264653", linestyle="--")
    axes[1, 0].set_xticks(locations)
    axes[1, 0].set_xticklabels(real_labels)
    axes[1, 0].set_ylabel("Effective/configured coefficient")
    axes[1, 0].set_title("Rotor-plus-airframe lateral drag")
    axes[1, 0].legend()

    trajectory_roles = (
        "real-calibration",
        "real-holdout-1",
        "real-holdout-2",
        "sim-as-flown-no-fac",
        "sim-real-rates",
        "sim-sitl-rates",
    )
    trajectory_labels = (
        "Real main",
        "Real holdout 1",
        "Real holdout 2",
        "As-flown allocator",
        "Earlier real-gain run",
        "Earlier SITL-gain run",
    )
    trajectory_rmse = [
        datasets[role]["bag"]["trajectory"]["position_rmse_norm_m"]
        for role in trajectory_roles
    ]
    axes[1, 1].bar(trajectory_labels, trajectory_rmse, color="#457b9d")
    axes[1, 1].set_yscale("log")
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].set_ylabel("Position RMSE norm (m, log scale)")
    axes[1, 1].set_title("Same recorded trajectory")

    for axis in axes.flat:
        axis.grid(True, axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def write_report(
    datasets: Dict[str, Dict[str, object]],
    parameters: Dict[str, object],
    output_path: Path,
) -> None:
    real_roles = ("real-calibration", "real-holdout-1", "real-holdout-2")
    real_names = {
        "real-calibration": "实机主飞行",
        "real-holdout-1": "实机留出 1",
        "real-holdout-2": "实机留出 2",
    }
    lines = [
        "# LZF 仿真真实性最终评估",
        "",
        "## 结论",
        "",
        "**当前 LZF 是一个有真实参数依据、适合悬停与低频趋势研究的工程模型，"
        "但还不是能复现实机激烈飞行动力学的数字孪生。**",
        "",
        "- 质量、旋翼位置、转速反馈、静态桨系数、负载压降、实测整机惯量和实测电机时间常数"
        "已经进入模型。",
        "- 独立实飞验证显示，稳态 RPM 误差约 `4.4%~4.7%`，低进动比轴向力误差约 "
        "`0.81~0.86 m/s²`。",
        "- 严格复现实飞参数后，旧 FAC 控制配置在定点悬停阶段即翻转；关闭 FAC 后可稳定悬停，"
        "但回放同一圈轨迹 `5.57 s` 后仍超过 60° 并坠落，而实机 18 圈最大位置误差仅 "
        "`0.58 m`。",
        "- 此前的 SITL 轨迹试验没有完全匹配实机的控制分配、`THR_MDL_FAC` 和姿态增益，"
        "只能作为预试验，不能再称为“精确实机配置”。",
        "- 更新实测惯量和电机时间常数后，PX4 原生悬停保持有界但出现 `8.82 Hz` 俯仰极限环；"
        "说明惯量数值已进入物理模型，但控制器、推力/力矩增益和执行机构动态尚未闭环匹配。",
        "- 因而不能用当前模型评估激烈轨迹的控制器稳定裕度、FAC 控制效果或实机安全边界。",
        "",
        "## 实机交叉验证",
        "",
        "| 数据集 | RPM 相对 RMSE | z 比力 RMSE | 有效 CT | 电池 RMSE | 去偏置电池 RMSE | 角速度 RMSE | 轨迹 RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for role in real_roles:
        item = datasets[role]
        ulog = item["ulog"]
        motor = ulog["esc"]["motor_speed_model_pooled"]
        propeller = ulog["propeller"]
        battery = ulog["battery"]["model"]
        rate = ulog["rate_tracking"]["pooled"]
        trajectory = item["bag"]["trajectory"]
        lines.append(
            f"| {real_names[role]} | {motor['relative_rmse_percent']:.2f}% | "
            f"{propeller['specific_thrust_model']['rmse']:.3f} m/s² | "
            f"{propeller['effective_static_ct']['median']:.6f} | "
            f"{battery['rmse']:.3f} V | {battery['bias_removed_rmse']:.3f} V | "
            f"{rate['rmse']:.3f} rad/s | "
            f"{trajectory['position_rmse_norm_m']:.3f} m |"
        )

    main = datasets["real-calibration"]
    holdout_1 = datasets["real-holdout-1"]
    holdout_2 = datasets["real-holdout-2"]
    configured_ct = float(parameters["static_ct"])
    ct_deviations = [
        100.0
        * (
            datasets[role]["ulog"]["propeller"]["effective_static_ct"]["median"]
            / configured_ct
            - 1.0
        )
        for role in real_roles
    ]
    motor_biases = np.asarray(
        [
            [
                motor["bias"]
                for motor in datasets[role]["ulog"]["esc"][
                    "motor_speed_model_per_motor"
                ]
            ]
            for role in real_roles
        ]
    )
    drag_x = [
        datasets[role]["bag"]["aerodynamics"]["rotor_drag"][0][
            "effective_to_configured_ratio"
        ]
        for role in real_roles
    ]
    drag_y = [
        datasets[role]["bag"]["aerodynamics"]["rotor_drag"][1][
            "effective_to_configured_ratio"
        ]
        for role in real_roles
    ]
    advance_ratio_p95 = [
        datasets[role]["bag"]["aerodynamics"]["advance_ratio_positive"]["p95"]
        for role in real_roles
    ]
    advance_ratio_max = [
        datasets[role]["bag"]["aerodynamics"]["advance_ratio_positive"]["max"]
        for role in real_roles
    ]
    low_soc_battery = holdout_2["ulog"]["battery"]

    lines.extend(
        [
            "",
            "## 分系统判断",
            "",
            "### 电机与 ESC",
            "",
            f"- 三次实飞的 RPM 模型误差为 "
            f"`{main['ulog']['esc']['motor_speed_model_pooled']['relative_rmse_percent']:.2f}% / "
            f"{holdout_1['ulog']['esc']['motor_speed_model_pooled']['relative_rmse_percent']:.2f}% / "
            f"{holdout_2['ulog']['esc']['motor_speed_model_pooled']['relative_rmse_percent']:.2f}%`，"
            "属于中等精度，尚未达到 3% 验收线。",
            f"- 四电机偏差具有稳定的个体差异；M1 在三次飞行中为 "
            f"`{motor_biases[:, 0].min():.0f}~{motor_biases[:, 0].max():.0f} rpm`，"
            "当前四电机共用一条映射，无法复现这种不对称。",
            f"- 实机 ESC 日志约 `{main['ulog']['rates_hz']['esc_status']:.1f} Hz`，"
            f"SITL 为 `{datasets['sim-sitl-rates']['ulog']['rates_hz']['esc_status']:.1f} Hz`；"
            "仿真没有 CAN/DShot 延迟、量化、丢包和估速噪声。",
            "- 电机升速和降速时间常数均已设为实测的 `24.568 ms`；当前实验没有区分两个方向，"
            "因此模型也暂不引入不对称。",
            "",
            "### 桨叶与来流",
            "",
            f"- 实飞反推的有效 CT 比配置值高 "
            f"`{min(ct_deviations):.1f}%~{max(ct_deviations):.1f}%`；"
            "当前模型在三次飞行中都略低估轴向力。",
            f"- 实飞可验证的正进动比范围很窄：95% 仅 "
            f"`{min(advance_ratio_p95):.3f}~{max(advance_ratio_p95):.3f}`，最大 "
            f"`{max(advance_ratio_max):.3f}`。UIUC 表中更高进动比部分仍未被这批飞行数据验证。",
            "- UIUC 数据主要在约 7~8 krpm 测得，而实机悬停约 16 krpm；当前直接使用无量纲表，"
            "尚未模拟 Reynolds 数和桨叶变形随转速的变化。",
            "",
            "### 横向气动",
            "",
            f"- 三次飞行辨识出的 x 方向有效阻力为配置的 "
            f"`{min(drag_x):.2f}~{max(drag_x):.2f}x`，y 方向为 "
            f"`{min(drag_y):.2f}~{max(drag_y):.2f}x`，重复性很强。",
            "- 该有效值同时包含桨盘和机身阻力，不能把全部差额直接写入 "
            "`rotorDragCoefficient`；当前模型缺少可独立标定的机身阻力项。",
            "",
            "### 电池",
            "",
            f"- 主飞行电压 RMSE `{main['ulog']['battery']['model']['rmse']:.3f} V`、"
            f"R² `{main['ulog']['battery']['model']['r_squared']:.3f}`；满电附近留出飞行 "
            f"`{holdout_1['ulog']['battery']['model']['rmse']:.3f} V`、"
            f"R² `{holdout_1['ulog']['battery']['model']['r_squared']:.3f}`。",
            f"- 低初始电量留出飞行的绝对 RMSE 为 "
            f"`{low_soc_battery['model']['rmse']:.3f} V`，偏差 "
            f"`{low_soc_battery['model']['bias']:.3f} V`；去掉常量偏置后仍有 "
            f"`{low_soc_battery['model']['bias_removed_rmse']:.3f} V`。",
            f"- 用该日志开头的 `remaining={low_soc_battery['remaining_start']:.3f}` "
            f"初始化后 RMSE 降为 "
            f"`{low_soc_battery['model_with_logged_initial_soc']['rmse']:.3f} V`，"
            "但同一做法会恶化另外两条日志，说明实机 `remaining` 本身也没有可靠对齐。"
            "当前模型缺少可持久化、可观测的初始 SOC。",
            "- 电流、内阻、效率、温度和容量 Ah 仍未模拟；SITL 的电压自洽误差很小是公式回代，"
            "不是独立真实性证据。",
            "",
            "### 刚体与控制动态",
            "",
            f"- 实机三次飞行的角速度跟踪 RMSE 为 "
            f"`{main['ulog']['rate_tracking']['pooled']['rmse']:.3f} / "
            f"{holdout_1['ulog']['rate_tracking']['pooled']['rmse']:.3f} / "
            f"{holdout_2['ulog']['rate_tracking']['pooled']['rmse']:.3f} rad/s`。",
            f"- 更新后的 PX4 原生悬停角速度跟踪 RMSE 为 "
            f"`{datasets['sim-hover']['ulog']['rate_tracking']['pooled']['rmse']:.3f} rad/s`，"
            f"其中俯仰轴为 "
            f"`{datasets['sim-hover']['ulog']['rate_tracking']['per_axis']['pitch']['rmse']:.3f} rad/s`；"
            "实测到 `8.82 Hz` 俯仰极限环和约 `9493 rpm` 的前后电机差动 RMS，"
            "因此当前控制动态仍不可信。",
            "- 整机关于重心的实测惯量已更新为 "
            "`Ixx=0.003242418, Iyy=0.003092245, Izz=0.005802551 kg·m²`。"
            "Gazebo 的 `base_link` 数值已扣除 IMU、GPS 和四个旋翼子链接的惯量及平行轴贡献，"
            "所以不能把模板中的补偿值误读为整机惯量。",
            "- 这组惯量与 `24.568 ms` 电机时间常数尚未经过新的 SITL chirp/轨迹闭环验证；"
            "下表中的历史复现结果均产生于更新前模型。",
            "- 实飞 rosbag 中外部控制器记录的 `mass=1.378 kg`，LZF 刚体为 `1.326 kg`，"
            "相差 3.9%。试验保留了实飞值以复现原控制器；该差异会影响推力标定，"
            "但不足以单独解释翻转。",
            "",
            "### 控制分配与电机接线",
            "",
            "- 三份实机 ULog 都保存了相同的旧控制分配位置：FRD 下四个旋翼为 "
            "`(±1, ±1, 0)`。这只是飞行时忘记更新的归一化正方形参数，不是机架尺寸记录。",
            "- 实际机架位置为相对整机重心的 FLU 坐标 "
            "`(0.085,-0.10,0.01), (-0.085,0.10,0.01), "
            "(0.085,0.10,0.01), (-0.085,-0.10,0.01)`；Gazebo 历史复现时始终保持这些真实物理位置。",
            "- 旧正方形分配矩阵的归一化滚转项为 `±0.7071`；按实际 `0.17 × 0.20 m` "
            "矩形几何计算时为 `±0.6010`，因此实飞当时的滚转电机差动比正确几何大约 17.6%。"
            "俯仰项均为 `±0.7071`；`PZ=0` 与实际 `-0.01 m` 对竖直旋翼的一阶滚转/俯仰力矩无影响。",
            "- 实机 PWM 功能顺序为 `[M3,M2,M1,M4] = [103,102,101,104]`。LZF 已同步该接线，"
            "Gazebo RPM 回传也按同一通道顺序发布，再由分析工具映射回逻辑 `[M1,M2,M3,M4]`。",
            "",
            "## 同轨迹闭环试验",
            "",
            "| 试验 | 结果 | 轨迹位置 RMSE | >1 m 时间 | >60° 时间 | 角速度 RMSE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    real_trajectory = main["bag"]["trajectory"]
    exact_fac = datasets["sim-exact-fac"]["ulog"]
    sim_real = datasets["sim-real-rates"]
    sim_sitl = datasets["sim-sitl-rates"]
    as_flown_exact = datasets["sim-as-flown-exact-fac"]["ulog"]
    as_flown_no_fac = datasets["sim-as-flown-no-fac"]
    sim_real_trajectory = sim_real["bag"]["trajectory"]
    sim_sitl_trajectory = sim_sitl["bag"]["trajectory"]
    as_flown_no_fac_trajectory = as_flown_no_fac["bag"]["trajectory"]
    lines.extend(
        [
            f"| 实机原始飞行（18 圈） | 全部完成；最大误差 "
            f"{real_trajectory['position_error_max_norm_m']:.3f} m | "
            f"{real_trajectory['position_rmse_norm_m']:.3f} m | - | - | "
            f"{main['ulog']['rate_tracking']['pooled']['rmse']:.3f} rad/s |",
            f"| 更新后：PX4 原生定点悬停 | 未翻转，但存在 8.82 Hz 俯仰极限环 | "
            f"- | - | 未超过 60° | "
            f"{datasets['sim-hover']['ulog']['rate_tracking']['pooled']['rmse']:.3f} rad/s |",
            f"| 更新前：旧 CA + 实机 PWM + FAC | 定点阶段翻转；进入悬停约 "
            f"5.0 s 后超过 60° | - | - | "
            f"{as_flown_exact['stability']['first_tilt_over_60_s_after_arm']:.2f} s* | "
            f"{as_flown_exact['rate_tracking']['pooled']['rmse']:.3f} rad/s |",
            f"| 更新前：旧 CA + 实机 PWM，关闭 FAC | 悬停稳定，轨迹中翻转坠落 | "
            f"{as_flown_no_fac_trajectory['position_rmse_norm_m']:.3f} m | "
            f"{as_flown_no_fac_trajectory['time_to_position_error_1m_s']:.2f} s | "
            f"{as_flown_no_fac_trajectory['time_to_tilt_over_60_deg_s']:.2f} s | "
            f"{as_flown_no_fac['ulog']['rate_tracking']['pooled']['rmse']:.3f} rad/s |",
            f"| 此前预试验：FAC，参数未完全对齐 | 定点阶段翻转；解锁后 "
            f"{exact_fac['stability']['first_tilt_over_60_s_after_arm']:.2f} s 超过 60° | "
            f"- | - | {exact_fac['stability']['first_tilt_over_60_s_after_arm']:.2f} s* | "
            f"{exact_fac['rate_tracking']['pooled']['rmse']:.3f} rad/s |",
            f"| 此前预试验：关闭 FAC，实机角速度增益 | 起飞/悬停稳定，轨迹后坠落 | "
            f"{sim_real_trajectory['position_rmse_norm_m']:.3f} m | "
            f"{sim_real_trajectory['time_to_position_error_1m_s']:.2f} s | "
            f"{sim_real_trajectory['time_to_tilt_over_60_deg_s']:.2f} s | "
            f"{sim_real['ulog']['rate_tracking']['pooled']['rmse']:.3f} rad/s |",
            f"| 此前预试验：关闭 FAC，SITL 默认增益 | 起飞/悬停稳定，轨迹后坠落 | "
            f"{sim_sitl_trajectory['position_rmse_norm_m']:.3f} m | "
            f"{sim_sitl_trajectory['time_to_position_error_1m_s']:.2f} s | "
            f"{sim_sitl_trajectory['time_to_tilt_over_60_deg_s']:.2f} s | "
            f"{sim_sitl['ulog']['rate_tracking']['pooled']['rmse']:.3f} rad/s |",
            "",
            "\\* FAC 试验未进入轨迹，表中阈值时间从解锁开始；轨迹试验阈值时间从轨迹第一帧开始。",
            "",
            "除“更新后：PX4 原生定点悬停”外，其余闭环试验使用的是惯量和电机时间常数"
            "更新前的 LZF，不能代表本次更新后的轨迹精度，必须重新采集 SITL 轨迹日志后"
            "才能作 A/B 判断。",
            "",
            "新试验把历史控制分配和 PWM 接线也纳入了复现：FAC 会把差异放大到悬停失稳；"
            "即使关闭 FAC，旧正方形分配在真实矩形机架上也无法完成这圈轨迹。"
            "不过，这仍不能证明 25% 的滚转分配偏差是唯一原因；未辨识惯量和电机动态、"
            "低估的横向总阻力、四电机个体差异、推力标定和理想化传感器/执行链路仍然耦合在结果中。",
            "",
            "## 当前适用边界",
            "",
            "| 用途 | 可信度 |",
            "|---|---|",
            "| 质量、几何和电机编号检查 | 高 |",
            "| 悬停 RPM、推力和电压变化趋势 | 中 |",
            "| 低进动比下的平均轴向推力 | 中 |",
            "| 不同初始 SOC 的绝对端电压 | 低 |",
            "| 横向高速气动与机身阻力 | 低 |",
            "| 角速度带宽、相位裕度和惯量响应 | 低 |",
            "| FAC、高速轨迹和失稳边界 | 低，不可用于实机等价验证 |",
            "",
            "## 改进优先级",
            "",
            "1. 实机后续飞行使用相对重心的实际 `CA_ROTOR*` 几何；历史 ULog 中的 `±1` "
            "只能用于复现当时错误配置，不能反向覆盖 Gazebo 物理尺寸。",
            "2. 用小幅 roll/pitch/yaw chirp 验证新惯量和 `24.568 ms` 电机动态，"
            "并继续辨识时延与力矩系数。",
            "3. 记录至少 200 Hz 的电机命令、四路机械 RPM、电压和电流阶跃，分别标定四电机的 "
            "`RPM(u,V)`、上升/下降时间常数和饱和区。",
            "4. 将机身阻力与旋翼阻力拆成两个模型，用多方向恒速段辨识，随后在未参与拟合的高速轨迹验证。",
            "5. 用实测初始开路电压/静置时间维护持久 SOC，并加入电流、内阻和温度；不要直接信任当前 "
            "`battery_status.remaining`。",
            "6. 完成上述动态建模后再回放同一轨迹；验收条件应至少是整圈不失稳、位置 RMSE 接近实机 "
            "`0.17~0.19 m`，再恢复 FAC 验证。",
            "",
            "## 说明",
            "",
            "- 历史复现只临时修改 PX4 控制分配参数；Gazebo 中的真实旋翼位置没有改动。",
            "- LZF 永久修改同步了实机 PWM/RPM 通道顺序，并加入本次实测惯量和电机时间常数；"
            "历史轨迹指标没有被当作更新后结果复用。",
            "- 仿真失败日志中的坠机后非物理关节 RPM 已在统计中剔除。",
            "- 横向阻力辨识是有效总阻力，不能直接解释为纯桨盘系数。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets: Dict[str, Dict[str, object]] = {}
    sources: Dict[str, str] = {}
    parameters = None

    for value in args.dataset:
        role, path, dataset, dataset_parameters = load_dataset(value)

        if role not in EXPECTED_ROLES:
            raise ValueError(f"unknown role {role!r}")

        datasets[role] = dataset
        sources[role] = str(path)

        if parameters is None:
            parameters = dataset_parameters
        elif parameters != dataset_parameters:
            raise ValueError(f"model parameters in {path} do not match")

    missing = sorted(set(EXPECTED_ROLES) - set(datasets))

    if missing:
        raise ValueError(f"missing datasets: {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_parameters": parameters,
        "source_metrics": sources,
        "datasets": datasets,
    }
    metrics_path = args.output_dir / "final_metrics.json"
    report_path = args.output_dir / "final_report.md"
    plot_path = args.output_dir / "final_summary.png"
    metrics_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(datasets, parameters, report_path)
    write_summary_plot(datasets, parameters, plot_path)
    print(f"metrics: {metrics_path}")
    print(f"report:  {report_path}")
    print(f"plot:    {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
