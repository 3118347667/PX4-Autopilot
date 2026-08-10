#!/usr/bin/env python3

"""Replay one recorded LZF PositionCommand lap with fresh ROS timestamps."""

import argparse
import copy
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and replay a single /setpoints_cmd lap from a ROS bag."
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("--lap", type=int, default=1, help="one-based lap number")
    parser.add_argument("--topic", default="/setpoints_cmd")
    parser.add_argument("--odom-topic", default="/mavros/local_position/odom")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--position-scale",
        type=float,
        default=1.0,
        help=(
            "scale the path about its first point; derivatives are scaled "
            "consistently with both position-scale and --speed"
        ),
    )
    parser.add_argument(
        "--vertical-excitation-peak-acceleration",
        type=float,
        default=0.0,
        metavar="M_S2",
        help=(
            "add a smooth asymmetric periodic z trajectory whose maximum "
            "upward acceleration is M_S2; zero disables it"
        ),
    )
    parser.add_argument(
        "--vertical-excitation-frequency",
        type=float,
        default=0.6,
        metavar="HZ",
        help="frequency of the optional asymmetric z excitation",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="repeat the selected lap continuously in the same ROS process",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="do not translate the trajectory start to the current odometry position",
    )
    parser.add_argument(
        "--export-controller-yaml",
        type=Path,
        help="write the recorded /px4ctrl_param values as a px4ctrl YAML file",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="write controller YAML and exit without replaying",
    )
    return parser.parse_args()


def controller_yaml(message) -> Dict[str, object]:
    return {
        "mass": message.mass,
        "gra": message.gra,
        "ctrl_freq_max": message.ctrl_freq_max,
        "use_outer_loop_FAC": message.use_outer_loop_FAC,
        "max_tilt_angle_deg": message.max_tilt_angle_deg,
        "manual_ctrl_mode": message.manual_ctrl_mode,
        "max_manual_vel": message.max_manual_vel,
        "max_manual_vel_z": message.max_manual_vel_z,
        "max_manual_yaw_rate": message.max_manual_yaw_rate,
        "manual_ctrl_lookahead_time": message.manual_ctrl_lookahead_time,
        "imu_acc_bias": {
            "x": message.imu_acc_bias_x,
            "y": message.imu_acc_bias_y,
            "z": message.imu_acc_bias_z,
        },
        "rc_reverse": {
            "roll": message.rc_reverse_roll,
            "pitch": message.rc_reverse_pitch,
            "yaw": message.rc_reverse_yaw,
            "throttle": message.rc_reverse_throttle,
        },
        "auto_takeoff_land": {
            "enable": True,
            "enable_auto_arm": True,
            # SITL has no RC input. This changes only the FSM input requirement.
            "no_RC": True,
            "takeoff_height": message.auto_takeoff_land_height,
            "takeoff_land_speed": message.auto_takeoff_land_speed,
        },
        "thrust_model": {
            "curve_a": message.thrust_model_curve_a,
            "curve_b": message.thrust_model_curve_b,
            "curve_c": message.thrust_model_curve_c,
            "solver_type": message.thrust_model_solver_type,
            "acc_per_unit_throttle": message.thrust_model_acc_per_unit_throttle,
            "hover_percentage": message.thrust_model_hover_percentage,
        },
        "actuator_dyn": {
            "xy": {
                "model_type": message.actuator_dyn_xy_model_type,
                "cutoff_freq": message.actuator_dyn_xy_cutoff_freq,
                "natural_freq": message.actuator_dyn_xy_natural_freq,
                "damping_ratio": message.actuator_dyn_xy_damping_ratio,
            },
            "z": {
                "model_type": message.actuator_dyn_z_model_type,
                "cutoff_freq": message.actuator_dyn_z_cutoff_freq,
                "natural_freq": message.actuator_dyn_z_natural_freq,
                "damping_ratio": message.actuator_dyn_z_damping_ratio,
            },
        },
        "dob": {
            "enable": message.dob_enable,
            "enable_x": message.dob_enable_x,
            "enable_y": message.dob_enable_y,
            "enable_z": message.dob_enable_z,
            "c_force": message.dob_c_force,
            "filter_cutoff_hz": message.dob_filter_cutoff_hz,
            "force_scale": message.dob_force_scale,
            "thrust_source": message.dob_thrust_source,
            "thrust_coeff": message.dob_thrust_coeff,
            "force_limit": message.dob_force_limit,
        },
        "gain": {
            "Kp0": message.gain_Kp0,
            "Kp1": message.gain_Kp1,
            "Kp2": message.gain_Kp2,
            "Kv0": message.gain_Kv0,
            "Kv1": message.gain_Kv1,
            "Kv2": message.gain_Kv2,
            "Ka0": message.gain_Ka0,
            "Ka1": message.gain_Ka1,
            "Ka2": message.gain_Ka2,
        },
        "msg_timeout": {
            "odom": message.msg_timeout_odom,
            "rc": message.msg_timeout_rc,
            "cmd": message.msg_timeout_cmd,
            "esc": message.msg_timeout_esc,
        },
        "battery": {
            "series": message.battery_series,
            "cell_full_voltage": message.battery_cell_full_voltage,
            "cell_cutoff_voltage": message.battery_cell_cutoff_voltage,
        },
    }


def load_laps(bag, topic: str) -> List[List[Tuple[float, object]]]:
    samples = [
        (record_time.to_sec(), copy.deepcopy(message))
        for _, message, record_time in bag.read_messages(topics=[topic])
    ]

    if not samples:
        raise RuntimeError(f"bag has no {topic} messages")

    laps: List[List[Tuple[float, object]]] = [[]]

    for sample in samples:
        if laps[-1] and sample[0] - laps[-1][-1][0] > 0.5:
            laps.append([])

        laps[-1].append(sample)

    return laps


def vertical_excitation_derivative(
    elapsed_s: float,
    order: int,
    peak_acceleration: float,
    frequency_hz: float,
) -> float:
    """Return one derivative of a C-infinity asymmetric vertical excitation.

    The acceleration is a normalized, zero-mean version of
    ``(1 + cos(theta))**16``.  Its positive peak is about 6.15 times the
    magnitude of its broad negative recovery segment.  Starting at
    ``theta=pi`` makes position, velocity and every odd derivative continuous
    with the unmodified trajectory.  The finite cosine series is integrated
    analytically, so position through crackle remain mutually consistent.
    """
    if peak_acceleration <= 0.0:
        return 0.0
    if order < 0 or order > 5:
        raise ValueError(f"unsupported vertical excitation derivative: {order}")

    exponent = 16
    omega = 2.0 * math.pi * frequency_hz
    theta = math.pi + omega * elapsed_s
    mean = math.comb(2 * exponent, exponent) / float(2**exponent)
    positive_peak = float(2**exponent) - mean
    harmonics = range(1, exponent + 1)
    coefficients = [
        math.comb(2 * exponent, exponent - harmonic)
        / float(2 ** (exponent - 1))
        for harmonic in harmonics
    ]

    if order == 0:
        # Twice integrating acceleration.  Subtract the value at theta=pi so
        # the excitation starts at zero position without changing derivatives.
        series = sum(
            coefficient
            * (math.cos(harmonic * theta) - math.cos(harmonic * math.pi))
            / float(harmonic * harmonic)
            for harmonic, coefficient in zip(harmonics, coefficients)
        )
        return -peak_acceleration * series / (positive_peak * omega * omega)
    if order == 1:
        series = sum(
            coefficient * math.sin(harmonic * theta) / float(harmonic)
            for harmonic, coefficient in zip(harmonics, coefficients)
        )
        return peak_acceleration * series / (positive_peak * omega)

    phase_order = order - 2
    phase = phase_order % 4
    total = 0.0
    for harmonic, coefficient in zip(harmonics, coefficients):
        angle = harmonic * theta
        if phase == 0:
            basis = math.cos(angle)
        elif phase == 1:
            basis = -math.sin(angle)
        elif phase == 2:
            basis = -math.cos(angle)
        else:
            basis = math.sin(angle)
        total += coefficient * harmonic**phase_order * basis
    return (
        peak_acceleration
        * omega**phase_order
        * total
        / positive_peak
    )


def main() -> int:
    args = parse_args()

    if not args.bag.is_file():
        print(f"error: missing bag: {args.bag}", file=sys.stderr)
        return 2

    try:
        import rosbag
    except ImportError as error:
        print(f"error: rosbag is unavailable: {error}", file=sys.stderr)
        return 2

    with rosbag.Bag(str(args.bag)) as bag:
        if args.export_controller_yaml is not None:
            parameter_message = next(
                (
                    message
                    for _, message, _ in bag.read_messages(
                        topics=["/px4ctrl_param"]
                    )
                ),
                None,
            )

            if parameter_message is None:
                print("error: bag has no /px4ctrl_param", file=sys.stderr)
                return 1

            args.export_controller_yaml.parent.mkdir(parents=True, exist_ok=True)
            args.export_controller_yaml.write_text(
                yaml.safe_dump(
                    controller_yaml(parameter_message),
                    sort_keys=False,
                    default_flow_style=False,
                ),
                encoding="utf-8",
            )
            print(f"controller YAML: {args.export_controller_yaml}")

        if args.export_only:
            return 0

        laps = load_laps(bag, args.topic)

    if args.lap < 1 or args.lap > len(laps):
        print(
            f"error: lap must be in [1, {len(laps)}], got {args.lap}",
            file=sys.stderr,
        )
        return 2

    if args.speed <= 0.0:
        print("error: --speed must be positive", file=sys.stderr)
        return 2
    if args.position_scale <= 0.0:
        print("error: --position-scale must be positive", file=sys.stderr)
        return 2
    if (
        not math.isfinite(args.vertical_excitation_peak_acceleration)
        or args.vertical_excitation_peak_acceleration < 0.0
    ):
        print(
            "error: --vertical-excitation-peak-acceleration must be finite "
            "and non-negative",
            file=sys.stderr,
        )
        return 2
    if (
        not math.isfinite(args.vertical_excitation_frequency)
        or args.vertical_excitation_frequency <= 0.0
    ):
        print(
            "error: --vertical-excitation-frequency must be finite and positive",
            file=sys.stderr,
        )
        return 2

    try:
        import rospy
        from nav_msgs.msg import Odometry
        from quadrotor_msgs.msg import PositionCommand
    except ImportError as error:
        print(
            "error: ROS messages are unavailable; source /opt/ros/noetic/setup.zsh "
            f"and comet_ws/devel/setup.zsh: {error}",
            file=sys.stderr,
        )
        return 2

    selected = laps[args.lap - 1]
    source_origin = copy.deepcopy(selected[0][1].position)
    rospy.init_node("replay_lzf_trajectory", anonymous=True)
    publisher = rospy.Publisher(args.topic, PositionCommand, queue_size=20)
    offset = [0.0, 0.0, 0.0]

    if not args.no_align:
        odometry = rospy.wait_for_message(args.odom_topic, Odometry, timeout=10.0)
        first = selected[0][1].position
        offset = [
            odometry.pose.pose.position.x - first.x,
            odometry.pose.pose.position.y - first.y,
            odometry.pose.pose.position.z - first.z,
        ]

    wait_deadline = rospy.Time.now() + rospy.Duration(5.0)

    while publisher.get_num_connections() == 0 and rospy.Time.now() < wait_deadline:
        rospy.sleep(0.05)

    source_start = selected[0][0]
    replay_count = 0

    while not rospy.is_shutdown():
        replay_count += 1
        wall_start = rospy.Time.now()
        print(
            f"replaying lap {args.lap}/{len(laps)} iteration {replay_count}: "
            f"{len(selected)} samples, "
            f"{selected[-1][0] - source_start:.3f} s, offset={offset}",
            flush=True,
        )

        for source_time, original in selected:
            target_elapsed = (source_time - source_start) / args.speed

            while not rospy.is_shutdown():
                remaining = target_elapsed - (rospy.Time.now() - wall_start).to_sec()

                if remaining <= 0.0:
                    break

                rospy.sleep(min(remaining, 0.002))

            if rospy.is_shutdown():
                return 1

            # rosbag can materialize a dynamic Python class when the recorded
            # PositionCommand MD5 differs from the currently sourced message.
            # Publishing that object directly makes rospy treat it as the first
            # field (Header) and fail before serialization. Rebuild the command
            # with the current class while preserving every common field.
            message = PositionCommand()
            for field in message.__slots__:
                if hasattr(original, field):
                    setattr(message, field, copy.deepcopy(getattr(original, field)))
            message.header.stamp = rospy.Time.now()
            message.position.x = (
                source_origin.x
                + args.position_scale * (message.position.x - source_origin.x)
                + offset[0]
            )
            message.position.y = (
                source_origin.y
                + args.position_scale * (message.position.y - source_origin.y)
                + offset[1]
            )
            message.position.z = (
                source_origin.z
                + args.position_scale * (message.position.z - source_origin.z)
                + offset[2]
            )

            # A polynomial trajectory transformed as p'(t)=s*p(q*t) has its
            # n-th spatial derivative multiplied by s*q**n.  Scaling publish
            # timestamps alone would make the commanded position inconsistent
            # with velocity/acceleration and invalidate controller validation.
            derivative_fields = (
                ("velocity", 1),
                ("acceleration", 2),
                ("jerk", 3),
                ("snap", 4),
                ("crackle", 5),
            )
            for field, order in derivative_fields:
                vector = getattr(message, field, None)
                if vector is None:
                    continue
                scale = args.position_scale * args.speed ** order
                vector.x *= scale
                vector.y *= scale
                vector.z *= scale
            excitation_arguments = (
                target_elapsed,
                args.vertical_excitation_peak_acceleration,
                args.vertical_excitation_frequency,
            )
            message.position.z += vertical_excitation_derivative(
                excitation_arguments[0], 0, *excitation_arguments[1:]
            )
            for field, order in derivative_fields:
                vector = getattr(message, field, None)
                if vector is not None:
                    vector.z += vertical_excitation_derivative(
                        excitation_arguments[0], order, *excitation_arguments[1:]
                    )
            message.yaw_dot *= args.speed
            message.yaw_ddot *= args.speed * args.speed
            publisher.publish(message)

        if not args.repeat:
            break

    print("trajectory replay complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
