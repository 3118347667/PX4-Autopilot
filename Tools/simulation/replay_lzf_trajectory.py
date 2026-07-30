#!/usr/bin/env python3

"""Replay one recorded LZF PositionCommand lap with fresh ROS timestamps."""

import argparse
import copy
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
            "use_esc_k1_update": message.thrust_model_use_esc_k1_update,
            "apply_esc_k1_update": message.thrust_model_apply_esc_k1_update,
            "esc_k2": message.thrust_model_esc_k2,
            "k1_rls_forgetting_factor": message.thrust_model_k1_rls_forgetting_factor,
            "k1_rls_init_covariance": message.thrust_model_k1_rls_init_covariance,
            "k1_min": message.thrust_model_k1_min,
            "k1_max": message.thrust_model_k1_max,
            "k1_min_delta_rpm": message.thrust_model_k1_min_delta_rpm,
            "k1_response_window": message.thrust_model_k1_response_window,
            "k1_delay_align": message.thrust_model_k1_delay_align,
            "k1_require_response_same_sign": message.thrust_model_k1_require_response_same_sign,
            "k1_instant_min": message.thrust_model_k1_instant_min,
            "k1_instant_max": message.thrust_model_k1_instant_max,
            "k1_min_delta_acc": message.thrust_model_k1_min_delta_acc,
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
    wall_start = rospy.Time.now()
    print(
        f"replaying lap {args.lap}/{len(laps)}: {len(selected)} samples, "
        f"{selected[-1][0] - source_start:.3f} s, offset={offset}"
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

        message = copy.deepcopy(original)
        message.header.stamp = rospy.Time.now()
        message.position.x += offset[0]
        message.position.y += offset[1]
        message.position.z += offset[2]
        publisher.publish(message)

    print("trajectory replay complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
