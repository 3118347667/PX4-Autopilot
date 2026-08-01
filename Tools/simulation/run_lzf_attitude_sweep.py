#!/usr/bin/env python3

"""Run one isolated LZF SITL attitude chirp and save its PX4 ULog."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PX4_ROOT = Path("/home/user/PX4-Autopilot")
PX4_BUILD = PX4_ROOT / "build/px4_sitl_default"
COMET_ROOT = Path("/home/user/comet_ws")
sys.path.insert(0, str(COMET_ROOT / "script"))

from run_lzf_trajectory_trial import (  # noqa: E402
    ManagedProcess,
    newest_ulog_after,
    remove_stale_px4_socket,
    ros_args,
    terminate_processes,
    wait_for_port,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axis-duration", type=float, default=20.0)
    parser.add_argument("--settle-duration", type=float, default=2.0)
    parser.add_argument("--start-frequency", type=float, default=0.5)
    parser.add_argument("--end-frequency", type=float, default=5.0)
    parser.add_argument("--amplitude-deg", type=float, default=4.0)
    parser.add_argument("--px4-instance", type=int, default=7)
    parser.add_argument("--gazebo-master-port", type=int, default=11346)
    parser.add_argument("--ros-master-port", type=int, default=11321)
    parser.add_argument(
        "--px4-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="PX4 parameter override applied before arming (repeatable)",
    )
    return parser.parse_args()


def parse_px4_parameters(items):
    parameters = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --px4-param {item!r}; expected NAME=VALUE")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"invalid --px4-param {item!r}; expected NAME=VALUE")
        parameters.append((name, float(value)))
    return parameters


def quaternion_from_roll_pitch(roll, pitch):
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    sin_roll = math.sin(half_roll)
    cos_roll = math.cos(half_roll)
    sin_pitch = math.sin(half_pitch)
    cos_pitch = math.cos(half_pitch)
    return (
        sin_roll * cos_pitch,
        cos_roll * sin_pitch,
        -sin_roll * sin_pitch,
        cos_roll * cos_pitch,
    )


def chirp(time_s, duration_s, start_hz, end_hz):
    rate_hz_s = (end_hz - start_hz) / duration_s
    phase = 2.0 * math.pi * (start_hz * time_s + 0.5 * rate_hz_s * time_s * time_s)
    return math.sin(phase)


def main():
    args = parse_args()
    try:
        px4_parameters = parse_px4_parameters(args.px4_param)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"error: output directory is not empty: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ros_home = args.output_dir / "ros_home"
    ros_log_dir = args.output_dir / "ros_logs"
    ros_home.mkdir(parents=True, exist_ok=True)
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROS_HOME"] = str(ros_home)
    os.environ["ROS_LOG_DIR"] = str(ros_log_dir)
    os.environ["ROS_MASTER_URI"] = f"http://127.0.0.1:{args.ros_master_port}"
    os.environ["ROS_HOSTNAME"] = "127.0.0.1"
    os.environ.pop("ROS_IP", None)

    sdf_path = (
        PX4_ROOT
        / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/lzf/lzf.sdf"
    )
    for path in (PX4_BUILD / "bin/px4", sdf_path):
        if not path.is_file():
            print(f"error: missing {path}", file=sys.stderr)
            return 2

    processes = []
    start_wall = time.time()
    result = {
        "status": "running",
        "started_unix_s": start_wall,
        "axis_duration_s": args.axis_duration,
        "settle_duration_s": args.settle_duration,
        "frequency_hz": [args.start_frequency, args.end_frequency],
        "amplitude_deg": args.amplitude_deg,
        "px4_instance": args.px4_instance,
        "gazebo_master_port": args.gazebo_master_port,
        "ros_master_port": args.ros_master_port,
        "px4_parameters": dict(px4_parameters),
    }

    try:
        roscore = ManagedProcess(
            "roscore",
            ros_args("roscore", "-p", str(args.ros_master_port)),
            args.output_dir / "roscore.log",
            cwd=COMET_ROOT,
        )
        processes.append(roscore)
        wait_for_port("127.0.0.1", args.ros_master_port, 45.0)
        roscore.require_running()

        mavlink_port = 14580 + args.px4_instance
        mavros = ManagedProcess(
            "MAVROS",
            ros_args(
                "roslaunch",
                "mavros",
                "px4.launch",
                f"fcu_url:=udp://:{14540 + args.px4_instance}@127.0.0.1:{mavlink_port}",
                f"tgt_system:={1 + args.px4_instance}",
            ),
            args.output_dir / "mavros.log",
            cwd=COMET_ROOT,
        )
        processes.append(mavros)
        time.sleep(2.0)
        mavros.require_running()

        sitl_env = os.environ.copy()
        sitl_env.update({
            "HEADLESS": "1",
            "PX4_NO_FOLLOW_MODE": "1",
            "GAZEBO_MASTER_URI": f"http://127.0.0.1:{args.gazebo_master_port}",
            "PX4_SIM_MODEL": "gazebo-classic_lzf",
            "PX4_SIM_WORLD": "none",
        })
        for name, path in (
            ("GAZEBO_PLUGIN_PATH", PX4_BUILD / "build_gazebo-classic"),
            (
                "GAZEBO_MODEL_PATH",
                PX4_ROOT / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models",
            ),
            ("LD_LIBRARY_PATH", PX4_BUILD / "build_gazebo-classic"),
        ):
            previous = sitl_env.get(name, "")
            sitl_env[name] = f"{path}:{previous}" if previous else str(path)

        world = (
            PX4_ROOT
            / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/empty.world"
        )
        gazebo = ManagedProcess(
            "Gazebo",
            ["gzserver", str(world)],
            args.output_dir / "gazebo.log",
            cwd=PX4_ROOT,
            env=sitl_env,
        )
        processes.append(gazebo)
        wait_for_port("127.0.0.1", args.gazebo_master_port, 45.0)

        isolated_sdf = args.output_dir / f"lzf_instance_{args.px4_instance}.sdf"
        sdf_text = sdf_path.read_text(encoding="utf-8")
        for old, new in (
            ("<mavlink_tcp_port>4560</mavlink_tcp_port>",
             f"<mavlink_tcp_port>{4560 + args.px4_instance}</mavlink_tcp_port>"),
            ("<mavlink_udp_port>14560</mavlink_udp_port>",
             f"<mavlink_udp_port>{14560 + args.px4_instance}</mavlink_udp_port>"),
            ("<qgc_udp_port>14550</qgc_udp_port>",
             f"<qgc_udp_port>{14550 + args.px4_instance}</qgc_udp_port>"),
            ("<sdk_udp_port>14540</sdk_udp_port>",
             f"<sdk_udp_port>{14540 + args.px4_instance}</sdk_udp_port>"),
        ):
            if old not in sdf_text:
                raise RuntimeError(f"missing expected SDF port: {old}")
            sdf_text = sdf_text.replace(old, new, 1)
        isolated_sdf.write_text(sdf_text, encoding="utf-8")

        with (args.output_dir / "gazebo_spawn.log").open("w", encoding="utf-8") as log:
            spawn = subprocess.run(
                [
                    "gz",
                    "model",
                    "--verbose",
                    f"--spawn-file={isolated_sdf}",
                    "--model-name=lzf",
                    "-x", "1.01",
                    "-y", "0.98",
                    "-z", "0.83",
                ],
                cwd=str(PX4_ROOT),
                env=sitl_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=45.0,
            )
        if spawn.returncode != 0:
            raise RuntimeError("Gazebo model spawn failed")

        rootfs = args.output_dir / "sitl_rootfs"
        rootfs.mkdir(parents=True, exist_ok=True)
        remove_stale_px4_socket(args.px4_instance)
        sitl = ManagedProcess(
            "PX4 SITL",
            [
                str(PX4_BUILD / "bin/px4"),
                "-i", str(args.px4_instance),
                "-w", str(rootfs),
                str(PX4_BUILD / "etc"),
            ],
            args.output_dir / "sitl.log",
            cwd=PX4_ROOT,
            env=sitl_env,
            writable_stdin=True,
        )
        processes.append(sitl)
        sitl_commands = [
            f"mavlink stream -u {mavlink_port} -s LOCAL_POSITION_NED -r 100",
            f"mavlink stream -u {mavlink_port} -s ATTITUDE -r 100",
            f"mavlink stream -u {mavlink_port} -s ATTITUDE_TARGET -r 100",
            f"mavlink stream -u {mavlink_port} -s ESC_STATUS -r 250",
        ]
        sitl_commands.extend(
            f"param set {name} {value:.9g}" for name, value in px4_parameters
        )
        sitl.write_lines(sitl_commands)

        import rospy
        from mavros_msgs.msg import AttitudeTarget, State
        from mavros_msgs.srv import CommandBool, SetMode
        from nav_msgs.msg import Odometry

        rospy.init_node("run_lzf_attitude_sweep", anonymous=True, disable_signals=True)
        observed = {
            "connected": False,
            "armed": False,
            "mode": "",
            "z": 0.0,
            "vz": 0.0,
            "odom_received": False,
        }

        def state_callback(message):
            observed["connected"] = bool(message.connected)
            observed["armed"] = bool(message.armed)
            observed["mode"] = str(message.mode)

        def odom_callback(message):
            observed["z"] = float(message.pose.pose.position.z)
            observed["vz"] = float(message.twist.twist.linear.z)
            observed["odom_received"] = True

        rospy.Subscriber("/mavros/state", State, state_callback, queue_size=20)
        rospy.Subscriber(
            "/mavros/local_position/odom", Odometry, odom_callback, queue_size=20
        )
        publisher = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=20
        )
        rospy.wait_for_service("/mavros/set_mode", timeout=45.0)
        rospy.wait_for_service("/mavros/cmd/arming", timeout=45.0)
        set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline and not (
            observed["connected"] and observed["odom_received"]
        ):
            for process in processes:
                process.require_running()
            time.sleep(0.05)
        if not (observed["connected"] and observed["odom_received"]):
            raise RuntimeError("timed out waiting for MAVROS state and odometry")

        initial_z = observed["z"]
        target_z = initial_z + 1.0
        hover_thrust = 0.25
        rate = rospy.Rate(100.0)

        def publish(roll, pitch, desired_z):
            message = AttitudeTarget()
            message.header.stamp = rospy.Time.now()
            message.type_mask = (
                AttitudeTarget.IGNORE_ROLL_RATE
                | AttitudeTarget.IGNORE_PITCH_RATE
                | AttitudeTarget.IGNORE_YAW_RATE
            )
            quaternion = quaternion_from_roll_pitch(roll, pitch)
            message.orientation.x = quaternion[0]
            message.orientation.y = quaternion[1]
            message.orientation.z = quaternion[2]
            message.orientation.w = quaternion[3]
            altitude_error = desired_z - observed["z"]
            message.thrust = max(
                0.12,
                min(0.42, hover_thrust + 0.04 * altitude_error - 0.035 * observed["vz"]),
            )
            publisher.publish(message)

        prestream_start = time.monotonic()
        while time.monotonic() - prestream_start < 2.0:
            publish(0.0, 0.0, initial_z)
            rate.sleep()
        if not set_mode(custom_mode="OFFBOARD").mode_sent:
            raise RuntimeError("PX4 rejected OFFBOARD mode")
        if not arm(True).success:
            raise RuntimeError("PX4 rejected arm command")
        offboard_monotonic = time.monotonic()

        takeoff_start = time.monotonic()
        while time.monotonic() - takeoff_start < 12.0:
            publish(0.0, 0.0, target_z)
            if abs(observed["z"] - target_z) < 0.08 and abs(observed["vz"]) < 0.08:
                if time.monotonic() - takeoff_start > 4.0:
                    break
            rate.sleep()
        if abs(observed["z"] - target_z) > 0.25:
            raise RuntimeError("altitude controller did not reach sweep height")

        settle_start = time.monotonic()
        while time.monotonic() - settle_start < args.settle_duration:
            publish(0.0, 0.0, target_z)
            rate.sleep()

        amplitude = math.radians(args.amplitude_deg)
        sweep_start = time.monotonic()
        result["sweep_start_after_offboard_s"] = sweep_start - offboard_monotonic
        for axis in ("roll", "pitch"):
            axis_start = time.monotonic()
            while True:
                elapsed = time.monotonic() - axis_start
                if elapsed >= args.axis_duration:
                    break
                value = amplitude * chirp(
                    elapsed,
                    args.axis_duration,
                    args.start_frequency,
                    args.end_frequency,
                )
                publish(value if axis == "roll" else 0.0,
                        value if axis == "pitch" else 0.0,
                        target_z)
                rate.sleep()
            settle_start = time.monotonic()
            while time.monotonic() - settle_start < args.settle_duration:
                publish(0.0, 0.0, target_z)
                rate.sleep()
        result["sweep_end_after_offboard_s"] = time.monotonic() - offboard_monotonic

        set_mode(custom_mode="AUTO.LAND")
        land_deadline = time.monotonic() + 25.0
        while time.monotonic() < land_deadline and observed["armed"]:
            time.sleep(0.1)
        if observed["armed"]:
            arm(False)
            result["forced_disarm_after_land_timeout"] = True
        else:
            result["forced_disarm_after_land_timeout"] = False

        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
    finally:
        terminate_processes(processes)
        ulog = newest_ulog_after(start_wall, [args.output_dir / "sitl_rootfs/log"])
        if ulog is not None:
            destination = args.output_dir / "flight.ulg"
            if ulog != destination:
                shutil.copy2(ulog, destination)
            result["ulog"] = str(destination)
        result["finished_unix_s"] = time.time()
        result["elapsed_wall_s"] = result["finished_unix_s"] - start_wall
        (args.output_dir / "trial.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
