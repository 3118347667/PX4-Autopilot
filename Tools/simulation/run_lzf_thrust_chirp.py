#!/usr/bin/env python3

"""Fly one PX4 SITL LZF collective-thrust chirp experiment."""

import argparse
import fcntl
import math
import os
from pathlib import Path
import struct
import time
from typing import Optional

# PX4 uses the MAVLink 2 MANUAL_CONTROL extensions for AUX1.
os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil


LOCK_PATH = Path("/tmp/lzf_thrust_chirp.lock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to a running LZF SITL instance, take off, trigger the "
            "PX4 Z-thrust chirp through AUX1, and land."
        )
    )
    parser.add_argument("--connection", default="udpin:0.0.0.0:14550")
    parser.add_argument(
        "--bootstrap-address",
        help=(
            "Optional host:port to seed an udpin peer before the first PX4 "
            "heartbeat, for example 127.0.0.1:14600"
        ),
    )
    parser.add_argument("--voltage", type=float, default=0.0)
    parser.add_argument(
        "--thr-mdl-fac",
        type=float,
        help="Optional THR_MDL_FAC value in the PX4-supported range [0, 1]",
    )
    parser.add_argument("--takeoff-altitude", type=float, default=2.5)
    parser.add_argument(
        "--hover-mode",
        choices=("LOITER", "ALTCTL"),
        default="LOITER",
        help="PX4 mode used during the chirp",
    )
    parser.add_argument("--hover-seconds", type=float, default=10.0)
    parser.add_argument("--start-frequency", type=float, default=0.25)
    parser.add_argument("--end-frequency", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--magnitude", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=45.0)
    return parser.parse_args()


class Vehicle:
    def __init__(
        self, connection: str, timeout: float, bootstrap_address: Optional[str] = None
    ):
        self.master = mavutil.mavlink_connection(
            connection,
            autoreconnect=True,
            source_system=255,
            source_component=190,
        )
        self.timeout = timeout

        if bootstrap_address:
            host, port = bootstrap_address.rsplit(":", 1)
            peer = (host, int(port))
            if not hasattr(self.master, "clients"):
                raise RuntimeError(
                    "--bootstrap-address requires an udpin connection"
                )
            self.master.clients.add(peer)
            self.master.clients_last_alive[peer] = time.time()
            self.heartbeat()

        heartbeat = None
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            candidate = self.master.recv_match(
                type="HEARTBEAT",
                blocking=True,
                timeout=0.5,
            )

            if candidate is not None and (
                candidate.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID
            ):
                heartbeat = candidate
                break

        if heartbeat is None:
            raise RuntimeError("timed out waiting for PX4 heartbeat")
        self.master.target_system = heartbeat.get_srcSystem()
        self.master.target_component = heartbeat.get_srcComponent()
        print(
            f"heartbeat: system={self.master.target_system} "
            f"component={self.master.target_component}"
        )

    def heartbeat(self) -> None:
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )

    @staticmethod
    def encode_parameter(value: float, parameter_type: int) -> float:
        formats = {
            mavutil.mavlink.MAV_PARAM_TYPE_UINT8: ">xxxB",
            mavutil.mavlink.MAV_PARAM_TYPE_INT8: ">xxxb",
            mavutil.mavlink.MAV_PARAM_TYPE_UINT16: ">xxH",
            mavutil.mavlink.MAV_PARAM_TYPE_INT16: ">xxh",
            mavutil.mavlink.MAV_PARAM_TYPE_UINT32: ">I",
            mavutil.mavlink.MAV_PARAM_TYPE_INT32: ">i",
        }
        if parameter_type == mavutil.mavlink.MAV_PARAM_TYPE_REAL32:
            return float(value)
        if parameter_type not in formats:
            raise RuntimeError(f"unsupported MAVLink parameter type {parameter_type}")
        packed = struct.pack(formats[parameter_type], int(round(value)))
        return struct.unpack(">f", packed)[0]

    @staticmethod
    def decode_parameter(value: float, parameter_type: int) -> float:
        formats = {
            mavutil.mavlink.MAV_PARAM_TYPE_UINT8: ">xxxB",
            mavutil.mavlink.MAV_PARAM_TYPE_INT8: ">xxxb",
            mavutil.mavlink.MAV_PARAM_TYPE_UINT16: ">xxH",
            mavutil.mavlink.MAV_PARAM_TYPE_INT16: ">xxh",
            mavutil.mavlink.MAV_PARAM_TYPE_UINT32: ">I",
            mavutil.mavlink.MAV_PARAM_TYPE_INT32: ">i",
        }
        if parameter_type == mavutil.mavlink.MAV_PARAM_TYPE_REAL32:
            return float(value)
        if parameter_type not in formats:
            raise RuntimeError(f"unsupported MAVLink parameter type {parameter_type}")
        packed = struct.pack(">f", float(value))
        return float(struct.unpack(formats[parameter_type], packed)[0])

    def read_parameter(self, name: str):
        encoded = name.encode("ascii")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self.master.mav.param_request_read_send(
                self.master.target_system,
                self.master.target_component,
                encoded,
                -1,
            )
            wait_until = min(deadline, time.monotonic() + 1.0)
            while time.monotonic() < wait_until:
                message = self.master.recv_match(
                    type=["PARAM_VALUE", "STATUSTEXT"],
                    blocking=True,
                    timeout=0.2,
                )
                if message is None:
                    continue
                if message.get_type() == "STATUSTEXT":
                    print(f"PX4: {message.text}")
                    continue
                parameter_id = message.param_id
                if isinstance(parameter_id, bytes):
                    parameter_id = parameter_id.decode("ascii", errors="ignore")
                parameter_id = parameter_id.rstrip("\x00")
                if parameter_id == name:
                    return (
                        self.decode_parameter(
                            float(message.param_value),
                            int(message.param_type),
                        ),
                        int(message.param_type),
                    )
        raise RuntimeError(f"timed out reading parameter {name}")

    def set_parameter(self, name: str, value: float) -> None:
        _, parameter_type = self.read_parameter(name)
        encoded_value = self.encode_parameter(value, parameter_type)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self.master.mav.param_set_send(
                self.master.target_system,
                self.master.target_component,
                name.encode("ascii"),
                encoded_value,
                parameter_type,
            )
            wait_until = min(deadline, time.monotonic() + 1.0)
            while time.monotonic() < wait_until:
                message = self.master.recv_match(
                    type=["PARAM_VALUE", "STATUSTEXT"],
                    blocking=True,
                    timeout=0.2,
                )
                if message is None:
                    continue
                if message.get_type() == "STATUSTEXT":
                    print(f"PX4: {message.text}")
                    continue
                parameter_id = message.param_id
                if isinstance(parameter_id, bytes):
                    parameter_id = parameter_id.decode("ascii", errors="ignore")
                parameter_id = parameter_id.rstrip("\x00")
                if parameter_id != name:
                    continue
                stored = self.decode_parameter(
                    float(message.param_value),
                    int(message.param_type),
                )
                if not math.isclose(
                    stored,
                    float(value),
                    rel_tol=1e-4,
                    abs_tol=1e-4,
                ):
                    raise RuntimeError(
                        f"PX4 stored {name}={stored}, requested {value}"
                    )
                print(f"parameter: {name}={stored}")
                return
        raise RuntimeError(f"timed out setting parameter {name}")

    def command(
        self,
        command: int,
        parameters,
        accepted=(mavutil.mavlink.MAV_RESULT_ACCEPTED,),
    ) -> None:
        values = list(parameters) + [0.0] * (7 - len(parameters))
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                command,
                0,
                *values[:7],
            )
            message = self.master.recv_match(
                type="COMMAND_ACK",
                blocking=True,
                timeout=1.0,
            )
            if message is None or message.command != command:
                continue
            if message.result == mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED:
                self.print_status_text(1.0)
                time.sleep(0.5)
                continue
            if message.result not in accepted:
                self.print_status_text(1.0)
                raise RuntimeError(
                    f"command {command} rejected with result {message.result}"
                )
            return
        raise RuntimeError(f"timed out waiting for command {command}")

    def print_status_text(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            message = self.master.recv_match(
                type="STATUSTEXT",
                blocking=True,
                timeout=0.1,
            )
            if message is not None:
                print(f"PX4: {message.text}")

    def set_mode(self, mode: str, manual_throttle: Optional[int] = None) -> None:
        mapping = self.master.mode_mapping()
        if not mapping or mode not in mapping:
            raise RuntimeError(f"PX4 mode {mode} is unavailable")
        self.master.set_mode_px4(*mapping[mode])
        deadline = time.monotonic() + self.timeout
        next_send = 0.0

        while time.monotonic() < deadline:
            now = time.monotonic()

            if manual_throttle is not None and now >= next_send:
                self.send_manual_control(False, manual_throttle)
                next_send = now + 0.1

            self.heartbeat()
            heartbeat = self.master.recv_match(
                type="HEARTBEAT",
                blocking=True,
                timeout=0.5,
            )
            if heartbeat is not None and (
                heartbeat.get_srcSystem() == self.master.target_system
                and heartbeat.get_srcComponent() == self.master.target_component
                and mavutil.mode_string_v10(heartbeat) == mode
            ):
                print(f"mode: {mode}")
                return
        raise RuntimeError(f"timed out entering {mode}")

    def arm(self) -> None:
        self.command(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            [1.0],
        )
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            heartbeat = self.master.recv_match(
                type="HEARTBEAT",
                blocking=True,
                timeout=0.5,
            )
            if heartbeat is not None and (
                heartbeat.get_srcSystem() == self.master.target_system
                and heartbeat.get_srcComponent() == self.master.target_component
                and (
                heartbeat.base_mode
                & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
            ):
                print("armed")
                return
        raise RuntimeError("timed out waiting for arming")

    def wait_for_altitude(self, altitude: float) -> None:
        deadline = time.monotonic() + self.timeout
        next_send = 0.0

        while time.monotonic() < deadline:
            now = time.monotonic()

            if now >= next_send:
                self.send_manual_control(False, 0)
                next_send = now + 0.1

            self.heartbeat()
            message = self.master.recv_match(
                type=["LOCAL_POSITION_NED", "STATUSTEXT"],
                blocking=True,
                timeout=0.5,
            )
            if message is None:
                continue
            if message.get_type() == "STATUSTEXT":
                print(f"PX4: {message.text}")
                continue
            current_altitude = -float(message.z)
            if current_altitude >= 0.95 * altitude:
                print(f"takeoff altitude reached: {current_altitude:.2f} m")
                return
        raise RuntimeError("timed out waiting for takeoff altitude")

    def is_armed(self) -> bool:
        heartbeat = self.master.recv_match(
            type="HEARTBEAT",
            blocking=True,
            timeout=1.0,
        )
        return bool(
            heartbeat is not None
            and heartbeat.get_srcSystem() == self.master.target_system
            and heartbeat.get_srcComponent() == self.master.target_component
            and heartbeat.base_mode
            & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )

    def send_manual_control(self, aux1_high: bool, throttle: int = 500) -> None:
        self.master.mav.manual_control_send(
            self.master.target_system,
            0,
            0,
            throttle,
            0,
            0,
            0,
            1 << 2,
            0,
            0,
            1000 if aux1_high else -1000,
        )

    def hold(
        self,
        duration: float,
        aux1_high: bool = False,
        throttle: int = 500,
    ) -> None:
        deadline = time.monotonic() + duration
        next_send = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                self.heartbeat()
                self.send_manual_control(aux1_high, throttle)
                next_send = now + 0.1
            message = self.master.recv_match(
                type="STATUSTEXT",
                blocking=True,
                timeout=0.05,
            )
            if message is not None:
                print(f"PX4: {message.text}")

    def land(self) -> None:
        self.set_mode("LAND")
        deadline = time.monotonic() + 2.0 * self.timeout
        while time.monotonic() < deadline:
            self.heartbeat()
            heartbeat = self.master.recv_match(
                type="HEARTBEAT",
                blocking=True,
                timeout=0.5,
            )
            if heartbeat is not None and (
                heartbeat.get_srcSystem() == self.master.target_system
                and heartbeat.get_srcComponent() == self.master.target_component
                and not (
                    heartbeat.base_mode
                    & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
            ):
                print("landed and disarmed")
                return
        raise RuntimeError("timed out waiting for landing")


def main() -> int:
    args = parse_args()
    if args.thr_mdl_fac is not None and not 0.0 <= args.thr_mdl_fac <= 1.0:
        raise ValueError("--thr-mdl-fac must be between 0 and 1")
    lock_file = LOCK_PATH.open("w")

    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"another LZF thrust chirp is already running ({LOCK_PATH})"
        ) from error

    vehicle = Vehicle(args.connection, args.timeout, args.bootstrap_address)
    parameters = [
        ("MPC_Z_CHIRP_EN", 1),
        ("MPC_Z_CHIRP_F0", args.start_frequency),
        ("MPC_Z_CHIRP_F1", args.end_frequency),
        ("MPC_Z_CHIRP_T", args.duration),
        ("MPC_Z_CHIRP_MAG", args.magnitude),
        ("MIS_TAKEOFF_ALT", args.takeoff_altitude),
        ("COM_RC_IN_MODE", 1),
        ("RC_CHAN_CNT", 8),
        ("RC_MAP_ROLL", 1),
        ("RC_MAP_PITCH", 2),
        ("RC_MAP_THROTTLE", 3),
        ("RC_MAP_YAW", 4),
        ("RC_MAP_AUX1", 8),
        ("SIM_BAT_V_OVR", args.voltage),
    ]
    if args.thr_mdl_fac is not None:
        parameters.append(("THR_MDL_FAC", args.thr_mdl_fac))

    for name, value in parameters:
        vehicle.set_parameter(name, value)

    armed = False
    try:
        print("establishing centered manual-control input")
        vehicle.hold(3.0, aux1_high=False, throttle=0)
        vehicle.set_mode("POSCTL", manual_throttle=0)
        vehicle.arm()
        armed = True
        vehicle.set_mode("TAKEOFF")
        vehicle.wait_for_altitude(args.takeoff_altitude)
        vehicle.hold(0.5, aux1_high=False)
        vehicle.set_mode(args.hover_mode, manual_throttle=500)
        print(f"settling for {args.hover_seconds:.1f} s")
        vehicle.hold(args.hover_seconds, aux1_high=False)

        print("triggering AUX1 rising edge")
        vehicle.hold(0.5, aux1_high=True)
        vehicle.hold(0.5, aux1_high=False)
        print(f"running {args.duration:.1f} s chirp")
        vehicle.hold(args.duration + 2.0, aux1_high=False)
        vehicle.land()
        armed = False
    finally:
        if armed and vehicle.is_armed():
            print("experiment aborted while armed; landing")
            vehicle.land()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
