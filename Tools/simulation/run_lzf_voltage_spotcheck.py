#!/usr/bin/env python3

"""Run isolated, fixed-voltage LZF collective-thrust spot checks.

The script deliberately does not attach to the usual instance-0 SITL.  For
each voltage it starts a fresh Gazebo Classic server and a fresh PX4 instance,
using a per-trial rootfs.  Consequently the parameters written by
``run_lzf_thrust_chirp.py`` cannot leak into either the next voltage or the
shared ``build/px4_sitl_default/rootfs`` directory.

The PX4 and Gazebo binaries must already have been built, for example with::

    make px4_sitl_default gazebo-classic_lzf

Existing output directories are never reused or overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


PX4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = PX4_ROOT / "build/px4_sitl_default"
DEFAULT_SDF = (
    PX4_ROOT
    / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/lzf/lzf.sdf"
)
DEFAULT_WORLD = (
    PX4_ROOT
    / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/empty.world"
)
RUN_CHIRP = PX4_ROOT / "Tools/simulation/run_lzf_thrust_chirp.py"
IDENTIFY_CHIRP = PX4_ROOT / "Tools/simulation/identify_lzf_thrust_chirp.py"
VALIDATE_CHAIN = PX4_ROOT / "Tools/simulation/validate_lzf_motor_prop_chain.py"
IDENTIFY_MAP = PX4_ROOT / "Tools/simulation/identify_lzf_static_gain_map.py"
WRAPPER_LOCK = Path("/tmp/lzf_voltage_spotcheck.lock")
CHIRP_LOCK = Path("/tmp/lzf_thrust_chirp.lock")


class SpotCheckError(RuntimeError):
    """A failure that should stop the voltage sequence without overwriting data."""


class ManagedProcess:
    """A child process whose process group and log belong only to this run."""

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        log_path: Path,
        *,
        cwd: Path,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.command = [str(item) for item in command]
        self.log_path = log_path
        self._log = log_path.open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        except Exception:
            self._log.close()
            raise

    def require_running(self) -> None:
        status = self.process.poll()
        if status is not None:
            self._log.flush()
            raise SpotCheckError(
                "{} exited with status {}; see {}".format(
                    self.name, status, self.log_path
                )
            )

    def stop(self, interrupt_timeout: float = 8.0) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGINT)
                self.process.wait(timeout=interrupt_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=3.0)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        self._log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for all raw logs, metadata, fits, and plots",
    )
    parser.add_argument(
        "--voltages",
        type=float,
        nargs="+",
        default=(18.0, 21.5, 23.2, 25.0),
        help="fixed pack voltages, run sequentially (default: 18 21.5 23.2 25)",
    )
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument(
        "--px4-instance",
        type=int,
        default=8,
        help=(
            "isolated PX4 instance in [1, 9] (default: 8); instance 0 is "
            "intentionally forbidden"
        ),
    )
    parser.add_argument(
        "--gazebo-master-port",
        type=int,
        default=11356,
        help="dedicated Gazebo master TCP port (default: 11356)",
    )
    parser.add_argument(
        "--gazebo-seed",
        type=int,
        default=7,
        help="Gazebo random seed, reset identically for every voltage",
    )
    parser.add_argument("--takeoff-altitude", type=float, default=2.5)
    parser.add_argument(
        "--hover-mode", choices=("LOITER", "ALTCTL"), default="LOITER"
    )
    parser.add_argument("--hover-seconds", type=float, default=8.0)
    parser.add_argument("--start-frequency", type=float, default=0.25)
    parser.add_argument("--end-frequency", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--magnitude", type=float, default=0.02)
    parser.add_argument(
        "--thr-mdl-fac",
        type=float,
        default=0.0,
        help="PX4 thrust linearization factor, kept identical at every voltage",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=60.0,
        help="timeout passed to run_lzf_thrust_chirp.py",
    )
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument(
        "--trial-timeout",
        type=float,
        default=360.0,
        help="overall timeout for one flight command",
    )
    parser.add_argument("--ulog-timeout", type=float, default=20.0)
    parser.add_argument("--analysis-timeout", type=float, default=600.0)
    parser.add_argument(
        "--mass",
        type=float,
        default=1.326,
        help="vehicle mass passed to the combined static-gain map",
    )
    parser.add_argument(
        "--rotor-count",
        type=int,
        default=4,
        help="rotor count passed to the combined static-gain map",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def capture(command: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_metadata(directory: Path) -> Dict[str, object]:
    return {
        "directory": str(directory),
        "commit": capture(("git", "rev-parse", "HEAD"), directory),
        "describe": capture(
            ("git", "describe", "--always", "--dirty", "--tags"), directory
        ),
        "status_porcelain": capture(
            ("git", "status", "--short", "--untracked-files=all"), directory
        ),
    }


def voltage_label(voltage: float) -> str:
    text = format(voltage, ".12g").replace("-", "m").replace(".", "p")
    return "v" + text


def prepend_path(environment: Dict[str, str], name: str, path: Path) -> None:
    previous = environment.get(name, "")
    environment[name] = str(path) + ((":" + previous) if previous else "")


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def model_voltage_limits(sdf: Path) -> Tuple[float, float]:
    root = ET.parse(str(sdf)).getroot()
    model = root.find("model")
    if model is None:
        raise SpotCheckError("LZF SDF has no <model>: {}".format(sdf))
    for plugin in model.findall("plugin"):
        minimum = plugin.findtext("motorSpeedVoltageMin")
        maximum = plugin.findtext("motorSpeedVoltageMax")
        if minimum is not None and maximum is not None:
            return float(minimum), float(maximum)
    raise SpotCheckError("LZF SDF has no motor voltage limits: {}".format(sdf))


def model_snapshot(sdf: Path) -> Dict[str, object]:
    root = ET.parse(str(sdf)).getroot()
    model = root.find("model")
    if model is None:
        return {}
    motors = [
        plugin
        for plugin in model.findall("plugin")
        if plugin.get("filename") == "libgazebo_motor_model.so"
    ]

    def values(name: str) -> List[str]:
        return [
            text
            for text in (plugin.findtext(name) for plugin in motors)
            if text is not None
        ]

    return {
        "motor_plugin_count": len(motors),
        "rotor_inertia_compensation_enabled": values(
            "rotorInertiaCompensationEnabled"
        ),
        "rotor_axial_inertia_kg_m2": values("rotorAxialInertia"),
        "rotor_velocity_slowdown_sim": values("rotorVelocitySlowdownSim"),
        "time_constant_up_s": values("timeConstantUp"),
        "time_constant_down_s": values("timeConstantDown"),
    }


def source_files(args: argparse.Namespace) -> List[Path]:
    build = args.build_dir
    gazebo_build = build / "build_gazebo-classic"
    candidates = [
        Path(__file__).resolve(),
        args.sdf,
        args.sdf.with_suffix(".sdf.jinja"),
        RUN_CHIRP,
        IDENTIFY_CHIRP,
        VALIDATE_CHAIN,
        IDENTIFY_MAP,
        PX4_ROOT
        / "ROMFS/px4fmu_common/init.d-posix/airframes/10020_gazebo-classic_lzf",
        build / "etc/init.d-posix/airframes/10020_gazebo-classic_lzf",
        build / "bin/px4",
        gazebo_build / "libgazebo_motor_model.so",
        gazebo_build / "libgazebo_mavlink_interface.so",
    ]
    return [path.resolve() for path in candidates]


def hash_sources(paths: Iterable[Path]) -> Dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def assert_sources_unchanged(expected: Dict[str, str]) -> None:
    changed = []
    for name, expected_hash in expected.items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            changed.append(name)
    if changed:
        raise SpotCheckError(
            "source/build inputs changed during the sequence: {}".format(
                ", ".join(changed)
            )
        )


def trial_ports(instance: int, gazebo_master_port: int) -> Dict[str, int]:
    return {
        "gazebo_master_tcp": gazebo_master_port,
        "simulator_tcp": 4560 + instance,
        "simulator_udp": 14560 + instance,
        "qgc_remote_udp": 14550 + instance,
        "offboard_remote_udp": 14540 + instance,
        "offboard_local_udp": 14580 + instance,
        "gcs_local_udp": 18570 + instance,
        "payload_local_udp": 14280 + instance,
        "gimbal_local_udp": 13030 + instance,
    }


def require_tcp_port_free(port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # The children bind wildcard addresses, so probe the same scope rather
        # than only loopback (which could miss a listener on another interface).
        probe.bind(("0.0.0.0", port))
    except OSError as error:
        raise SpotCheckError("TCP port {} is already in use".format(port)) from error
    finally:
        probe.close()


def require_udp_port_free(port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("0.0.0.0", port))
    except OSError as error:
        raise SpotCheckError("UDP port {} is already in use".format(port)) from error
    finally:
        probe.close()


def require_ports_free(ports: Dict[str, int]) -> None:
    require_tcp_port_free(ports["gazebo_master_tcp"])
    require_tcp_port_free(ports["simulator_tcp"])
    for name in (
        "simulator_udp",
        "offboard_remote_udp",
        "offboard_local_udp",
        "gcs_local_udp",
        "payload_local_udp",
        "gimbal_local_udp",
    ):
        require_udp_port_free(ports[name])


def remove_stale_px4_socket(instance: int) -> None:
    socket_path = Path("/tmp/px4-sock-{}".format(instance))
    if not socket_path.exists():
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.2)
        probe.connect(str(socket_path))
    except OSError:
        socket_path.unlink(missing_ok=True)
    else:
        raise SpotCheckError("PX4 instance {} is already running".format(instance))
    finally:
        probe.close()


def require_chirp_idle() -> None:
    lock_file = CHIRP_LOCK.open("a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    except BlockingIOError as error:
        raise SpotCheckError(
            "another thrust chirp is already running ({})".format(CHIRP_LOCK)
        ) from error
    finally:
        lock_file.close()


def wait_for_tcp(
    host: str,
    port: int,
    timeout: float,
    processes: Sequence[ManagedProcess],
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            process.require_running()
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise SpotCheckError("timed out waiting for TCP {}:{}".format(host, port))


def make_isolated_sdf(source: Path, destination: Path, ports: Dict[str, int]) -> None:
    text = source.read_text(encoding="utf-8")
    replacements = (
        (
            "<mavlink_tcp_port>4560</mavlink_tcp_port>",
            "<mavlink_tcp_port>{}</mavlink_tcp_port>".format(
                ports["simulator_tcp"]
            ),
        ),
        (
            "<mavlink_udp_port>14560</mavlink_udp_port>",
            "<mavlink_udp_port>{}</mavlink_udp_port>".format(
                ports["simulator_udp"]
            ),
        ),
        (
            "<qgc_udp_port>14550</qgc_udp_port>",
            "<qgc_udp_port>{}</qgc_udp_port>".format(ports["qgc_remote_udp"]),
        ),
        (
            "<sdk_udp_port>14540</sdk_udp_port>",
            "<sdk_udp_port>{}</sdk_udp_port>".format(
                ports["offboard_remote_udp"]
            ),
        ),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise SpotCheckError("expected exactly one SDF tag {}".format(old))
        text = text.replace(old, new, 1)
    destination.write_text(text, encoding="utf-8")


def run_logged_command(
    name: str,
    command: Sequence[str],
    log_path: Path,
    *,
    cwd: Path,
    timeout: float,
    dependencies: Sequence[ManagedProcess] = (),
    env: Optional[Dict[str, str]] = None,
) -> None:
    print("{}: {}".format(name, shlex.join([str(item) for item in command])))
    process = ManagedProcess(name, command, log_path, cwd=cwd, env=env)
    deadline = time.monotonic() + timeout
    try:
        while process.process.poll() is None:
            for dependency in dependencies:
                dependency.require_running()
            if time.monotonic() >= deadline:
                raise SpotCheckError(
                    "{} exceeded {:.1f} s; see {}".format(name, timeout, log_path)
                )
            time.sleep(0.25)
        status = process.process.returncode
        if status != 0:
            raise SpotCheckError(
                "{} exited with status {}; see {}".format(name, status, log_path)
            )
    finally:
        process.stop()


def stable_unique_ulog(log_root: Path, timeout: float) -> Path:
    deadline = time.monotonic() + timeout
    previous: Optional[Tuple[Path, int]] = None
    stable_observations = 0
    while time.monotonic() < deadline:
        candidates = sorted(log_root.glob("**/*.ulg")) if log_root.exists() else []
        if len(candidates) > 1:
            raise SpotCheckError(
                "expected exactly one ULog in {}, found {}: {}".format(
                    log_root,
                    len(candidates),
                    ", ".join(str(path) for path in candidates),
                )
            )
        if len(candidates) == 1:
            candidate = candidates[0]
            observation = (candidate, candidate.stat().st_size)
            if observation[1] > 0 and observation == previous:
                stable_observations += 1
            else:
                stable_observations = 0
            previous = observation
            if stable_observations >= 2:
                return candidate
        time.sleep(0.5)
    raise SpotCheckError("no stable, unique ULog appeared in {}".format(log_root))


def copy_recovered_ulog(log_root: Path, destination: Path) -> Optional[Path]:
    candidates = sorted(log_root.glob("**/*.ulg")) if log_root.exists() else []
    if len(candidates) != 1 or candidates[0].stat().st_size == 0:
        return None
    if not destination.exists():
        shutil.copy2(str(candidates[0]), str(destination))
    return candidates[0]


def validate_expected_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SpotCheckError(
            "analysis did not create non-empty files: {}".format(", ".join(missing))
        )


def chirp_command(
    args: argparse.Namespace,
    voltage: float,
    ports: Dict[str, int],
) -> List[str]:
    return [
        sys.executable,
        str(RUN_CHIRP),
        "--connection",
        "udpin:0.0.0.0:{}".format(ports["offboard_remote_udp"]),
        "--bootstrap-address",
        "127.0.0.1:{}".format(ports["offboard_local_udp"]),
        "--voltage",
        format(voltage, ".12g"),
        "--thr-mdl-fac",
        format(args.thr_mdl_fac, ".12g"),
        "--takeoff-altitude",
        format(args.takeoff_altitude, ".12g"),
        "--hover-mode",
        args.hover_mode,
        "--hover-seconds",
        format(args.hover_seconds, ".12g"),
        "--start-frequency",
        format(args.start_frequency, ".12g"),
        "--end-frequency",
        format(args.end_frequency, ".12g"),
        "--duration",
        format(args.duration, ".12g"),
        "--magnitude",
        format(args.magnitude, ".12g"),
        "--timeout",
        format(args.command_timeout, ".12g"),
    ]


def run_trial(
    args: argparse.Namespace,
    voltage: float,
    expected_sources: Dict[str, str],
    ports: Dict[str, int],
) -> Dict[str, object]:
    label = voltage_label(voltage)
    trial_dir = args.output_dir / label
    trial_dir.mkdir(parents=False, exist_ok=False)
    rootfs = trial_dir / "sitl_rootfs"
    rootfs.mkdir()
    log_root = rootfs / "log"
    isolated_sdf = trial_dir / "lzf_isolated.sdf"
    flight = trial_dir / "flight.ulg"
    command = chirp_command(args, voltage, ports)
    metadata: Dict[str, object] = {
        "status": "running",
        "label": label,
        "voltage_v": voltage,
        "started_utc": utc_now(),
        "started_unix_s": time.time(),
        "rootfs": str(rootfs),
        "shared_rootfs_used": False,
        "ports": ports,
        "chirp_command": command,
    }
    write_json(trial_dir / "trial.json", metadata)
    processes: List[ManagedProcess] = []
    source_ulog: Optional[Path] = None

    try:
        assert_sources_unchanged(expected_sources)
        require_ports_free(ports)
        require_chirp_idle()
        remove_stale_px4_socket(args.px4_instance)
        make_isolated_sdf(args.sdf, isolated_sdf, ports)
        metadata["isolated_sdf_sha256"] = sha256_file(isolated_sdf)

        environment = os.environ.copy()
        environment.update(
            {
                "HEADLESS": "1",
                "PX4_NO_FOLLOW_MODE": "1",
                "PX4_SIM_SPEED_FACTOR": "1",
                "GAZEBO_MASTER_URI": "http://127.0.0.1:{}".format(
                    args.gazebo_master_port
                ),
                "PX4_SIM_MODEL": "gazebo-classic_lzf",
                "PX4_SIM_WORLD": "none",
            }
        )
        environment.pop("PX4_SIM_HOSTNAME", None)
        environment.pop("PX4_SIM_HOST_ADDR", None)
        environment.pop("PX4_SYS_AUTOSTART", None)
        # rcS applies every inherited PX4_PARAM_* variable after loading the
        # airframe.  Remove them so a caller's shell cannot silently change one
        # voltage relative to another or invalidate the current-model baseline.
        for name in list(environment):
            if name.startswith("PX4_PARAM_"):
                environment.pop(name)
        prepend_path(
            environment,
            "GAZEBO_PLUGIN_PATH",
            args.build_dir / "build_gazebo-classic",
        )
        prepend_path(
            environment,
            "GAZEBO_MODEL_PATH",
            PX4_ROOT
            / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models",
        )
        prepend_path(
            environment,
            "LD_LIBRARY_PATH",
            args.build_dir / "build_gazebo-classic",
        )

        gazebo = ManagedProcess(
            "Gazebo Classic server",
            (
                "gzserver",
                "--verbose",
                "--seed",
                str(args.gazebo_seed),
                str(args.world),
            ),
            trial_dir / "gazebo.log",
            cwd=PX4_ROOT,
            env=environment,
        )
        processes.append(gazebo)
        wait_for_tcp(
            "127.0.0.1",
            args.gazebo_master_port,
            args.startup_timeout,
            processes,
        )

        spawn_command = [
            "gz",
            "model",
            "--verbose",
            "--spawn-file={}".format(isolated_sdf),
            "--model-name=lzf_spot_{}".format(label),
            "-x",
            "1.01",
            "-y",
            "0.98",
            "-z",
            "0.83",
        ]
        run_logged_command(
            "Gazebo model spawn",
            spawn_command,
            trial_dir / "gazebo_spawn.log",
            cwd=PX4_ROOT,
            env=environment,
            timeout=args.startup_timeout,
            dependencies=processes,
        )

        px4 = ManagedProcess(
            "PX4 SITL",
            (
                str(args.build_dir / "bin/px4"),
                "-i",
                str(args.px4_instance),
                "-w",
                str(rootfs),
                str(args.build_dir / "etc"),
            ),
            trial_dir / "sitl.log",
            cwd=PX4_ROOT,
            env=environment,
        )
        processes.append(px4)

        run_logged_command(
            "{} fixed-voltage thrust chirp".format(label),
            command,
            trial_dir / "chirp.log",
            cwd=PX4_ROOT,
            timeout=args.trial_timeout,
            dependencies=processes,
            env=environment,
        )
        source_ulog = stable_unique_ulog(log_root, args.ulog_timeout)
        shutil.copy2(str(source_ulog), str(flight))
        if sha256_file(source_ulog) != sha256_file(flight):
            raise SpotCheckError("copied ULog hash mismatch for {}".format(label))
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        for process in reversed(processes):
            process.stop()
        if source_ulog is None:
            source_ulog = copy_recovered_ulog(log_root, flight)
        metadata["finished_utc"] = utc_now()
        metadata["finished_unix_s"] = time.time()
        metadata["elapsed_wall_s"] = (
            metadata["finished_unix_s"] - metadata["started_unix_s"]
        )
        if source_ulog is not None and flight.is_file():
            metadata["ulog"] = {
                "source": str(source_ulog),
                "copy": str(flight),
                "size_bytes": flight.stat().st_size,
                "sha256": sha256_file(flight),
            }
        for parameter_name in ("parameters.bson", "parameters_backup.bson"):
            parameter_path = rootfs / parameter_name
            if parameter_path.is_file():
                metadata.setdefault("isolated_parameter_files", {})[
                    parameter_name
                ] = {
                    "path": str(parameter_path),
                    "sha256": sha256_file(parameter_path),
                }
        write_json(trial_dir / "trial.json", metadata)

    metadata["status"] = "analyzing"
    write_json(trial_dir / "trial.json", metadata)
    try:
        assert_sources_unchanged(expected_sources)
        time_domain = trial_dir / "time_domain"
        chain = trial_dir / "motor_prop_chain"
        time_domain.mkdir()
        chain.mkdir()
        fit_command = [
            sys.executable,
            str(IDENTIFY_CHIRP),
            str(flight),
            "--output-dir",
            str(time_domain),
            "--label",
            label,
        ]
        run_logged_command(
            "{} time-domain identification".format(label),
            fit_command,
            trial_dir / "identify.log",
            cwd=PX4_ROOT,
            timeout=args.analysis_timeout,
        )
        fit_files = [
            time_domain / "{}_thrust_chirp_fit.{}".format(label, suffix)
            for suffix in ("json", "md", "png")
        ]
        validate_expected_files(fit_files)

        chain_label = "{}_motor_prop_chain".format(label)
        chain_command = [
            sys.executable,
            str(VALIDATE_CHAIN),
            str(flight),
            "--sdf",
            str(args.sdf),
            "--output-dir",
            str(chain),
            "--label",
            chain_label,
        ]
        run_logged_command(
            "{} motor/propeller validation".format(label),
            chain_command,
            trial_dir / "motor_prop_chain.log",
            cwd=PX4_ROOT,
            timeout=args.analysis_timeout,
        )
        chain_files = [
            chain / "{}.{}".format(chain_label, suffix)
            for suffix in ("json", "md", "png")
        ]
        validate_expected_files(chain_files)

        fit_result = json.loads(fit_files[0].read_text(encoding="utf-8"))
        selected = fit_result.get("selected_model", {})
        metadata["status"] = "completed"
        metadata["analysis"] = {
            "time_domain": [str(path) for path in fit_files],
            "motor_prop_chain": [str(path) for path in chain_files],
            "selected_model": selected,
        }
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        metadata["error"] = "keyboard interrupt during analysis"
        raise
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = "analysis failed: {}".format(error)
        raise
    finally:
        metadata["finished_utc"] = utc_now()
        metadata["finished_unix_s"] = time.time()
        metadata["elapsed_wall_s"] = (
            metadata["finished_unix_s"] - metadata["started_unix_s"]
        )
        write_json(trial_dir / "trial.json", metadata)
    return {
        "label": label,
        "voltage_v": voltage,
        "trial": str(trial_dir / "trial.json"),
        "ulog": str(flight),
        "ulog_sha256": sha256_file(flight),
        "fit_json": str(fit_files[0]),
        "selected_model": selected,
        "chain_json": str(chain_files[0]),
    }


def run_combined_map(
    args: argparse.Namespace,
    results: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    output = args.output_dir / "combined_static_gain_map"
    output.mkdir()
    command = [
        sys.executable,
        str(IDENTIFY_MAP),
        "--sdf",
        str(args.sdf),
        "--output-dir",
        str(output),
        "--voltages",
        *[format(float(result["voltage_v"]), ".12g") for result in results],
        "--mass",
        format(args.mass, ".12g"),
        "--rotor-count",
        str(args.rotor_count),
    ]
    for result in results:
        command.extend(("--hover-fit", str(result["fit_json"])))
    run_logged_command(
        "combined static-gain map",
        command,
        args.output_dir / "combined_static_gain_map.log",
        cwd=PX4_ROOT,
        timeout=args.analysis_timeout,
    )
    files = [
        output / "lzf_static_gain_grid.csv",
        output / "lzf_dynamic_gain_checks.csv",
        output / "lzf_static_gain_map.json",
        output / "lzf_static_gain_map.md",
        output / "lzf_static_gain_map.png",
    ]
    validate_expected_files(files)
    return {"command": command, "artifacts": [str(path) for path in files]}


def validate_args(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.build_dir = args.build_dir.expanduser().resolve()
    args.sdf = args.sdf.expanduser().resolve()
    args.world = args.world.expanduser().resolve()
    if args.output_dir.exists():
        raise SpotCheckError(
            "refusing to overwrite existing output path: {}".format(args.output_dir)
        )
    shared_rootfs = (args.build_dir / "rootfs").resolve()
    if path_is_within(args.output_dir, shared_rootfs):
        raise SpotCheckError(
            "output must not be inside the shared SITL rootfs: {}".format(
                shared_rootfs
            )
        )
    if not 1 <= args.px4_instance <= 9:
        raise SpotCheckError("--px4-instance must be in [1, 9]")
    if not 1024 <= args.gazebo_master_port <= 65535:
        raise SpotCheckError("--gazebo-master-port must be in [1024, 65535]")
    if args.gazebo_seed < 0:
        raise SpotCheckError("--gazebo-seed must be non-negative")
    if not args.voltages or any(
        not math.isfinite(value) for value in args.voltages
    ):
        raise SpotCheckError("--voltages must contain finite values")
    labels = [voltage_label(value) for value in args.voltages]
    if len(set(labels)) != len(labels):
        raise SpotCheckError("--voltages contains duplicate output labels")
    if not 0.0 <= args.thr_mdl_fac <= 1.0:
        raise SpotCheckError("--thr-mdl-fac must be in [0, 1]")
    if args.takeoff_altitude <= 0.0 or args.hover_seconds < 0.0:
        raise SpotCheckError("takeoff altitude must be positive and hover non-negative")
    if not 0.0 < args.start_frequency < args.end_frequency:
        raise SpotCheckError("chirp frequencies must satisfy 0 < start < end")
    if args.duration <= 0.0 or not 0.0 < args.magnitude <= 0.1:
        raise SpotCheckError("duration must be positive and magnitude in (0, 0.1]")
    if min(
        args.command_timeout,
        args.startup_timeout,
        args.trial_timeout,
        args.ulog_timeout,
        args.analysis_timeout,
    ) <= 0.0:
        raise SpotCheckError("all timeouts must be positive")
    if args.mass <= 0.0 or args.rotor_count <= 0:
        raise SpotCheckError("mass and rotor count must be positive")

    required = source_files(args) + [args.world]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SpotCheckError("missing required files: {}".format(", ".join(missing)))
    for executable in ("gzserver", "gz"):
        if shutil.which(executable) is None:
            raise SpotCheckError("required executable is unavailable: {}".format(executable))
    voltage_min, voltage_max = model_voltage_limits(args.sdf)
    outside = [
        value
        for value in args.voltages
        if value < voltage_min or value > voltage_max
    ]
    if outside:
        raise SpotCheckError(
            "voltages outside model range [{}, {}] V: {}".format(
                voltage_min, voltage_max, outside
            )
        )
    require_ports_free(trial_ports(args.px4_instance, args.gazebo_master_port))
    remove_stale_px4_socket(args.px4_instance)
    require_chirp_idle()


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except (OSError, ValueError, ET.ParseError, SpotCheckError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2

    wrapper_lock = WRAPPER_LOCK.open("a+")
    try:
        fcntl.flock(wrapper_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            "error: another voltage spot check is running ({})".format(
                WRAPPER_LOCK
            ),
            file=sys.stderr,
        )
        wrapper_lock.close()
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = source_files(args)
    expected_sources = hash_sources(paths)
    ports = trial_ports(args.px4_instance, args.gazebo_master_port)
    gazebo_submodule = (
        PX4_ROOT / "Tools/simulation/gazebo-classic/sitl_gazebo-classic"
    )
    manifest: Dict[str, object] = {
        "status": "running",
        "started_utc": utc_now(),
        "started_unix_s": time.time(),
        "invocation": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "protocol": {
            "voltages_v": args.voltages,
            "thr_mdl_fac": args.thr_mdl_fac,
            "takeoff_altitude_m": args.takeoff_altitude,
            "hover_mode": args.hover_mode,
            "hover_seconds": args.hover_seconds,
            "chirp_start_hz": args.start_frequency,
            "chirp_end_hz": args.end_frequency,
            "chirp_duration_s": args.duration,
            "chirp_magnitude": args.magnitude,
        },
        "isolation": {
            "px4_instance": args.px4_instance,
            "mav_sys_id": args.px4_instance + 1,
            "gazebo_master_port": args.gazebo_master_port,
            "gazebo_seed": args.gazebo_seed,
            "ports": ports,
            "fresh_rootfs_per_voltage": True,
            "shared_build_rootfs_used": False,
            "global_process_kill_used": False,
        },
        "model": model_snapshot(args.sdf),
        "source_sha256": expected_sources,
        "git": {
            "px4": git_metadata(PX4_ROOT),
            "gazebo_classic": git_metadata(gazebo_submodule),
        },
        "results": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)

    exit_code = 0
    try:
        for voltage in args.voltages:
            print("\n=== LZF fixed-voltage spot check: {} V ===".format(voltage))
            result = run_trial(args, voltage, expected_sources, ports)
            manifest["results"].append(result)
            write_json(args.output_dir / "manifest.json", manifest)
        assert_sources_unchanged(expected_sources)
        manifest["combined_static_gain_map"] = run_combined_map(
            args, manifest["results"]
        )
        assert_sources_unchanged(expected_sources)
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["error"] = "keyboard interrupt"
        exit_code = 130
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        print("error: {}".format(error), file=sys.stderr)
        exit_code = 1
    finally:
        manifest["finished_utc"] = utc_now()
        manifest["finished_unix_s"] = time.time()
        manifest["elapsed_wall_s"] = (
            manifest["finished_unix_s"] - manifest["started_unix_s"]
        )
        write_json(args.output_dir / "manifest.json", manifest)
        fcntl.flock(wrapper_lock, fcntl.LOCK_UN)
        wrapper_lock.close()

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
