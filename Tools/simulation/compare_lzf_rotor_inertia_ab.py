#!/usr/bin/env python3

"""Compare enabled/disabled LZF rotor-inertia compensation flight artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag
from pyulog import ULog
from scipy.signal import welch
from scipy.spatial.transform import Rotation


ARMED_STATE = 2
OFFBOARD_NAV_STATE = 14
AXES = ("roll", "pitch", "yaw")
TARGET_BAND_HZ = (1.8, 2.6)


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--enabled-dir", type=Path, required=True)
  parser.add_argument("--disabled-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  return parser.parse_args()


def topic(ulog, name):
  matches = [
      item.data for item in ulog.data_list
      if item.name == name and item.multi_id == 0
  ]
  if not matches:
    raise RuntimeError(f"ULog has no {name} instance 0")
  return matches[0]


def seconds(data):
  return np.asarray(data["timestamp"], dtype=float) * 1e-6


def contiguous_segments(time_s, selected):
  selected_indices = np.flatnonzero(selected)
  if not selected_indices.size:
    return []
  split = np.flatnonzero(np.diff(selected_indices) > 1) + 1
  groups = np.split(selected_indices, split)
  return [(time_s[group[0]], time_s[group[-1]]) for group in groups]


def bag_window(path):
  command_time = []
  state = []
  throttle_time = []
  throttle = []

  with rosbag.Bag(str(path)) as bag:
    for topic_name, message, record_time in bag.read_messages(
        topics=["/setpoints_cmd", "/mavros/state", "/debugPx4ctrl"]
    ):
      time_s = record_time.to_sec()
      if topic_name == "/setpoints_cmd":
        command_time.append(time_s)
      elif topic_name == "/mavros/state":
        state.append((time_s, bool(message.armed), str(message.mode)))
      else:
        throttle_time.append(time_s)
        throttle.append(float(message.des_thr))

  if len(command_time) < 2:
    raise RuntimeError(f"{path} has no usable /setpoints_cmd interval")

  command_start = command_time[0]
  command_end = command_time[-1]
  preceding_offboard = [
      item[0] for item in state
      if item[0] <= command_start and item[1] and item[2] == "OFFBOARD"
  ]
  if not preceding_offboard:
    raise RuntimeError(f"{path} has no armed OFFBOARD state before trajectory")

  offboard_start = preceding_offboard[-1]
  throttle_time = np.asarray(throttle_time, dtype=float)
  throttle = np.asarray(throttle, dtype=float)
  throttle_mask = (throttle_time >= command_start) & (throttle_time <= command_end)
  return {
      "start_offset_s": command_start - offboard_start,
      "end_offset_s": command_end - offboard_start,
      "duration_s": command_end - command_start,
      "throttle_time_s": throttle_time[throttle_mask] - command_start,
      "throttle": throttle[throttle_mask],
  }


def spectrum(time_s, values):
  time_s = np.asarray(time_s, dtype=float)
  values = np.asarray(values, dtype=float)
  finite = np.isfinite(time_s) & np.isfinite(values)
  time_s = time_s[finite]
  values = values[finite]
  if time_s.size < 32:
    raise RuntimeError("not enough samples for spectrum")

  sample_rate = (time_s.size - 1) / (time_s[-1] - time_s[0])
  frequency, density = welch(
      values - np.mean(values),
      fs=sample_rate,
      nperseg=min(4096, values.size),
  )
  search = (frequency >= 0.5) & (frequency <= 10.0)
  band = (frequency >= TARGET_BAND_HZ[0]) & (frequency <= TARGET_BAND_HZ[1])
  total = np.trapz(density[search], frequency[search])
  band_power = np.trapz(density[band], frequency[band])
  peak_index = np.flatnonzero(search)[np.argmax(density[search])]
  return {
      "dominant_frequency_hz": float(frequency[peak_index]),
      "target_band_rms": float(np.sqrt(max(band_power, 0.0))),
      "target_band_power_fraction": float(band_power / total) if total > 0.0 else 0.0,
  }


def rms(values):
  values = np.asarray(values, dtype=float)
  return float(np.sqrt(np.mean(values * values)))


def analyze_trial(directory):
  bag_path = directory / "flight.bag"
  ulog_path = directory / "flight.ulg"
  score = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
  window = bag_window(bag_path)
  ulog = ULog(str(ulog_path))

  status = topic(ulog, "vehicle_status")
  status_time = seconds(status)
  offboard = (
      (np.asarray(status["arming_state"], dtype=int) == ARMED_STATE)
      & (np.asarray(status["nav_state"], dtype=int) == OFFBOARD_NAV_STATE)
  )
  segments = contiguous_segments(status_time, offboard)
  if not segments:
    raise RuntimeError(f"{ulog_path} has no armed OFFBOARD interval")
  offboard_start, offboard_end = max(segments, key=lambda value: value[1] - value[0])
  start = offboard_start + window["start_offset_s"]
  end = start + window["duration_s"]
  if end > offboard_end + 1.0:
    raise RuntimeError("mapped trajectory extends beyond the ULog OFFBOARD interval")

  angular_velocity = topic(ulog, "vehicle_angular_velocity")
  rate_time = seconds(angular_velocity)
  rate_mask = (rate_time >= start) & (rate_time <= end)
  selected_rate_time = rate_time[rate_mask]
  rates = np.column_stack([
      angular_velocity[f"xyz[{index}]"] for index in range(3)
  ])[rate_mask]

  rate_setpoint = topic(ulog, "vehicle_rates_setpoint")
  setpoint_time = seconds(rate_setpoint)
  setpoint_values = np.column_stack([
      rate_setpoint["roll"], rate_setpoint["pitch"], rate_setpoint["yaw"]
  ])
  interpolated_setpoint = np.column_stack([
      np.interp(selected_rate_time, setpoint_time, setpoint_values[:, index])
      for index in range(3)
  ])

  attitude = topic(ulog, "vehicle_attitude")
  attitude_time = seconds(attitude)
  attitude_mask = (attitude_time >= start) & (attitude_time <= end)
  quaternion = np.column_stack([
      attitude["q[1]"], attitude["q[2]"], attitude["q[3]"], attitude["q[0]"]
  ])[attitude_mask]
  rotation = Rotation.from_quat(quaternion).as_matrix()
  tilt = np.degrees(np.arccos(np.clip(rotation[:, 2, 2], -1.0, 1.0)))

  actuator = topic(ulog, "actuator_motors")
  actuator_time = seconds(actuator)
  actuator_mask = (actuator_time >= start) & (actuator_time <= end)
  motors = np.column_stack([
      actuator[f"control[{index}]"] for index in range(4)
  ])[actuator_mask]
  motors = motors[np.all(np.isfinite(motors), axis=1)]

  rate_metrics = {}
  for index, axis in enumerate(AXES):
    error = rates[:, index] - interpolated_setpoint[:, index]
    rate_metrics[axis] = {
        "actual_rms_rad_s": rms(rates[:, index]),
        "setpoint_rms_rad_s": rms(interpolated_setpoint[:, index]),
        "tracking_error_rms_rad_s": rms(error),
        "actual_spectrum": spectrum(selected_rate_time, rates[:, index]),
        "error_spectrum": spectrum(selected_rate_time, error),
    }

  throttle = window["throttle"]
  return {
      "source": str(directory),
      "mapped_ulog_window_s": [float(start), float(end)],
      "trajectory_duration_s": window["duration_s"],
      "position_rmse_m": score["position"]["rmse_3d"],
      "velocity_rmse_m_s": score["velocity"]["rmse_3d"],
      "tilt_deg": {
          "rms": rms(tilt),
          "p99": float(np.quantile(tilt, 0.99)),
          "max": float(np.max(tilt)),
      },
      "angular_rate": rate_metrics,
      "collective_throttle": {
          "mean": float(np.mean(throttle)),
          "rms": rms(throttle),
          "maximum": float(np.max(throttle)),
          "saturation_fraction": float(np.mean((throttle <= 0.001) | (throttle >= 0.999))),
          "spectrum": spectrum(window["throttle_time_s"], throttle),
      },
      "motor_command": {
          "minimum": float(np.min(motors)),
          "maximum": float(np.max(motors)),
          "saturation_fraction": float(np.mean((motors <= 0.01) | (motors >= 0.99))),
      },
  }


def write_report(result, path):
  enabled = result["enabled"]
  disabled = result["disabled"]
  lines = [
      "# LZF rotor-inertia compensation A/B",
      "",
      "| Metric | Enabled | Disabled |",
      "| --- | ---: | ---: |",
      f"| Position RMSE (m) | {enabled['position_rmse_m']:.4f} | {disabled['position_rmse_m']:.4f} |",
      f"| Velocity RMSE (m/s) | {enabled['velocity_rmse_m_s']:.4f} | {disabled['velocity_rmse_m_s']:.4f} |",
      f"| Tilt RMS (deg) | {enabled['tilt_deg']['rms']:.3f} | {disabled['tilt_deg']['rms']:.3f} |",
      f"| Tilt max (deg) | {enabled['tilt_deg']['max']:.3f} | {disabled['tilt_deg']['max']:.3f} |",
      f"| Collective saturation | {enabled['collective_throttle']['saturation_fraction']:.6f} | {disabled['collective_throttle']['saturation_fraction']:.6f} |",
      f"| Motor saturation | {enabled['motor_command']['saturation_fraction']:.6f} | {disabled['motor_command']['saturation_fraction']:.6f} |",
      "",
      "## Angular-rate tracking and spectrum",
      "",
      "| Axis/metric | Enabled | Disabled |",
      "| --- | ---: | ---: |",
  ]
  for axis in AXES:
    enabled_rate = enabled["angular_rate"][axis]
    disabled_rate = disabled["angular_rate"][axis]
    lines.extend([
        f"| {axis} error RMS (rad/s) | {enabled_rate['tracking_error_rms_rad_s']:.4f} | {disabled_rate['tracking_error_rms_rad_s']:.4f} |",
        f"| {axis} actual dominant frequency (Hz) | {enabled_rate['actual_spectrum']['dominant_frequency_hz']:.3f} | {disabled_rate['actual_spectrum']['dominant_frequency_hz']:.3f} |",
        f"| {axis} 1.8-2.6 Hz RMS (rad/s) | {enabled_rate['actual_spectrum']['target_band_rms']:.4f} | {disabled_rate['actual_spectrum']['target_band_rms']:.4f} |",
        f"| {axis} 1.8-2.6 Hz power fraction | {enabled_rate['actual_spectrum']['target_band_power_fraction']:.4f} | {disabled_rate['actual_spectrum']['target_band_power_fraction']:.4f} |",
    ])
  lines.extend([
      "",
      "The ULog trajectory window is aligned from the latest armed OFFBOARD state",
      "sample preceding the first `/setpoints_cmd` sample in each ROS bag.",
  ])
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
  args = parse_args()
  result = {
      "target_frequency_band_hz": list(TARGET_BAND_HZ),
      "enabled": analyze_trial(args.enabled_dir),
      "disabled": analyze_trial(args.disabled_dir),
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  json_path = args.output_dir / "comparison.json"
  report_path = args.output_dir / "comparison.md"
  json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  write_report(result, report_path)
  print(json_path)
  print(report_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
