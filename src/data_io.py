from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import time
from quaternion import Quaternion
from astropy.time import Time
from PIL import Image

MIRUA = "MIRUA"
FUSEA = "FUSEA"

GPS_EPOCH_TO_UNIX = 315964800
GPS_UTC_OFFSET = 18

def gps_to_unix(gps_seconds: float) -> float:
    t = Time(gps_seconds, format='gps')
    return t.unix + 17.15


class DataIO:

    def __init__(
        self,
        simulator_ground_truth: str | Path = "data/simulator_ground_truth",
        simulated_angular_vel: str | Path = "data/simulated_angular_vel",
        star_tracker_orientations_raw: str | Path = "data/star_tracker_orientations_raw",
        star_tracker_orientations_extrapolated: str | Path = "data/star_tracker_orientations_extrapolated",
        star_tracker_orientations_combined: str | Path = "data/star_tracker_orientations_combined",
        star_tracker_imu: str | Path = "data/star_tracker_imu",
        star_tracker_validity: str |Path = "data/star_tracker_validity",
        star_tracker_images: str | Path = "data/raw_images",
        fast_gyro: str | Path = "data/fast_gyro",
    ):
        self.simulator_ground_truth = Path(simulator_ground_truth)
        self.simulated_angular_vel = Path(simulated_angular_vel)
        self.fast_gyro = Path(fast_gyro)
        self.star_tracker_orientations_raw = Path(star_tracker_orientations_raw)
        self.star_tracker_orientations_extrapolated = Path(star_tracker_orientations_extrapolated)
        self.star_tracker_orientations_combined = Path(star_tracker_orientations_combined)
        self.star_tracker_imu = Path(star_tracker_imu)
        self.star_tracker_validity = Path(star_tracker_validity)
        self.star_tracker_path_raw = Path("../Coning Test Data/Spool")
        self.star_tracker_images = Path(star_tracker_images)

        self.simulator_ground_truth.mkdir(parents=True, exist_ok=True)
        self.simulated_angular_vel.mkdir(parents=True, exist_ok=True)
        self.star_tracker_orientations_raw.mkdir(parents=True, exist_ok=True)
        self.star_tracker_orientations_extrapolated.mkdir(parents=True, exist_ok=True)
        self.star_tracker_orientations_combined.mkdir(parents=True, exist_ok=True)
        self.star_tracker_imu.mkdir(parents=True, exist_ok=True)
        self.star_tracker_validity.mkdir(parents=True, exist_ok=True)
        self.star_tracker_images.mkdir(parents=True, exist_ok=True)
        self.fast_gyro.mkdir(parents=True, exist_ok=True)

    def save_image(self, image, timestamp):
        unix_ts = gps_to_unix(timestamp)
        filename = self.star_tracker_images / f"{unix_ts}.png"
        Image.fromarray(image).save(filename)
    
    def save_quaternions(self, sequence, fpath, start_timestamp):
        with fpath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "q1", "q2", "q3", "q4"])
            for q, t in sequence:
                arr = np.asarray(getattr(q, "quat", q), dtype=float).reshape(-1)
                if arr.size != 4:
                    raise ValueError("Quaternion must have 4 components [q1,q2,q3,q4].")
                writer.writerow([float(start_timestamp + t), *arr.tolist()])
        return fpath
    
    def read_quaternions(self, fpath, offset_time=True):
        out = []
        with fpath.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = Quaternion(float(row["q1"]), float(row["q2"]), float(row["q3"]), float(row["q4"]))
                t = float(row["timestamp"])
                out.append((q, t))
        if not out:
            return out
        if offset_time:
            t0 = out[0][1]
            out = [(q, t - t0) for (q, t) in out]
        return out
    
    def save_simulated_ground_truth(self, sequence: List[Tuple["Quaternion", float]], identifier, start_timestamp=0) -> Path:
        fpath = self._file_path(self.simulator_ground_truth, "orientations", identifier)
        self.save_quaternions(sequence, fpath, start_timestamp)


    def read_simulated_ground_truth(self, identifier: int, offset_time=True) -> List[Tuple["Quaternion", float]]:
        fpath = self._file_path(self.simulator_ground_truth, "orientations", identifier)
        if not fpath.exists():
            raise FileNotFoundError(f"No orientation file for identifier {identifier}: {fpath}")
        return self.read_quaternions(fpath, offset_time)
    

    def save_angular_velocities(self, sequence, fpath, timestamp_start):
        with fpath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "wx", "wy", "wz"])
            for w, t in sequence:
                wx, wy, wz = [float(v) for v in w]
                writer.writerow([float(t + timestamp_start), wx, wy, wz])
        return fpath
    
    def save_validity(self, sequence, fpath):
        with fpath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "valid"])
            for valid, t in sequence:
                writer.writerow([float(t), valid])
        return fpath
    
    def read_validity(self, fpath, offset_time=True):
        out = []
        with fpath.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                valid = row["valid"] == "True"
                t = float(row["timestamp"])
                out.append((valid, t))
        if not out:
            return out
        if offset_time:
            t0 = out[0][1]
            out = [(valid, t - t0) for (valid, t) in out]
        return out

    def read_angular_velocities(self, fpath, offset_time=True):
        out = []
        with fpath.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append((np.array([float(row["wx"]), float(row["wy"]), float(row["wz"])]), float(row["timestamp"])))
        if not out:
            return out
        if offset_time:
            t0 = out[0][1]
            out = [(omega, t - t0) for (omega, t) in out]
        return out


    def save_simulated_angular_velocities(self, sequence, identifier, timestamp_start=0) -> Path:
        fpath = self._file_path(self.simulated_angular_vel, "angular_velocities", identifier)
        self.save_angular_velocities(sequence, fpath, timestamp_start)


    def read_simulated_angular_velocities(self, identifier: int, offset_time=True) -> List[Tuple[Tuple[float, float, float], float]]:
        fpath = self._file_path(self.simulated_angular_vel, "angular_velocities", identifier)
        if not fpath.exists():
            raise FileNotFoundError(f"No angular velocity file for identifier {identifier}: {fpath}")
        return self.read_angular_velocities(fpath, offset_time)

    def save_fast_gyro(self, sequence, identifier, timestamp_start=0) -> Path:
        fpath = self._file_path(self.fast_gyro, "fast_gyro", identifier)
        self.save_angular_velocities(sequence, fpath, timestamp_start)

    def read_fast_gyro(self, identifier: int, offset_time=True) -> List[Tuple[Tuple[float, float, float], float]]:
        fpath = self._file_path(self.fast_gyro, "fast_gyro", identifier)
        if not fpath.exists():
            raise FileNotFoundError(f"No fast gyro file for identifier {identifier}: {fpath}")
        return self.read_angular_velocities(fpath, offset_time)
    
    def read_star_tracker_quaternions_raw(self, identifier, offset_time=True):
        fpath = self._file_path(self.star_tracker_orientations_raw, "orientations", identifier)
        if not fpath.exists():
            raise FileNotFoundError(f"No star tracker quaternion file for identifier {identifier}: {fpath}")
        return self.read_quaternions(fpath, offset_time)
    
    def read_star_tracker_quaternions_extrapolated(self, identifier, offset_time=True):
        fpath = self._file_path(self.star_tracker_orientations_extrapolated, "orientations", identifier)
        if not fpath.exists():
            raise FileNotFoundError(f"No star tracker quaternion file for identifier {identifier}: {fpath}")
        return self.read_quaternions(fpath, offset_time)
    
    def read_star_tracker_quaternions_combined(self, identifier, offset_time=True):
        fpath = self._file_path(self.star_tracker_orientations_combined, "orientations", identifier)
        if not fpath.exists():
            raise FileNotFoundError(f"No star tracker quaternion file for identifier {identifier}: {fpath}")
        return self.read_quaternions(fpath, offset_time)
    
    def read_star_tracker_angular_velocities(self, identifier, offset_time=True):
        fpath = self._file_path(self.star_tracker_imu, "angular_velocities", identifier)
        if not fpath.exists():
            raise FileNotFoundError(f"No angular velocity file for identifier {identifier}: {fpath}")
        return self.read_angular_velocities(fpath, offset_time)

    def save_star_tracker_quaternions_raw(self, sequence, identifier, start_timestamp):
        fpath = self._file_path(self.star_tracker_orientations_raw, "orientations", identifier)
        self.save_quaternions(sequence, fpath, start_timestamp)
    
    def save_star_tracker_quaternions_extrapolated(self, sequence, identifier, start_timestamp):
        fpath = self._file_path(self.star_tracker_orientations_extrapolated, "orientations", identifier)
        self.save_quaternions(sequence, fpath, start_timestamp)
    
    def save_star_tracker_quaternions_combined(self, sequence, identifier, start_timestamp):
        fpath = self._file_path(self.star_tracker_orientations_combined, "orientations", identifier)
        self.save_quaternions(sequence, fpath, start_timestamp)
    
    def save_star_tracker_imu(self, sequence, identifier, timestamp_start):
        fpath = self._file_path(self.star_tracker_imu, "angular_velocities", identifier)
        self.save_angular_velocities(sequence, fpath, timestamp_start)
    
    def save_star_tracker_validity(self, sequence, identifier):
        fpath = self._file_path(self.star_tracker_validity, "validity", identifier)
        self.save_validity(sequence, fpath)
    
    def read_star_tracker_validity(self, identifier, offset_time=True):
        fpath = self._file_path(self.star_tracker_validity, "validity", identifier)
        if not fpath.exists():
            raise FileNotFoundError(f"No angular velocity file for identifier {identifier}: {fpath}")
        return self.read_validity(fpath, offset_time)
    
    def offset_measurements(self, measurements, offset):
        out = [(value, t - offset) for (value, t) in measurements]
        return out

    def read_star_tracker_raw_data(self, identifier: str):
        imu_filename = f"{MIRUA}{identifier}.txt"
        quat_fileneme = f"{FUSEA}{identifier}.txt"
        imu_filepath = self.star_tracker_path_raw / imu_filename
        quat_filepath = self.star_tracker_path_raw / quat_fileneme

        out_imu = []
        out_quat_raw = []
        out_quat_extrapolated = []
        out_quat_combined = []
        out_validity = []

        t0_imu = 0
        t0_quat = 0

        i = 0
        dt = 1 / 80
        with imu_filepath.open("r", newline="") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                elements = line.strip().split()
                t = gps_to_unix(float(elements[0]))
                wx = np.deg2rad(float(elements[1]))
                wy = np.deg2rad(float(elements[2]))
                wz = np.deg2rad(float(elements[3]))
                out_imu.append((np.array([wx, wy, wz]), t + dt * (i+0.5)))
                i += 1
                if i == 10:
                    i = 0
        
        if out_imu:
            t0_imu = out_imu[0][1]
            out_imu = [(w, t - t0_imu) for (w, t) in out_imu]
                
        with quat_filepath.open("r", newline="") as file:
            extrapolated_quat = Quaternion.I()
            while True:
                line = file.readline()
                if not line:
                    break
                elements = line.strip().split()
                t = gps_to_unix(float(elements[4]))
                valid = int(elements[17]) == 1
                out_validity.append((valid, t))

                q1 = float(elements[19])
                q2 = float(elements[20])
                q3 = float(elements[21])
                q4 = float(elements[22])
                q = Quaternion(q1, q2, q3, q4)
                out_quat_raw.append((q,t))

                if valid:
                    out_quat_combined.append((q, t))
                else:
                    out_quat_combined.append((extrapolated_quat, t))

                # Extrapolated orientations
                t_ext = gps_to_unix(float(elements[30]))
                q1_ext = float(elements[31])
                q2_ext = float(elements[32])
                q3_ext = float(elements[33])
                q4_ext = float(elements[34])
                extrapolated_quat = Quaternion(q1_ext, q2_ext, q3_ext, q4_ext)
                out_quat_extrapolated.append((extrapolated_quat, t_ext))

                
        if out_quat_raw:
            t0_quat = out_quat_raw[0][1]
            out_quat_raw = [(q, t - t0_quat) for (q, t) in out_quat_raw]
            out_quat_extrapolated = [(q, t - t0_quat) for (q, t) in out_quat_extrapolated]
            out_quat_combined = [(q, t - t0_quat) for (q, t) in out_quat_combined]

        return out_imu, t0_imu, out_quat_raw, out_quat_extrapolated, out_quat_combined, t0_quat, out_validity

    def save_star_tracker_data(self, read_identifier, write_identifier):
        imu_data, t0_imu, out_quat_raw, out_quat_extrapolated, out_quat_combined, t0_quat, out_validity = self.read_star_tracker_raw_data(read_identifier)
        self.save_star_tracker_quaternions_raw(out_quat_raw, write_identifier, t0_quat)
        self.save_star_tracker_quaternions_extrapolated(out_quat_extrapolated, write_identifier, t0_quat)
        self.save_star_tracker_quaternions_combined(out_quat_combined, write_identifier, t0_quat)
        self.save_star_tracker_imu(imu_data, write_identifier, t0_imu)
        self.save_star_tracker_validity(out_validity, write_identifier)

    def _file_path(self, path: Path, prefix: str, identifier: int) -> Path:
        return path / f"{prefix}_{identifier}.csv"


if __name__ == "__main__":
    dataIo = DataIO()

    read_identifiers = [
        '251106_20294048',
        '251106_20303002',
        '251106_20312646',
        '251106_20333015',
        '251106_20341679',
        '251106_20345626',
        '251106_20353606',
        '251106_20362204',
        '251106_20371202',
        '251106_20380038',
        '251106_20390645',
        '251106_20404395',
        '251106_20422600',
        '251106_20441845',
        '251106_20460555',
        '251106_20475173',
        '251106_20493113',
        '251106_20512877',
        '251106_20541532'
    ]

    for i, read_id in enumerate(read_identifiers):
        dataIo.save_star_tracker_data(read_id, i)
    