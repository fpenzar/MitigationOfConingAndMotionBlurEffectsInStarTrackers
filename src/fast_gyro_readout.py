from __future__ import annotations

import os, sys, glob, ctypes, time, struct, math
from collections import deque
import usb.backend.libusb1 as libusb1
from pyftdi.spi import SpiController
from data_io import DataIO
from typing import List, Tuple
from threading import Thread, Event
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---------- libusb preload (Windows) ----------
_cands = glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "libusb_package", "**", "libusb-1.0.dll"), recursive=True)
if not _cands:
    raise FileNotFoundError("libusb-1.0.dll not found. Run: pip install libusb-package")
_dll = _cands[0]
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(_dll))
ctypes.CDLL(_dll)
os.environ["PYUSB_BACKEND"] = "libusb1"
os.environ["PYUSB_LIBRARY"] = _dll
print("PyUSB backend:", libusb1.get_backend(find_library=lambda _: _dll))


class ASM330LHHPoller:
    """Poll ASM330LHH gyro OUT registers as fast as possible (no FIFO) and save SI (rad/s)."""

    # Registers
    RD = 0x80
    REG_WHO_AM_I = 0x0F   # expect 0x6B
    REG_CTRL1_XL = 0x10
    REG_CTRL2_G  = 0x11
    REG_CTRL3_C  = 0x12   # IF_INC(bit2), BDU(bit6)
    REG_CTRL4_C  = 0x13
    REG_CTRL6_C  = 0x15
    REG_STATUS   = 0x1E   # bit1=GDA

    REG_OUTX_L_G = 0x22   # then X_H,Y_L,Y_H,Z_L,Z_H

    # ODR codes
    ODR_833HZ  = 0x7
    ODR_1667HZ = 0x8
    ODR_3333HZ = 0x9
    ODR_6667HZ = 0xA

    def __init__(self,
                 spi_url: str = 'ftdi://ftdi:232h/1',
                 spi_mode: int = 3,
                 spi_freq_hz: int = 12_000_000,
                 odr_code: int = ODR_1667HZ,     # start safely
                 gyr_fs_dps: int = 125):
        self.spi_url = spi_url
        self.spi_mode = spi_mode
        self.spi_freq_hz = spi_freq_hz
        self.odr_code = odr_code
        self.gyr_fs_dps = gyr_fs_dps

        self.bias = np.array([0, 0, 0])

        # Sensitivity (mdps/LSB)
        self.GYR_SENS_mdps_per_lsb = {125:4.375, 250:8.75, 500:17.5, 1000:35.0, 2000:70.0}[gyr_fs_dps]

        self.ctl = None
        self.port = None

    # SPI helpers
    def _w8(self, reg, val): self.port.write(bytes([reg & 0x7F, val]))
    def _rN(self, reg, n):   return self.port.exchange(bytes([reg | self.RD]), n)
    def _r8(self, reg):      return self._rN(reg, 1)[0]

    @staticmethod
    def _i16_le(lo, hi):
        v = (hi << 8) | lo
        return v - 0x10000 if v & 0x8000 else v

    # lifecycle
    def open(self):
        self.ctl = SpiController()
        self.ctl.configure(self.spi_url)
        self.port = self.ctl.get_port(cs=0, freq=self.spi_freq_hz, mode=self.spi_mode)

        # lower latency for snappier reads
        try:
            ftdi = self.ctl._ftdi
            ftdi.set_latency_timer(1)
            ftdi.set_timeouts(read_timeout=200, write_timeout=200)
        except Exception:
            pass

        wid = self._r8(self.REG_WHO_AM_I)
        if wid != 0x6B:
            if self.spi_mode != 0:
                print("Unexpected WHO_AM_I. Retrying with SPI mode 0…")
                self.close()
                self.spi_mode = 0
                self.open()
                return
            raise RuntimeError(f"Unexpected WHO_AM_I: 0x{wid:02X}")

        self._configure()

    def close(self):
        if self.ctl:
            try: self.ctl.terminate()
            except Exception: pass
        self.ctl = None
        self.port = None

    def _configure(self):
        # Soft reset, then enable IF_INC+BDU
        self._w8(self.REG_CTRL3_C, 0x01)
        time.sleep(0.05)
        self._w8(self.REG_CTRL3_C, (1 << 6) | (1 << 2))   # BDU=1, IF_INC=1

        # LPF1 enable, gyro LPF1 BW ~49Hz
        self._w8(self.REG_CTRL4_C, 0x04 | 0x02)
        self._w8(self.REG_CTRL6_C, 0x05)

        # Gyro ODR & FS (FS_125 via bit1)
        if self.gyr_fs_dps == 125:
            g = (self.odr_code << 4) | (1 << 1)
        else:
            fs_bits_g = {250:0, 500:1, 1000:2, 2000:3}[self.gyr_fs_dps] << 2
            g = (self.odr_code << 4) | fs_bits_g
        self._w8(self.REG_CTRL2_G, g)

        time.sleep(0.01)
    
    def calculate_bias(self, end_calibrating: Event):
        samples = []
        t_start = time.perf_counter()

        try:
            while not end_calibrating.is_set():
                status = self._r8(self.REG_STATUS)
                if status & 0x02:  # GDA (gyro data available)
                    gx, gy, gz = self.read_raw()              # rad/s in sensor frame
                    samples.append((gx, gy, gz))
                else:
                    # time.sleep(0.00001)
                    pass
        except KeyboardInterrupt:
            pass

        dur = time.perf_counter() - t_start
        n = len(samples)

        if n == 0:
            print("No samples captured during calibration. Bias set to zeros.")
            self.bias = np.array([0.0, 0.0, 0.0], dtype=float)
            return self.bias

        arr = np.asarray(samples, dtype=float)  # shape (N, 3)
        bias = arr.mean(axis=0)                 # rad/s
        std = arr.std(axis=0)

        self.bias = bias.astype(float)
        print(f"Bias and std estimated from {n} samples over {dur:.3f}s.")
        print(f"Bias [rad/s]: {self.bias}")
        print(f"Bias [deg/s]: {np.degrees(self.bias)}")
        print(f"Std [rad/s]: {std}")
        print(f"Std [deg/s]: {np.degrees(std)}")

        return self.bias
    
    def remap_axis(self, wx, wy, wz):
        return (-wx, wz, wy)
    
    def read_raw(self):
        b = self._rN(self.REG_OUTX_L_G, 6)
        gx_raw = self._i16_le(b[0], b[1])
        gy_raw = self._i16_le(b[2], b[3])
        gz_raw = self._i16_le(b[4], b[5])

        # raw -> mdps -> dps -> rad/s
        mdps = self.GYR_SENS_mdps_per_lsb
        gx = (gx_raw * (mdps / 1000.0)) * (math.pi / 180.0)
        gy = (gy_raw * (mdps / 1000.0)) * (math.pi / 180.0)
        gz = (gz_raw * (mdps / 1000.0)) * (math.pi / 180.0)

        return gx, gy, gz
    
    def calibrate(self, gx, gy, gz):
        omega = np.array([gx, gy, gz], dtype=float)
        return tuple(omega - self.bias)

    # main record loop
    def record(self, duration_s: float | None = None, max_samples: int | None = None,
               start_event: Event | None = None, end_event: Event | None = None,
               end_calibrating: Event | None = None,
               ) -> List[Tuple[Tuple[float, float, float], float]]:
        """
        Poll gyro OUT registers whenever GDA=1.
        Returns [((wx, wy, wz), timestamp), ...] with SI (rad/s).
        Timestamps are host-side (time.perf_counter()).
        """

        if end_calibrating is not None:
            print("Calibrating fast gyro")
            self.calculate_bias(end_calibrating)

        if start_event is not None:
            start_event.wait()
        
        seq: List[Tuple[Tuple[float, float, float], float]] = []
        timestamp_start = time.time()
        t_start = time.perf_counter()
        t_end = (time.perf_counter() + duration_s) if duration_s is not None else None

        print("Polling fast gyro registers…")
        try:
            while True:
                if end_event is not None and end_event.is_set():
                    break

                if t_end is not None and time.perf_counter() >= t_end:
                    break
                if max_samples is not None and len(seq) >= max_samples:
                    break

                status = self._r8(self.REG_STATUS)
                if status & 0x02:  # GDA
                    # timestamp ASAP around the read
                    t = time.perf_counter() - t_start
                    gx, gy, gz = self.calibrate(*self.read_raw())
                    seq.append((self.remap_axis(gx, gy, gz), t))
                else:
                    # tiny pause to avoid 100% CPU if no new sample yet
                    # (keep this very small so we don't miss many samples)
                    # time.sleep(0.00001)
                    pass
        except KeyboardInterrupt:
            pass

        print(f"Captured {len(seq)} gyro samples.")
        return seq, timestamp_start

    # context manager
    def __enter__(self): self.open(); return self
    def __exit__(self, *args): self.close()


if __name__ == "__main__":
    with ASM330LHHPoller(
        spi_url='ftdi://ftdi:232h:FT83B9FY/1',
        spi_mode=3,
        spi_freq_hz=12_000_000,
        odr_code=ASM330LHHPoller.ODR_3333HZ,
        gyr_fs_dps=125
    ) as imu:
        seq, _ = imu.record(duration_s=5.0)

    # dio = DataIO()
    # out_path = dio.save_fast_gyro(seq)
    # print(f"Saved gyro (SI) to: {out_path}")

