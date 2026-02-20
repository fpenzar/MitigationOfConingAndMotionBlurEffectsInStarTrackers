import time
import numpy as np
from pipython import GCSDevice, pitools
from gyro_simulator import GyroSimulator
from visualizer3d import Visualizer3D
from threading import Thread, Event
from fast_gyro_readout import ASM330LHHPoller

CHAIN_SERIAL = "PI C-863 Mercury SN 0165500726"
SOLO_SERIAL = "PI C-863 Mercury  SN 0023550694"
PITCH_OFFSET = 90
# PITCH_OFFSET = 0
# ROLL_OFFSET = -90
ROLL_OFFSET = 0


def _eulers_from_quats(sequence):
    """
    sequence: list[(Quaternion, t)]
    returns: times (np.array), eulers_deg_unwrapped shape (N,3) in degrees
    """
    if not sequence:
        return np.array([]), np.empty((0,3))

    ts = np.array([float(t) for (_, t) in sequence], dtype=float)
    eulers_rad = np.array([q.to_euler() for (q, _) in sequence], dtype=float)  # (N,3)
    # unwrap each channel to avoid discontinuities
    eulers_unwrapped_rad = np.column_stack([
        np.unwrap(eulers_rad[:, 0]),
        np.unwrap(eulers_rad[:, 1]),
        np.unwrap(eulers_rad[:, 2]),
    ])
    eulers_deg = np.degrees(eulers_unwrapped_rad)
    return ts, eulers_deg


def set_to_starting_pos(roll, pitch, yaw, roll_axis, pitch_axis, yaw_axis):
    print("Setting to starting positions")
    roll.MOV(roll_axis, ROLL_OFFSET)
    pitch.MOV(pitch_axis, PITCH_OFFSET)
    yaw.MOV(yaw_axis, 0)
    pitools.waitontarget(roll, roll_axis)
    pitools.waitontarget(pitch, pitch_axis)
    pitools.waitontarget(yaw, yaw_axis)

def set_to_zero(roll, pitch, yaw, roll_axis, pitch_axis, yaw_axis):
    roll.MOV(roll_axis, 0)
    pitch.MOV(pitch_axis, 0)
    yaw.MOV(yaw_axis, 0)
    pitools.waitontarget(roll, roll_axis)
    pitools.waitontarget(pitch, pitch_axis)
    pitools.waitontarget(yaw, yaw_axis)

def print_positions(roll, pitch, yaw, roll_axis, pitch_axis, yaw_axis, rewrite=True):
    roll_pos = roll.qPOS(roll_axis)[roll_axis]
    pitch_pos = pitch.qPOS(pitch_axis)[pitch_axis]
    yaw_pos = yaw.qPOS(yaw_axis)[yaw_axis]
    if rewrite:
        end = "\r"
    else:
        end = "\n"
    print(f"Roll: {roll_pos - ROLL_OFFSET:.3f} Pitch: {pitch_pos - PITCH_OFFSET:.3f} Yaw: {yaw_pos:.3f}", end=end)


def execute_on_picontrollers(
    sequence,
    chain_serial,
    solo_serial,
    chain_addresses=(1, 2),
    axes=('1','1','1'),
    wait_each_step=False,
    zero_all_motors=False,
    should_print=False,
    start_evt: Event | None = None,
    end_evt: Event | None = None,
    end_calibrating: Event | None = None, 
):
    """
    Replays the ground-truth quaternion trajectory using relative MVR moves between samples.
    Timing follows the provided timestamps (no constant frequency assumed).
    Mapping (by convention):
      roll  -> chain addr chain_addresses[0]
      pitch -> chain addr chain_addresses[1]
      yaw   -> solo USB
    """
    if not sequence:
        return

    # Normalize timestamps to start at 0
    t_raw, e_deg = _eulers_from_quats(sequence)
    t0 = t_raw[0]
    t_rel = t_raw - t0

    # Connect controllers
    with GCSDevice('C-863') as chain:
        print("Connecting to daisy chain...")
        chain.OpenUSBDaisyChain(chain_serial)
        dcid = chain.dcid

        with GCSDevice('C-863') as roll, GCSDevice('C-863') as pitch, GCSDevice('C-863') as yaw:
            print("Connecting to individual controllers...")
            yaw.ConnectDaisyChainDevice(chain_addresses[0], dcid)
            roll.ConnectDaisyChainDevice(chain_addresses[1], dcid)
            pitch.ConnectUSB(solo_serial)

            # Set the end_calibrating event because the axis are about to start moving
            if end_calibrating is not None:
                end_calibrating.set()

            roll_axis, pitch_axis, yaw_axis = axes

            for dev, ax in ((roll, roll_axis), (pitch, pitch_axis), (yaw, yaw_axis)):
                dev.VEL(ax, 7.5)   # deg/s
                dev.ACC(ax, 50.0)  # deg/s^2
                dev.DEC(ax, 50.0)
            
            # Reset to starting configuration
            if zero_all_motors:
                set_to_zero(roll, pitch, yaw, roll_axis, pitch_axis, yaw_axis)
                return
            # Move to starting position
            set_to_starting_pos(roll, pitch, yaw, roll_axis, pitch_axis, yaw_axis)
            # Prime previous angles with the first sample
            prev_rpy = e_deg[0, :].copy()
            # A small delay before starting
            # time.sleep(1.5)

            # Real time pacing using monotonic clock
            start = time.monotonic()
            # Signal for the viz thread to execute
            if start_evt is not None:
                start_evt.set()
            # Move to first step - following steps are relative to this first one
            roll.MOV(roll_axis, float(e_deg[0, 0]) + ROLL_OFFSET)
            pitch.MOV(pitch_axis, float(e_deg[0, 1]) + PITCH_OFFSET)
            yaw.MOV(yaw_axis, float(e_deg[0, 2]))
            # Play from the second sample
            for k in range(1, len(t_rel)):
                target_t = t_rel[k]
                while True:
                    now = time.monotonic()
                    dt = target_t - (now - start)
                    if dt <= 0:
                        break
                    time.sleep(min(0.005, dt))
                rpy = e_deg[k, :]
                delta = rpy - prev_rpy
                # Get rid of small angles
                eps = 1e-10
                delta = np.where(np.abs(delta) < eps, 0.0, delta)

                # send relative moves (MVR) per axis
                if delta[0] != 0.0:
                    roll.MVR(roll_axis, float(delta[0]))
                if delta[1] != 0.0:
                    pitch.MVR(pitch_axis, float(delta[1]))
                if delta[2] != 0.0:
                    yaw.MVR(yaw_axis, float(delta[2]))

                if wait_each_step:
                    # original, less smooth behavior: wait after each triplet
                    pitools.waitontarget(roll, roll_axis)
                    pitools.waitontarget(pitch, pitch_axis)
                    pitools.waitontarget(yaw, yaw_axis)

                prev_rpy = rpy

                if should_print and k % 10 == 0:
                    print_positions(roll, pitch, yaw, roll_axis, pitch_axis, yaw_axis, True)

            if not wait_each_step:
                pitools.waitontarget(roll, roll_axis)
                pitools.waitontarget(pitch, pitch_axis)
                pitools.waitontarget(yaw, yaw_axis)
            
            if end_evt is not None:
                end_evt.set()
            
            print("Final positions:")
            print_positions(roll, pitch, yaw, roll_axis, pitch_axis, yaw_axis, False)

    print("Playback complete.")


def visualizer_thread(sequence, start_evt: Event):
    viz = Visualizer3D(window_size=(640, 480))
    viz.create_window()
    start_evt.wait()
    viz.execute_live_maneuver(sequence)
    # Keep window open
    # viz.keep_running()


def run_motion_and_visualization(sequence, chain_serial, solo_serial, should_print=False):
    start_evt = Event()
    end_evt = Event()

    # Start VISUALIZER in its own thread
    t_viz = Thread(target=visualizer_thread,
                   args=(sequence, start_evt),
                   daemon=True)
    t_viz.start()

    execute_on_picontrollers(
        sequence=sequence,
        chain_serial=chain_serial,
        solo_serial=solo_serial,
        chain_addresses=(1, 2),
        axes=('1', '1', '1'),
        wait_each_step=False,
        zero_all_motors=False,
        should_print=should_print,
        start_evt=start_evt,
        end_evt=end_evt
    )


def scenario_1():
    gyro = GyroSimulator(80)
    t = 5.0
    w = (np.pi / 4.0) / t

    gyro.set_maneuver(np.array([1, 1, 0]), 0, t)
    # gyro.set_maneuver(np.array([0, 1, 0]), w, t)
    # gyro.set_maneuver(np.array([0, 0, 1]), w, t)

    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)

    return orientations


def coning_1():
    gyro = GyroSimulator(5)
    omega = 2.0
    half_angle_deg = 10.0
    duration = 10
    dt = 0.05
    orientations = gyro.coning(omega, half_angle_deg, duration, dt, use_offset=False)
    return orientations


def reset_to_zero(chain_serial, solo_serial):
    gyro = GyroSimulator(80)
    t = 5.0
    w = (np.pi / 4.0) / t
    gyro.set_maneuver(np.array([1, 0, 0]), 0, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(50)
    execute_on_picontrollers(orientations, chain_serial, solo_serial, zero_all_motors=True)


if __name__ == "__main__":
    orientations = scenario_1()
    # orientations = coning_1()

    # run_motion_and_visualization(orientations, CHAIN_SERIAL, SOLO_SERIAL, should_print=False)

    # The setup should be reset back to 0 at the end of the day
    reset_to_zero(CHAIN_SERIAL, SOLO_SERIAL)
