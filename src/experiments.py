import numpy as np
from data_io import DataIO
from visualizer_graph import VisualizerGraph
from fast_gyro_readout import ASM330LHHPoller
from pimikromove import execute_on_picontrollers, CHAIN_SERIAL, SOLO_SERIAL
from gyro_integrator import GyroIntegrator
from gyro_simulator import GyroSimulator
from threading import Thread, Event
from visualizer3d import Visualizer3D, request_window_close
from orientations_extrapolator import OrientationsExtrapolator
from error_calculator import ErrorCalculator
import time
from bisect import bisect_left


def closest_startracker_quat(init_time: float,
                             star_tracker_orientations):
    """
    Return the Quaternion from star_tracker_orientations whose timestamp is closest to init_time.
    If there's a tie, returns the earlier one.
    """
    if not star_tracker_orientations:
        raise ValueError("star_tracker_orientations is empty")

    # Sort by timestamp (noop if already sorted)
    seq = sorted(star_tracker_orientations, key=lambda x: x[1])
    ts = [t for _, t in seq]

    i = bisect_left(ts, init_time)
    if i == 0:
        return seq[0][0]
    if i == len(ts):
        return seq[-1][0]

    before_q, before_t = seq[i-1]
    after_q,  after_t  = seq[i]

    return before_q if abs(init_time - before_t) <= abs(after_t - init_time) else after_q

def align_imu_to_startracker( star_tracker_orientations, imu_measurements):
    """
    Returns a subset of imu_measurements that start at or after
    the first star tracker timestamp.
    """
    if not star_tracker_orientations or not imu_measurements:
        return []

    start_time = star_tracker_orientations[0][1]
    aligned = [(w, t) for (w, t) in imu_measurements if t >= start_time]
    return aligned


def visualizer_thread(sequence, start_evt: Event, end_evt: Event):
    viz = Visualizer3D(window_size=(640, 480))
    viz.create_window()
    start_evt.wait()
    viz.execute_live_maneuver(sequence)
    viz.keep_running(end_evt=end_evt)


def fast_gyro_readout_thread(start_evt: Event, end_evt: Event, end_calibrating: Event, identifier, save=True):
    with ASM330LHHPoller(
        spi_url='ftdi://ftdi:232h/1',
        spi_mode=3,
        spi_freq_hz=12_000_000,
        odr_code=ASM330LHHPoller.ODR_3333HZ,
        gyr_fs_dps=125
    ) as imu:
        seq, timestamp_start = imu.record(start_event=start_evt, end_event=end_evt, end_calibrating=end_calibrating)
    
    if save:
        dio = DataIO()
        dio.save_fast_gyro(seq, identifier=identifier, timestamp_start=timestamp_start)


def test(start_evt: Event, end_evt: Event):
    time.sleep(3)
    start_evt.set()
    time.sleep(5)
    end_evt.set()


def run_experiment(sequence, save=True):
    start_evt = Event()
    end_evt = Event()
    end_calibrating = Event()

    identifier = int(time.time())

    t_gyro = Thread(target=fast_gyro_readout_thread,
                    args=(start_evt, end_evt, end_calibrating, identifier, save),
                    daemon=True)
    t_gyro.start()
    t_viz = Thread(target=visualizer_thread,
                   args=(sequence, start_evt, end_evt),
                   daemon=True)
    t_viz.start()

    execute_on_picontrollers(
        sequence=sequence,
        chain_serial=CHAIN_SERIAL,
        solo_serial=SOLO_SERIAL,
        chain_addresses=(1, 2),
        axes=('1', '1', '1'),
        wait_each_step=False,
        zero_all_motors=False,
        should_print=False,
        start_evt=start_evt,
        end_evt=end_evt,
        end_calibrating=end_calibrating
    )

    # test(start_evt, end_evt)

    # Close the 3D visualization window
    request_window_close()

    t_gyro.join(timeout=2.0)
    t_viz.join(timeout=2.0)

    if save:
        dataIO = DataIO()
        dataIO.save_simulated_ground_truth(sequence, identifier)


def experiment_0():
    """
    Get the camera in the correct orientation
    """
    gyro = GyroSimulator(80)
    t = 5
    w = (np.pi / 2.0) / t
    gyro.set_maneuver(np.array([0, 0, 1]), 0, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations


def experiment_1():
    """
    Spin around the z-axis, and back
    """
    gyro = GyroSimulator(80)
    t = 5
    w = (np.pi / 2.0) / t
    gyro.set_maneuver(np.array([0, 0, 1]), w, t)
    gyro.set_maneuver(np.array([0, 0, 1]), -w, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations

def experiment_2():
    """
    Spin around the x-axis, and back
    """
    gyro = GyroSimulator(80)
    t = 6.0
    w = (np.pi / 4.0) / t
    gyro.set_maneuver(np.array([1, 0, 0]), w, t)
    gyro.set_maneuver(np.array([1, 0, 0]), -w, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations

def experiment_3():
    """
    Spin around the z-x axis, and back
    """
    gyro = GyroSimulator(80)
    t = 5.0
    w = (np.pi / 4.0) / t
    gyro.set_maneuver(np.array([1, 0, 1]), w, t)
    gyro.set_maneuver(np.array([1, 0, 1]), -w, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations

def experiment_4():
    """
    Spin around the z-y axis, there and back
    """
    gyro = GyroSimulator(80)
    t = 5.0
    w = (np.pi / 4.0) / t
    gyro.set_maneuver(np.array([0, 1, 1]), w, t)
    gyro.set_maneuver(np.array([0, 1, 1]), -w, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations

def experiment_5():
    """
    Spin around the z-y-x axis
    """
    gyro = GyroSimulator(80)
    t = 5.0
    w = (np.pi / 4.0) / t
    gyro.set_maneuver(np.array([1, 1, 1]), w, t)
    gyro.set_maneuver(np.array([1, 1, 1]), -w, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations

def experiment_6():
    """
    Spin around z-axis, then change to spin around y axis
    """
    gyro = GyroSimulator(80)
    t = 5.0
    w = (np.pi / 4.0) / t
    gyro.set_maneuver(np.array([0, 0, 1]), w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), -w, t)
    gyro.set_maneuver(np.array([0, 0, 1]), -w, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations

def experiment_7():
    """
    Spin around y-axis, then change to spin around z axis
    """
    gyro = GyroSimulator(80)
    t = 5.0
    w = (np.pi / 4.0) / t
    gyro.set_maneuver(np.array([0, 1, 0]), w, t)
    gyro.set_maneuver(np.array([0, 0, 1]), w, t)
    gyro.set_maneuver(np.array([0, 0, 1]), -w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), -w, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations

def experiment_8():
    """
    Spin around x-axis, then spin around y axis
    """
    gyro = GyroSimulator(80)
    t = 5.0
    w = (np.pi / 4.0) / t
    gyro.set_maneuver(np.array([1, 0, 0]), w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), -w, t)
    gyro.set_maneuver(np.array([1, 0, 0]), -w, t)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(25)
    return orientations

def experiment_9():
    """
    Coning 1
    """
    gyro = GyroSimulator(80)
    omega = 0.3
    half_angle_deg = 10.0
    duration = 60
    dt = 0.05
    orientations = gyro.coning(omega, half_angle_deg, duration, dt, use_offset=False)
    return orientations

def experiment_10():
    """
    Coning 2
    """
    gyro = GyroSimulator(80)
    omega = -0.35
    half_angle_deg = 10.0
    duration = 60
    dt = 0.05
    orientations = gyro.coning(omega, half_angle_deg, duration, dt, use_offset=False)
    return orientations

def experiment_11():
    """
    Coning 3
    """
    gyro = GyroSimulator(80)
    omega = 0.15
    half_angle_deg = 20.0
    duration = 60
    dt = 0.05
    orientations = gyro.coning(omega, half_angle_deg, duration, dt, use_offset=False)
    return orientations

def experiment_12():
    """
    Coning 4
    """
    gyro = GyroSimulator(80)
    omega = -0.1
    half_angle_deg = 20.0
    duration = 60
    dt = 0.05
    orientations = gyro.coning(omega, half_angle_deg, duration, dt, use_offset=False)
    return orientations


def load(identifier, fast_gyro=False, visualize=True, offset_value=0, on_valid_only=False):
    dataIO = DataIO()
    orientations_extrapolator = OrientationsExtrapolator()
    error_calculator = ErrorCalculator()

    offset_time = False
    star_tracker_orientations_raw = dataIO.read_star_tracker_quaternions_raw(identifier, offset_time=offset_time)
    star_tracker_orientations_combined = dataIO.read_star_tracker_quaternions_combined(identifier, offset_time=offset_time)
    star_tracker_orientations_extrapolated = dataIO.read_star_tracker_quaternions_extrapolated(identifier, offset_time=offset_time)
    star_tracker_validity = dataIO.read_star_tracker_validity(identifier, offset_time=offset_time)

    mimu_measurements = dataIO.read_star_tracker_angular_velocities(identifier, offset_time=offset_time)
    mimu_measurements = dataIO.offset_measurements(mimu_measurements, offset_value)
    mimu_measurements = align_imu_to_startracker(star_tracker_orientations_raw, mimu_measurements)
    init_time_mimu = mimu_measurements[0][1]
    init_quat_mimu = closest_startracker_quat(init_time_mimu, star_tracker_orientations_raw)

    if fast_gyro:
        fast_gyro_measurements = dataIO.read_fast_gyro(identifier, offset_time=offset_time)
        fast_gyro_measurements = align_imu_to_startracker(star_tracker_orientations_raw, fast_gyro_measurements)
        init_time_fast_gyro = fast_gyro_measurements[0][1]
        init_quat_fast_gyro = closest_startracker_quat(init_time_fast_gyro, star_tracker_orientations_raw)
        gyro_integrator_fast_gyro_quats = GyroIntegrator(init_quat_fast_gyro, init_time=init_time_fast_gyro)
        for w, t in fast_gyro_measurements:
            gyro_integrator_fast_gyro_quats.update_direct_quaternion(w, t)
        fast_gyro_orientations_quats = gyro_integrator_fast_gyro_quats.get_subsampled_orientations(samples=700)

    gyro_integrator_quats = GyroIntegrator(init_quat_mimu, init_time=init_time_mimu)
    gyro_integrator_noconing = GyroIntegrator(init_quat_mimu, init_time=init_time_mimu)
    gyro_integrator_coning = GyroIntegrator(init_quat_mimu, init_time=init_time_mimu)        

    for w, t in mimu_measurements:
        gyro_integrator_quats.update_direct_quaternion(w, t)
        gyro_integrator_noconing.update_rotation_vector(w, t, coning=False)
        gyro_integrator_coning.update_rotation_vector(w, t, coning=True)
    
    mimu_orientations_quats = gyro_integrator_quats.get_subsampled_orientations()
    mimu_orientations_noconing = gyro_integrator_noconing.orientations
    mimu_orientations_coning = gyro_integrator_coning.orientations

    direct_quat_extrapolated = orientations_extrapolator.extrapolate(star_tracker_orientations_raw,
                                                                     star_tracker_validity,
                                                                     mimu_measurements,
                                                                     OrientationsExtrapolator.DIRECT_QUATERNION,
                                                                     on_valid_only=on_valid_only)
    coning_appx_extrapolated = orientations_extrapolator.extrapolate(star_tracker_orientations_raw,
                                                                     star_tracker_validity,
                                                                     mimu_measurements,
                                                                     OrientationsExtrapolator.CONING_APPX,
                                                                     on_valid_only=on_valid_only)
    noconing_extrapolated = orientations_extrapolator.extrapolate(star_tracker_orientations_raw,
                                                                     star_tracker_validity,
                                                                     mimu_measurements,
                                                                     OrientationsExtrapolator.NOCONING,
                                                                     on_valid_only=on_valid_only)
    if fast_gyro:
        fast_gyro_direct_quat_extrapolated = orientations_extrapolator.extrapolate(star_tracker_orientations_raw,
                                                                                        star_tracker_validity,
                                                                                        fast_gyro_measurements,
                                                                                        OrientationsExtrapolator.DIRECT_QUATERNION,
                                                                                        on_valid_only=on_valid_only)
        fast_gyro_coning_appx_extrapolated = orientations_extrapolator.extrapolate(star_tracker_orientations_raw,
                                                                                        star_tracker_validity,
                                                                                        fast_gyro_measurements,
                                                                                        OrientationsExtrapolator.CONING_APPX,
                                                                                        on_valid_only=on_valid_only)
        fast_gyro_direct_noconing_extrapolated = orientations_extrapolator.extrapolate(star_tracker_orientations_raw,
                                                                                        star_tracker_validity,
                                                                                        fast_gyro_measurements,
                                                                                        OrientationsExtrapolator.NOCONING,
                                                                                        on_valid_only=on_valid_only)

    # vis = Visualizer3D()
    # vis.create_window()
    # vis.execute_live_maneuver(mimu_orientations_quats)
    # vis.keep_running()

    visualizer_graph = VisualizerGraph(visualize=visualize)
    star_tracker_orientations = star_tracker_orientations_raw
    if fast_gyro:
        labels = ["Star Tracker - Ground Truth", "Direct Quat. Update", "Rotation Vector Update", "Simple Kinematic Int.",
             "Ext. Gyro Direct Quat.", "Ext. Gyro Rotation Vector", "Ext. Gyro Simple Kinematic"]    
        orientations = [star_tracker_orientations, noconing_extrapolated,
                        direct_quat_extrapolated, coning_appx_extrapolated,
                        fast_gyro_direct_noconing_extrapolated, fast_gyro_direct_quat_extrapolated, 
                        fast_gyro_coning_appx_extrapolated]
        
        orientations = [star_tracker_orientations, 
                        direct_quat_extrapolated, coning_appx_extrapolated, noconing_extrapolated,
                        fast_gyro_direct_quat_extrapolated, fast_gyro_coning_appx_extrapolated, 
                        fast_gyro_direct_noconing_extrapolated]
    
    else:
        # labels = ["Star Tracker", "MIMU NOCONING", "MIMU QUAT", "MIMU CONING", "MIMU EXTRAPOLATED"]
        # orientations = [star_tracker_orientations, noconing_extrapolated,
        #              direct_quat_extrapolated, coning_appx_extrapolated,
        #             star_tracker_orientations_extrapolated]
        
        labels = ["RAW", "MIMU EXTRAPOLATED", "NO CONING"]
        orientations = [star_tracker_orientations_raw, star_tracker_orientations_extrapolated, noconing_extrapolated]

    visualizer_graph.visualize(
        *orientations,
        labels=labels,
        validity=star_tracker_validity,
        unwrap=False
    )

    # visualizer_graph.visualize_error(
    #     *orientations,
    #     labels=labels
    # )
    
    direct_quat_errors = orientations_extrapolator.calculate_accumulated_error(star_tracker_orientations_raw,
                                                                               star_tracker_validity,
                                                                               direct_quat_extrapolated)
    coning_appx_errors = orientations_extrapolator.calculate_accumulated_error(star_tracker_orientations_raw,
                                                                               star_tracker_validity,
                                                                               coning_appx_extrapolated)
    noconing_errors = orientations_extrapolator.calculate_accumulated_error(star_tracker_orientations_raw,
                                                                               star_tracker_validity,
                                                                               noconing_extrapolated)
    
    mimu_extrapolated_errors = orientations_extrapolator.calculate_accumulated_error(star_tracker_orientations_raw,
                                                                                     star_tracker_validity,
                                                                                     star_tracker_orientations_extrapolated)
    
    if fast_gyro:
        fast_gyro_coning_errors = orientations_extrapolator.calculate_accumulated_error(star_tracker_orientations_raw,
                                                                                        star_tracker_validity,
                                                                                        fast_gyro_coning_appx_extrapolated)
        fast_gyro_noconing_errors = orientations_extrapolator.calculate_accumulated_error(star_tracker_orientations_raw,
                                                                                        star_tracker_validity,
                                                                                        fast_gyro_direct_noconing_extrapolated)
        fast_gyro_quat_errors = orientations_extrapolator.calculate_accumulated_error(star_tracker_orientations_raw,
                                                                                        star_tracker_validity,
                                                                                        fast_gyro_direct_quat_extrapolated)
        
        error_sequences, avg_errors = error_calculator.get_aligned_errors(direct_quat_errors,
                                                                            coning_appx_errors,
                                                                            noconing_errors,
                                                                            fast_gyro_quat_errors,
                                                                            fast_gyro_coning_errors,
                                                                            fast_gyro_noconing_errors)

        visualizer_graph.visualize_error_sequences(*error_sequences,
                                                    labels=["Direct Quat. Update", "Rotation Vector Update", "Simple Kinematic Int.",
                                                            "Ext. Gyro Direct Quat.", "Ext. Gyro Rotation Vector", "Ext. Gyro Simple Kinematic"],
                                                    average_errors=avg_errors)
    else:
        error_sequences, avg_errors = error_calculator.get_aligned_errors(direct_quat_errors,
                                                                            coning_appx_errors,
                                                                            noconing_errors,
                                                                            mimu_extrapolated_errors)
        visualizer_graph.visualize_error_sequences(*error_sequences,
                                                labels=["Direct Quat", "Coning Approx.", "No Coning", "Mimu extrapolated"],
                                                average_errors=avg_errors)
    
    return error_sequences, avg_errors


def average_errors():
    visualizer = VisualizerGraph()
    names = ["Direct Quat. Update", "Rotation Vector Update", "Simple Kinematic Int.",
             "Ext. Gyro Direct Quat. Update", "Ext. Gyro Rotation Vector Update", "Ext. Gyro Simple Kinematic Int."]
    # names = ["DIRECT QUAT", "CONING APPROX.", "NO CONING",
    #          "MIMU EXTRAPOLATED"]
    all_errors = []
    for i in range(2, 19):
        print(f"{i-1}/17")
        _, avg_errors = load(f"{i}", fast_gyro=True, visualize=True, offset_value=0.01, on_valid_only=False)
        for i, avg_error in enumerate(avg_errors):
            if len(all_errors) == i:
                all_errors.append([])
            all_errors[i].append(avg_error)

    calculated_errors = []
    for name, errors in zip(names, all_errors):
        avg_error = 0
        total_time = sum(t for _, t in errors)
        for error, time in errors:
            avg_error += (error * (time/total_time))
        print(f"{name} average error rate: {round(avg_error, 6)} deg/s")
        calculated_errors.append(avg_error)
    
    visualizer.plot_average_errors(names, *calculated_errors)

def find_time_offset():
    xs = np.arange(0, 0.02 + 1e-9, 0.005)
    results = []  # (t, total_avg_error)

    for t in xs:
        print(f"\nTesting offset t = {t:.2f}")

        all_errors = []  # list of lists, one per error component

        for scenario in range(2, 19):
            print(f"  scenario {scenario-1}/17", end="\r")
            _, avg_errors = load(
                f"{scenario}",
                fast_gyro=False,
                visualize=False,
                offset_value=t
            )

            for i, err in enumerate(avg_errors):
                if len(all_errors) <= i:
                    all_errors.append([])
                all_errors[i].append(err[0])

        # convert to array: shape (num_error_types, num_scenarios)
        all_errors = np.array(all_errors)

        # total average error (mean over error types and scenarios)
        total_avg_error = all_errors.mean()

        results.append((t, total_avg_error))

    # sort by error
    results.sort(key=lambda x: x[1])

    print("\n=== Best time offsets ===")
    for t, err in results[:5]:
        print(f"t = {t:.5f}, total_avg_error = {err:.6f}")

    return results

def main():
    should_save = True
    orientations = experiment_11()

    # viz = Visualizer3D(window_size=(640, 480))
    # viz.create_window()
    # viz.execute_live_maneuver(orientations)

    # run_experiment(orientations, should_save)

    # exp. 5 has star tracker values which are presented as valid, but are in fact invalid
    # load("18", fast_gyro=False)
    average_errors()
    # find_time_offset()

if __name__ == "__main__":
    main()
