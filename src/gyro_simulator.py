import numpy as np
from gyro_integrator import GyroIntegrator
from quaternion import Quaternion, almost
import copy


class GyroSimulatorManeuver:
    
    def __init__(self, rotation_axis: np.ndarray, angular_speed: float, duration: float):
        self.rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        self.angular_speed = angular_speed
        self.duration = duration
        self.omega = self.calculate_omega()

    def calculate_omega(self):
        omega = self.angular_speed * self.rotation_axis
        return omega


class GyroSimulator:

    def __init__(self, output_frequency_hz=1, noise_std=0.0):
        self.attitude_maneuvers = []
        self.output_frequency_hz = output_frequency_hz
        self.internal_gyro_integrator = GyroIntegrator()
        self.time = 0
        self.noise_std = noise_std
        self.ground_truth_attitudes = []
    
    def ground_truth(self):
        return self.internal_gyro_integrator.orientation

    def set_maneuver(self, axis: np.ndarray, angular_speed: float, duration: float):
        maneuver = GyroSimulatorManeuver(axis, angular_speed, duration)
        self.attitude_maneuvers.append(maneuver)
        self.time += duration
        self.internal_gyro_integrator.update_direct_quaternion(maneuver.omega, self.time)
        self.ground_truth_attitudes.append((copy.deepcopy(self.internal_gyro_integrator.orientation), self.time))
    
    def get_ground_truth_attitudes(self):
        self.ground_truth_attitudes
        
    def reset_maneuvers(self):
        self.attitude_maneuvers = []
        self.internal_gyro_integrator.reset()
        self.time = 0
        self.ground_truth_attitudes = []
    
    def set_noise(self, noise_std):
        self.noise_std = noise_std
    
    def execute_attitude_maneuvers(self, noise=False):
        """
        Returns a list of (omega, timestamp) readings from the simulated gyro. 
        The maneuvers to execture are set via set_maneuver function. 
        """
        timeswitches = []
        start = 0
        for maneuver in self.attitude_maneuvers:
            timeswitches.append(maneuver.duration + start)
            start += maneuver.duration
        
        outputs = []
        timer = 0
        dt = 1 / self.output_frequency_hz
        i = 0

        while(i < len(self.attitude_maneuvers)):
            if (timer + dt) <= timeswitches[i]:
                timer += dt
                outputs.append((self.attitude_maneuvers[i].omega, timer))
                continue

            # case when timer + dt > timeswitches[i]
            average_omega = np.array([0, 0, 0])
            j = 0
            previous_timer = timer
            # account for maneuvers which end within the (timer + dt) interval
            while ((i+j) < len(timeswitches) and timeswitches[i+j] <= (timer + dt)):
                omega = self.attitude_maneuvers[i+j].omega
                omega_weighted = omega * ((timeswitches[i+j] - previous_timer) / dt)
                previous_timer = timeswitches[i+j]
                average_omega = omega_weighted + average_omega
                j += 1
            
            # account for the maneuver which ends after the (timer + dt) interval
            if (i+j < len(timeswitches)):
                omega = self.attitude_maneuvers[i+j].omega
                omega_weighted = omega * ((timer + dt - previous_timer) / dt)
                average_omega = omega_weighted + average_omega
            # if the sampling interval ends after all maneuvers, don't report it
            elif (timeswitches[-1] < (timer + dt)):
                break

            # update the outer loop
            i += j
            timer += dt
            outputs.append((average_omega, timer))
        
        # add Gaussian noise if requested
        if noise and self.noise_std > 0:
            noisy_outputs = []
            for omega, t in outputs:
                noise_vec = np.random.normal(0, self.noise_std, size=3)
                noisy_outputs.append((omega + noise_vec, t))
            outputs = noisy_outputs

        return outputs

    def coning(self, omega: float, half_angle_deg: float, duration: float, dt: float, use_offset=False):
        """
        Reference: A New Strapdown Attitude Algorithm, Appendix: Coning Motion 
        """
        outputs = []
        s = np.sin(0.5*np.deg2rad(half_angle_deg))
        c = np.cos(0.5*np.deg2rad(half_angle_deg))
        k = int(duration / dt)
        offset = Quaternion.from_euler(0, 0, np.deg2rad(-90))
        for t in range(k+1):
            time = dt*t
            Qx = Quaternion(0, s*np.cos(omega*time), s*np.sin(omega*time), c)
            Qy = Quaternion(s*np.sin(omega*time), 0, s*np.cos(omega*time), c)
            Qz = Quaternion(s*np.cos(omega*time), s*np.sin(omega*time), 0, c)
            Q = Qz
            if use_offset:
                # add a rotation around z for visualization purposes
                rotation = Quaternion.chain_rotations(offset, Q)
            else:
                rotation = Q
            outputs.append((rotation, time))
        return outputs
    
    def coning_angular_velocities(self, omega: float, half_angle_deg: float, duration: float, noise: bool = False):
        """
        Reference: A New Strapdown Attitude Algorithm, Appendix: Coning Motion
        Average of sin, cos in a [a,b] interval -> favg = (1 / (b-a)) * integral_a_to_b(f(x)dx)

        The norm of the angular velocities is always 2*omega*sin(half_angle_rad/2) rad/s
        """
        dt = 1.0 / float(self.output_frequency_hz)
        s_half = np.sin(0.5 * np.deg2rad(half_angle_deg))
        s = np.sin(np.deg2rad(half_angle_deg))
        c = np.cos(0.5 * np.deg2rad(half_angle_deg))

        outputs = []
        t = 0.0

        while (t + dt) <= duration:
            if omega == 0.0:
                avg_wx = 0.0
                avg_wy = 0.0
                avg_wz = 0.0
            else:
                sin_avg = (np.cos(omega * t) - np.cos(omega * (t + dt))) / (omega * dt)
                cos_avg = (np.sin(omega * (t + dt)) - np.sin(omega * t)) / (omega * dt)

                # X-axis is coning
                # avg_wx = -2.0 * omega * s_half * s_half
                # avg_wy = -omega * s * sin_avg
                # avg_wz =  omega * s * cos_avg

                # Z-axis is coning
                avg_wx = -omega * s * sin_avg
                avg_wy =  omega * s * cos_avg
                avg_wz = -2.0 * omega * s_half * s_half
                
            w = np.array([avg_wx, avg_wy, avg_wz], dtype=float)
            t += dt
            outputs.append((w, t))

        # add Gaussian noise if requested
        if noise and self.noise_std > 0:
            noisy_outputs = []
            for omega, t in outputs:
                noise_vec = np.random.normal(0, self.noise_std, size=3)
                noisy_outputs.append((omega + noise_vec, t))
            outputs = noisy_outputs

        return outputs

    
    def execute_ground_truth_attitude_maneuvers(self, reporting_frequency_hz=25):
        """
        Returns a list with elements (Quaternion, timestamp), where Quaternion represents the 
        orientation at that timestamp.
        """
        integrator = GyroIntegrator()

        timeswitches = []
        start = 0
        for maneuver in self.attitude_maneuvers:
            timeswitches.append(maneuver.duration + start)
            start += maneuver.duration
        
        outputs = []
        timer = 0
        dt = 1 / reporting_frequency_hz
        i = 0

        while(i < len(self.attitude_maneuvers)):
            if (timer + dt) <= timeswitches[i]:
                integrator.update_direct_quaternion(self.attitude_maneuvers[i].omega, timer + dt)
                outputs.append((copy.deepcopy(integrator.orientation), timer + dt))
                timer += dt
                continue
            
            integrator.update_direct_quaternion(self.attitude_maneuvers[i].omega, timeswitches[i])
            outputs.append((copy.deepcopy(integrator.orientation), timeswitches[i]))
            timer = timeswitches[i]
            i += 1

        return outputs


def test_simple_attitude_maneuvers():
    gyro = GyroSimulator(5)
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    gyro.set_maneuver(ex, np.pi, 1)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[0][0], np.array([np.pi, 0, 0]))

    gyro.reset_maneuvers()
    gyro.set_maneuver(ey, np.pi / 2, 1)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[0][0], np.array([0, np.pi / 2, 0]))

    gyro.reset_maneuvers()
    gyro.set_maneuver(ez, np.pi / 2, 1)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[0][0], np.array([0, 0, np.pi / 2]))

def test_combined_rotation_axis():
    gyro = GyroSimulator(5)
    axis = np.array([1, 1, 0])
    angular_speed = 2

    gyro.set_maneuver(axis, angular_speed, 1)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[0][0], outputs[2][0])
    assert almost(outputs[0][0], np.array([2/np.sqrt(2), 2/np.sqrt(2), 0]))

    axis = np.array([1, 1, 2])
    angular_speed = -2
    gyro.reset_maneuvers()
    gyro.set_maneuver(axis, angular_speed, 1)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[0][0], outputs[2][0])
    assert almost(outputs[0][0], np.array([(-2*1)/np.sqrt(6), (-2*1)/np.sqrt(6), (-2*2)/np.sqrt(6)]))

def test_combined_rotations():
    gyro = GyroSimulator(1)

    axis1 = np.array([1, 0, 0])
    angular_speed = 1
    gyro.set_maneuver(axis1, angular_speed, 1)

    axis2 = np.array([0, 1, 0])
    angular_speed = 2
    gyro.set_maneuver(axis2, angular_speed, 1)

    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[0][0], np.array([1, 0, 0]))
    assert almost(outputs[1][0], np.array([0, 2, 0]))

def test_averaging_rotations():
    gyro = GyroSimulator(5)

    axis = np.array([1, 0, 0])
    angular_speed = 1
    gyro.set_maneuver(axis, angular_speed, 0.95)
    angular_speed = 2
    gyro.set_maneuver(axis, angular_speed, 1)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[4][0], np.array([1.25, 0, 0]))

    gyro.reset_maneuvers()
    axis = np.array([1, 0, 0])
    angular_speed1 = 1
    angular_speed2 = 2
    angular_speed3 = 5
    gyro.set_maneuver(axis, angular_speed1, 0.25)
    gyro.set_maneuver(axis, angular_speed2, 0.1)
    gyro.set_maneuver(axis, angular_speed3, 1)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[1][0], np.array([2.5, 0, 0]))

    gyro.reset_maneuvers()
    axis = np.array([1, 0, 0])
    angular_speed1 = 1
    angular_speed2 = 2
    angular_speed3 = 5
    gyro.set_maneuver(axis, angular_speed1, 0.05)
    gyro.set_maneuver(axis, angular_speed2, 0.1)
    gyro.set_maneuver(axis, angular_speed3, 0.05)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[0][0], np.array([2.5, 0, 0]))

    gyro.reset_maneuvers()
    axis = np.array([1, 0, 0])
    angular_speed1 = 1
    angular_speed2 = 2
    angular_speed3 = 5
    angular_speed4 = 7
    gyro.set_maneuver(axis, angular_speed1, 0.25)
    gyro.set_maneuver(axis, angular_speed2, 0.08)
    gyro.set_maneuver(axis, angular_speed3, 0.02)
    gyro.set_maneuver(axis, angular_speed4, 1)
    outputs = gyro.execute_attitude_maneuvers()
    assert almost(outputs[1][0], np.array([3.3, 0, 0]))

def test_ground_truth():
    gyro = GyroSimulator(5)
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    axis = np.array([0, 0, 1])
    angular_speed = -np.pi/2
    gyro.set_maneuver(axis, angular_speed, 1)
    orientation = gyro.ground_truth()
    assert almost(Quaternion.rotate(ex, orientation), -ey)
    assert almost(Quaternion.rotate(ey, orientation), ex)
    assert almost(Quaternion.rotate(ez, orientation), ez)

    gyro.reset_maneuvers()
    angular_speed1 = np.pi
    gyro.set_maneuver(ex, angular_speed1, 1)
    angular_speed2 = np.pi/2
    gyro.set_maneuver(ey, angular_speed2, 1)
    orientation = gyro.ground_truth()
    assert almost(Quaternion.rotate(ex, orientation), ez)
    assert almost(Quaternion.rotate(ey, orientation), -ey)
    assert almost(Quaternion.rotate(ez, orientation), ex)

def test_execute_ground_truth_attitude_maneuvers():
    gyro = GyroSimulator(5)
    axis = np.array([0, 0, 1])
    angular_speed = -np.pi/2
    gyro.set_maneuver(axis, angular_speed, 1)
    angular_speed2 = np.pi/2
    gyro.set_maneuver(axis, angular_speed2, 1)
    orientations = gyro.execute_ground_truth_attitude_maneuvers(20)
    euler_angles = [orientation.to_euler() for orientation, _ in orientations]
    assert almost(euler_angles[19], np.array([0, 0, -np.pi/2]))
    assert almost(euler_angles[39], np.array([0, 0, 0]))

if __name__ == "__main__":
    test_simple_attitude_maneuvers()
    test_combined_rotation_axis()
    test_combined_rotations()
    test_averaging_rotations()
    test_ground_truth()
    test_execute_ground_truth_attitude_maneuvers()

    print("All gyro simulator tests passed ✅")
