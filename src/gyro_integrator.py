import numpy as np
from quaternion import Quaternion, almost
import copy


class GyroIntegrator:

    def __init__(self, initial_orientation=Quaternion.I(), init_time=0, fast_loop_samples=20):
        self.orientation = initial_orientation
        self.timestamp = init_time
        self.orientations = [(copy.deepcopy(self.orientation), self.timestamp)]

        # update_rotation_vector params
        self.fast_loop_samples = fast_loop_samples
        self.fast_loop_counter = 0
        self.rotation_vector = np.array([0, 0, 0])
        self.delta_alpha = np.array([0, 0, 0])
        self.prev_delta_alpha = np.array([0, 0, 0])
        self.alpha = np.array([0, 0, 0])
        self.prev_alpha = np.array([0, 0, 0])
        self.beta = np.array([0, 0, 0])
        self.prev_beta = np.array([0, 0, 0])
        self.delta_beta = np.array([0, 0, 0])

        # update_direct_quaternion params
        self.prev_delta_theta = np.array([0, 0, 0])
    

    def update_orientation(self, rotation_vector):
        """
        FLOP complexity: 32 mul, 20 add, 2 div, 2 sqrt, 1 sin, 1 cos
        """
        dq = Quaternion.from_rotation_vector(rotation_vector)
        self.orientation = Quaternion.chain_rotations(self.orientation, dq)
        self.orientation.normalize()
        self.orientations.append((copy.deepcopy(self.orientation), self.timestamp))
    

    def update_direct_quaternion(self, omega_body: np.ndarray, time: float):
        """
        Calculates the quaternion from the angular velocities each time and chains it to the previous
        rotation.
        Reference: Geometric Integration of Quaternions
        Reference: Runge-Kutta Approximations for Direct Coning Compensation Applying Lie Theory, Single speed calculation

        FLOP complexity (without quaternion, single call to the function):
        12 mul, 6 add

        with quaternion:
        => 26 add, 44 mul, 2 div, 2 sqrt, 1 sin, 1 cos
        20 x updates =  520 add, 880 mul, 40 div, 40 sqrt, 20 sin, 20 cos
        """
        dt = time - self.timestamp
        self.timestamp = time
        if dt <= 0:
            return

        delta_theta = omega_body * dt
        rotation_vector = delta_theta + 1/12 * np.cross(self.prev_delta_theta, delta_theta)
        self.update_orientation(rotation_vector)
        self.prev_delta_theta = delta_theta


    def update_rotation_vector(self, omega_body: np.ndarray, time: float, coning=True, force_orientation=False):
        """
        Updates the rotation vector. Set coning flag to take into account coning errors.   
        The rotation vector is then converted into a quaternion after every
        self.fast_loop_samples calls to this function.
        Reference: Strapdown Inertial Navigation Integration Part 1 Attitude Algorithms

        FLOP complexity (without quaternion, single call to the function):
        no coning: 3 add, 3 mul
        coning: 13 add, 15 mul

        FLOP complexity (with 20 measurements and update to the quaternion):
        no coning: 80 add, 92 mul, 2 div, 2 sqrt, 1 sin, 1 cos
        coning: 280 add, 332 mul, 2 div, 2 sqrt, 1 sin, 1 cos

        INFO:
        Selective coning is not feasable, since a check would have to be performed between
        omega_body and previous omega to see if they are parallel (cross product and linalg.norm).
        But at that point, it is cheaper to do the regular coning algorithm
        """
        dt = time - self.timestamp
        self.timestamp = time
        if dt <= 0:
            return
        
        self.fast_loop_counter += 1

        # Integrate the angular velocity into incremental rotation, delta_alpha
        self.delta_alpha = omega_body * dt
        # Add delta_alpha to the previous alpha (linear rotation)
        self.alpha = self.prev_alpha + self.delta_alpha

        if coning:
            # Calculate the non-commutative cross product of the previous and current rotations
            self.delta_beta = 0.5 * np.cross(self.prev_alpha + self.prev_delta_alpha/6, self.delta_alpha)

            # Sum the non-commutative product from this step into total coning compensation part
            self.beta = self.prev_beta + self.delta_beta

        # Update the variables from this step
        self.prev_alpha = self.alpha
        self.prev_beta = self.beta
        self.prev_delta_alpha = self.delta_alpha
        
        # Convert the accumulated rotation vector into a quaternion and chain rotations
        if self.fast_loop_counter == self.fast_loop_samples or force_orientation:
            if coning:
                rotation_vector = self.alpha + self.beta
            else:
                rotation_vector = self.alpha
            self.update_orientation(rotation_vector)

            # Reset the variables
            self.reset_rotation_vector_vars()


    def reset_rotation_vector_vars(self):
        self.prev_alpha = np.array([0, 0, 0])
        self.prev_beta = np.array([0, 0, 0])
        self.prev_delta_alpha = np.array([0, 0, 0])
        self.fast_loop_counter = 0
    

    def set_orientation(self, orientation: Quaternion, time: float):
        if time < self.orientations[-1][1]:
            print("ERROR: cannot set orientation at a time before last update time")
            return
        self.orientation = copy.deepcopy(orientation) 
        self.orientation.normalize()
        self.orientations.append((copy.deepcopy(self.orientation), time))
        self.reset_rotation_vector_vars()
    

    def get_subsampled_orientations(self, samples=None):
        if samples == None:
            samples = self.fast_loop_samples
        subsampled_orientations = self.orientations[::samples]
        return subsampled_orientations
        

    def reset_orientation(self, orientation=Quaternion.I()):
        self.orientation = orientation
    
    def reset_time(self, time=0):
        self.timestamp = time
    
    def reset(self, orientation=Quaternion.I(), time=0, fast_loop_samples=None):
        self.reset_orientation(orientation)
        self.reset_time(time)
        self.orientations = [(copy.deepcopy(self.orientation), self.timestamp)]
        self.reset_rotation_vector_vars()
        self.prev_delta_theta = np.array([0, 0, 0])
        if fast_loop_samples != None:
            self.fast_loop_samples = fast_loop_samples


def test_single_axis_update():
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    integrator = GyroIntegrator()
    omega_body = np.array([np.pi / 4, 0, 0])
    
    time = 2
    integrator.update_direct_quaternion(omega_body, time)
    orientation = integrator.orientation
    assert almost(integrator.orientation.to_euler(True), np.array([90, 0, 0]))
    assert almost(Quaternion.rotate(ex, orientation), ex)
    assert almost(Quaternion.rotate(ey, orientation), ez)
    assert almost(Quaternion.rotate(ez, orientation), -ey)

    time = 3
    omega_body = np.array([0, np.pi / 4, 0])
    integrator.update_direct_quaternion(omega_body, time)
    assert almost(integrator.orientation.to_euler(True), np.array([90, 0, 45]))


def test_combined_rotation_axis_update():
    integrator = GyroIntegrator()
    rate = (2 * np.pi) / np.sqrt(3)
    omega_body = np.array([rate, rate, rate])
    time = 1/3
    integrator.update_direct_quaternion(omega_body, time)
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])
    orientation = integrator.orientation
    assert almost(Quaternion.rotate(ex, orientation), ey)
    assert almost(Quaternion.rotate(ey, orientation), ez)
    assert almost(Quaternion.rotate(ez, orientation), ex)

    integrator.reset_orientation(Quaternion.I())
    integrator.reset_time()

    rate = (np.pi / 2) / np.sqrt(2)
    omega_body = np.array([0, rate, rate])
    time = 2
    integrator.update_direct_quaternion(omega_body, time)
    orientation = integrator.orientation
    assert almost(Quaternion.rotate(ex, orientation), -ex)
    assert almost(Quaternion.rotate(ey, orientation), ez)
    assert almost(Quaternion.rotate(ez, orientation), ey)


def test_chained_combined_rotation_axis_update():
    integrator = GyroIntegrator()
    rate = (2 * np.pi) / np.sqrt(3)
    omega_body = np.array([rate, rate, rate])
    time = 1/3
    integrator.update_direct_quaternion(omega_body, time)
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])
    orientation = integrator.orientation
    assert almost(Quaternion.rotate(ex, orientation), ey)
    assert almost(Quaternion.rotate(ey, orientation), ez)
    assert almost(Quaternion.rotate(ez, orientation), ex)

    rate = (np.pi / 2) / np.sqrt(2)
    omega_body = np.array([0, rate, rate])
    time = 2 + 1/3
    integrator.update_direct_quaternion(omega_body, time)
    orientation = integrator.orientation
    assert almost(Quaternion.rotate(ex, orientation), -ey)
    assert almost(Quaternion.rotate(ey, orientation), ex)
    assert almost(Quaternion.rotate(ez, orientation), ez)


if __name__ == "__main__":
    test_single_axis_update()
    test_combined_rotation_axis_update()
    test_chained_combined_rotation_axis_update()

    print("All gyro integrator tests passed ✅")