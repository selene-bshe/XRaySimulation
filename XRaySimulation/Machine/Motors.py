"""
Anyway, in the end, I think it is better for me to just
create a module to mimic the motion of the motors I used
for the SD table

To be compatible with all other modules, we use unit
1. um
2. fs
3. rad

In my definition, the relation between the coordinate I used in this code and the
coordinate used by XPP is the following:

 my definition                  XPP                         physical
 0                          y axis                       vertical
 1                          x axis                       horizontal
 2                          z axis                       x-ray proportion direction
"""

"""
When I am implementing this module, 
I am very short in time.
Therefore, I have assumed that the sole purpose of this module is for the TG experiment.
Even though I have tried a bit to make it more general purpose, 
it is pretty much a failure.
If you plan to use this module for your own work, you need to think twice.
With high probability, you need to spend a significant amount of time to revise the code. 
"""

import numpy as np
from XRaySimulation import util


class xyMotor:
    def __init__(self,
                 upperLim=25000.0,
                 lowerLim=-25000.0,
                 res=5.0,
                 backlash=100.0,
                 speed_um_per_ps=1 * 1000 / 1e12,
                 dimension=None,  # The linear dimension of the motor for the visualization
                 height=4e3,
                 color='black',
                 ):
        """
        By default, the motion axis is along the x axis.

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param speed_um_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        # With respect to the default positive direction, whether change the motor motion direction
        if dimension is None:
            dimension = [100e3, 100e3]
        self.type = "Linear xy"

        self.control_location = 0.0
        self.control_positive = 1.
        self.control_speed = speed_um_per_ps
        self.control_backlash = backlash

        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim
        self.res = res

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])

        self.motion_dir = np.zeros(3, dtype=np.float64)
        self.motion_dir[1] = 1.0
        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface
        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.array([0, 0, 0, ], dtype=np.float64)  # The center of the top mounting surface

        self.color = color  # For visualization

    def shift(self, displacement):

        # Change the linear stage platform center
        self.top_mount_pos += displacement
        self.bottom_mount_pos += displacement
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.top_mount_pos = np.ascontiguousarray(rot_mat.dot(self.top_mount_pos))
        self.bottom_mount_pos = np.ascontiguousarray(rot_mat.dot(self.bottom_mount_pos))

        self.top_mount_dir = np.ascontiguousarray(rot_mat.dot(self.top_mount_dir))
        self.bottom_mount_dir = np.ascontiguousarray(rot_mat.dot(self.bottom_mount_dir))

        self.motion_dir = np.ascontiguousarray(rot_mat.dot(self.motion_dir))
        self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))

    def user_move_abs(self, target):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location
            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction
                # Step 3: change the physical location

                # Get the physical displacement of the table
                physical_motion = delta + self.res * (np.random.rand() - 0.5)
                physical_motion = physical_motion * self.control_positive * self.motion_dir

                # Move the stage table
                self.top_mount_pos = self.top_mount_pos + physical_motion

                # Step 4: Change the status in the control system
                self.control_location = target

                print("Motor moved to {:.4f} um".format(self.control_location))

                # The motion time
                motion_time = delta / self.control_speed

                return motion_time, physical_motion

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):
                    # Get the physical displacement of the table
                    physical_motion = self.control_backlash + delta + self.res * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.motion_dir

                    motion_record = np.copy(physical_motion)

                    # Move the stage table
                    self.top_mount_pos = self.top_mount_pos + physical_motion

                    # Get the physical displacement of the table
                    physical_motion = -self.control_backlash + self.res * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.motion_dir

                    motion_record += physical_motion

                    # Move the stage table
                    self.top_mount_pos = self.top_mount_pos + physical_motion

                    # Step 4: Change the status in the control system
                    self.control_location = target
                    print("Motor moved to {:.4f} um".format(self.control_location))

                    motion_time = (2 * self.control_backlash + delta) / self.control_speed
                    return motion_time, motion_record

                else:
                    print("The target location {:.2f} um plus back clash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target location {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

    def user_getPosition(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False


class zMotor:
    def __init__(self,
                 upperLim=7e3,
                 lowerLim=-7e3,
                 res=0.5,
                 backlash=100.0,
                 speed_um_per_ps=1e3 / 1e12,
                 dimension=None,  # The linear dimension of the motor for the visualization
                 height=80e3,
                 color='black',
                 ):
        """
        By default, the motion axis is along the x axis.

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param speed_um_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        # With respect to the default positive direction, whether change the motor motion direction
        if dimension is None:
            dimension = [100e3, 100e3]
        self.type = "Linear z"

        self.control_location = 0.0
        self.control_positive = 1.
        self.control_speed = speed_um_per_ps
        self.control_backlash = backlash

        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim
        self.res = res

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])

        self.motion_dir = np.zeros(3, dtype=np.float64)
        self.motion_dir[0] = 1.0
        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface
        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.array([0, 0, 0, ], dtype=np.float64)  # The center of the top mounting surface

        self.color = color  # For visualization

    def shift(self, displacement):

        # Change the linear stage platform center
        self.top_mount_pos += displacement
        self.bottom_mount_pos += displacement
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        self.top_mount_pos = np.ascontiguousarray(rot_mat.dot(self.top_mount_pos))
        self.bottom_mount_pos = np.ascontiguousarray(rot_mat.dot(self.bottom_mount_pos))

        self.top_mount_dir = np.ascontiguousarray(rot_mat.dot(self.top_mount_dir))
        self.bottom_mount_dir = np.ascontiguousarray(rot_mat.dot(self.bottom_mount_dir))

        self.motion_dir = np.ascontiguousarray(rot_mat.dot(self.motion_dir))
        self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))

    def user_move_abs(self, target):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location
            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction
                # Step 3: change the physical location

                # Get the physical displacement of the table
                physical_motion = delta + self.res * (np.random.rand() - 0.5)
                physical_motion = physical_motion * self.control_positive * self.motion_dir

                # Move the stage table
                self.top_mount_pos = self.top_mount_pos + physical_motion

                # Step 4: Change the status in the control system
                self.control_location = target

                print("Motor moved to {:.4f} um".format(self.control_location))
                # The motion time
                motion_time = delta / self.control_speed

                return motion_time, physical_motion

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):
                    # Get the physical displacement of the table
                    physical_motion = self.control_backlash + delta + self.res * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.motion_dir
                    motion_record = np.copy(physical_motion)
                    # Move the stage table
                    self.top_mount_pos = self.top_mount_pos + physical_motion

                    # Get the physical displacement of the table
                    physical_motion = -self.control_backlash + self.res * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.motion_dir
                    motion_record += physical_motion
                    # Move the stage table
                    self.top_mount_pos = self.top_mount_pos + physical_motion

                    # Step 4: Change the status in the control system
                    self.control_location = target
                    print("Motor moved to {:.4f} um".format(self.control_location))

                    motion_time = (2 * self.control_backlash + delta) / self.control_speed
                    return motion_time, motion_record

                else:
                    print("The target location {:.2f} um plus back clash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target location {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

    def user_getPosition(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False


class RotationMotor:
    def __init__(self,
                 upperLim=np.deg2rad(180),
                 lowerLim=-np.deg2rad(-180),
                 res=1.0,
                 backlash=0.05,
                 speed_rad_per_ps=1 * 1000 / 1e12,
                 dimension=None,
                 height=60e3,
                 color='grey'
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param speed_rad_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        if dimension is None:
            dimension = [100e3, 100e3]

        self.type = "Rotation"
        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim
        self.control_location = 0.0  # rad

        self.control_backlash = backlash
        self.control_speed = speed_rad_per_ps
        self.res = res

        self.deg0direction = np.zeros(3, dtype=np.float64)
        self.deg0direction[1] = 1.0
        self.rotation_axis = np.zeros(3, dtype=np.float64)
        self.rotation_axis[0] = 1.0
        self.rotation_center = np.zeros(3, dtype=np.float64)
        self.rotation_center[0] = height

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface
        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.array([0, 0, 0, ], dtype=np.float64)  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])
        self.color = color

    def shift(self, displacement):

        # Change the linear stage platform center
        self.rotation_center += np.copy(displacement)
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.deg0direction = np.ascontiguousarray(rot_mat.dot(self.deg0direction))
        self.rotation_center = np.ascontiguousarray(rot_mat.dot(self.rotation_center))
        self.rotation_axis = np.ascontiguousarray(rot_mat.dot(self.rotation_axis))

        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)

        self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))

    def user_move_abs(self, target, getMotionTime=True):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location

            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction

                # Step 3: change the physical status of the motor
                # Get the rotation matrix
                rotMat = util.get_rotmat_around_axis(
                    angleRadian=delta + self.res * (np.random.rand() - 0.5),
                    axis=self.rotation_axis)

                # Update the zero deg direction
                self.deg0direction = np.dot(rotMat, self.deg0direction)

                # Step 4 : change the control system information
                self.control_location = target

                print("Motor moved to {:.5f} deg".format(np.rad2deg(self.control_location)))

                motion_time = delta / self.control_speed
                return motion_time, rotMat

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):

                    rotMat1 = util.get_rotmat_around_axis(
                        angleRadian=self.control_backlash + delta + self.res * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)

                    self.deg0direction = np.dot(rotMat1, self.deg0direction)

                    rotMat2 = util.get_rotmat_around_axis(
                        angleRadian=- self.control_backlash + self.res * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)
                    self.deg0direction = np.dot(rotMat2, self.deg0direction)
                    self.control_location = target
                    print("Motor moved to {:.5f} deg".format(np.rad2deg(self.control_location)))
                    motion_time = (2 * self.control_backlash + delta) / self.control_speed
                    return motion_time, np.dot(rotMat2, rotMat1)

                else:
                    print("The target location {:.2f} rad plus backlash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target location {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

        if getMotionTime:
            return 0

    def user_getPosition(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False


class SwivalMotor:
    def __init__(self,
                 upperLim=np.deg2rad(5),
                 lowerLim=-np.deg2rad(-5),
                 res=np.deg2rad(0.000756),
                 backlash=0.05,
                 speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                 dimension=None,
                 rot_center_height=70e3,
                 height=26e3,
                 color='grey'
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param speed_rad_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        if dimension is None:
            dimension = [70e3, 70e3]

        self.type = "Swival"
        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim
        self.control_location = 0.0  # rad

        self.control_backlash = backlash
        self.control_speed = speed_rad_per_ps
        self.res = res

        self.deg0direction = np.zeros(3, dtype=np.float64)
        self.deg0direction[0] = 1.0
        self.rotation_axis = np.zeros(3, dtype=np.float64)
        self.rotation_axis[2] = 1.0
        self.rotation_center = np.zeros(3, dtype=np.float64)
        self.rotation_center[0] = rot_center_height + height

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface
        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.zeros(3, dtype=np.float64)  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])
        self.color = color

    def shift(self, displacement):

        # Change the linear stage platform center
        self.rotation_center += np.copy(displacement)
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.deg0direction = np.ascontiguousarray(rot_mat.dot(self.deg0direction))
        self.rotation_center = np.ascontiguousarray(rot_mat.dot(self.rotation_center))
        self.rotation_axis = np.ascontiguousarray(rot_mat.dot(self.rotation_axis))

        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)
        self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))

    def user_move_abs(self, target):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location

            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction

                # Step 3: change the physical status of the motor
                # Get the rotation matrix
                rotMat = util.get_rotmat_around_axis(
                    angleRadian=delta + self.res * (np.random.rand() - 0.5),
                    axis=self.rotation_axis)

                # Update the zero deg direction
                self.deg0direction = np.dot(rotMat, self.deg0direction)

                # Step 4 : change the control system information
                self.control_location = target

                print("Motor moved to {:.5f} degree".format(np.deg2rad(self.control_location)))
                motion_time = delta / self.control_speed
                return motion_time, rotMat

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):

                    rotMat1 = util.get_rotmat_around_axis(
                        angleRadian=self.control_backlash + delta + self.res * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)

                    self.deg0direction = np.dot(rotMat1, self.deg0direction)

                    rotMat2 = util.get_rotmat_around_axis(
                        angleRadian=- self.control_backlash + self.res * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)
                    self.deg0direction = np.dot(rotMat2, self.deg0direction)

                    self.control_location = target
                    print("Motor moved to {:.5f} degree".format(np.deg2rad(self.control_location)))
                    motion_time = (2 * self.control_backlash + delta) / self.control_speed
                    return motion_time, np.dot(rotMat2, rotMat1)

                else:
                    print("The target location {:.2f} rad plus backlash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target location {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

        if get_motion_time:
            return 0

    def user_get_position(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False


class Breadboard:
    def __init__(self, hole_num_x, hole_num_z, thickness=12.7e3, gauge='metric'):
        self.holes_top = np.zeros((hole_num_x, hole_num_z, 3))
        self.holes_top[:, :, 1] = np.arange(hole_num_x)[:, np.newaxis]
        self.holes_top[:, :, 2] = np.arange(hole_num_z)[np.newaxis, :]
        if gauge == 'metric':
            self.holes_top *= 25e3
        elif gauge == 'imperial':
            self.holes_top *= 25.4e3

        self.holes_bottom = np.copy(self.holes_top)
        self.holes_top[:, :, 0] += thickness

        self.normal = np.array([1, 0, 0], dtype=np.float64)

    def shift(self, displacement):
        self.holes_top += displacement[np.newaxis, np.newaxis, :]
        self.holes_bottom += displacement[np.newaxis, np.newaxis, :]

    def rotate(self, rot_mat):
        self.holes_top = np.dot(self.holes_top, rot_mat)
        self.holes_bottom = np.dot(self.holes_top, rot_mat)
        self.normal = np.dot(rot_mat, self.normal)

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))


class AdaptorPlate:
    def __init__(self, height=10e3, dimension=None, color='black'):
        if dimension is None:
            dimension = (10e4, 10e4)

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface

        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.zeros(3, dtype=np.float64)  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])

        self.color = color  # For visualization

    def shift(self, displacement):
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)

        self.boundary = np.dot(self.boundary, rot_mat.T)

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))


class L_Bracket:
    def __init__(self, height=10e3, dimension=None, color='black'):
        if dimension is None:
            dimension = (10e4, 10e4)

        self.top_mount_dir = np.array([0, 1.0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface

        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.zeros(3)  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])

        self.color = color  # For visualization

    def shift(self, displacement):
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)

        self.boundary = np.dot(self.boundary, rot_mat)

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))


def install_motors_on_motor_or_adaptors(motor_tower, motor_or_adaptor):
    """
    :param motor_or_adaptor:
    :param motor_tower:
    :return:
    """
    # Step 2 move the motor such that the center of the bottom mounting surface of the first motor is
    # the same as the top mounting surface of the new motor or adaptor.
    displacement = np.copy(motor_or_adaptor.top_mount_pos - motor_tower[0].bottom_mount_pos)
    for motor in motor_tower:
        motor.shift(displacement=displacement)

    # Add the new object to the motor-tower
    motor_tower = [motor_or_adaptor, ] + motor_tower
    return motor_tower


def install_motors_on_breadboard(motor_stack, breadboard, diag_hole_idx1, diag_hole_idx2):
    """
    Calculate the location to install the motor stack

    :param motor_stack:
    :param breadboard:
    :param diag_hole_idx1:
    :param diag_hole_idx2:
    :return:
    """
    # Get the location where to install the motor stack
    position = breadboard.holes_top[diag_hole_idx1[0], diag_hole_idx1[1]]
    position += breadboard.holes_top[diag_hole_idx2[0], diag_hole_idx2[1]]
    position /= 2.

    displacement = position - motor_stack[0].bottom_mount_pos

    for motor in motor_stack:
        motor.shift(displacement=displacement)

    return motor_stack


def get_motors_with_model_for_axis(model, rot_center_height=70e3, color='k', axis='x'):
    if model == "XA10A":
        print("Create a XA10A motor, moving along x axis.")
        motor_obj = xyMotor(upperLim=12.5 * 1000,
                            lowerLim=-12.5 * 1000,
                            res=1,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[100e3, 100e3],
                            height=30e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "UTS100CC":
        print("Create a {} motor, moving along x axis.".format(model))
        motor_obj = xyMotor(upperLim=50 * 1000,
                            lowerLim=-50 * 1000,
                            res=2,
                            backlash=100,
                            speed_um_per_ps=2 * 1000 / 1e12,
                            dimension=[100e3, 100e3],
                            height=32e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

    elif model == "XA10A-L101":
        print("Create a XA10A motor, moving along x axis.")
        motor_obj = xyMotor(upperLim=50 * 1000,
                            lowerLim=-50 * 1000,
                            res=2,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[190e3, 100e3],
                            height=50e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass


    elif model == "XA07A":
        print("Create a XA10A motor, moving along x axis.")
        motor_obj = xyMotor(upperLim=10 * 1000,
                            lowerLim=-10 * 1000,
                            res=1,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[70e3, 70e3],
                            height=21e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "ABL1000WB":
        print("Create a ABL1000WB motor, moving along x axis.")
        motor_obj = xyMotor(upperLim=25 * 1000,
                            lowerLim=-25 * 1000,
                            res=0.002,
                            backlash=1,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[307e3, 185e3],
                            height=75e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            print("Warning, cannot create ABL1000WB along y axis automatically.")
            print("Please create this motor manually.")
        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "RA10A":
        print("Create a {} motor, rotating around y axis.".format(model))
        motor_obj = RotationMotor(upperLim=10 * np.pi,
                                  lowerLim=-10 * np.pi,
                                  res=np.deg2rad(0.002),
                                  backlash=0.03,
                                  speed_rad_per_ps=np.deg2rad(0.1) / 1e12,
                                  dimension=[100e3, 100e3],
                                  height=60e3,
                                  color=color)

    elif model == "RA05A":
        print("Create a {} motor, rotating around y axis.".format(model))
        motor_obj = RotationMotor(upperLim=np.deg2rad(180),
                                  lowerLim=-np.deg2rad(-180),
                                  res=np.deg2rad(0.002),
                                  backlash=0.03,
                                  speed_rad_per_ps=np.deg2rad(0.1) / 1e12,
                                  dimension=[100e3, 100e3],
                                  height=60e3,
                                  color=color)
        if axis == 'x':
            print("Rotate motor to rotate around x axis")
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == 'y':
            pass
        elif axis == "z":
            print("Rotate motor to rotate around z axis")
            rot_mat = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [-1, 0, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "ZA10A":
        print("Create a XA10A motor, moving along y axis.")
        motor_obj = zMotor(upperLim=7e3,
                           lowerLim=7e3,
                           res=1,
                           backlash=100,
                           speed_um_per_ps=1 * 1000 / 1e12,
                           dimension=[100e3, 100e3],
                           height=30e3,
                           color=color)

    elif model == "SA07A":
        print("Create a SA07A motor, rotating around z axis.")
        motor_obj = SwivalMotor(upperLim=np.deg2rad(5),
                                lowerLim=-np.deg2rad(-5),
                                res=np.deg2rad(0.000756),
                                backlash=0.05,
                                speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                dimension=[70e3, 70e3],
                                rot_center_height=rot_center_height,
                                height=26e3,
                                color=color)
        if axis == 'x':
            print("Rotate motor to rotate around x axis")

            rot_mat = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == 'y':
            print("Rotate motor to rotate around y axis")

            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == "z":
            pass

    elif model == "SA05A-R2S01":
        print("Create a {} motor pair. The normal is pointing along y".format(model))
        motor_obj1 = SwivalMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(-5),
                                 res=np.deg2rad(0.002126),
                                 backlash=0.05,
                                 speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                 dimension=[50e3, 50e3],
                                 rot_center_height=68,
                                 height=18e3,
                                 color=color)
        motor_obj2 = SwivalMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(-5),
                                 res=np.deg2rad(0.002126),
                                 backlash=0.05,
                                 speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                 dimension=[50e3, 50e3],
                                 rot_center_height=50,
                                 height=18e3,
                                 color=color)
        rot_mat = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])
        motor_obj2.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(motor_obj2.bottom_mount_pos))
        motor_obj = install_motors_on_motor_or_adaptors(motor_tower=[motor_obj2, ],
                                                        motor_or_adaptor=motor_obj1)
        if axis == 'x':
            print("Rotate motor to face x axis")
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            motor_obj1.rotate_wrt_point(rot_mat=rot_mat,
                                        ref_point=np.copy(motor_obj1.bottom_mount_pos))
            motor_obj2.rotate_wrt_point(rot_mat=rot_mat,
                                        ref_point=np.copy(motor_obj1.bottom_mount_pos))
        elif axis == 'y':
            print("Warning, cannot create {} along y axis automatically.".format(model))
            print("Please create this motor manually.")
        elif axis == "z":
            print("Warning, cannot create {} along z axis automatically.".format(model))
            print("Please create this motor manually.")

    else:
        print("Motor with model {} has not been defined in this simulator.".format(model))
        motor_obj = 0

    return motor_obj


class CrystalTower_x_y_theta_chi:
    def __init__(self, crystal, crystal_loc):
        # Create the instance of each motors and adaptors
        self.adaptor1 = AdaptorPlate(height=55e3, dimension=[100e3, 100e3])
        self.x = get_motors_with_model_for_axis(model="XA10A")
        self.y = get_motors_with_model_for_axis(model="ZA10A")
        self.adaptor2 = AdaptorPlate(height=10e3, dimension=[70e3, 70e3])
        self.th = get_motors_with_model_for_axis(model="RA10A")
        self.chi = get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="z")
        self.adaptor3 = AdaptorPlate(height=39e3, dimension=[70e3, 70e3])
        self.optics = crystal

        # Create the list of all components in this tower
        self.optics.shift(displacement=self.adaptor3.top_mount_pos - np.copy(crystal_loc))
        self.all_obj = [self.adaptor3, self.optics]

        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.chi)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.th)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor2)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor1)

        # Define a holder that contains all the motor objects
        self.all_motors = [self.x, self.y, self.th, self.chi]
        if self.optics.type == "Channel cut with two surfaces":
            self.obj_to_plot = [self.x, self.y, self.th, self.chi, ] + self.optics.crystal_list
            self.all_optics = self.optics.crystal_list
        elif self.optics.type == "Crystal: Bragg Reflection":
            self.obj_to_plot = [self.x, self.y, self.th, self.chi, self.optics]
            self.all_optics = [self.optics, ]

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """
        # Shift all the motors and crystals with it
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def th_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.th.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.rotation_center)
        return motion_time

    def chi_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.chi.user_move_abs(target=target)
        for item in self.all_obj[6:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.rotation_center)
        return motion_time


class Tower_x_y_pi:
    def __init__(self, mirror):
        # Create the instance of each motors and adaptors
        self.adaptor1 = AdaptorPlate(height=55e3, dimension=[100e3, 100e3])
        self.x = get_motors_with_model_for_axis(model="XA10A")
        self.y = get_motors_with_model_for_axis(model="ZA10A")
        self.adaptor2 = AdaptorPlate(height=10e3, dimension=[70e3, 70e3])
        self.pi = get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="x")
        self.adaptor3 = AdaptorPlate(height=39e3, dimension=[70e3, 70e3])
        self.optics = mirror

        # Create the list of all components in this tower
        self.optics.shift(displacement=self.adaptor3.top_mount_pos - np.copy(mirror.surface_point))
        self.all_obj = [self.adaptor3, self.optics]

        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.pi)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor2)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor1)

        # Define a holder that contains all the motor objects
        self.all_motors = [self.x, self.y, self.pi]
        self.obj_to_plot = [self.x, self.y, self.pi, self.optics]
        self.all_optics = [self.optics, ]

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def pi_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.pi.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center)
        return motion_time


class CrystalTower_miniSD_Scan:
    """
    This is simple implementation of the delay scan tower of the miniSD table.
    It is composed of an air-bearing stage and a few other stages.
    """

    def __init__(self,
                 channelCut1, crystal_loc1,
                 channelCut2, crystal_loc2):
        # Create the instance of each motors and adaptors
        self.adaptor1 = AdaptorPlate(height=10e3, dimension=[307e3, 270e3])
        self.delay = get_motors_with_model_for_axis(model="XA10A")
        self.adaptor2 = AdaptorPlate(height=10e3, dimension=[233e3, 288e3])
        self.th1 = get_motors_with_model_for_axis(model="RA10A")
        self.th2 = get_motors_with_model_for_axis(model="RA10A")
        self.chi = get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="z")
        self.x = get_motors_with_model_for_axis(model="XA07A")
        self.adaptor3 = AdaptorPlate(height=39e3, dimension=[70e3, 70e3])
        self.adaptor4 = AdaptorPlate(height=44e3, dimension=[70e3, 70e3])
        self.optics1 = channelCut1
        self.optics2 = channelCut2

        # Create the list of all components in this tower
        self.optics1.shift(displacement=self.adaptor3.top_mount_pos - crystal_loc1)
        self.optics2.shift(displacement=self.adaptor4.top_mount_pos - crystal_loc2)

        self.tower1 = [self.adaptor3, self.optics1]
        self.tower1 = install_motors_on_motor_or_adaptors(motor_tower=self.tower1, motor_or_adaptor=self.x)
        self.tower1 = install_motors_on_motor_or_adaptors(motor_tower=self.tower1, motor_or_adaptor=self.th1)

        self.tower2 = [self.adaptor4, self.optics2]
        self.tower2 = install_motors_on_motor_or_adaptors(motor_tower=self.tower2, motor_or_adaptor=self.chi)
        self.tower2 = install_motors_on_motor_or_adaptors(motor_tower=self.tower2, motor_or_adaptor=self.th2)

        # Adjust the relative position between the two towers with respect to the adaptor2
        displacement = np.array([0., 0., 89e3]) + self.adaptor2.top_mount_pos - self.tower1[0].bottom_mount_pos
        for item in self.tower1:
            # print(item)
            item.shift(displacement=displacement)

        displacement = np.array([0., 0., -89e3]) + self.adaptor2.top_mount_pos - self.tower2[0].bottom_mount_pos
        for item in self.tower2:
            item.shift(displacement=displacement)

        self.all_obj = [self.adaptor2, ] + self.tower1 + self.tower2
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.delay)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor1)

        # Define a holder that contains all the motor objects
        self.all_motors = [self.delay, self.th1, self.th2, self.chi, self.x]
        self.obj_to_plot = ([self.delay, self.th1, self.th2, self.chi, self.x] +
                            self.optics1.crystal_list + self.optics2.crystal_list)
        self.all_optics = self.optics1.crystal_list + self.optics2.crystal_list

    def x_umv(self, target):
        motion_time, displacement = self.delay.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def th1_umv(self, target):
        motion_time, rotMat = self.th1.user_move_abs(target=target)
        for item in self.tower1[1:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th1.rotation_center)
        return motion_time

    def th2_umv(self, target):
        motion_time, rotMat = self.th2.user_move_abs(target=target)
        for item in self.tower2[1:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th2.rotation_center)
        return motion_time

    def chi_umv(self, target):
        motion_time, rotMat = self.chi.user_move_abs(target=target)
        for item in self.tower2[2:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.chi.rotation_center)
        return motion_time

    def x1_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.tower1[1:]:
            item.shift(displacement=displacement)
        return motion_time


class Grating_tower:
    """
    This is a temporary implementation for the TG project.
    For more general purpose usage, one needs to use something different.
    """

    def __init__(self,
                 grating_1,
                 grating_m1):
        # Create the instance of each motor and adaptors
        self.adaptor1 = AdaptorPlate(height=14e3, dimension=[70e3, 70e3])
        self.x = get_motors_with_model_for_axis(model="XA07A")
        self.adaptor2 = L_Bracket(height=159e3, dimension=[70e3, 70e3])
        self.y = get_motors_with_model_for_axis(model="XA07A", axis='y')
        self.pi = get_motors_with_model_for_axis(model='RA05A', axis='x')
        (self.roll, self.yaw) = get_motors_with_model_for_axis(model='SA05A-R2S01', axis='x')
        self.adaptor3 = AdaptorPlate(height=50e3, dimension=[50e3, 50e3])

        # Install adaptors and motors
        # Rotate the adaptor 3 to install it on the roll yaw motor
        rot_mat = np.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])
        self.adaptor3.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.adaptor3.bottom_mount_pos))

        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=[self.adaptor3, ], motor_or_adaptor=self.yaw)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.roll)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.pi)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor2)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor1)

        # Install the two gratings
        displacement = self.adaptor3.top_mount_pos - grating_1.surface_point
        grating_1.shift(displacement=displacement)
        displacement = self.adaptor3.top_mount_pos - grating_m1.surface_point
        grating_m1.shift(displacement=displacement)
        self.grating_1 = grating_1
        self.grating_m1 = grating_m1

        # Add the gratings to the grating tower
        self.all_obj += [self.grating_1, self.grating_m1]

        self.all_motors = [self.x, self.y, self.pi, self.roll, self.yaw]
        self.obj_to_plot = [self.x, self.y, self.pi, self.roll, self.yaw, self.grating_1]
        self.all_optics = [self.grating_1, self.grating_m1]

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umv(self, target):
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[4:]:
            item.shift(displacement=displacement)
        return motion_time

    def pi_umv(self, target):
        motion_time, rotMat = self.pi.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center)
        return motion_time

    def roll_umv(self, target):
        motion_time, rotMat = self.roll.user_move_abs(target=target)
        for item in self.all_obj[6:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center)
        return motion_time

    def yaw_umv(self, target):
        motion_time, rotMat = self.yaw.user_move_abs(target=target)
        for item in self.all_obj[6:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center)
        return motion_time


class Mirror_tower1:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self, mirror):
        # Create the instance of each motors
        self.adapter1 = AdaptorPlate(height=20e3, dimension=[100e3, 200e3])
        self.z = get_motors_with_model_for_axis(model="XA10A-L101", axis='z')
        self.x = get_motors_with_model_for_axis(model="XA10A", axis='x')
        self.adapter2 = AdaptorPlate(height=10e3, dimension=[100e3, 250e3])
        self.roll = get_motors_with_model_for_axis(model="SA07A", rot_center_height=96e3, axis="z")
        self.adapter3 = L_Bracket(height=10e3, dimension=[70e3, 70e3])
        self.yaw = get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="z")
        self.adapter4 = AdaptorPlate(height=70e3, dimension=[70e3, 70e3])
        self.optics = mirror

        # Install mirror to the adaptor 4
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.pi / 2, axis=np.array([0., 0., 1]))
        self.optics.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.optics.surface_point))
        self.adapter4.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.adapter4.bottom_mount_pos))
        self.yaw.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.yaw.bottom_mount_pos))

        displacement = self.adapter4.top_mount_pos + np.array([86.3e3 + 5e3, 0, 0]) - self.optics.surface_point
        self.optics.shift(displacement=displacement)

        self.all_obj = [self.adapter4, self.optics]
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.yaw)

        # Adjust the dimension of adaptor 3
        self.adapter3.top_mount_pos += np.array([0, 75.35e3, 0])
        self.adapter3.top_mount_pos[0] = 19e3
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adapter3)

        # Install other components
        # Adjust the dimension of adaptor 2
        self.adapter2.bottom_mount_pos[2] = -75e3
        self.adapter2.top_mount_pos[2] = 125e3 - 35e3
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.roll)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adapter2)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.z)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adapter1)

        self.all_motors = [self.z, self.x, self.roll, self.yaw]
        self.obj_to_plot = [self.z, self.x, self.roll, self.yaw, self.optics]
        self.all_optics = [self.optics, ]

    def z_umv(self, target):
        motion_time, displacement = self.z.user_move_abs(target=target)
        for item in self.all_obj[2]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[3]:
            item.shift(displacement=displacement)
        return motion_time

    def roll_umv(self, target):
        motion_time, rotMat = self.roll.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center)
        return motion_time

    def yaw_umv(self, target):
        motion_time, rotMat = self.yaw.user_move_abs(target=target)
        for item in self.all_obj[7:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center)
        return motion_time


class Mirror_tower2:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self,
                 mirror):
        # Create the instance of each motors
        self.adapter1 = AdaptorPlate(height=20e3, dimension=[100e3, 200e3])
        self.z = get_motors_with_model_for_axis(model="XA10A-L101", axis='z')
        self.x = get_motors_with_model_for_axis(model="XA10A", axis='x')
        self.adapter2 = AdaptorPlate(height=10e3, dimension=[100e3, 250e3])
        self.roll = get_motors_with_model_for_axis(model="SA07A", rot_center_height=96e3, axis="z")
        self.adapter3 = L_Bracket(height=10e3, dimension=[70e3, 70e3])
        self.yaw = get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="z")
        self.adapter4 = AdaptorPlate(height=70e3, dimension=[70e3, 70e3])
        self.optics = mirror

        # Install mirror to the adaptor 4
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.pi / 2, axis=np.array([0., 0., 1]))
        self.optics.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.optics.surface_point))
        self.adapter4.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.adapter4.bottom_mount_pos))
        self.yaw.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.yaw.bottom_mount_pos))

        displacement = self.adapter4.top_mount_pos + np.array([86.3e3 + 5e3, 0, 0]) - self.optics.surface_point
        self.optics.shift(displacement=displacement)

        self.all_obj = [self.adapter4, self.optics]
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.yaw)

        # Adjust the dimension of adaptor 3
        self.adapter3.top_mount_pos += np.array([0, 75.35e3, 0])
        self.adapter3.top_mount_pos[0] = 19e3
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adapter3)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.roll)

        # Everything above is copied from the mirror tower 1 class
        # Here I need to rotate the components to get the correct geometry
        # First rotate around the y axis
        rot_mat = np.array([[1.0, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1], ])
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=self.roll.bottom_mount_pos)

        # Adjust the dimension of adaptor 2
        self.adapter2.bottom_mount_pos[2] = -75e3
        self.adapter2.top_mount_pos[2] = 125e3 - 35e3
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adapter2)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.z)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adapter1)

        self.all_motors = [self.z, self.x, self.roll, self.yaw]
        self.obj_to_plot = [self.z, self.x, self.roll, self.yaw, self.optics]
        self.all_optics = [self.optics, ]

    def z_umv(self, target):
        motion_time, displacement = self.z.user_move_abs(target=target)
        for item in self.all_obj[2]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[3]:
            item.shift(displacement=displacement)
        return motion_time

    def roll_umv(self, target):
        motion_time, rotMat = self.roll.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center)
        return motion_time

    def yaw_umv(self, target):
        motion_time, rotMat = self.yaw.user_move_abs(target=target)
        for item in self.all_obj[7:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center)
        return motion_time


class Silicon_tower:
    def __init__(self, crystal):
        # Create the instance of each motors and adaptors
        self.adaptor1 = AdaptorPlate(height=22.7e3, dimension=[100e3, 100e3])
        self.y = get_motors_with_model_for_axis(model="ZA10A")
        self.z = get_motors_with_model_for_axis(model="XA10A", axis='z')
        self.x = get_motors_with_model_for_axis(model="XA10A")

        self.adaptor2 = AdaptorPlate(height=10e3, dimension=[75e3, 200e3])
        self.adaptor2.bottom_mount_pos[2] = 100e3 - 35e3
        self.adaptor2.top_mount_pos[2] = -(100e3 - 35e3)
        self.adaptor2.top_mount_pos[0] = 0.0

        self.adaptor3 = AdaptorPlate(height=245e3, dimension=[70e3, 70e3])
        tilt_angle3 = np.deg2rad(10)
        self.adaptor3.top_mount_dir = np.array([np.cos(np.deg2rad(tilt_angle3)),
                                                0, np.sin(np.deg2rad(tilt_angle3))])

        (self.roll, self.yaw) = get_motors_with_model_for_axis(model='SA05A-R2S01', axis='y')
        self.adaptor4 = AdaptorPlate(height=50e3, dimension=[20e3, 20e3])
        self.optics = crystal
        # print(crystal.boundary)
        # print(crystal.normal)

        # Install the crystal on the top of the first adaptor
        self.optics.shift(displacement=self.adaptor4.top_mount_pos - self.optics.surface_point)
        self.all_obj = [self.adaptor4, self.optics]
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.yaw)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.roll)

        # Rotate around the x axis such that it matches the angle
        rot_mat = np.array([[np.cos(tilt_angle3), 0, np.sin(tilt_angle3)],
                            [0, 1, 0],
                            [-np.sin(tilt_angle3), 0, np.cos(tilt_angle3)]])
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=self.roll.bottom_mount_pos)

        # Install the setup on adaptor 3
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor3)

        # Rotate everything around the z axis by 180 deg
        rot_mat = np.eye(3)
        rot_mat[0, 0] = -1
        rot_mat[1, 1] = -1
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=self.adaptor3.bottom_mount_pos)

        # Install the other components
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor2)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.z)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor1)

        self.all_motors = [self.y, self.z, self.x, self.roll, self.yaw]
        self.obj_to_plot = [self.y, self.z, self.x, self.roll, self.yaw, self.optics]
        self.all_optics = [self.optics, ]

    def y_umv(self, target):
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def z_umv(self, target):
        motion_time, displacement = self.z.user_move_abs(target=target)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[4:]:
            item.shift(displacement=displacement)
        return motion_time

    def roll_umv(self, target):
        motion_time, rotMat = self.roll.user_move_abs(target=target)
        for item in self.all_obj[7:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center)
        return motion_time

    def yaw_umv(self, target):
        motion_time, rotMat = self.yaw.user_move_abs(target=target)
        for item in self.all_obj[8:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center)
        return motion_time


class TG_Sample_tower:
    def __init__(self, sample, yag_sample, yag1, yag2, yag3):

        # Create the instance of each motors and adaptors
        self.adaptor1 = AdaptorPlate(height=10e3, dimension=[100e3, 100e3])
        self.x = get_motors_with_model_for_axis(model="UTS100CC")
        self.adaptor2 = AdaptorPlate(height=24.3e3, dimension=[100e3, 100e3])
        self.y = get_motors_with_model_for_axis(model="ZA10A")
        self.z = get_motors_with_model_for_axis(model="XA10A", axis='z')
        self.adaptor3 = AdaptorPlate(height=30e3, dimension=[100e3, 100e3])
        self.adaptor4 = L_Bracket(height=35e3, dimension=[50e3, 50e3])
        self.th = get_motors_with_model_for_axis(model="RA05A", axis='y')
        self.adaptor5 = AdaptorPlate(height=72.7e3, dimension=[10e3, 10e3])

        self.sample = sample
        self.yag_sample = yag_sample
        self.yag1 = yag1
        self.yag2 = yag2
        self.yag3 = yag3

        # Currently the samples are pointing z axis, rotate them such that they are facing y axis
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.pi / 2, axis=np.array([0, 1, 0], dtype=np.float64))
        self.sample.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.sample.surface_point))
        self.yag_sample.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.yag_sample.surface_point))

        # Install sample and sample_yag on the adaptor 5
        displacement = np.copy(self.adaptor5.top_mount_pos - self.sample.surface_point)
        self.sample.shift(displacement=displacement)
        self.yag_sample.shift(displacement=displacement + np.array([0., 10e3, 0]))

        # Rotate around the z axis to make the component horizontal
        self.all_obj = [self.adaptor5, self.yag_sample, self.sample]
        self.all_obj = install_motors_on_motor_or_adaptors(self.all_obj, self.th)
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.pi / 2, axis=np.array([0, 0, 1], dtype=np.float64))
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.th.bottom_mount_pos))

        self.all_obj = install_motors_on_motor_or_adaptors(self.all_obj, self.adaptor4)

        rot_mat = util.get_rotmat_around_axis(angleRadian=np.deg2rad(-5), axis=np.array([1, 0, 0], dtype=np.float64))
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.adaptor4.bottom_mount_pos))

        # Assemble the small sample tower to the big sample and yag tower
        displacement = np.array([0, -67.5e3, -70.35e3]) + np.copy(self.adaptor3.top_mount_pos)
        for item in self.all_obj:
            item.shift(displacement=displacement)

        self.yag1.shift(displacement=np.array([35.7e3, 67.5e3, -70.35e3]) + np.copy(self.adaptor3.top_mount_pos))
        self.yag2.shift(displacement=np.array([45.7e3, 67.5e3, -70.35e3]) + np.copy(self.adaptor3.top_mount_pos))
        self.yag3.shift(displacement=np.array([45.7e3, 77.5e3, -70.35e3]) + np.copy(self.adaptor3.top_mount_pos))
        self.all_obj = [self.adaptor3, self.yag1, self.yag2, self.yag3] + self.all_obj

        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.z)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor2)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor1)

        self.all_motors = [self.x, self.y, self.z, self.th]
        self.obj_to_plot = [self.x, self.y, self.z, self.th, self.sample,
                            self.yag_sample, self.yag1, self.yag2, self.yag3]
        self.all_optics = [self.sample, self.yag_sample, self.yag1, self.yag2, self.yag3]

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umv(self, target):
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[4:]:
            item.shift(displacement=displacement)
        return motion_time

    def z_umv(self, target):
        motion_time, displacement = self.z.user_move_abs(target=target)
        for item in self.all_obj[4:]:
            item.shift(displacement=displacement)
        return motion_time

    def th_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.th.user_move_abs(target=target)
        for item in self.all_obj[11:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=np.copy(self.th.rotation_center))
        return motion_time
