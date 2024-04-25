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
        self.bottom_mount_pos = np.array([0, 0, 0, ])  # The center of the top mounting surface

        self.color = color  # For visualization

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.top_mount_pos += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.top_mount_pos = np.ascontiguousarray(rot_mat.dot(self.top_mount_pos))
        self.motion_dir = np.ascontiguousarray(rot_mat.dot(self.motion_dir))

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

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

                print("Motor moved to {:.2f} um".format(self.control_location))

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
                    print("Motor moved to {:.2f} um".format(self.control_location))

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
        self.bottom_mount_pos = np.array([0, 0, 0, ])  # The center of the top mounting surface

        self.color = color  # For visualization

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.top_mount_pos += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.top_mount_pos = np.ascontiguousarray(rot_mat.dot(self.top_mount_pos))
        self.motion_dir = np.ascontiguousarray(rot_mat.dot(self.motion_dir))

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

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

                print("Motor moved to {:.2f} um".format(self.control_location))
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
                    print("Motor moved to {:.2f} um".format(self.control_location))

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
        self.bottom_mount_pos = np.array([0, 0, 0, ])  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])
        self.color = color

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.rotation_center += np.copy(displacement)
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.deg0direction = np.ascontiguousarray(rot_mat.dot(self.deg0direction))
        self.rotation_center = np.ascontiguousarray(rot_mat.dot(self.rotation_center))
        self.rotation_axis = np.ascontiguousarray(rot_mat.dot(self.rotation_axis))

        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

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

                print("Motor moved to {:.2f} deg".format(np.deg2rad(self.control_location)))

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
                    print("Motor moved to {:.2f} deg".format(np.deg2rad(self.control_location)))
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
        self.bottom_mount_pos = np.array([0, 0, 0, ])  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])
        self.color = color

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.rotation_center += np.copy(displacement)
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.deg0direction = np.ascontiguousarray(rot_mat.dot(self.deg0direction))
        self.rotation_center = np.ascontiguousarray(rot_mat.dot(self.rotation_center))
        self.rotation_axis = np.ascontiguousarray(rot_mat.dot(self.rotation_axis))

        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

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

                print("Motor moved to {:.2f} degree".format(np.deg2rad(self.control_location)))
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
                    print("Motor moved to {:.2f} degree".format(np.deg2rad(self.control_location)))
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

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
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
        self.bottom_mount_pos = np.array([0, 0, 0, ])  # The center of the top mounting surface

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

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
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

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface

        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.array([0, 0, 0, ])  # The center of the top mounting surface

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

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))


def install_motors_on_motor_or_adaptors(motor_tower, motor_or_adaptor, rot_mat=np.eye(3)):
    """
    This function tries to solve the challenge of imposing geometric relation
    between different motors.

    This will not solve all the problems.
    However, I think it will solve at least some issues.

    Currently, this function tries to install the motor A
    at the center of the mounting surface of motor B.

    :param motor_or_adaptor:
    :param motor_tower:
    :return:
    """
    # Step 1 rotate the motor tower
    ref_point = np.copy(motor_tower[0].bottom_mounting_point)
    for motor in motor_tower:
        motor.rotate_wrt_point(rot_mat=rot_mat, ref_point=ref_point)

    # Step 2 move the motor such that the center of the bottom mounting surface of the first motor is
    # the same as the top mounting surface of the new motor or adaptor.
    displacement = motor_or_adaptor.top_mount_pos - motor_tower[0].bottom_mount_pos
    for motor in motor_tower:
        motor.shift(displacement)

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
    position = breadboard.holes_top[diag_hole_idx1[0], diag_hole_idx1[0]]
    position += breadboard.holes_top[diag_hole_idx2[0], diag_hole_idx2[0]]
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
            rot_mat = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    if model == "XA07A":
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
            rot_mat = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    if model == "ABL1000WB":
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
        print("Create a RA10A motor, rotating around z axis.")
        motor_obj = RotationMotor(upperLim=np.deg2rad(180),
                                  lowerLim=-np.deg2rad(-180),
                                  res=np.deg2rad(0.002),
                                  backlash=0.03,
                                  speed_rad_per_ps=np.deg2rad(0.1) / 1e12,
                                  dimension=[100e3, 100e3],
                                  height=60e3,
                                  color=color)
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
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == 'y':
            rot_mat = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == "z":
            pass
    else:
        print("Motor with model {} has not been defined in this simulator.".format(model))
        motor_obj = 0

    return motor_obj


# --------------------------------------------------------------------
#    Here, I define a few commonly used motor composition.
#    Even though they do not have any scientific generality
#    my gut feeling is that they should have a long enough
#    lifetime that deserve such a position in the main
#    body of this simulation package.
# --------------------------------------------------------------------
class CrystalTower_x_y_theta_chi:
    def __init__(self, channelCut, crystal_loc):
        # Create the instance of each motors and adaptors
        self.adaptor1 = AdaptorPlate(height=55e3, dimension=[100e3, 100e3])
        self.x = get_motors_with_model_for_axis(model="XA10A")
        self.y = get_motors_with_model_for_axis(model="ZA10A")
        self.adaptor2 = AdaptorPlate(height=10e3, dimension=[70e3, 70e3])
        self.th = get_motors_with_model_for_axis(model="RA10A")
        self.chi = get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="z")
        self.adaptor3 = AdaptorPlate(height=39e3, dimension=[70e3, 70e3])
        self.optics = channelCut

        # Create the list of all components in this tower
        self.optics.shift(displacement=self.adaptor3.top_mount_pos - crystal_loc)
        self.all_obj = [self.adaptor3, self.optics]

        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.chi)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.th)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor2)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor1)

        # Define a holder that contains all the motor objects
        self.all_motor_obj = [self.x, self.y, self.th, self.chi]

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """
        # Shift all the motors and crystals with it
        motion_time, displacement = self.x.user_move_abs(target=target, getMotionTime=True)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, displacement = self.y.user_move_abs(target=target, getMotionTime=True)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def th_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.th.user_move_abs(target=target, getMotionTime=True)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.rotation_center)
        return motion_time

    def chi_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.chi.user_move_abs(target=target, getMotionTime=True)
        for item in self.all_obj[6:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.rotation_center)
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
            item.shift(displacement=displacement)

        displacement = np.array([0., 0., -89e3]) + self.adaptor2.top_mount_pos - self.tower2[0].bottom_mount_pos
        for item in self.tower2:
            item.shift(displacement=displacement)

        self.all_obj = [self.adaptor2, ] + self.tower1 + self.tower2
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.delay)
        self.all_obj = install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.adaptor1)

        # Define a holder that contains all the motor objects
        self.all_motor_obj = [self.delay, self.th1, self.th2, self.chi, self.x]

    def x_umv(self, target):
        motion_time, displacement = self.delay.user_move_abs(target=target, getMotionTime=True)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def th1_umv(self, target):
        motion_time, rotMat = self.th1.user_move_abs(target=target, getMotionTime=True)
        for item in self.tower1[1:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th1.rotation_center)
        return motion_time

    def th2_umv(self, target):
        motion_time, rotMat = self.th2.user_move_abs(target=target, getMotionTime=True)
        for item in self.tower2[1:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th2.rotation_center)
        return motion_time

    def chi_umv(self, target):
        motion_time, rotMat = self.chi.user_move_abs(target=target, getMotionTime=True)
        for item in self.tower2[2:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.chi.rotation_center)
        return motion_time

    def x1_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target, getMotionTime=True)
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
        # Create the instance of each motors

        self.x = xyMotor(upperLim=12.5 * 1000,
                         lowerLim=-12.5 * 1000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = xyMotor(upperLim=25000,
                         lowerLim=-25000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.pi = RotationMotor(upperLim=np.deg2rad(360),
                                lowerLim=-np.deg2rad(360),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                speed_rad_per_ps=0.01 / 1e12, )

        self.roll = RotationMotor(upperLim=np.deg2rad(5),
                                  lowerLim=-np.deg2rad(5),
                                  res=1e-6,
                                  backlash=np.deg2rad(-0.005),
                                  speed_rad_per_ps=0.1 / 1e12, )

        self.yaw = RotationMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=1e-6,
                                 backlash=np.deg2rad(-0.005),
                                 speed_rad_per_ps=0.1 / 1e12, )

        self.grating_1 = grating_1
        self.grating_m1 = grating_m1

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.x.motion_dir = np.zeros(3, dtype=np.float64)
        self.x.motion_dir[1] = 1.0  #
        # Define the installation location of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.motion_dir = np.zeros(3, dtype=np.float64)
        self.y.motion_dir[0] = 1.0  #
        # Define the installation location of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.pi.deg0direction = np.zeros(3, dtype=np.float64)
        self.pi.deg0direction[1] = 1.0  #
        self.pi.rotation_axis = np.zeros(3, dtype=np.float64)
        self.pi.rotation_axis[0] = 1.0
        self.pi.rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation location of the x stage
        pi_stage_center = np.zeros(3, dtype=np.float64)
        pi_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage + the height of the y stage
        self.pi.shift(displacement=pi_stage_center)

        self.roll.deg0direction = np.zeros(3, dtype=np.float64)
        self.roll.deg0direction[0] = 1.0  #
        self.roll.rotation_axis = np.zeros(3, dtype=np.float64)
        self.roll.rotation_axis[2] = 1.0
        self.roll.rotation_center = np.zeros(3, dtype=np.float64)
        self.roll.rotation_center[1] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation location of the x stage
        roll_stage_center = np.zeros(3, dtype=np.float64)
        roll_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.roll.shift(displacement=roll_stage_center)

        self.yaw.deg0direction = np.zeros(3, dtype=np.float64)
        self.yaw.deg0direction[0] = 1.0  #
        self.yaw.rotation_axis = np.zeros(3, dtype=np.float64)
        self.yaw.rotation_axis[2] = 1.0
        self.yaw.rotation_center = np.zeros(3, dtype=np.float64)
        self.yaw.rotation_center[1] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation location of the x stage
        yaw_stage_center = np.zeros(3, dtype=np.float64)
        yaw_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.roll.shift(displacement=yaw_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        self.grating_1.shift(displacement=crystalSurface)
        self.grating_m1.shift(displacement=crystalSurface)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.motion_dir * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.roll.shift(displacement=displacement, include_boundary=True)
        self.yaw.shift(displacement=displacement, include_boundary=True)
        self.grating_1.shift(displacement=displacement, include_boundary=True)
        self.grating_m1.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.motion_dir * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.roll.shift(displacement=displacement, include_boundary=True)
        self.yaw.shift(displacement=displacement, include_boundary=True)
        self.grating_1.shift(displacement=displacement, include_boundary=True)
        self.grating_m1.shift(displacement=displacement, include_boundary=True)

    def pi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.pi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.pi.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.pi.user_move_abs(target=target, getMotionTime=True)
        self.roll.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center, include_boundary=True)
        self.yaw.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center, include_boundary=True)
        self.grating_1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center,
                                        include_boundary=True)
        self.grating_m1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center,
                                         include_boundary=True)

    def roll_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.roll.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.roll.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.roll.user_move_abs(target=target, getMotionTime=True)
        self.yaw.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center, include_boundary=True)
        self.grating_1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center,
                                        include_boundary=True)
        self.grating_m1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center,
                                         include_boundary=True)

    def yaw_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.yaw.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.yaw.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.yaw.user_move_abs(target=target, getMotionTime=True)
        self.grating_1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center,
                                        include_boundary=True)
        self.grating_m1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center,
                                         include_boundary=True)


class Mirror_tower1:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self,
                 mirror,
                 crystal_loc):
        # Create the instance of each motors

        self.x = xyMotor(upperLim=12.5 * 1000,
                         lowerLim=-12.5 * 1000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = xyMotor(upperLim=25000,
                         lowerLim=-25000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.pi = RotationMotor(upperLim=np.deg2rad(5),
                                lowerLim=-np.deg2rad(5),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                speed_rad_per_ps=0.1 / 1e12, )

        self.optics = mirror

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.x.motion_dir = np.zeros(3, dtype=np.float64)
        self.x.motion_dir[1] = 1.0  #
        # Define the installation location of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.motion_dir = np.zeros(3, dtype=np.float64)
        self.y.motion_dir[0] = 1.0  #
        # Define the installation location of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.pi.deg0direction = np.zeros(3, dtype=np.float64)
        self.pi.deg0direction[0] = 1.0  #
        self.pi.rotation_axis = np.zeros(3, dtype=np.float64)
        self.pi.rotation_axis[2] = 1.0
        self.pi.rotation_center = np.zeros(3, dtype=np.float64)
        self.pi.rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation location of the x stage
        pi_stage_center = np.zeros(3, dtype=np.float64)
        pi_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.pi.shift(displacement=pi_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface += crystal_loc
        self.optics.shift(displacement=crystalSurface)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.motion_dir * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.motion_dir * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def pi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.pi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.pi.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.pi.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center, include_boundary=True)


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
                 mirror,
                 crystal_loc):
        # Create the instance of each motors

        self.z = xyMotor(upperLim=12.5 * 1000,
                         lowerLim=-12.5 * 1000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = xyMotor(upperLim=25000,
                         lowerLim=-25000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.yaw = RotationMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=1e-6,
                                 backlash=np.deg2rad(-0.005),
                                 speed_rad_per_ps=0.1 / 1e12, )

        self.optics = mirror

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.z.motion_dir = np.zeros(3, dtype=np.float64)
        self.z.motion_dir[1] = 1.0  #
        # Define the installation location of the x stage
        z_stage_center = np.zeros(3, dtype=np.float64)
        self.z.shift(displacement=z_stage_center)

        self.y.motion_dir = np.zeros(3, dtype=np.float64)
        self.y.motion_dir[0] = 1.0  #
        # Define the installation location of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.yaw.deg0direction = np.zeros(3, dtype=np.float64)
        self.yaw.deg0direction[0] = 1.0  #
        self.yaw.rotation_axis = np.zeros(3, dtype=np.float64)
        self.yaw.rotation_axis[2] = 1.0
        self.yaw.rotation_center = np.zeros(3, dtype=np.float64)
        self.yaw.rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation location of the x stage
        yaw_stage_center = np.zeros(3, dtype=np.float64)
        yaw_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.yaw.shift(displacement=yaw_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface += crystal_loc
        self.optics.shift(displacement=crystalSurface)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def z_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.z.motion_dir * (target - self.z.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.z.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.yaw.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.motion_dir * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.yaw.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def yaw_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.yaw.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.yaw.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.yaw.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center, include_boundary=True)


class Silicon_tower:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self,
                 crystal,
                 crystal_loc):
        # Create the instance of each motors

        self.x = xyMotor(upperLim=12.5 * 1000,
                         lowerLim=-12.5 * 1000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = xyMotor(upperLim=25000,
                         lowerLim=-25000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.z = xyMotor(upperLim=25000,
                         lowerLim=-25000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.roll = RotationMotor(upperLim=np.deg2rad(360),
                                  lowerLim=-np.deg2rad(360),
                                  res=1e-6,
                                  backlash=np.deg2rad(-0.005),
                                  speed_rad_per_ps=0.01 / 1e12, )

        self.pi = RotationMotor(upperLim=np.deg2rad(5),
                                lowerLim=-np.deg2rad(5),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                speed_rad_per_ps=0.1 / 1e12, )

        self.optics = crystal

        self.all_mostors = [self.x, self.y, self.z, self.roll, self.pi]
        self.all_mostors_and_optics = [self.x, self.y, self.z, self.roll, self.pi, self.optics]

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.z.motion_dir = np.zeros(3, dtype=np.float64)
        self.z.motion_dir[2] = 1.0  #
        # Define the installation location of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.z.shift(displacement=x_stage_center)

        self.x.motion_dir = np.zeros(3, dtype=np.float64)
        self.x.motion_dir[1] = 1.0  #
        # Define the installation location of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.motion_dir = np.zeros(3, dtype=np.float64)
        self.y.motion_dir[0] = 1.0  #
        # Define the installation location of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.roll.deg0direction = np.zeros(3, dtype=np.float64)
        self.roll.deg0direction[1] = 1.0  #
        self.roll.rotation_axis = np.zeros(3, dtype=np.float64)
        self.roll.rotation_axis[0] = 1.0
        self.roll.rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation location of the x stage
        roll_stage_center = np.zeros(3, dtype=np.float64)
        roll_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage + the height of the y stage
        self.roll.shift(displacement=roll_stage_center)

        self.pi.deg0direction = np.zeros(3, dtype=np.float64)
        self.pi.deg0direction[0] = 1.0  #
        self.pi.rotation_axis = np.zeros(3, dtype=np.float64)
        self.pi.rotation_axis[1] = 1.0
        self.pi.rotation_center = np.zeros(3, dtype=np.float64)
        self.pi.rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation location of the x stage
        pi_stage_center = np.zeros(3, dtype=np.float64)
        pi_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.pi.shift(displacement=pi_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface += crystal_loc
        self.optics.shift(displacement=crystalSurface)

    def y_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.y.motion_dir * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.x.shift(displacement=displacement, include_boundary=True)
        self.z.shift(displacement=displacement, include_boundary=True)
        self.roll.shift(displacement=displacement, include_boundary=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def x_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.x.motion_dir * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.z.shift(displacement=displacement, include_boundary=True)
        self.roll.shift(displacement=displacement, include_boundary=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def roll_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.roll.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.roll.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.roll.user_move_abs(target=target, getMotionTime=True)
        self.pi.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center, include_boundary=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center,
                                     include_boundary=True)

    def pi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.pi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.pi.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.pi.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center, include_boundary=True)


class TG_Sample_tower:
    """
    This class is probability only useful for the TG experiment.
    Therefore, when initializing this class, I do not allow for an arbitrary crystal location
    since there is almost no possibility of using this for a new application.
    """

    def __init__(self,
                 sample,
                 yag_sample,
                 yag1, yag2, yag3, ):
        # Create the instance of each motors

        self.x = xyMotor(upperLim=12.5 * 1000,
                         lowerLim=-12.5 * 1000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = xyMotor(upperLim=12.5 * 1000,
                         lowerLim=-12.5 * 1000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.z = xyMotor(upperLim=12.5 * 1000,
                         lowerLim=-12.5 * 1000,
                         res=5,
                         backlash=100,
                         speed_um_per_ps=1 * 1000 / 1e12, )

        self.th = RotationMotor(upperLim=np.deg2rad(360),
                                lowerLim=-np.deg2rad(360),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                speed_rad_per_ps=0.01 / 1e12, )

        self.sample = sample
        self.yag_sample = yag_sample
        self.yag1 = yag1
        self.yag2 = yag2
        self.yag3 = yag3

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.x.motion_dir = np.zeros(3, dtype=np.float64)
        self.x.motion_dir[1] = 1.0  #
        # Define the installation location of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.motion_dir = np.zeros(3, dtype=np.float64)
        self.y.motion_dir[0] = 1.0  #
        # Define the installation location of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.z.motion_dir = np.zeros(3, dtype=np.float64)
        self.z.motion_dir[0] = 1.0  #
        # Define the installation location of the x stage
        z_stage_center = np.zeros(3, dtype=np.float64)
        z_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.z.shift(displacement=z_stage_center)

        self.th.deg0direction = np.zeros(3, dtype=np.float64)
        self.th.deg0direction[1] = 1.0  #
        self.th.rotation_axis = np.zeros(3, dtype=np.float64)
        self.th.rotation_axis[0] = 1.0
        self.th.rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation location of the x stage
        th_stage_center = np.zeros(3, dtype=np.float64)
        th_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage + the height of the y stage
        self.th.shift(displacement=th_stage_center)

        # Move the crystal such that the
        sampleSurface = np.zeros(3, dtype=np.float64)
        sampleSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        self.sample.shift(displacement=sampleSurface)

        yag_sample_location = np.zeros(3)
        self.yag_sample.shift(displacement=yag_sample_location)

        yag1_location = np.zeros(3)
        self.yag1.shift(displacement=yag1_location)
        yag2_location = np.zeros(3)
        self.yag2.shift(displacement=yag2_location)
        yag3_location = np.zeros(3)
        self.yag3.shift(displacement=yag3_location)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.motion_dir * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.z.shift(displacement=displacement, include_boundary=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.sample.shift(displacement=displacement, include_boundary=True)
        self.yag_sample.shift(displacement=displacement, include_boundary=True)
        self.yag1.shift(displacement=displacement, include_boundary=True)
        self.yag2.shift(displacement=displacement, include_boundary=True)
        self.yag3.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.motion_dir * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.z.shift(displacement=displacement, include_boundary=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.sample.shift(displacement=displacement, include_boundary=True)
        self.yag_sample.shift(displacement=displacement, include_boundary=True)
        self.yag1.shift(displacement=displacement, include_boundary=True)
        self.yag2.shift(displacement=displacement, include_boundary=True)
        self.yag3.shift(displacement=displacement, include_boundary=True)

    def z_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.z.motion_dir * (target - self.z.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.z.user_move_abs(target=target, getMotionTime=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.sample.shift(displacement=displacement, include_boundary=True)
        self.yag_sample.shift(displacement=displacement, include_boundary=True)
        self.yag1.shift(displacement=displacement, include_boundary=True)
        self.yag2.shift(displacement=displacement, include_boundary=True)
        self.yag3.shift(displacement=displacement, include_boundary=True)

    def th_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.th.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.th.rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.th.user_move_abs(target=target, getMotionTime=True)
        self.sample.shift(displacement=displacement, include_boundary=True)
        self.yag_sample.shift(displacement=displacement, include_boundary=True)
