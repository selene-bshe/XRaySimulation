import numpy as np

import sys

sys.path.append("../../../../XRaySimulation")

from XRaySimulation import Pulse, DeviceSimu, util, Crystal
from XRaySimulation.Machine import Motors

class XppController_TG:
    def __abs__(self):

        # Step 1 Create all the optics

        # Step 2 Create all motion stack
        # For the CC branch
        self.t1 = Motors.CrystalTower_x_y_theta_chi()
        self.t6 = Motors.CrystalTower_x_y_theta_chi()

        # For the VCC branch
        self.t2 = Motors.CrystalTower_x_y_theta_chi()
        self.t3 = Motors.CrystalTower_x_y_theta_chi()
        self.t45 = Motors.CrystalTower_miniSD_Scan()

        # Step 3 Move the devices to their rough position


        # Step 4 Rotate the crystals such that they are at the ideal location


        # Step 5 Add diodes


        # Step 6 Add cameras


    def plot_motors(self):
        pass

    def get_diode(self):
        pass

    def get_camera(self):
        pass

def parser(commandline):
    pass