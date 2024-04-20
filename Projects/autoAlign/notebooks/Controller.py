import numpy as np

import sys

sys.path.append("../../../../XRaySimulation")

from XRaySimulation import Pulse, DeviceSimu, util, Crystal
from XRaySimulation.Machine import Motors


class XppController_TG:
    def __abs__(self):
        # Step 1 Create all the optics
        optics_container = get_optics()

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


# Create the optics I do not want to add to many dependence with these temporary files
# I understand this is not the best practice for software development.
# However, I am not a software engineer and the highest priority here is to
# finish this simualtion in time. Therefore, I'll try to do it in the fast but dirty way.
def get_optics(Ec=9.8):
    # Define the crystal property for the simulation
    si220 = {'d': 1.9201 * 1e-4,
             "chi0": complex(-0.97631E-05, 0.14871E-06),
             "chih_sigma": complex(0.59310E-05, -0.14320E-06),
             "chihbar_sigma": complex(0.59310E-05, -0.14320E-06),
             "chih_pi": complex(0.46945E-05, -0.11201E-06),
             "chihbar_pi": complex(0.46945E-05, -0.11201E-06),
             }

    si111 = {'d': 3.1355 * 1e-4,
             "chi0": complex(-0.10826E-04, 0.18209E-06),
             "chih_sigma": complex(0.57174E-05, - 0.12694E-06),
             "chihbar_sigma": complex(0.57174E-05, - 0.12694E-06),
             "chih_pi": complex(0.52222E-05, -0.11545E-06),
             "chihbar_pi": complex(0.52222E-05, -0.11545E-06),
             }

    g1_period = 1  # um
    g2_period = 1  # um
    tg_g_period = 1  # um
    # Crystal

    # Define gratings
    g1_cc = Crystal.RectangleGrating(a=g1_period / 2.,
                                     b=g1_period / 2.,
                                     direction=np.zeros(3),
                                     surface_point=np.zeros(3),
                                     order=1.)
    g1_vcc = Crystal.RectangleGrating(a=g1_period / 2.,
                                      b=g1_period / 2.,
                                      direction=np.zeros(3),
                                      surface_point=np.zeros(3),
                                      order=-1.)

    g2_cc = Crystal.RectangleGrating(a=g2_period / 2.,
                                     b=g2_period / 2.,
                                     direction=np.zeros(3),
                                     surface_point=np.zeros(3),
                                     order=-1.)
    g2_vcc = Crystal.RectangleGrating(a=g2_period / 2.,
                                      b=g2_period / 2.,
                                      direction=np.zeros(3),
                                      surface_point=np.zeros(3),
                                      order=1.)

    tg_g_a = Crystal.RectangleGrating(a=tg_g_period / 2.,
                                      b=tg_g_period / 2.,
                                      surface_point=np.zeros(3),
                                      order=1.)
    tg_g_b = Crystal.RectangleGrating(a=tg_g_period / 2.,
                                      b=tg_g_period / 2.,
                                      surface_point=np.zeros(3),
                                      order=-1.)

    # Define total reflection mirrors
    tg_mirror_pump_a = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([0, 1, 0]))
    tg_mirror_pump_b = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([0, -1, 0]))

    tg_mirror_probe = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([-1, 0, 0]))

    # ------------------------------------------------
    #    Get VCC
    # Define Bragg crystals
    vcc_channel_cut_config = ["lower left", 'upper left', 'upper left', 'lower left']
    vcc_channel_cut_angles = np.deg2rad(np.array([[0, -5], [5., 0], [0, 5], [-5, 0]]))
    vcc_channel_cut_edge_length_list = np.array([[50e3, 65.25e3],
                                                 [65.25e3, 50e3, ],
                                                 [50e3, 65.25e3],
                                                 [65.25e3, 50e3, ],
                                                 ])
    vcc_channel_cut_locations = np.zeros((3, 4))

    vcc_channel_cuts = [Crystal.ChannelCut(crystal_type="Silicon",
                                           miller_index="220",
                                           energy_keV=Ec,
                                           thickness_list=np.array([1e4, 1e4]),
                                           gap=13.595e3,
                                           surface_center_offset=32.5e3,
                                           edge_length_list=vcc_channel_cut_edge_length_list[_x],
                                           asymmetry_angle_list=vcc_channel_cut_angles[_x],
                                           first_surface_loc=vcc_channel_cut_config[_x],
                                           source=None,
                                           crystal_property=si220)
                        for _x in range(4)]
    # Change the location of the crystals
    for idx in range(4):
        vcc_channel_cuts[idx].shift(displacement=vcc_channel_cut_locations[idx])

    # --------------------------------------------------
    #   Get CC
    cc_channel_cut_config = ["upper left", 'lower left', ]
    cc_channel_cut_angles = np.deg2rad(np.array([[0, 0], [0, 0]]))
    cc_channel_cut_edge_length_list = np.array([[40e3, 100e3],
                                                [120e3, 15e3]])
    cc_channel_cut_center_offset = [30e3, 52.5e3]
    cc_channel_cut_gap = [25.15e3, 25.8e3]
    cc_channel_cut_locations = np.array([[0, 0, 0],
                                         [0, -cc_channel_cut_gap[1], 1050e3 - cc_channel_cut_center_offset[1]], ])

    cc_channel_cuts = [Crystal.ChannelCut(crystal_type="Silicon",
                                          miller_index="220",
                                          energy_keV=Ec,
                                          thickness_list=np.array([1e4, 1e4]),
                                          gap=cc_channel_cut_gap[_x],
                                          surface_center_offset=cc_channel_cut_center_offset[_x],
                                          edge_length_list=cc_channel_cut_edge_length_list[_x],
                                          asymmetry_angle_list=cc_channel_cut_angles[_x],
                                          first_surface_loc=cc_channel_cut_config[_x],
                                          source=None,
                                          crystal_property=si220)
                       for _x in range(2)]

    # Get the silicon 111 for the TG probe
    tg_si111 = Crystal.CrystalBlock3D(h=np.array([- np.pi * 2 / si111['d'], 0, 0], dtype=np.float64),
                                      normal=np.array([1., 0, 0.]),
                                      surface_point=np.zeros(3),
                                      thickness=1e4,
                                      chi_dict=si111,
                                      edge_length=2e4, )

    optics_dict = {"g1 cc": g1_cc,
                   "g1 vcc": g1_vcc,
                   "g2 cc": g2_cc,
                   "g2 vcc": g2_vcc,
                   "tg g a": tg_g_a,
                   "tg g b": tg_g_b,
                   "tg mirror pump a": tg_mirror_pump_a,
                   "tg mirror pump b": tg_mirror_pump_b,
                   "tg mirror probe": tg_mirror_probe,
                   "tg si111": tg_si111,
                   "cc1": cc_channel_cuts[0],
                   "cc2": cc_channel_cuts[1],
                   "vcc1": vcc_channel_cuts[0],
                   "vcc2": vcc_channel_cuts[1],
                   "vcc3": vcc_channel_cuts[2],
                   "vcc4": vcc_channel_cuts[3], }
    return optics_dict
