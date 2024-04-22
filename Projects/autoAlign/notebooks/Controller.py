import numpy as np

import sys

sys.path.append("../../../../XRaySimulation")

from XRaySimulation import Crystal
from XRaySimulation.Machine import Motors, ScintillatorCamera


class XppController_TG:
    """
    With this object, I define a lot of ways to access each motors.
    This certainly makes this object prone to error.
    However, I have little time to find a better solution.
    If you intend to use this future for your own work,
    you definitely need to rethink about the logic to make it compatible
    for your own applications

    """

    def __abs__(self):
        # Step 1 Create all the optics and motors
        motors, optics = assemble_motors_and_optics(Ec=9.8)

        # Step 2 Create properties associate with each component
        self._motor_stacks = motors
        self._optics = optics

        self.t1 = motors['t1']
        self.t2 = motors['t1']
        self.t3 = motors['t1']
        self.t45 = motors['t1']
        self.t6 = motors['t1']
        self.g1 = motors['t1']
        self.g2 = motors['t1']
        self.tg_g = motors['t1']
        self.m1 = motors['t1']
        self.m2a = motors['t1']
        self.m2b = motors['t1']
        self.si = motors['si']

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

    # Create the YAG crystals
    #  Later, I'll install the YAG camera. However, at this moment, I would like to use a
    # simple implementation of the yag crystal as a place-holder to make the simulation work.
    yag_sample = Crystal.YAG()
    yag1 = Crystal.YAG()
    yag2 = Crystal.YAG()
    yag3 = Crystal.YAG()

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
                   "vcc4": vcc_channel_cuts[3],
                   "yag sample": yag_sample,
                   "yag1": yag1,
                   "yag2": yag2,
                   "yag3": yag3,
                   }
    return optics_dict


def assemble_motors_and_optics(Ec=9.8):
    # Get all the optics
    optics_all = get_optics(Ec=Ec)

    # Get all the motors
    t1 = Motors.CrystalTower_x_y_theta_chi(channelCut=optics_all['cc1'],
                                           crystal_loc=np.copy(optics_all['cc1'].crystal_list[0].surface_point, ))
    t6 = Motors.CrystalTower_x_y_theta_chi(channelCut=optics_all['cc2'],
                                           crystal_loc=np.copy(optics_all['cc2'].crystal_list[0].surface_point, ))

    # For the VCC branch
    t2 = Motors.CrystalTower_x_y_theta_chi(channelCut=optics_all['vcc1'],
                                           crystal_loc=np.copy(optics_all['vcc1'].crystal_list[0].surface_point, ))
    t3 = Motors.CrystalTower_x_y_theta_chi(channelCut=optics_all['vcc2'],
                                           crystal_loc=np.copy(optics_all['vcc2'].crystal_list[0].surface_point, ))
    t45 = Motors.CrystalTower_miniSD_Scan(channelCut1=optics_all['vcc3'],
                                          crystal_loc1=np.copy(optics_all['vcc3'].crystal_list[0].surface_point, ),
                                          channelCut2=optics_all['vcc4'],
                                          crystal_loc2=np.copy(optics_all['vcc4'].crystal_list[0].surface_point, ),
                                          )

    # Get the grating tower
    g1 = Motors.Grating_tower(grating_1=optics_all['g1 cc'],
                              grating_m1=optics_all['g1 vcc'],
                              )
    g2 = Motors.Grating_tower(grating_1=optics_all['g2 cc'],
                              grating_m1=optics_all['g2 vcc'], )

    tg_g = Motors.Grating_tower(grating_1=optics_all['tg g a'],
                                grating_m1=optics_all['tg g b'], )

    # Get the Mirror tower
    m1 = Motors.Mirror_tower1(mirror=optics_all['tg mirror probe'],
                              crystal_loc=np.copy(optics_all['tg mirror probe'].surface_point),
                              )
    m2a = Motors.Mirror_tower2(mirror=optics_all['tg mirror pump a'],
                               crystal_loc=np.copy(optics_all['tg mirror pump a'].surface_point), )
    m2b = Motors.Mirror_tower2(mirror=optics_all['tg mirror pump b'],
                               crystal_loc=np.copy(optics_all['tg mirror pump b'].surface_point), )

    # Get the silicon tower
    si = Motors.Silicon_tower(crystal=optics_all['tg si111'],
                              crystal_loc=np.copy(optics_all['tg si111'].surface_point),
                              )

    # Get the sample tower
    sample = Motors.TG_Sample_tower(sample=optics_all['yag sample'],
                                    yag_sample=optics_all['yag sample'],
                                    yag1=optics_all['yag1'],
                                    yag2=optics_all['yag2'],
                                    yag3=optics_all['yag3']
                                    )

    motor_stacks = {'t1': t1,
                    't2': t2,
                    't3': t3,
                    't45': t45,
                    't6': t6,
                    'g1': g1,
                    'g2': g2,
                    'tg g': tg_g,
                    'm1': m1,
                    'm2a': m2a,
                    'm2b': m2b,
                    "si": si,
                    'sample': sample}

    # TODO: Need to specify the installation location with respect to the optical breadboard.

    return motor_stacks, optics_all
