import os

import numpy as np

import sys
import h5py

sys.path.append("../../../../XRaySimulation")

from XRaySimulation import Crystal, DeviceSimu, util
from XRaySimulation.Machine import Motors, ScintillatorCamera

# The following modules are loaded as a temporary solution
import rayTracingCalculation


class XppController_TG:
    """
    With this object, I define a lot of ways to access each motors.
    This certainly makes this object prone to error.
    However, I have little time to find a better solution.
    If you intend to use this future for your own work,
    you definitely need to rethink about the logic to make it compatible
    for your own applications

    """

    def __init__(self):
        # Step 1 Create all the optics and motors
        motors, optics = assemble_motors_and_optics(Ec=9.8)

        # Step 2 Create properties associate with each component
        self._motor_stacks = motors
        self._optics = optics

        self.t1 = motors['t1']
        self.t2 = motors['t2']
        self.t3 = motors['t3']
        self.t45 = motors['t45']
        self.t6 = motors['t6']
        self.g1 = motors['g1']
        self.g2 = motors['g2']
        self.tg_g = motors['tg g']
        self.m1 = motors['m1']
        self.m2a = motors['m2a']
        self.m2b = motors['m2b']
        self.si = motors['si']
        self.sample = motors['sample']

        self.all_towers = [self.t1, self.t2, self.t3, self.t45, self.t6,
                           self.g1, self.g2, self.tg_g,
                           self.m1, self.m2a, self.m2b, self.si, self.sample, ]

        # Install the crystal towers on the breadboard
        self.breadboard1 = Motors.Breadboard(hole_num_x=23, hole_num_z=55, gauge='metric')
        # controller.breadboard2 = Motors.Breadboard(hole_num_x=17, hole_num_z=17, gauge='metric')
        self.breadboard3 = Motors.Breadboard(hole_num_x=34, hole_num_z=55, gauge='metric')

        # Install SD table
        self.breadboard1.shift(displacement=np.array([-220e3 - 12.7e3, -225e3, 0, ]))
        Motors.install_motors_on_breadboard(motor_stack=self.t1.all_obj, breadboard=self.breadboard1,
                                            diag_hole_idx1=(7, 0), diag_hole_idx2=(11, 5))
        Motors.install_motors_on_breadboard(motor_stack=self.t2.all_obj, breadboard=self.breadboard1,
                                            diag_hole_idx1=(7, 8), diag_hole_idx2=(11, 14))
        Motors.install_motors_on_breadboard(motor_stack=self.t3.all_obj, breadboard=self.breadboard1,
                                            diag_hole_idx1=(7, 16), diag_hole_idx2=(11, 22))
        Motors.install_motors_on_breadboard(motor_stack=self.t45.all_obj, breadboard=self.breadboard1,
                                            diag_hole_idx1=(7, 27), diag_hole_idx2=(11, 37))
        Motors.install_motors_on_breadboard(motor_stack=self.t6.all_obj, breadboard=self.breadboard1,
                                            diag_hole_idx1=(7, 42), diag_hole_idx2=(11, 47))

        # Install mirror1
        displacement = np.array([0.0, 0.0, 4e6]) - self.m1.optics.surface_point
        for item in self.m1.all_obj:
            item.shift(displacement=displacement)

        # Install sample table
        self.breadboard3.shift(displacement=np.array([-254e3, -212.5e3, 7e6]))
        Motors.install_motors_on_breadboard(motor_stack=self.m2a.all_obj, breadboard=self.breadboard3,
                                            diag_hole_idx1=(0, 0), diag_hole_idx2=(4, 12))
        Motors.install_motors_on_breadboard(motor_stack=self.m2b.all_obj, breadboard=self.breadboard3,
                                            diag_hole_idx1=(13, 0), diag_hole_idx2=(17, 12))
        Motors.install_motors_on_breadboard(motor_stack=self.sample.all_obj, breadboard=self.breadboard3,
                                            diag_hole_idx1=(5, 23), diag_hole_idx2=(8, 28))

        # print("mounting", self.sample.all_obj[0].top_mount_pos, self.sample.all_obj[0].bottom_mount_pos, )
        # for optics in self.sample.all_optics:
        #    print(optics.surface_point)

        Motors.install_motors_on_breadboard(motor_stack=self.si.all_obj, breadboard=self.breadboard3,
                                            diag_hole_idx1=(5, 24), diag_hole_idx2=(10, 28))

        displacement = np.array([412.7e3 - 254e3, 0.0, 0.0])
        for item in self.si.all_obj:
            item.shift(displacement=displacement)

        # Install the gratings
        # Assume that there is no need to align the gratings
        displacement = np.array([0.0, 0.0, -1.9e6]) - self.g1.grating_1.surface_point
        for item in self.g1.all_obj:
            item.shift(displacement=displacement)

        displacement = np.array([0.0, 0.0, 3.8e6]) - self.g2.grating_1.surface_point
        for item in self.g2.all_obj:
            item.shift(displacement=displacement)

        displacement = np.array([0.0, 0.0, 1.5e6]) - self.tg_g.grating_1.surface_point
        for item in self.tg_g.all_obj:
            item.shift(displacement=displacement)

        # Add the shutter
        self.cc_shutter = True
        self.vcc_shutter = True

        # Step 4 Rotate the crystals such that they are at the ideal location
        pass

        # Step 5 Add diodes

        # Step 6 Add cameras
        self.pixel_num_x = 2048
        self.pixel_num_y = 2048

        # Add record
        self.record = []

    # def save_operation_record(controller, file_name=None):
    #    if file_name is None:
    #        file_name = "~/Desktop/operation_record_{}.h5".format(util.time_stamp())
    #    with h5py.File(file_name, 'wb') as target:
    #        target.create_dataset(name='t1x', data=np.array(controller.record['t1x']))

    def align_crystals(self):
        pass

    def plot_motors(self, ax):
        for tower in self.all_towers:
            for item in tower.all_motors:
                ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 1] / 1000, c='black')

    def plot_optics(self, ax):
        for tower in self.all_towers:
            for item in tower.all_optics:
                ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 1] / 1000, c='blue')

    def get_diode(self):
        pass

    def get_camera(self):
        pass

    def show_cc(self):
        self.cc_shutter = True
        self.vcc_shutter = False

    def show_vcc(self):
        self.vcc_shutter = True
        self.cc_shutter = False

    def show_both(self):
        self.vcc_shutter = True
        self.cc_shutter = True

    def show_neither(self):
        self.vcc_shutter = False
        self.cc_shutter = False


def get_optics(Ec=9.8):
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
    tg_mirror_pump_a = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([0, 1.0, 0]))
    tg_mirror_pump_b = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([0, -1.0, 0]))

    tg_mirror_probe = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([-1.0, 0, 0]))

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
    # Shift the crystal such that the rotation center is at 0
    vcc_channel_cuts[1].shift(displacement=np.copy(vcc_channel_cuts[1].crystal_list[1].surface_point))
    vcc_channel_cuts[1].shift(displacement=np.copy(vcc_channel_cuts[3].crystal_list[1].surface_point))

    # --------------------------------------------------
    #   Get CC
    cc_channel_cut_config = ["upper left", 'lower left', ]
    cc_channel_cut_angles = np.deg2rad(np.array([[0, 0], [0, 0]]))
    cc_channel_cut_edge_length_list = np.array([[40e3, 100e3],
                                                [120e3, 15e3]])
    cc_channel_cut_center_offset = [30e3, 52.5e3]
    cc_channel_cut_gap = [25.15e3, 25.8e3]

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
    cc_channel_cuts[1].shift(displacement=np.copy(vcc_channel_cuts[1].crystal_list[1].surface_point))

    # Get the silicon 111 for the TG probe
    tg_si111 = Crystal.CrystalBlock3D(h=np.array([- np.pi * 2 / si111['d'], 0, 0], dtype=np.float64),
                                      normal=np.array([1., 0, 0.]),
                                      surface_point=np.zeros(3),
                                      thickness=1e4,
                                      chi_dict=si111,
                                      edge_length=2e4, )
    tg_si111.boundary = np.array([[0, -10e3, -10e3, ],
                                  [0, -10e3, 10e3, ],
                                  [0, 10e3, 10e3, ],
                                  [0, 10e3, -10e3, ],
                                  [0, -10e3, -10e3, ], ])

    # Create the YAG crystals
    #  Later, I'll install the YAG camera. However, at this moment, I would like to use a
    # simple implementation of the yag crystal as a place-holder to make the simulation work.
    sample = Crystal.YAG()
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
                   "sample": sample,
                   }
    return optics_dict


def assemble_motors_and_optics(Ec=9.8):
    # Get all the optics
    optics_all = get_optics(Ec=Ec)

    # Get all the motors
    t1 = Motors.CrystalTower_x_y_theta_chi(channelCut=optics_all['cc1'],
                                           crystal_loc=np.copy(optics_all['cc1'].crystal_list[0].surface_point, ))
    t6 = Motors.CrystalTower_x_y_theta_chi(channelCut=optics_all['cc2'],
                                           crystal_loc=np.copy(optics_all['cc2'].crystal_list[1].surface_point, ))

    # For the VCC branch
    t2 = Motors.CrystalTower_x_y_theta_chi(channelCut=optics_all['vcc1'],
                                           crystal_loc=np.copy(optics_all['vcc1'].crystal_list[0].surface_point, ))
    t3 = Motors.CrystalTower_x_y_theta_chi(channelCut=optics_all['vcc2'],
                                           crystal_loc=np.copy(optics_all['vcc2'].crystal_list[1].surface_point, ))
    t45 = Motors.CrystalTower_miniSD_Scan(channelCut2=optics_all['vcc3'],
                                          crystal_loc2=np.copy(optics_all['vcc3'].crystal_list[0].surface_point, ),
                                          channelCut1=optics_all['vcc4'],
                                          crystal_loc1=np.copy(optics_all['vcc4'].crystal_list[1].surface_point, ),
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
    m1 = Motors.Tower_x_y_pi(mirror=optics_all['tg mirror probe'], )
    m2a = Motors.Mirror_tower1(mirror=optics_all['tg mirror pump a'],
                               crystal_loc=np.copy(optics_all['tg mirror pump a'].surface_point), )
    m2b = Motors.Mirror_tower2(mirror=optics_all['tg mirror pump b'],
                               crystal_loc=np.copy(optics_all['tg mirror pump b'].surface_point), )

    # Get the silicon tower
    si = Motors.Silicon_tower(crystal=optics_all['tg si111'], )

    # Get the sample tower
    sample = Motors.TG_Sample_tower(sample=optics_all['sample'],
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
