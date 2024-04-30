"""
This notebook tries to mimic the installation condition of the setup.
"""

import sys

sys.path.append("../../../../XRaySimulation")

import numpy as np
from matplotlib import patches

from XRaySimulation import Crystal, DeviceSimu, util, Pulse
from XRaySimulation.Machine import Motors, ScintillatorCamera

# The following modules are loaded as a temporary solution
import MotorStack

si220 = {'d': 1.9201 * 1e-4,
         "chi0": complex(-0.10169E-04, 0.16106E-06),
         "chih_sigma": complex(0.61786E-05, - 0.15508E-06),
         "chihbar_sigma": complex(0.61786E-05, -0.15508E-06),
         "chih_pi": complex(0.48374E-05, -0.11996E-06),
         "chihbar_pi": complex(0.48374E-05, -0.11996E-06),
         }

si111 = {'d': 3.1355 * 1e-4,
         "chi0": complex(-0.10169E-04, 0.16106E-06),
         "chih_sigma": complex(0.53693E-05, -0.11228E-06),
         "chihbar_sigma": complex(0.53693E-05, -0.11228E-06),
         "chih_pi": complex(0.49322E-05, -0.10272E-06),
         "chihbar_pi": complex(0.49322E-05, -0.10272E-06),
         }

dia111 = {'d': 2.0593 * 1e-4,
          "chi0": complex(-0.15217E-04, 0.13392E-07),
          "chih_sigma": complex(0.55417E-05, -0.93083E-08),
          "chihbar_sigma": complex(0.55417E-05, -0.93083E-08),
          "chih_pi": complex(0.44959E-05, - 0.74602E-08),
          "chihbar_pi": complex(0.44959E-05, -0.74602E-08),
          }

g1_period = 1  # um
g2_period = 1  # um
tg_g_period = 1  # um


def get_optics():
    # Define gratings
    g1_cc = Crystal.RectangleGrating(a=g1_period / 2.,
                                     b=g1_period / 2.,
                                     direction=np.array([-1., 0., 0.], dtype=np.float64),
                                     surface_point=np.zeros(3),
                                     order=1.)

    g1_vcc = Crystal.RectangleGrating(a=g1_period / 2.,
                                      b=g1_period / 2.,
                                      direction=np.array([-1., 0., 0.], dtype=np.float64),
                                      surface_point=np.zeros(3),
                                      order=-1.)

    g2_cc = Crystal.RectangleGrating(a=g2_period / 2.,
                                     b=g2_period / 2.,
                                     direction=np.array([0., 1., 0.], dtype=np.float64),
                                     surface_point=np.zeros(3),
                                     order=-1.)
    g2_vcc = Crystal.RectangleGrating(a=g2_period / 2.,
                                      b=g2_period / 2.,
                                      direction=np.array([0., 1., 0.], dtype=np.float64),
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
    tg_mirror_pump_a = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([-1.0, 0, 0]))
    tg_mirror_pump_b = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([-1.0, 0.0, 0]))
    tg_mirror_probe = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([-1.0, 0, 0]))

    # ------------------------------------------
    #   Get crystal for XPP mono
    # ------------------------------------------
    mono_miscut = [np.deg2rad(0.0), np.deg2rad(0.0)]
    mono_diamond = [Crystal.CrystalBlock3D(h=np.array([0., 2. * np.pi / dia111['d'], 0.]),
                                           normal=np.array(
                                               [0., -np.cos(mono_miscut[x]), np.sin(mono_miscut[x])]),
                                           surface_point=np.zeros(3, dtype=np.float64),
                                           thickness=10e3,
                                           chi_dict=dia111,
                                           edge_length=20e3) for x in range(2)]
    mono_diamond[1].rotate_wrt_point(rot_mat=np.array([[1, 0, 0],
                                                       [0, -1, 0],
                                                       [0, 0, -1]], dtype=np.float64),
                                     ref_point=np.copy(mono_diamond[1].surface_point))

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
    vcc_channel_cuts[3].shift(displacement=np.copy(vcc_channel_cuts[3].crystal_list[1].surface_point))

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
    tg_si111 = Crystal.CrystalBlock3D(h=np.array([np.pi * 2 / si111['d'], 0, 0], dtype=np.float64),
                                      normal=np.array([-1., 0, 0.]),
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
                   "xpp mono": mono_diamond,
                   }
    return optics_dict


def assemble_motors_and_optics():
    # Get all the optics
    optics_all = get_optics()

    # Get the XPP mono
    monoT1 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['xpp mono'][0],
                                                   crystal_loc=np.copy(optics_all['xpp mono'][0].surface_point, ))

    monoT2 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['xpp mono'][1],
                                                   crystal_loc=np.copy(optics_all['xpp mono'][1].surface_point, ))
    # Get all the motors
    t1 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['cc1'],
                                               crystal_loc=np.copy(optics_all['cc1'].crystal_list[0].surface_point, ))
    t6 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['cc2'],
                                               crystal_loc=np.copy(optics_all['cc2'].crystal_list[1].surface_point, ))

    # For the VCC branch
    t2 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['vcc1'],
                                               crystal_loc=np.copy(optics_all['vcc1'].crystal_list[0].surface_point, ))
    t3 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['vcc2'],
                                               crystal_loc=np.copy(optics_all['vcc2'].crystal_list[1].surface_point, ))
    t45 = MotorStack.CrystalTower_miniSD_Scan(channelCut1=optics_all['vcc3'],
                                              crystal_loc1=np.copy(optics_all['vcc3'].crystal_list[0].surface_point, ),
                                              channelCut2=optics_all['vcc4'],
                                              crystal_loc2=np.copy(optics_all['vcc4'].crystal_list[1].surface_point, ),
                                              )

    # Get the grating tower
    g1 = MotorStack.Grating_tower(grating_1=optics_all['g1 cc'],
                                  grating_m1=optics_all['g1 vcc'],
                                  )
    g2 = MotorStack.Grating_tower(grating_1=optics_all['g2 cc'],
                                  grating_m1=optics_all['g2 vcc'], )

    tg_g = MotorStack.Grating_tower(grating_1=optics_all['tg g a'],
                                    grating_m1=optics_all['tg g b'], )

    # Get the Mirror tower
    m1 = MotorStack.Tower_x_y_pi(mirror=optics_all['tg mirror probe'], )
    m2a = MotorStack.Mirror_tower1(mirror=optics_all['tg mirror pump a'])
    m2b = MotorStack.Mirror_tower2(mirror=optics_all['tg mirror pump b'])

    # Get the silicon tower
    si = MotorStack.Silicon_tower(crystal=optics_all['tg si111'], )

    # Get the sample tower
    sample = MotorStack.TG_Sample_tower(sample=optics_all['sample'],
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
                    'sample': sample,
                    'mono t1': monoT1,
                    'mono t2': monoT2}

    return motor_stacks, optics_all


class XppController_TG:
    """
    With this object, I define a lot of ways to access each motors.
    This certainly makes this object prone to error.
    However, I have little time to find a better solution.
    If you intend to use this future for your own work,
    you definitely need to rethink about the logic to make it compatible
    for your own applications

    """

    def __init__(self, photon_kev=9.8):

        fwhm = 200  # um

        # Define a reference pulse for the alignment

        self.gaussian_pulse = Pulse.GaussianPulse3D()
        self.gaussian_pulse.set_pulse_properties(central_energy=photon_kev,
                                                 polar=[1., 0., 0.],
                                                 sigma_x=fwhm / 2. / np.sqrt(np.log(2)) / util.c,
                                                 sigma_y=fwhm / 2. / np.sqrt(np.log(2)) / util.c,
                                                 sigma_z=9.,
                                                 x0=np.array([0., -500e3, -30e6]))
        self.wavelength = np.pi * 2 / util.kev_to_wavevec_length(energy=photon_kev)

        # Step 1 Create all the optics and motors
        motors, optics = assemble_motors_and_optics()

        # Step 2 Create properties associate with each component
        self._motor_stacks = motors
        self._optics = optics

        self.mono_t1 = motors['mono t1']
        self.mono_t2 = motors['mono t2']

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

        self.all_towers = [self.mono_t1, self.mono_t2,
                           self.t1, self.t2, self.t3, self.t45, self.t6,
                           self.g1, self.g2, self.tg_g,
                           self.m1, self.m2a, self.m2b, self.si, self.sample, ]

        # Insatll the XPP mono
        bragg = util.get_bragg_angle(wave_length=self.wavelength, plane_distance=dia111['d'])
        # Assume that the gap size is 50 cm, then the z offset is gap / np.tan(2 * bragg)
        gap = 500e3
        z_offset = gap / np.tan(2 * bragg)

        # Shift the pulse and mono tower 1
        displacement = np.array([0, -gap, -z_offset], dtype=np.float64)
        for item in self.mono_t1.all_obj:
            item.shift(displacement=displacement)

        # Shift the installation path of the xpp mono
        displacement = np.array([0, 0, -10e6], dtype=np.float64)
        for item in self.mono_t1.all_obj:
            item.shift(displacement=displacement)
        for item in self.mono_t2.all_obj:
            item.shift(displacement=displacement)

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
        displacement = np.array([0e3, 0.0, 4e6]) - self.m1.optics.surface_point
        for item in self.m1.all_obj:
            item.shift(displacement=displacement)

        # Install sample table
        self.breadboard3.shift(displacement=np.array([-254e3, -212.5e3, 7e6]))
        Motors.install_motors_on_breadboard(motor_stack=self.m2a.all_obj, breadboard=self.breadboard3,
                                            diag_hole_idx1=(0, 7), diag_hole_idx2=(4, 19))
        Motors.install_motors_on_breadboard(motor_stack=self.m2b.all_obj, breadboard=self.breadboard3,
                                            diag_hole_idx1=(13, 7), diag_hole_idx2=(17, 19))
        Motors.install_motors_on_breadboard(motor_stack=self.si.all_obj, breadboard=self.breadboard3,
                                            diag_hole_idx1=(5, 25), diag_hole_idx2=(10, 30))
        Motors.install_motors_on_breadboard(motor_stack=self.sample.all_obj, breadboard=self.breadboard3,
                                            diag_hole_idx1=(4, 23), diag_hole_idx2=(7, 28))

        displacement = np.array([50e3, 0.0, 0.0])
        for item in self.sample.all_obj:
            item.shift(displacement=displacement)

        # print("test", self.si.optics.surface_point)
        displacement = np.array([412.7e3 + 60e3, 25e3, 0.0])
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

        # Step 5 Add diodes

        # Step 6 Add cameras
        self.pixel_num_x = 2048
        self.pixel_num_y = 2048

        # Add record
        self.record = []
        self.mono_t1_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.mono_t2_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]

        self.t1_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t2_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t3_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t4_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t5_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t6_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]

    def align_xpp_mono(self):

        # Get the geometry bragg angle
        bragg = util.get_bragg_angle(wave_length=np.pi * 2 / self.gaussian_pulse.klen0, plane_distance=dia111['d'])

        # Step 1, move the mono1 th to the geometric path
        _ = self.mono_t1.th_umv(target=-bragg)
        _ = self.mono_t2.th_umv(target=-bragg)

        # Step 2, get the rocking curve around the motion axis for the two crystals.
        (angles1, reflect_sigma1,
         reflect_pi1, b_factor1, kout1) = DeviceSimu.get_rocking_curve_around_axis(
            kin=self.gaussian_pulse.k0,
            scan_range=np.deg2rad(0.2),
            scan_number=10 ** 3,
            rotation_axis=self.mono_t1.th.rotation_axis,
            h_initial=self.mono_t1.optics.h,
            normal_initial=self.mono_t1.optics.normal,
            thickness=self.mono_t1.optics.thickness,
            chi_dict=self.mono_t1.optics.chi_dict, )

        # Get the target bragg peak
        fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                  curve_values=np.square(np.abs(reflect_sigma1)),
                                                  center=True,
                                                  get_index=True)

        # Move the crystal to the target path
        _ = self.mono_t1.th_umv(target=-bragg + angle_adjust)

        # Align the second crystal
        kin1 = np.copy(kout1[index])

        (angles2, reflect_sigma2,
         reflect_pi2, b_factor2, kout2) = DeviceSimu.get_rocking_curve_around_axis(
            kin=kin1,
            scan_range=np.deg2rad(0.2),
            scan_number=10 ** 3,
            rotation_axis=self.mono_t2.th.rotation_axis,
            h_initial=self.mono_t2.optics.h,
            normal_initial=self.mono_t2.optics.normal,
            thickness=self.mono_t2.optics.thickness,
            chi_dict=self.mono_t2.optics.chi_dict, )

        # Get the target bragg peak
        fwhm2, angle_adjust2, index2 = util.get_fwhm(coordinate=angles2,
                                                     curve_values=np.square(np.abs(reflect_sigma2)),
                                                     center=True,
                                                     get_index=True)
        _ = self.mono_t2.th_umv(target=-bragg + angle_adjust2)

        self.mono_t1_rocking = [angles1 - angle_adjust, np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
        self.mono_t2_rocking = [angles2 - angle_adjust2, np.square(np.abs(reflect_sigma2)) / np.abs(b_factor2)]

        # Adjust the path of the XPP mono such that the exit X x-ray pulse is at the (0,0, ...)
        # on the second crystal
        trajectory, kout, _ = self.get_raytracing_trajectory(path='mono')
        # Get the ideal location of the second crsytal
        dir = kout[-2] / np.linalg.norm(kout[-2])
        location = dir * (0 - self.gaussian_pulse.x0[1]) / dir[1]
        # print(location)
        location += trajectory[-3]
        displacement = location - self.mono_t2.optics.surface_point
        for item in self.mono_t2.all_obj:
            item.shift(displacement=np.copy(displacement))

        return ((angles1 + angle_adjust, np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1), kout1),
                (angles2 + angle_adjust2, np.square(np.abs(reflect_sigma2)) / np.abs(b_factor2), kout2),)

    def align_miniSD(self):

        # Get the kout after the XPP mono
        _, kout, _ = DeviceSimu.get_lightpath(device_list=[self.mono_t1.optics, self.mono_t2.optics],
                                              kin=self.gaussian_pulse.k0,
                                              initial_point=self.gaussian_pulse.x0,
                                              final_plane_point=np.array([0, 0, 10e6]),
                                              final_plane_normal=np.array([0, 0, -1]))
        kout = kout[-1]

        # Get the geometry bragg angle
        bragg = util.get_bragg_angle(wave_length=np.pi * 2 / self.gaussian_pulse.klen0, plane_distance=si220['d'])
        bragg_list = [bragg, -bragg, bragg, bragg, -bragg, -bragg]

        # Step 1, move the mono1 th to the geometric path
        _ = self.t1.th_umv(target=bragg_list[0])
        _ = self.t2.th_umv(target=bragg_list[1])
        _ = self.t3.th_umv(target=bragg_list[2])
        _ = self.t45.th1_umv(target=bragg_list[3])
        _ = self.t45.th2_umv(target=bragg_list[4])
        _ = self.t6.th_umv(target=bragg_list[5])

        # Fine adjustment according to dynamical diffraction theory
        kin = np.copy(kout + self.g1.grating_m1.momentum_transfer)
        combo = [[self.t1, self.t1_rocking, bragg_list[0]],
                 [self.t6, self.t6_rocking, bragg_list[-1]], ]
        for tower in combo:
            # Step 2, get the rocking curve around the motion axis for the two crystals.
            (angles1, reflect_sigma1,
             reflect_pi1, b_factor1, kout1) = DeviceSimu.get_rocking_curve_channelcut_around_axis(
                kin=kin,
                scan_range=np.deg2rad(0.2),
                scan_number=10 ** 3,
                rotation_axis=tower[0].th.rotation_axis,
                channelcut=tower[0].optics, )

            # Get the target bragg peak
            fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                      curve_values=np.square(np.abs(reflect_sigma1)),
                                                      center=True,
                                                      get_index=True)
            # Move the crystal to the target path
            _ = tower[0].th_umv(target=tower[2] + angle_adjust)

            # Record the current rocking curve
            tower[1][:] = [np.copy(angles1 - angle_adjust), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
            kin = np.copy(kout1[index])

        # Align vcc2 and vcc3
        # Fine adjustment according to dynamical diffraction theory
        kin = np.copy(kout + self.g1.grating_1.momentum_transfer)
        combo = [[self.t2, self.t2_rocking, bragg_list[1]],
                 [self.t3, self.t3_rocking, bragg_list[2]], ]
        for tower in combo:
            # Step 2, get the rocking curve around the motion axis for the two crystals.
            (angles1, reflect_sigma1,
             reflect_pi1, b_factor1, kout1) = DeviceSimu.get_rocking_curve_channelcut_around_axis(
                kin=kin,
                scan_range=np.deg2rad(0.2),
                scan_number=10 ** 3,
                rotation_axis=tower[0].th.rotation_axis,
                channelcut=tower[0].optics, )

            # Get the target bragg peak
            fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                      curve_values=np.square(np.abs(reflect_sigma1)),
                                                      center=True,
                                                      get_index=True)
            # Move the crystal to the target path
            _ = tower[0].th_umv(target=tower[2] + angle_adjust)

            # Record the current rocking curve
            tower[1][:] = [np.copy(angles1 - angle_adjust), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
            kin = np.copy(kout1[index])

        # Align vcc4 and vcc5
        # Fine adjustment according to dynamical diffraction theory
        combo = [[self.t45.th1, self.t45.optics1, self.t45.th1_umv, self.t4_rocking, bragg_list[3]],
                 [self.t45.th2, self.t45.optics2, self.t45.th2_umv, self.t5_rocking, bragg_list[4]], ]
        for tower in combo:
            # Step 2, get the rocking curve around the motion axis for the two crystals.
            (angles1, reflect_sigma1,
             reflect_pi1, b_factor1, kout1) = DeviceSimu.get_rocking_curve_channelcut_around_axis(
                kin=kin,
                scan_range=np.deg2rad(0.2),
                scan_number=10 ** 3,
                rotation_axis=tower[0].rotation_axis,
                channelcut=tower[1])

            # Get the target bragg peak
            fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                      curve_values=np.square(np.abs(reflect_sigma1)),
                                                      center=True,
                                                      get_index=True)
            # Move the crystal to the target path
            _ = tower[2](target=tower[4] + angle_adjust)

            # Record the current rocking curve
            tower[3][:] = (np.copy(angles1 - angle_adjust), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1))
            kin = np.copy(kout1[index])

    def get_miniSD_rocking(self):
        # Get the kout after the XPP mono
        _, kout, _ = DeviceSimu.get_lightpath(device_list=[self.mono_t1.optics, self.mono_t2.optics],
                                              kin=self.gaussian_pulse.k0,
                                              initial_point=self.gaussian_pulse.x0,
                                              final_plane_point=np.array([0, 0, 10e6]),
                                              final_plane_normal=np.array([0, 0, -1]))
        kout = kout[-1]

        # Fine adjustment according to dynamical diffraction theory
        kin = np.copy(kout + self.g1.grating_m1.momentum_transfer)
        combo = [[self.t1, self.t1_rocking],
                 [self.t6, self.t6_rocking], ]
        for tower in combo:
            # Step 2, get the rocking curve around the motion axis for the two crystals.
            (angles1, reflect_sigma1, reflect_pi1, b_factor1, kout1
             ) = DeviceSimu.get_rocking_curve_channelcut_around_axis(
                kin=kin, scan_range=np.deg2rad(0.2), scan_number=10 ** 3,
                rotation_axis=tower[0].th.rotation_axis, channelcut=tower[0].optics, )

            # Get the target bragg peak
            (fwhm, angle_adjust, index
             ) = util.get_fwhm(coordinate=angles1, curve_values=np.square(np.abs(reflect_sigma1)),
                               center=True, get_index=True)

            # Record the current rocking curve
            tower[1][:] = [np.copy(angles1), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
            kin = np.copy(kout1[index])

        # Align vcc2 and vcc3
        # Fine adjustment according to dynamical diffraction theory
        kin = np.copy(kout + self.g1.grating_1.momentum_transfer)
        combo = [[self.t2, self.t2_rocking],
                 [self.t3, self.t3_rocking], ]
        for tower in combo:
            # Step 2, get the rocking curve around the motion axis for the two crystals.
            (angles1, reflect_sigma1,
             reflect_pi1, b_factor1, kout1) = DeviceSimu.get_rocking_curve_channelcut_around_axis(
                kin=kin,
                scan_range=np.deg2rad(0.2),
                scan_number=10 ** 3,
                rotation_axis=tower[0].th.rotation_axis,
                channelcut=tower[0].optics, )

            # Get the target bragg peak
            fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                      curve_values=np.square(np.abs(reflect_sigma1)),
                                                      center=True,
                                                      get_index=True)

            # Record the current rocking curve
            tower[1][:] = [np.copy(angles1), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
            kin = np.copy(kout1[index])

        # Align vcc4 and vcc5
        # Fine adjustment according to dynamical diffraction theory
        combo = [[self.t45.th1, self.t45.optics1, self.t45.th1_umv, self.t4_rocking],
                 [self.t45.th2, self.t45.optics2, self.t45.th2_umv, self.t5_rocking], ]
        for tower in combo:
            # Step 2, get the rocking curve around the motion axis for the two crystals.
            (angles1, reflect_sigma1,
             reflect_pi1, b_factor1, kout1) = DeviceSimu.get_rocking_curve_channelcut_around_axis(
                kin=kin,
                scan_range=np.deg2rad(0.2),
                scan_number=10 ** 3,
                rotation_axis=tower[0].rotation_axis,
                channelcut=tower[1])

            # Get the target bragg peak
            fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                      curve_values=np.square(np.abs(reflect_sigma1)),
                                                      center=True,
                                                      get_index=True)

            # Record the current rocking curve
            tower[3][:] = (np.copy(angles1), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1))
            kin = np.copy(kout1[index])

    def get_raytracing_trajectory(self, path="mono", get_path_length='True', virtual_sample_plane=None):

        if path == "cc":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_m1]
                           + self.t1.optics.crystal_list + self.t6.optics.crystal_list
                           + [self.sample.yag1, ])
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=
                                                                    np.copy(self.sample.yag1.surface_point),
                                                                    final_plane_normal=
                                                                    np.copy(self.sample.yag1.normal))

        elif path == "cc sample":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_m1]
                           + self.t1.optics.crystal_list + self.t6.optics.crystal_list
                           + [self.sample.sample, ])

            if virtual_sample_plane is None:
                virtual_sample_plane = np.copy(self.sample.sample.normal)
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=
                                                                    np.copy(self.sample.sample.surface_point),
                                                                    final_plane_normal=virtual_sample_plane)

        elif path == "vcc":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_1]
                           + self.t2.optics.crystal_list + self.t3.optics.crystal_list
                           + self.t45.optics1.crystal_list + self.t45.optics2.crystal_list
                           + [self.sample.yag1, ])

            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=
                                                                    np.copy(self.sample.yag1.surface_point),
                                                                    final_plane_normal=
                                                                    np.copy(self.sample.yag1.normal))
        elif path == "probe m1 only":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_1]
                           + self.t2.optics.crystal_list + self.t3.optics.crystal_list
                           + self.t45.optics1.crystal_list + self.t45.optics2.crystal_list
                           + [self.m1.optics, self.sample.yag1])

            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=
                                                                    np.copy(self.sample.yag1.surface_point),
                                                                    final_plane_normal=
                                                                    np.copy(self.sample.yag1.normal))
        elif path == "probe":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_1]
                           + self.t2.optics.crystal_list + self.t3.optics.crystal_list
                           + self.t45.optics1.crystal_list + self.t45.optics2.crystal_list
                           + [self.m1.optics, self.si.optics, self.sample.yag1])

            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=
                                                                    np.copy(self.sample.yag1.surface_point),
                                                                    final_plane_normal=
                                                                    np.copy(self.sample.yag1.normal))

        elif path == "pump a":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_m1]
                           + self.t1.optics.crystal_list + self.t6.optics.crystal_list
                           + [self.tg_g.grating_m1, self.m2a.optics, self.sample.yag1])
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=np.copy(
                                                                        self.sample.yag1.surface_point),
                                                                    final_plane_normal=np.copy(self.sample.yag1.normal))

        elif path == "pump a no mirror":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_m1]
                           + self.t1.optics.crystal_list + self.t6.optics.crystal_list
                           + [self.tg_g.grating_m1, self.sample.yag1])
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=np.copy(
                                                                        self.sample.yag1.surface_point),
                                                                    final_plane_normal=np.copy(self.sample.yag1.normal))

        elif path == "pump b":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_m1]
                           + self.t1.optics.crystal_list + self.t6.optics.crystal_list
                           + [self.tg_g.grating_1, self.m2b.optics, self.sample.yag1])
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=np.copy(
                                                                        self.sample.yag1.surface_point),
                                                                    final_plane_normal=np.copy(self.sample.yag1.normal))

        elif path == "pump b no mirror":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_m1]
                           + self.t1.optics.crystal_list + self.t6.optics.crystal_list
                           + [self.tg_g.grating_1, self.sample.yag1])
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=np.copy(
                                                                        self.sample.yag1.surface_point),
                                                                    final_plane_normal=np.copy(self.sample.yag1.normal))

        elif path == "mono":
            defice_list = [self.mono_t1.optics, self.mono_t2.optics]
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=np.array([0, 0, -7e6]),
                                                                    final_plane_normal=np.array([0, 0, -1]))

        elif path == "probe sample":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_1]
                           + self.t2.optics.crystal_list + self.t3.optics.crystal_list
                           + self.t45.optics1.crystal_list + self.t45.optics2.crystal_list
                           + [self.m1.optics, self.si.optics, self.sample.sample])

            if virtual_sample_plane is None:
                virtual_sample_plane = np.copy(self.sample.sample.normal)
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=
                                                                    np.copy(self.sample.sample.surface_point),
                                                                    final_plane_normal=virtual_sample_plane)

        elif path == "pump a sample":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_m1]
                           + self.t1.optics.crystal_list + self.t6.optics.crystal_list
                           + [self.tg_g.grating_m1, self.m2a.optics, self.sample.sample])

            if virtual_sample_plane is None:
                virtual_sample_plane = np.copy(self.sample.sample.normal)
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=
                                                                    np.copy(self.sample.sample.surface_point),
                                                                    final_plane_normal=virtual_sample_plane)

        elif path == "pump b sample":
            defice_list = ([self.mono_t1.optics, self.mono_t2.optics, self.g1.grating_m1]
                           + self.t1.optics.crystal_list + self.t6.optics.crystal_list
                           + [self.tg_g.grating_1, self.m2b.optics, self.sample.yag1])

            if virtual_sample_plane is None:
                virtual_sample_plane = np.copy(self.sample.sample.normal)
            trajectory, kout, pathlength = DeviceSimu.get_lightpath(device_list=defice_list,
                                                                    kin=self.gaussian_pulse.k0,
                                                                    initial_point=self.gaussian_pulse.x0,
                                                                    final_plane_point=
                                                                    np.copy(self.sample.sample.surface_point),
                                                                    final_plane_normal=virtual_sample_plane)

        else:
            print("Warning, the specified path option is not defined.")
            trajectory = 0
            kout = 0
            pathlength = 0

        if get_path_length:
            return trajectory, kout, pathlength
        else:
            return trajectory, kout

    def plot_motors(self, ax, color='black', axis="xz"):
        if axis == "xz":
            for tower in self.all_towers:
                for item in tower.all_motors:
                    ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 1] / 1000, c=color)
        elif axis == 'yz':
            for tower in self.all_towers:
                for item in tower.all_motors:
                    ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 0] / 1000, c=color)
        elif axis == 'xy':
            for tower in self.all_towers:
                for item in tower.all_motors:
                    ax.plot(item.boundary[:, 1] / 1000, item.boundary[:, 0] / 1000, c=color)

    def plot_optics(self, ax, color='black', axis="xz"):
        if axis == 'xz':
            for tower in self.all_towers:
                for item in tower.all_optics:
                    ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 1] / 1000, c=color)
        elif axis == 'yz':
            for tower in self.all_towers:
                for item in tower.all_optics:
                    ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 0] / 1000, c=color)
        elif axis == 'xy':
            for tower in self.all_towers:
                for item in tower.all_optics:
                    ax.plot(item.boundary[:, 1] / 1000, item.boundary[:, 0] / 1000, c=color)

    def plot_mono_rocking(self, ax_mono_t1, ax_mono_t2):

        ax_mono_t1.plot(np.rad2deg(self.mono_t1_rocking[0]) * 1e3,
                        self.mono_t1_rocking[1], c='b', label='mono t1')
        ax_mono_t1.set_xlim([- 5, 5])
        ax_mono_t1.set_xlabel("relative th (mdeg)")
        ax_mono_t1.set_ylabel("R")
        ax_mono_t1.set_title("mono T1")

        ax_mono_t2.plot(np.rad2deg(self.mono_t2_rocking[0]) * 1e3,
                        self.mono_t2_rocking[1], c='r', label='mono t2')
        ax_mono_t2.set_xlim([- 5, 5])
        ax_mono_t2.set_xlabel("relative th (mdeg)")
        ax_mono_t2.set_ylabel("R")
        ax_mono_t2.set_title("mono T2")

    def plot_mono_optics(self, ax, show_trajectory=False):

        self.plot_motors(ax=ax, color='black')
        self.plot_optics(ax=ax, color='blue')

        if show_trajectory:
            mono_traj, mono_kout, mono_pathlength = self.get_raytracing_trajectory(path="mono")
            ax.plot(mono_traj[:, 2] / 1e3, mono_traj[:, 1] / 1e3, 'g', label='vcc')

        ax.set_aspect('equal')
        ax.set_title("Mono after alignment")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("x (mm)")
        ax.set_xlim([-10e3 - 800, -10e3 + 100])
        ax.set_ylim([- 600, 100])

    def plot_miniSD_table(self, ax, xlim=None, ylim=None, show_trajectory=False):
        if xlim is None:
            xlim = [-100, 1200]
        if ylim is None:
            ylim = [-100, 100]

        self.plot_motors(ax=ax, color='black')
        self.plot_optics(ax=ax, color='blue')

        ax.set_aspect('equal')
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("x (mm)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if show_trajectory:
            vcc_traj, vcc_kout, vcc_path = self.get_raytracing_trajectory(path="vcc")
            cc_traj, cc_kout, cc_path = self.get_raytracing_trajectory(path="cc")

            ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 1] / 1e3, 'g', label='vcc')
            ax.plot(cc_traj[:, 2] / 1e3, cc_traj[:, 1] / 1e3, 'r', label='cc')

    def plot_miniSD_rocking(self, ax_list):

        # Get the current rocking curve
        self.get_miniSD_rocking()
        print("Get the most updated rocking curve around current location.")

        # Start plotting
        record_to_plot = [self.t1_rocking, self.t2_rocking, self.t3_rocking,
                          self.t4_rocking, self.t5_rocking, self.t6_rocking]
        for idx in range(6):
            record = record_to_plot[idx]
            ax_list[idx].plot(np.rad2deg(record[0]) * 1000, record[1], label='t{}'.format(idx + 1))
            ax_list[idx].set_xlabel('relative th (mdeg)')
            ax_list[idx].legend()
            ax_list[idx].set_xlim([-5, 5])

    def plot_beam_on_yag(self, ax):

        vcc_traj, vcc_kout, vcc_pathlength = self.get_raytracing_trajectory(path="vcc")
        probe_m1_traj, probe_m1_kout, probe_m1_pathlength = self.get_raytracing_trajectory(path="probe m1 only")
        probe_traj, kout, probe_pathlength = self.get_raytracing_trajectory(path="probe")

        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(path="cc")
        pump_a_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a no mirror')
        pump_a_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a')
        pump_b_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b no mirror')
        pump_b_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b')

        vcc_spot = patches.Rectangle((vcc_traj[-1][1] / 1e3 - 0.75, vcc_traj[-1][0] / 1e3 - 0.75),
                                     width=1.5, height=1.5, fill=False, edgecolor='green', label='vcc')
        probe_m1_spot = patches.RegularPolygon(
            xy=(probe_m1_traj[-1][1] / 1e3, probe_m1_traj[-1][0] / 1e3),
            numVertices=3, radius=1., fill=False, edgecolor='black', label='m1')
        probe_spot = patches.Circle((probe_traj[-1][1] / 1e3, probe_traj[-1][0] / 1e3),
                                    radius=0.5, fill=False, edgecolor='orange', label='probe')

        cc_spot = patches.Rectangle((pump_ref_traj[-1][1] / 1e3 - 0.75, pump_ref_traj[-1][0] / 1e3 - 0.75),
                                    width=1.5, height=1.5, fill=False, edgecolor='pink', label='cc')
        pump_no_m1a_spot = patches.RegularPolygon(
            xy=(pump_a_no_mirror_traj[-1][1] / 1e3, pump_a_no_mirror_traj[-1][0] / 1e3),
            numVertices=3, radius=1., fill=False, edgecolor='pink', label='m2a')
        pump_no_m2b_spot = patches.RegularPolygon(
            xy=(pump_b_no_mirror_traj[-1][1] / 1e3, pump_b_no_mirror_traj[-1][0] / 1e3),
            numVertices=3, radius=1., fill=False, edgecolor='brown', label='m2b')
        pump_m1a_spot = patches.Circle((pump_a_traj[-1][1] / 1e3, pump_a_traj[-1][0] / 1e3),
                                       radius=0.5, fill=False, edgecolor='red', label='tg a')
        pump_m2b_spot = patches.Circle((pump_b_traj[-1][1] / 1e3, pump_b_traj[-1][0] / 1e3),
                                       radius=0.5, fill=False, edgecolor='purple', label='tg b')

        for item in self.sample.all_optics:
            ax.plot(item.boundary[:, 1] / 1000, item.boundary[:, 0] / 1000, color='blue')

        ax.add_patch(vcc_spot)
        ax.add_patch(probe_m1_spot)
        ax.add_patch(probe_spot)

        ax.add_patch(cc_spot)
        ax.add_patch(pump_no_m1a_spot)
        ax.add_patch(pump_no_m2b_spot)
        ax.add_patch(pump_m1a_spot)
        ax.add_patch(pump_m2b_spot)

        ax.set_title("X-ray coming out of the screen")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc=(1, 0))

    def plot_beam_on_sample_yag(self, ax, aspect=None):

        # Calculate the interaction point
        probe_sample_traj, probe_kout, probe_path = self.get_raytracing_trajectory(path="probe sample")
        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(path="cc sample")
        pump_a_sample_traj, pump_a_kout, pump_a_path = self.get_raytracing_trajectory(path="pump a sample")
        pump_b_sample_traj, pump_b_kout, pump_b_path = self.get_raytracing_trajectory(path="pump b sample")

        # Define the rotation matrix
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.deg2rad(5), axis=np.array([1.0, 0, 0]))
        rot_center = np.copy(self.sample.sample.surface_point)
        # print(rot_mat)

        # Define the object
        tmp = np.dot(probe_sample_traj - rot_center, rot_mat.T)
        # print(tmp)
        probe_spot = patches.Circle((tmp[-1][2] / 1e3, tmp[-1][0] / 1e3),
                                    radius=1, fill=False, edgecolor='green', label='probe')

        tmp = np.dot(pump_ref_traj - rot_center, rot_mat.T)
        pump_ref_spot = patches.Circle((tmp[-1][2] / 1e3, tmp[-1][0] / 1e3),
                                       radius=0.75, fill=False, edgecolor='black', label='cc')

        tmp = np.dot(pump_a_sample_traj - rot_center, rot_mat.T)
        pump_a_spot = patches.Circle((tmp[-1][2] / 1e3, tmp[-1][0] / 1e3),
                                     radius=0.5, fill=False, edgecolor='red', label='pump a')

        tmp = np.dot(pump_b_sample_traj - rot_center, rot_mat.T)
        pump_b_spot = patches.Circle((tmp[-1][2] / 1e3, tmp[-1][0] / 1e3),
                                     radius=0.5, fill=False, edgecolor='purple', label='pump b')

        ax.plot(np.dot(self.sample.sample.boundary - rot_center, rot_mat.T)[:, 2] / 1e3,
                np.dot(self.sample.sample.boundary - rot_center, rot_mat.T)[:, 0] / 1e3,
                color='purple',
                )
        ax.plot(np.dot(self.sample.yag_sample.boundary - rot_center, rot_mat.T)[:, 2] / 1e3,
                np.dot(self.sample.yag_sample.boundary - rot_center, rot_mat.T)[:, 0] / 1e3,
                color='blue', )
        ax.add_patch(probe_spot)
        ax.add_patch(pump_ref_spot)
        ax.add_patch(pump_a_spot)
        ax.add_patch(pump_b_spot)

        ax.set_title('Zyla 2')

        if aspect:
            ax.set_aspect(aspect)

        ax.set_xlabel("horizontal (mm)")
        ax.set_ylabel("vertical (mm)")
        ax.legend()

    def plot_m1_traj(self, ax, axis='yz', xlim=None, ylim=None):

        # Get the most updated trajectory
        vcc_traj, vcc_kout, vcc_pathlength = self.get_raytracing_trajectory(path="vcc")
        probe_m1_traj, probe_m1_kout, probe_m1_pathlength = self.get_raytracing_trajectory(path="probe m1 only")
        # probe_traj, kout, probe_pathlength = self.get_raytracing_trajectory(path="probe")

        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(path="cc")
        # pump_a_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a no mirror')
        # pump_a_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a')
        # pump_b_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b no mirror')
        # pump_b_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b')
        print("Perform ray tracing calculation at current motor position.")

        if xlim is None:
            xlim = [3900, 4100]
        if ylim is None:
            ylim = [-1, 1]

        if axis == 'yz':
            ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 0] / 1e3,
                    color='g', label='vcc')
            ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 0] / 1e3,
                    color='r', label='cc')
            ax.plot(probe_m1_traj[:, 2] / 1e3, probe_m1_traj[:, 0] / 1e3,
                    color='g', linestyle='--', label='probe m1')
            self.plot_optics(ax=ax, axis=axis, color='blue')

        elif axis == 'xz':
            ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 1] / 1e3,
                    color='g', label='vcc')
            ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 1] / 1e3,
                    color='r', label='cc')
            ax.plot(probe_m1_traj[:, 2] / 1e3, probe_m1_traj[:, 1] / 1e3,
                    color='g', linstyle='--', label='probe m1')
            self.plot_optics(ax=ax, axis=axis, color='blue')

        else:
            print("Please check the source code for the option for axis argument.")
            print("The current one \'{}\' is not defined".format(axis))

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_xlabel("{} axis (mm)".format(axis[1]))
        ax.set_ylabel("{} axis (mm)".format(axis[0]))
        ax.set_title('Mirror 1')
        ax.legend()

    def plot_si_traj(self, ax, axis='yz', xlim=None, ylim=None):

        # Get the most updated trajectory
        vcc_traj, vcc_kout, vcc_pathlength = self.get_raytracing_trajectory(path="vcc")
        probe_m1_traj, probe_m1_kout, probe_m1_pathlength = self.get_raytracing_trajectory(path="probe m1 only")
        probe_traj, kout, probe_pathlength = self.get_raytracing_trajectory(path="probe")

        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(path="cc")
        # pump_a_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a no mirror')
        # pump_a_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a')
        # pump_b_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b no mirror')
        # pump_b_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b')
        print("Perform ray tracing calculation at current motor position.")

        if xlim is None:
            xlim = [probe_traj[-1, 2] / 1e3 - 50, probe_traj[-1, 2] / 1e3 + 5]
        if ylim is None:
            ylim = [probe_traj[-1, 0] / 1e3 - 15, probe_traj[-1, 0] / 1e3 + 20]

        if axis == 'yz':
            ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 0] / 1e3,
                    color='g', label='vcc')
            ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 0] / 1e3,
                    color='r', label='cc')
            ax.plot(probe_m1_traj[:, 2] / 1e3, probe_m1_traj[:, 0] / 1e3,
                    color='g', linestyle='--', label='probe m1')
            ax.plot(probe_traj[:, 2] / 1e3, probe_traj[:, 0] / 1e3,
                    color='g', linestyle='dotted', label='probe')

            self.plot_optics(ax=ax, axis=axis, color='blue')

        elif axis == 'xz':
            ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 1] / 1e3,
                    color='g', label='vcc')
            ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 1] / 1e3,
                    color='r', label='cc')
            ax.plot(probe_m1_traj[:, 2] / 1e3, probe_m1_traj[:, 1] / 1e3,
                    color='g', linstyle='--', label='probe m1')
            ax.plot(probe_traj[:, 2] / 1e3, probe_traj[:, 1] / 1e3,
                    color='g', linestyle='dotted', label='probe')

            self.plot_optics(ax=ax, axis=axis, color='blue')

        else:
            print("Please check the source code for the option for axis argument.")
            print("The current one \'{}\' is not defined".format(axis))

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_xlabel("{} axis (mm)".format(axis[1]))
        ax.set_ylabel("{} axis (mm)".format(axis[0]))
        ax.set_title('silicon')
        ax.legend(loc=(1, 0))

    def plot_tg_traj(self, ax, axis='yz', xlim=None, ylim=None):

        # Get the most updated trajectory
        vcc_traj, vcc_kout, vcc_pathlength = self.get_raytracing_trajectory(path="vcc")
        probe_m1_traj, probe_m1_kout, probe_m1_pathlength = self.get_raytracing_trajectory(path="probe m1 only")
        probe_traj, kout, probe_pathlength = self.get_raytracing_trajectory(path="probe")

        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(path="cc")
        pump_a_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a no mirror')
        pump_a_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a')
        pump_b_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b no mirror')
        pump_b_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b')
        print("Perform ray tracing calculation at current motor position.")

        if xlim is None:
            xlim = [7400, 7700]
        if ylim is None:
            ylim = [-5, 5]

        if axis == 'yz':
            ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 0] / 1e3,
                    color='g', label='vcc')
            ax.plot(probe_traj[:, 2] / 1e3, probe_traj[:, 0] / 1e3,
                    color='g', linestyle='dotted', label='probe')

            ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 0] / 1e3,
                    color='r', label='cc')
            ax.plot(pump_a_no_mirror_traj[:, 2] / 1e3, pump_a_no_mirror_traj[:, 0] / 1e3,
                    linestyle='--', color='r', label='pump a')
            ax.plot(pump_a_traj[:, 2] / 1e3, pump_a_traj[:, 0] / 1e3,
                    linestyle='dotted', color='r', label='pump a')
            ax.plot(pump_b_no_mirror_traj[:, 2] / 1e3, pump_b_no_mirror_traj[:, 0] / 1e3,
                    linestyle='--', color='r', label='pump b')
            ax.plot(pump_b_traj[:, 2] / 1e3, pump_b_traj[:, 0] / 1e3,
                    linestyle='dotted', color='r', label='pump b')

            self.plot_optics(ax=ax, axis=axis, color='blue')

        elif axis == 'xz':
            ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 1] / 1e3,
                    color='g', label='vcc')
            ax.plot(probe_traj[:, 2] / 1e3, probe_traj[:, 1] / 1e3,
                    color='g', linestyle='dotted', label='probe')

            ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 1] / 1e3,
                    color='r', label='cc')
            ax.plot(pump_a_no_mirror_traj[:, 2] / 1e3, pump_a_no_mirror_traj[:, 1] / 1e3,
                    linestyle='--', color='r', label='pump a')
            ax.plot(pump_a_traj[:, 2] / 1e3, pump_a_traj[:, 1] / 1e3,
                    linestyle='dotted', color='r', label='pump a')
            ax.plot(pump_b_no_mirror_traj[:, 2] / 1e3, pump_b_no_mirror_traj[:, 1] / 1e3,
                    linestyle='--', color='r', label='pump b')
            ax.plot(pump_b_traj[:, 2] / 1e3, pump_b_traj[:, 1] / 1e3,
                    linestyle='dotted', color='r', label='pump b')

            self.plot_optics(ax=ax, axis=axis, color='blue')

        else:
            print("Please check the source code for the option for axis argument.")
            print("The current one \'{}\' is not defined".format(axis))

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_xlabel("{} axis (mm)".format(axis[1]))
        ax.set_ylabel("{} axis (mm)".format(axis[0]))
        ax.set_title('Sample')
        ax.legend(loc=(1, 0))

    def get_beam_position_on_yag(self):
        vcc_traj, vcc_kout, vcc_pathlength = self.get_raytracing_trajectory(path="vcc")
        probe_m1_traj, probe_m1_kout, probe_m1_pathlength = self.get_raytracing_trajectory(path="probe m1 only")
        probe_traj, kout, probe_pathlength = self.get_raytracing_trajectory(path="probe")

        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(path="cc")
        pump_a_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a no mirror')
        pump_a_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump a')
        pump_b_no_mirror_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b no mirror')
        pump_b_traj, kout, pump_a_path = self.get_raytracing_trajectory(path='pump b')

        return {'vcc': vcc_traj[-1],
                'probe m1': probe_m1_traj[-1],
                'probe': probe_traj[-1],

                'cc': pump_ref_traj[-1],
                'pump a no mirror': pump_a_no_mirror_traj[-1],
                'pump b no mirror': pump_b_no_mirror_traj[-1],
                'pump a': pump_a_traj[-1],
                'pump b': pump_b_traj[-1],
                }

    def get_beam_position_on_sample_yag(self):
        probe_sample_traj, probe_kout, probe_path = self.get_raytracing_trajectory(path="probe sample")
        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(path="cc sample")
        pump_a_sample_traj, pump_a_kout, pump_a_path = self.get_raytracing_trajectory(path="pump a sample")
        pump_b_sample_traj, pump_b_kout, pump_b_path = self.get_raytracing_trajectory(path="pump b sample")

        return {'probe': probe_sample_traj[-1],
                'cc': pump_ref_traj[-1],
                'pump a': pump_a_sample_traj[-1],
                'pump b': pump_b_sample_traj[-1],
                }

    def get_sample_path_length(self):
        probe_sample_traj, probe_kout, probe_path = self.get_raytracing_trajectory(path="probe sample")
        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(path="cc sample")
        pump_a_sample_traj, pump_a_kout, pump_a_path = self.get_raytracing_trajectory(path="pump a sample")
        pump_b_sample_traj, pump_b_kout, pump_b_path = self.get_raytracing_trajectory(path="pump b sample")

        return {'probe': probe_path,
                'cc': pump_ref_path,
                'pump a': pump_a_path,
                'pump b': pump_b_path,
                }

    def get_arrival_time(self):
        probe_sample_traj, probe_kout, probe_path = self.get_raytracing_trajectory(
            path="probe sample", virtual_sample_plane=np.array([0.0, 0.0, -1.0]))
        pump_ref_traj, pump_ref_kout, pump_ref_path = self.get_raytracing_trajectory(
            path="cc sample", virtual_sample_plane=np.array([0.0, 0.0, -1.0]))
        pump_a_sample_traj, pump_a_kout, pump_a_path = self.get_raytracing_trajectory(
            path="pump a sample", virtual_sample_plane=np.array([0.0, 0.0, -1.0]))
        pump_b_sample_traj, pump_b_kout, pump_b_path = self.get_raytracing_trajectory(
            path="pump b sample", virtual_sample_plane=np.array([0.0, 0.0, -1.0]))

        return {'probe': probe_path,
                'cc': pump_ref_path,
                'pump a': pump_a_path,
                'pump b': pump_b_path,
                }

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

    # def save_operation_record(controller, file_name=None):
    #    if file_name is None:
    #        file_name = "~/Desktop/operation_record_{}.h5".format(util.time_stamp())
    #    with h5py.File(file_name, 'wb') as target:
    #        target.create_dataset(name='t1x', data=np.array(controller.record['t1x']))
