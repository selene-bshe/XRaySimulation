import os

import numpy as np

import sys

sys.path.append("../../../../XRaySimulation")

from XRaySimulation import Crystal, DeviceSimu
from XRaySimulation.Machine import Motors, ScintillatorCamera

# The following modules are loaded as a temporary solution
import rayTracingCalculation
import eFieldProcessing


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

        # For a quick and temporary solution.
        # I'll start with the aligned optics
        expSimu = rayTracingCalculation.get_miniSD_and_TG_trajectory()
        (t2x_coef,
         t3x_coef,
         t4x_coef,
         t5x_coef,
         probe1z_coef,
         probe1alpha_coef,
         probe2z_coef,
         pump1z_coef,
         pump1alpha_coef,
         pump2z_coef,
         pump2alpha_coef) = rayTracingCalculation.get_trajectory_dependence_on_various_parameters()

        # self.cc1 = optics['cc1']
        # self.cc2 = optics['cc2']
        # self.vcc1 = optics['vcc1']
        # self.vcc2 = optics['vcc2']
        # self.vcc3 = optics['vcc3']
        # self.vcc4 = optics['vcc4']

        self.cc1 = expSimu['devices']['cc'][0]
        self.cc2 = expSimu['devices']['cc'][1]
        self.vcc1 = expSimu['devices']['vcc'][0]
        self.vcc2 = expSimu['devices']['vcc'][1]
        self.vcc3 = expSimu['devices']['vcc'][2]
        self.vcc4 = expSimu['devices']['vcc'][3]

        # This a temporary entry which contains the aligned optics from the old method
        self.expSimu = expSimu

        # Add the shutter
        self.cc_shutter = True
        self.vcc_shutter = True

        # Step 3 Move the devices to their rough position

        # Step 4 Rotate the crystals such that they are at the ideal location

        # Step 5 Add diodes

        # Step 6 Add cameras
        self.pixel_num_x = 2048
        self.pixel_num_y = 2048

        self.pixel_pos_x = np.linspace(- 1024 * 6.5 / 10, 1024 * 6.5 / 10, 2048) + expSimu['cc']['trajectory'][-1, 1]
        self.pixel_pos_y = np.linspace(- 1024 * 6.5 / 10, 1024 * 6.5 / 10, 2048) + expSimu['cc']['trajectory'][-1, 0]

    def plot_motors(self):
        pass

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

    def get_measurement(self, sase_field, kGrid, coor_dict):
        """
        This is a temporary solution.
        I'll just calculate all the quantity that I need.
        Later, this will be replaced with something more reasonable.

        :return:
        """
        simulation_statistics = {}
        measurements = {}

        # Get the incident pulse energy  # always use the summation of the electric field as the energy
        measurements.update({'ipm2': np.sum(np.square(np.abs(sase_field)))})
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(sase_field)),
                                        tag="sase intensity")

        sase_spec = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sase_field)))

        # Collect the statistics of these fields
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(sase_spec)),
                                        tag="sase spectrum")

        # ---------------------------------------------------------------------------
        # Get the electric field after CC1
        # Step 1 get the light path after the first CC crystal
        observation_point = np.array([0, 0, 6e4])

        # Get the trajectory through the targeted optics
        device_list = [self.expSimu['devices']["sd grating cc"][0], ]
        for idx in range(1):
            device_list += self.expSimu['devices']["cc"][idx].crystal_list

        # Get the electric field after the first CC
        outputDict, coor_dict = eFieldProcessing.get_efield_with_interpolation(observation_point=observation_point,
                                                                               device_list=device_list,
                                                                               gaussian_pulse=self.expSimu['devices'][
                                                                                   'pulse'],
                                                                               spec_in=sase_spec,
                                                                               kin_grid=kGrid,
                                                                               coordinate_dict=coor_dict,
                                                                               flag_interpolation=False,
                                                                               mode="xyz 3D")
        measurements.update({'d1': np.sum(np.square(np.abs(outputDict['field_grid'])))})
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['field_grid'])),
                                        tag="cc1 intensity")
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['spectrum_grid'])),
                                        tag="cc1 spectrum")

        # ---------------------------------------------------------------------------
        # Get the electric field after CC2
        # Step 1 get the light path after the first CC crystal
        observation_point = np.array([0, 0, 1.1e6])

        # Get the trajectory through the targeted optics
        device_list = [self.expSimu['devices']["sd grating cc"][0], ]
        for idx in range(2):
            device_list += self.expSimu['devices']["cc"].crystal_list

        # Get the electric field after the first CC
        outputDict, coor_dict = eFieldProcessing.get_efield_with_interpolation(observation_point=observation_point,
                                                                               device_list=device_list,
                                                                               gaussian_pulse=self.expSimu['devices'][
                                                                                   'pulse'],
                                                                               spec_in=sase_spec,
                                                                               kin_grid=kGrid,
                                                                               coordinate_dict=coor_dict,
                                                                               flag_interpolation=False,
                                                                               mode="xyz 3D")

        # Get the d6 output assuming that there is no shutter and no influence from the VCC branch.
        measurements.update({'d6 raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['field_grid'])),
                                        tag="cc2 intensity")
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['spectrum_grid'])),
                                        tag="cc2 spectrum")

        # ---------------------------------------------------------------------------
        # Get the CC pulse on the sample without focusing
        observation_point = np.array([0, 0, 10e6])

        # Get the trajectory through the targeted optics
        device_list = [self.expSimu['devices']["sd grating cc"][0], ]
        for idx in range(2):
            device_list += self.expSimu['devices']["cc"][idx].crystal_list
        device_list += [self.expSimu['devices']["sd grating cc"][1], ]

        # Get the electric field after the first CC
        outputDict, coor_dict = eFieldProcessing.get_efield_with_interpolation(observation_point=observation_point,
                                                                               device_list=device_list,
                                                                               gaussian_pulse=self.expSimu['devices'][
                                                                                   'pulse'],
                                                                               spec_in=sase_spec,
                                                                               kin_grid=kGrid,
                                                                               coordinate_dict=coor_dict,
                                                                               flag_interpolation=False,
                                                                               mode="xyz 3D")

        measurements.update({'cc sample raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['field_grid'])),
                                        tag="cc sample intensity")
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['spectrum_grid'])),
                                        tag="cc sample spectrum")

        (cc_trajectory_local,
         cc_kout_list_local,
         cc_path_local) = DeviceSimu.get_lightpath(device_list=device_list,
                                                   kin=self.expSimu['devices'][
                                                       'pulse'].k0,
                                                   initial_point=self.expSimu['devices'][
                                                       'pulse'].x0,
                                                   final_plane_point=np.array([0, 0, 10e6]),
                                                   final_plane_normal=np.array([0, 0, 1]))

        # ------------------------------------------------------------------------------
        #   VCC 1
        # ------------------------------------------------------------------------------
        # Step 1 get the light path after the first CC crystal
        observation_point = np.array([0, 0, 3e5])

        # Get the trajectory through the targeted optics
        device_list = [self.expSimu['devices']["sd grating vcc"][0], ]
        for idx in range(1):
            device_list += self.expSimu['devices']["vcc"][idx].crystal_list

        # Get the electric field after the first CC
        (outputDict,
         coor_dict) = eFieldProcessing.get_efield_with_interpolation(observation_point=observation_point,
                                                                     device_list=device_list,
                                                                     gaussian_pulse=self.expSimu['devices']['pulse'],
                                                                     spec_in=sase_spec,
                                                                     kin_grid=kGrid,
                                                                     coordinate_dict=coor_dict,
                                                                     coordinate_info_new={"nx": 4,
                                                                                          'ny': 128,
                                                                                          'nz': 1024,
                                                                                          'dx': 4,
                                                                                          'dy': 1,
                                                                                          'dz': coor_dict['zCoor'][1] -
                                                                                                coor_dict['zCoor'][
                                                                                                    0], },
                                                                     flag_interpolation=True,
                                                                     mode="xyz 3D")

        measurements.update({'d2': np.sum(np.square(np.abs(outputDict['field_grid'])))})
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['field_grid'])),
                                        tag="vcc1 intensity")
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['spectrum_grid'])),
                                        tag="vcc1 spectrum")
        # ------------------------------------------------------------------------------
        #   VCC 2
        # ------------------------------------------------------------------------------
        # Step 1 get the light path after the first CC crystal
        observation_point = np.array([0, 0, 5e5])

        # Get the trajectory through the targeted optics
        device_list = [self.expSimu['devices']["sd grating vcc"][0], ]
        for idx in range(2):
            device_list += self.expSimu['devices']["vcc"][idx].crystal_list

        # Get the electric field after the first CC
        (outputDict,
         coor_dict) = eFieldProcessing.get_efield_with_interpolation(observation_point=observation_point,
                                                                     device_list=device_list,
                                                                     gaussian_pulse=self.expSimu['devices']['pulse'],
                                                                     spec_in=sase_spec,
                                                                     kin_grid=kGrid,
                                                                     coordinate_dict=coor_dict,
                                                                     coordinate_info_new={"nx": 4,
                                                                                          'ny': 128,
                                                                                          'nz': 1024,
                                                                                          'dx': 4,
                                                                                          'dy': 1,
                                                                                          'dz': coor_dict['zCoor'][1] -
                                                                                                coor_dict['zCoor'][
                                                                                                    0], },
                                                                     flag_interpolation=True,
                                                                     mode="xyz 3D")

        measurements.update({'d3': np.sum(np.square(np.abs(outputDict['field_grid'])))})
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['field_grid'])),
                                        tag="vcc2 intensity")
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['spectrum_grid'])),
                                        tag="vcc2 spectrum")
        # ------------------------------------------------------------------------------
        #   VCC 3
        # ------------------------------------------------------------------------------
        # Step 1 get the light path after the first CC crystal
        observation_point = np.array([0, 0, 7e5])

        # Get the trajectory through the targeted optics
        device_list = [self.expSimu['devices']["sd grating vcc"][0], ]
        for idx in range(3):
            device_list += self.expSimu['devices']["vcc"][idx].crystal_list

        # Get the electric field after the first CC
        (outputDict,
         coor_dict) = eFieldProcessing.get_efield_with_interpolation(observation_point=observation_point,
                                                                     device_list=device_list,
                                                                     gaussian_pulse=self.expSimu['devices']['pulse'],
                                                                     spec_in=sase_spec,
                                                                     kin_grid=kGrid,
                                                                     coordinate_dict=coor_dict,
                                                                     coordinate_info_new={"nx": 4,
                                                                                          'ny': 128,
                                                                                          'nz': 1024,
                                                                                          'dx': 4,
                                                                                          'dy': 1,
                                                                                          'dz': coor_dict['zCoor'][1] -
                                                                                                coor_dict['zCoor'][
                                                                                                    0], },
                                                                     flag_interpolation=True,
                                                                     mode="xyz 3D")

        measurements.update({'d4 raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})

        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['field_grid'])),
                                        tag="vcc3 intensity")
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['spectrum_grid'])),
                                        tag="vcc3 spectrum")

        # ------------------------------------------------------------------------------
        #   VCC 4
        # ------------------------------------------------------------------------------
        # Step 1 get the light path after the first CC crystal
        observation_point = np.array([0, 0, 9e5])

        # Get the trajectory through the targeted optics
        device_list = [self.expSimu['devices']["sd grating vcc"][0], ]
        for idx in range(4):
            device_list += self.expSimu['devices']["vcc"][idx].crystal_list

        # Get the electric field after the first CC
        (outputDict,
         coor_dict) = eFieldProcessing.get_efield_with_interpolation(observation_point=observation_point,
                                                                     device_list=device_list,
                                                                     gaussian_pulse=self.expSimu['devices']['pulse'],
                                                                     spec_in=sase_spec,
                                                                     kin_grid=kGrid,
                                                                     coordinate_dict=coor_dict,
                                                                     coordinate_info_new={"nx": 4,
                                                                                          'ny': 128,
                                                                                          'nz': 1024,
                                                                                          'dx': 4,
                                                                                          'dy': 1,
                                                                                          'dz': coor_dict['zCoor'][1] -
                                                                                                coor_dict['zCoor'][
                                                                                                    0], },
                                                                     flag_interpolation=True,
                                                                     mode="xyz 3D")

        measurements.update({'d5 raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['field_grid'])),
                                        tag="vcc4 intensity")
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['spectrum_grid'])),
                                        tag="vcc4 spectrum")

        # ------------------------------------------------------------------------------
        #   VCC sample
        # ------------------------------------------------------------------------------
        # Step 1 get the light path after the first CC crystal
        observation_point = np.array([0, 0, 10e6])

        # Get the trajectory through the targeted optics
        device_list = [self.expSimu['devices']["sd grating vcc"][0], ]
        for idx in range(4):
            device_list += self.expSimu['devices']["vcc"][idx].crystal_list
        device_list += [self.expSimu['devices']["sd grating vcc"][1], ]

        # Get the electric field after the first CC
        (outputDict,
         coor_dict) = eFieldProcessing.get_efield_with_interpolation(observation_point=observation_point,
                                                                     device_list=device_list,
                                                                     gaussian_pulse=self.expSimu['devices']['pulse'],
                                                                     spec_in=sase_spec,
                                                                     kin_grid=kGrid,
                                                                     coordinate_dict=coor_dict,
                                                                     coordinate_info_new={"nx": 4,
                                                                                          'ny': 128,
                                                                                          'nz': 1024,
                                                                                          'dx': 4,
                                                                                          'dy': 1,
                                                                                          'dz': coor_dict['zCoor'][1] -
                                                                                                coor_dict['zCoor'][
                                                                                                    0], },
                                                                     flag_interpolation=True,
                                                                     mode="xyz 3D")

        measurements.update({'vcc sample raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['field_grid'])),
                                        tag="vcc sample intensity")
        eFieldProcessing.get_statistics(summary_dict=simulation_statistics,
                                        array=np.square(np.abs(outputDict['spectrum_grid'])),
                                        tag="vcc sample spectrum")

        # Get the current interaction point with the YAG screen with the
        (vcc_trajectory_local,
         vcc_kout_list_local,
         vcc_path_local) = DeviceSimu.get_lightpath(device_list=device_list,
                                                    kin=self.expSimu['devices'][
                                                        'pulse'].k0,
                                                    initial_point=self.expSimu['devices'][
                                                        'pulse'].x0,
                                                    final_plane_point=np.array([0, 0, 10e6]),
                                                    final_plane_normal=np.array([0, 0, 1]))

        # -----------------------------------------------------------
        #    Calculate the influence of the shutter on the diode output
        # -----------------------------------------------------------
        if self.cc_shutter:
            measurements.update({'d6': 0 + measurements['d6 raw']})
            measurements.update({'dsample': 0 + measurements['cc sample raw']})
        else:
            measurements.update({'d6': 0})
            measurements.update({'dsample': 0})

        if self.vcc_shutter:
            measurements.update({'d4': measurements['d4 raw']})
            measurements.update({'d5': measurements['d5 raw']})
            measurements['d6'] += measurements['d5 raw']
            measurements['dsample'] += measurements['vcc sample raw']
        else:
            measurements.update({'d4': 0})
            measurements.update({'d5': 0})

        # -----------------------------------------------------------
        #    Calculate the spatial profile of the beam on the screen based on the ray-tracing calculation
        # -----------------------------------------------------------
        yag_image = np.zeros((self.pixel_num_x, self.pixel_num_y))
        if self.cc_shutter:
            yag_image += DeviceSimu.get_intensity_on_YAG(intensity=simulation_statistics['cc sample intensity']['xy'],
                                                         intensity_coor=coor_dict,
                                                         intensity_loc=cc_trajectory_local[-1][:2],
                                                         pixel_coor=(self.pixel_pos_x, self.pixel_pos_y))
        if self.vcc_shutter:
            yag_image += DeviceSimu.get_intensity_on_YAG(intensity=simulation_statistics['vcc sample intensity']['xy'],
                                                         intensity_coor=coor_dict,
                                                         intensity_loc=vcc_trajectory_local[-1][:2],
                                                         pixel_coor=(self.pixel_pos_x, self.pixel_pos_y))
        measurements.update({"zyla": yag_image})

        return measurements, simulation_statistics


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
