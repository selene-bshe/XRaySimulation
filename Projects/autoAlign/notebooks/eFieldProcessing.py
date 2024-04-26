import sys

import numpy as np

sys.path.append("../../../../XRaySimulation")

from XRaySimulation import DeviceSimu
import XRaySimulation.GPU.GPUMultiDevice as gMultiDevice


def get_statistics(summary_dict, array, tag):
    arrayShape = np.array(array.shape)
    summary_dict[tag] = {}
    summary_dict[tag]['yz'] = np.square(np.abs(array[arrayShape[0] // 2, :, :]))
    summary_dict[tag]['xz'] = np.square(np.abs(array[:, arrayShape[1] // 2, :]))
    summary_dict[tag]['xy'] = np.square(np.abs(array[:, :, arrayShape[2] // 2]))

    summary_dict[tag]['z'] = np.square(np.abs(array[arrayShape[0] // 2, arrayShape[1] // 2, :]))
    summary_dict[tag]['x'] = np.square(np.abs(array[arrayShape[0] // 2, :, arrayShape[2] // 2]))
    summary_dict[tag]['y'] = np.square(np.abs(array[:, arrayShape[1] // 2, arrayShape[2] // 2]))


def get_efield_with_interpolation(observation_point,
                                  device_list,
                                  gaussian_pulse,
                                  spec_in,
                                  kin_grid,
                                  coordinate_dict,
                                  coordinate_info_new=None,
                                  flag_interpolation=False,
                                  mode="xyz 3D"):
    (trajectory,
     kout,
     path) = DeviceSimu.get_lightpath(device_list=device_list,
                                      kin=gaussian_pulse.k0,
                                      initial_point=gaussian_pulse.x0,
                                      final_plane_point=observation_point,
                                      final_plane_normal=np.array([0., 0., 1.]))
    input_pulse_shape = np.array(spec_in.shape)
    entry_num = np.prod(input_pulse_shape)

    # -----------------------------------------------------------------
    #   The analysis below is very similar for different cases and therefore can be considered
    #   to be combined in to a single function
    # -----------------------------------------------------------------
    # Get the output electric field spectrum
    outputDict = gMultiDevice.get_multicrystal_reflection(kin_grid=np.reshape(kin_grid, newshape=(entry_num, 3)),
                                                          spectrum_in=np.reshape(spec_in, newshape=entry_num),
                                                          device_list=device_list,
                                                          total_path=path,
                                                          initial_position=gaussian_pulse.x0,
                                                          d_num=512,
                                                          batch_num=int(1),
                                                          flag_reflectivity=False,
                                                          flag_jacobian=False,
                                                          flag_kout=True,
                                                          flag_kout_length=True,
                                                          flag_phase=False)

    # Collect statistics of the spectrum
    outputSpec = np.reshape(outputDict['spectrum_grid'], newshape=input_pulse_shape)
    outputDict['spectrum_grid'] = outputSpec

    # Get the electric field
    outputField = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(outputSpec)))
    # Roll the electric field a little bit
    offset = np.array(np.unravel_index(np.abs(outputField).argmax(), outputField.shape))
    # print("The ray-tracing calculation differ from the actual beam center by a few pixels:", offset)

    outputDict['field_grid'] = np.roll(outputField, shift=-offset + np.array(outputField.shape) // 2, axis=(0, 1, 2))

    # Because almost for sure we have the screen facing towards the Z direction.
    # I only consider the interpolation that generate the electric field in the x,y,t or x,y,z coordinate
    # the same coordiante as the inital pulse.
    # For the theory, find it in Haoyuan Li's thesis

    # The purpose of this interpolation is multiple
    # 1. Demonstrate the pulse front tilt issue after asymmetric channel-cut crystals
    #        for this purpose, the interpolation is within the xpp x xpp z plane, or the y-z plane in this simulation
    # 2. Get the accurate electric field and use that to calculate the TG fringe
    #        for this purpose, the interpolation is within the xpp x xpp z plane, or the y-z plane in this simulation
    # 3. Get the probe pulse electric field and see its spatial overlap with the TG fringe
    #        for this purpose, the interpolation is within the xpp y xpp z plane, or the x-z plane in this simulation

    # Below, I try to implement two kinds of interpolation
    # one is the 2D interpolation. The interpolation dimension is the same as that
    # explained above. The other one is the 3D interpolation.
    # The 3D interpolation is more time-consuming and more accurate.
    # Ideally, in one simulation, I would need to compare the two cases and
    # choose the correct one to implement.

    # This function is so fundamental, I believe I need to create a basic function
    # for this purpose.
    if flag_interpolation:
        (field_fit,
         new_coor_dict) = DeviceSimu.get_interpolated_eField(kvec_array=np.reshape(outputDict['kout_grid'],
                                                                                   newshape=(input_pulse_shape[0],
                                                                                             input_pulse_shape[1],
                                                                                             input_pulse_shape[2],
                                                                                             3), ),
                                                             coor_dict=coordinate_dict,
                                                             efield_array=outputDict['field_grid'],
                                                             k0=gaussian_pulse.klen0,
                                                             mode=mode,
                                                             coor_info_new=coordinate_info_new,
                                                             affine_mat=None)
        del outputDict
        outputDict = {}
        outputDict['field_grid'] = field_fit
        outputDict['spectrum_grid'] = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(field_fit)))

        return outputDict, new_coor_dict
    else:
        return outputDict, coordinate_dict


def get_measurement(controller, sase_field, kGrid, coor_dict):
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
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(sase_field)),
                   tag="sase intensity")

    sase_spec = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sase_field)))

    # Collect the statistics of these fields
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(sase_spec)),
                   tag="sase spectrum")

    # ---------------------------------------------------------------------------
    # Get the electric field after CC1
    # Step 1 get the light path after the first CC crystal
    observation_point = np.array([0, 0, 6e4])

    # Get the trajectory through the targeted optics
    device_list = [controller.expSimu['devices']["sd grating cc"][0], ]
    for idx in range(1):
        device_list += controller.expSimu['devices']["cc"][idx].crystal_list

    # Get the electric field after the first CC
    outputDict, coor_dict = get_efield_with_interpolation(observation_point=observation_point,
                                                          device_list=device_list,
                                                          gaussian_pulse=controller.expSimu['devices'][
                                                              'pulse'],
                                                          spec_in=np.copy(sase_spec),
                                                          kin_grid=np.copy(kGrid),
                                                          coordinate_dict=coor_dict,
                                                          flag_interpolation=False,
                                                          mode="xyz 3D")
    measurements.update({'d1': np.sum(np.square(np.abs(outputDict['field_grid'])))})
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['field_grid'])),
                   tag="cc1 intensity")
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['spectrum_grid'])),
                   tag="cc1 spectrum")

    # ---------------------------------------------------------------------------
    # Get the electric field after CC2
    # Step 1 get the light path after the first CC crystal
    observation_point = np.array([0, 0, 1.1e6])

    # Get the trajectory through the targeted optics
    device_list = [controller.expSimu['devices']["sd grating cc"][0], ]
    for idx in range(2):
        device_list += controller.expSimu['devices']["cc"][idx].crystal_list

    # Get the electric field after the first CC
    outputDict, coor_dict = get_efield_with_interpolation(observation_point=observation_point,
                                                          device_list=device_list,
                                                          gaussian_pulse=controller.expSimu['devices'][
                                                              'pulse'],
                                                          spec_in=np.copy(sase_spec),
                                                          kin_grid=np.copy(kGrid),
                                                          coordinate_dict=coor_dict,
                                                          flag_interpolation=False,
                                                          mode="xyz 3D")

    # Get the d6 output assuming that there is no shutter and no influence from the VCC branch.
    measurements.update({'d6 raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['field_grid'])),
                   tag="cc2 intensity")
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['spectrum_grid'])),
                   tag="cc2 spectrum")

    # ---------------------------------------------------------------------------
    # Get the CC pulse on the sample without focusing
    observation_point = np.array([0, 0, 10e6])

    # Get the trajectory through the targeted optics
    device_list = [controller.expSimu['devices']["sd grating cc"][0], ]
    for idx in range(2):
        device_list += controller.expSimu['devices']["cc"][idx].crystal_list
    device_list += [controller.expSimu['devices']["sd grating cc"][1], ]

    # Get the electric field after the first CC
    outputDict, coor_dict = get_efield_with_interpolation(observation_point=observation_point,
                                                          device_list=device_list,
                                                          gaussian_pulse=controller.expSimu['devices'][
                                                              'pulse'],
                                                          spec_in=np.copy(sase_spec),
                                                          kin_grid=np.copy(kGrid),
                                                          coordinate_dict=coor_dict,
                                                          flag_interpolation=False,
                                                          mode="xyz 3D")

    measurements.update({'cc sample raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['field_grid'])),
                   tag="cc sample intensity")
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['spectrum_grid'])),
                   tag="cc sample spectrum")

    (cc_trajectory_local,
     cc_kout_list_local,
     cc_path_local) = DeviceSimu.get_lightpath(device_list=device_list,
                                               kin=controller.expSimu['devices'][
                                                   'pulse'].k0,
                                               initial_point=controller.expSimu['devices'][
                                                   'pulse'].x0,
                                               final_plane_point=np.array([0, 0, 10e6]),
                                               final_plane_normal=np.array([0, 0, 1]))

    # ------------------------------------------------------------------------------
    #   VCC 1
    # ------------------------------------------------------------------------------
    # Step 1 get the light path after the first CC crystal
    observation_point = np.array([0, 0, 3e5])

    # Get the trajectory through the targeted optics
    device_list = [controller.expSimu['devices']["sd grating vcc"][0], ]
    for idx in range(1):
        device_list += controller.expSimu['devices']["vcc"][idx].crystal_list

    # Get the electric field after the first CC
    (outputDict,
     coor_dict) = get_efield_with_interpolation(observation_point=observation_point,
                                                device_list=device_list,
                                                gaussian_pulse=controller.expSimu['devices']['pulse'],
                                                spec_in=np.copy(sase_spec),
                                                kin_grid=np.copy(kGrid),
                                                coordinate_dict=coor_dict,
                                                coordinate_info_new={"nx": 32,
                                                                     'ny': 32,
                                                                     'nz': 1024,
                                                                     'dx': 16,
                                                                     'dy': 16,
                                                                     'dz': coor_dict['zCoor'][1] -
                                                                           coor_dict['zCoor'][
                                                                               0], },
                                                flag_interpolation=True,
                                                mode="xyz 3D")

    measurements.update({'d2': np.sum(np.square(np.abs(outputDict['field_grid'])))})
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['field_grid'])),
                   tag="vcc1 intensity")
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['spectrum_grid'])),
                   tag="vcc1 spectrum")
    # ------------------------------------------------------------------------------
    #   VCC 2
    # ------------------------------------------------------------------------------
    # Step 1 get the light path after the first CC crystal
    observation_point = np.array([0, 0, 5e5])

    # Get the trajectory through the targeted optics
    device_list = [controller.expSimu['devices']["sd grating vcc"][0], ]
    for idx in range(2):
        device_list += controller.expSimu['devices']["vcc"][idx].crystal_list

    # Get the electric field after the first CC
    (outputDict,
     coor_dict) = get_efield_with_interpolation(observation_point=observation_point,
                                                device_list=device_list,
                                                gaussian_pulse=controller.expSimu['devices']['pulse'],
                                                spec_in=np.copy(sase_spec),
                                                kin_grid=np.copy(kGrid),
                                                coordinate_dict=coor_dict,
                                                coordinate_info_new={"nx": 32,
                                                                     'ny': 32,
                                                                     'nz': 1024,
                                                                     'dx': 16,
                                                                     'dy': 16,
                                                                     'dz': coor_dict['zCoor'][1] -
                                                                           coor_dict['zCoor'][
                                                                               0], },
                                                flag_interpolation=True,
                                                mode="xyz 3D")

    measurements.update({'d3': np.sum(np.square(np.abs(outputDict['field_grid'])))})
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['field_grid'])),
                   tag="vcc2 intensity")
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['spectrum_grid'])),
                   tag="vcc2 spectrum")
    # ------------------------------------------------------------------------------
    #   VCC 3
    # ------------------------------------------------------------------------------
    # Step 1 get the light path after the first CC crystal
    observation_point = np.array([0, 0, 7e5])

    # Get the trajectory through the targeted optics
    device_list = [controller.expSimu['devices']["sd grating vcc"][0], ]
    for idx in range(3):
        device_list += controller.expSimu['devices']["vcc"][idx].crystal_list

    # Get the electric field after the first CC
    (outputDict,
     coor_dict) = get_efield_with_interpolation(observation_point=observation_point,
                                                device_list=device_list,
                                                gaussian_pulse=controller.expSimu['devices']['pulse'],
                                                spec_in=np.copy(sase_spec),
                                                kin_grid=np.copy(kGrid),
                                                coordinate_dict=coor_dict,
                                                coordinate_info_new={"nx": 32,
                                                                     'ny': 32,
                                                                     'nz': 1024,
                                                                     'dx': 16,
                                                                     'dy': 16,
                                                                     'dz': coor_dict['zCoor'][1] -
                                                                           coor_dict['zCoor'][
                                                                               0], },
                                                flag_interpolation=True,
                                                mode="xyz 3D")

    measurements.update({'d4 raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})

    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['field_grid'])),
                   tag="vcc3 intensity")
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['spectrum_grid'])),
                   tag="vcc3 spectrum")

    # ------------------------------------------------------------------------------
    #   VCC 4
    # ------------------------------------------------------------------------------
    # Step 1 get the light path after the first CC crystal
    observation_point = np.array([0, 0, 9e5])

    # Get the trajectory through the targeted optics
    device_list = [controller.expSimu['devices']["sd grating vcc"][0], ]
    for idx in range(4):
        device_list += controller.expSimu['devices']["vcc"][idx].crystal_list

    # Get the electric field after the first CC
    (outputDict,
     coor_dict) = get_efield_with_interpolation(observation_point=observation_point,
                                                device_list=device_list,
                                                gaussian_pulse=controller.expSimu['devices']['pulse'],
                                                spec_in=np.copy(sase_spec),
                                                kin_grid=np.copy(kGrid),
                                                coordinate_dict=coor_dict,
                                                coordinate_info_new={"nx": 32,
                                                                     'ny': 32,
                                                                     'nz': 1024,
                                                                     'dx': 16,
                                                                     'dy': 16,
                                                                     'dz': coor_dict['zCoor'][1] -
                                                                           coor_dict['zCoor'][
                                                                               0], },
                                                flag_interpolation=True,
                                                mode="xyz 3D")

    measurements.update({'d5 raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['field_grid'])),
                   tag="vcc4 intensity")
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['spectrum_grid'])),
                   tag="vcc4 spectrum")

    # ------------------------------------------------------------------------------
    #   VCC sample
    # ------------------------------------------------------------------------------
    # Step 1 get the light path after the first CC crystal
    observation_point = np.array([0, 0, 10e6])

    # Get the trajectory through the targeted optics
    device_list = [controller.expSimu['devices']["sd grating vcc"][0], ]
    for idx in range(4):
        device_list += controller.expSimu['devices']["vcc"][idx].crystal_list
    device_list += [controller.expSimu['devices']["sd grating vcc"][1], ]

    # Get the electric field after the first CC
    (outputDict,
     coor_dict) = get_efield_with_interpolation(observation_point=observation_point,
                                                device_list=device_list,
                                                gaussian_pulse=controller.expSimu['devices']['pulse'],
                                                spec_in=np.copy(sase_spec),
                                                kin_grid=np.copy(kGrid),
                                                coordinate_dict=coor_dict,
                                                coordinate_info_new={"nx": 32,
                                                                     'ny': 32,
                                                                     'nz': 1024,
                                                                     'dx': 16,
                                                                     'dy': 16,
                                                                     'dz': coor_dict['zCoor'][1] -
                                                                           coor_dict['zCoor'][
                                                                               0], },
                                                flag_interpolation=True,
                                                mode="xyz 3D")

    measurements.update({'vcc sample raw': np.sum(np.square(np.abs(outputDict['field_grid'])))})
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['field_grid'])),
                   tag="vcc sample intensity")
    get_statistics(summary_dict=simulation_statistics,
                   array=np.square(np.abs(outputDict['spectrum_grid'])),
                   tag="vcc sample spectrum")

    # Get the current interaction point with the YAG screen with the
    (vcc_trajectory_local,
     vcc_kout_list_local,
     vcc_path_local) = DeviceSimu.get_lightpath(device_list=device_list,
                                                kin=controller.expSimu['devices'][
                                                    'pulse'].k0,
                                                initial_point=controller.expSimu['devices'][
                                                    'pulse'].x0,
                                                final_plane_point=np.array([0, 0, 10e6]),
                                                final_plane_normal=np.array([0, 0, 1]))

    # -----------------------------------------------------------
    #    Calculate the influence of the shutter on the diode output
    # -----------------------------------------------------------
    if controller.cc_shutter:
        measurements.update({'d6': 0 + measurements['d6 raw']})
        measurements.update({'dsample': 0 + measurements['cc sample raw']})
    else:
        measurements.update({'d6': 0})
        measurements.update({'dsample': 0})

    if controller.vcc_shutter:
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
    yag_image = np.zeros((controller.pixel_num_x, controller.pixel_num_y))
    if controller.cc_shutter:
        yag_image += np.reshape(
            DeviceSimu.get_intensity_on_YAG(intensity=simulation_statistics['cc sample intensity']['xy'],
                                            intensity_coor=coor_dict,
                                            intensity_loc=cc_trajectory_local[-1][:2],
                                            pixel_coor={'xCoor': controller.pixel_pos_x,
                                                        'yCoor': controller.pixel_pos_y}),
            (controller.pixel_num_x, controller.pixel_num_y))
        print(cc_trajectory_local[-1][:2])
    if controller.vcc_shutter:
        yag_image += np.reshape(
            DeviceSimu.get_intensity_on_YAG(intensity=simulation_statistics['vcc sample intensity']['xy'],
                                            intensity_coor=coor_dict,
                                            intensity_loc=vcc_trajectory_local[-1][:2],
                                            pixel_coor={'xCoor': controller.pixel_pos_x,
                                                        'yCoor': controller.pixel_pos_y}),
            (controller.pixel_num_x, controller.pixel_num_y))
        print(vcc_trajectory_local[-1][:2])
    measurements.update({"zyla": yag_image})

    return measurements, simulation_statistics
