import numpy as np
import sys

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


def get_efield_with_interpolation(observation_point, device_list, gaussian_pulse, spec_in, kin_grid, coordinate_dict,
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
    print("The ray-tracing calculation differ from the actual beam center by a few pixels:", offset)

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
                                                             efield_array=outputField,
                                                             k0=gaussian_pulse.klen0,
                                                             mode=mode,
                                                             coor_info_new=None,
                                                             affine_mat=None)
        del outputDict
        outputDict = {}
        outputDict['field_grid'] = field_fit
        outputDict['spectrum_grid'] = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(field_fit)))

        return outputDict, new_coor_dict
    else:
        return outputDict, coordinate_dict
