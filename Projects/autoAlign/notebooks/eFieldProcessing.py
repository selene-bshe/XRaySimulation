import numpy as np
import sys

sys.path.append("../../../../XRaySimulation")

from XRaySimulation import Pulse, DeviceSimu, util, Crystal
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


def get_efield_with_interpolation(observation_point, device_list, gaussian_pulse, spec_in, kin_grid,
                                  flag_interpolation=False, interpolation_purpose="vcc"):
    (cc_trajectory_local,
     cc_kout_list_local,
     cc_path_local) = DeviceSimu.get_lightpath(device_list=device_list,
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
                                                          total_path=cc_path_local,
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

    # Get the electric field
    outputField = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(outputSpec)))

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

    if flag_interpolation:
        if interpolation_purpose == "vcc":
            pass
        elif interpolation_purpose == "TG pump":
            pass
        elif interpolation_purpose == "TG probe":
            pass
        elif interpolation_purpose == "xyz":
            pass
        elif interpolation_purpose == "beam frame":
            pass
        else:
            print("No interpolation is applied. Currently this function cannot handle a general interpolation request.")
            print("Please check the source code for this function to understand the current capability boundary.")
            