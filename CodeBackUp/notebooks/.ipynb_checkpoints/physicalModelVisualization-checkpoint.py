import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm

plt.rcParams.update({'font.size': 14})


def showCrystalEdges(simulation_summary):
    #############################################
    #  The intersection point on the lower branch
    #############################################
    fig, axes = plt.subplots(6, 2)
    fig.set_figheight(27)
    fig.set_figwidth(9)

    for n1 in range(2):
        for n2 in range(2):
            ax = axes[n1, n2]

            for x in range(4):
                for idx in range(2):
                    ax.plot(simulation_summary['devices']['vcc'][x].crystal_list[idx].boundary[:, 2] / 1000,
                            simulation_summary['devices']['vcc'][x].crystal_list[idx].boundary[:, 1] / 1000,
                            c='k')

            for x in range(2):
                for idx in range(2):
                    ax.plot(simulation_summary['devices']['cc'][x].crystal_list[idx].boundary[:, 2] / 1000,
                            simulation_summary['devices']['cc'][x].crystal_list[idx].boundary[:, 1] / 1000,
                            c='k')

            # Plot a horizontal reference line
            ax.plot(np.arange(-50, 1100), np.zeros(1150), c='b', linestyle='--')

            # Plot the miniSD trajectory
            ax.plot(simulation_summary['vcc']['trajectory'][:, 2] / 1000,
                    simulation_summary['vcc']['trajectory'][:, 1] / 1000, 'g')

            ax.plot(simulation_summary['cc']['trajectory'][:, 2] / 1000,
                    simulation_summary['cc']['trajectory'][:, 1] / 1000, 'r')

            ax.set_aspect("equal")
            ax.set_xlabel("z axis (mm)")
            ax.set_ylabel("x axis (mm)")

            idx = 2 * n1 + n2 + 2
            ax.set_xlim(
                [simulation_summary['cc']['trajectory'][idx, 2] / 1000 - 10,
                 simulation_summary['cc']['trajectory'][idx, 2] / 1000 + 10])
            ax.set_ylim(
                [simulation_summary['cc']['trajectory'][idx, 1] / 1000 - 10,
                 simulation_summary['cc']['trajectory'][idx, 1] / 1000 + 10])

            ax.set_title("CC {}, surface {}".format(n1 + 1, n2 + 1))

    for n1 in range(2, 6):
        for n2 in range(2):
            ax = axes[n1, n2]

            for x in range(4):
                for idx in range(2):
                    ax.plot(simulation_summary['devices']['vcc'][x].crystal_list[idx].boundary[:, 2] / 1000,
                            simulation_summary['devices']['vcc'][x].crystal_list[idx].boundary[:, 1] / 1000,
                            c='k')

            for x in range(2):
                for idx in range(2):
                    ax.plot(simulation_summary['devices']['cc'][x].crystal_list[idx].boundary[:, 2] / 1000,
                            simulation_summary['devices']['cc'][x].crystal_list[idx].boundary[:, 1] / 1000,
                            c='k')

            # Plot a horizontal reference line
            ax.plot(np.arange(-50, 1100), np.zeros(1150), c='b', linestyle='--')

            # Plot the miniSD trajectory
            ax.plot(simulation_summary['vcc']['trajectory'][:, 2] / 1000,
                    simulation_summary['vcc']['trajectory'][:, 1] / 1000, 'g')

            ax.plot(simulation_summary['cc']['trajectory'][:, 2] / 1000,
                    simulation_summary['cc']['trajectory'][:, 1] / 1000, 'r')

            ax.set_aspect("equal")
            ax.set_xlabel("z axis (mm)")
            ax.set_ylabel("x axis (mm)")

            idx = 2 * (n1 - 2) + n2 + 2
            ax.set_xlim(
                [simulation_summary['vcc']['trajectory'][idx, 2] / 1000 - 10,
                 simulation_summary['vcc']['trajectory'][idx, 2] / 1000 + 10])
            ax.set_ylim(
                [simulation_summary['vcc']['trajectory'][idx, 1] / 1000 - 10,
                 simulation_summary['vcc']['trajectory'][idx, 1] / 1000 + 10])

            ax.set_title("VCC {}, surface {}".format(n1 - 1, n2 + 1))

    plt.tight_layout()
    plt.show()


def showSDandTGtrajectory(simulation_summary):
    fig, axes = plt.subplots(ncols=1, nrows=6)

    fig.set_figheight(20)
    fig.set_figwidth(10)

    # --------------------------------------------------------------------------
    # Show the trajectory of the miniSD region
    cmap = colormaps.get_cmap('spring')
    for x in range(4):
        for idx in range(2):
            axes[0].plot(simulation_summary['devices']['vcc'][x].crystal_list[idx].boundary[:, 2] / 1000,
                         simulation_summary['devices']['vcc'][x].crystal_list[idx].boundary[:, 1] / 1000,
                         c='k')
            # c=cmap((x * 2 + idx) / 8))

    for x in range(2):
        for idx in range(2):
            axes[0].plot(simulation_summary['devices']['cc'][x].crystal_list[idx].boundary[:, 2] / 1000,
                         simulation_summary['devices']['cc'][x].crystal_list[idx].boundary[:, 1] / 1000,
                         c='k')

    # Plot a horizontal reference line
    axes[0].plot(np.arange(-50, 1100), np.zeros(1150), c='b', linestyle='--')

    # Plot the miniSD trajectory
    axes[0].plot(simulation_summary['vcc']['trajectory'][:, 2] / 1000,
                 simulation_summary['vcc']['trajectory'][:, 1] / 1000, 'g', label='vcc')

    axes[0].plot(simulation_summary['cc']['trajectory'][:, 2] / 1000,
                 simulation_summary['cc']['trajectory'][:, 1] / 1000, 'r', label='cc')

    axes[0].set_aspect("equal")
    axes[0].set_ylim([-100, 100])
    axes[0].set_xlim([-100, 1200])
    axes[0].set_xlabel("z axis (mm)")
    axes[0].set_ylabel("x axis (mm)")
    axes[0].set_title('SD table')
    axes[0].legend()

    # --------------------------------------------------------------------------
    # Show the TG part in the x-z plane

    # Plot a horizontal reference line
    axes[1].plot(np.arange(-50 - 100, 700) * 15, np.zeros(850) * 10, c='b', linestyle='--')

    # Plot the miniSD trajectory
    axes[1].plot(simulation_summary['TG probe']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG probe']['trajectory'][:, 1] / 1000, 'r', label='probe')

    axes[1].plot(simulation_summary['TG pump a']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump a']['trajectory'][:, 1] / 1000, 'g', label='pump a')

    axes[1].plot(simulation_summary['TG pump b']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump b']['trajectory'][:, 1] / 1000, 'g--', label='pump b')

    axes[1].set_xlabel("z axis (mm)")
    axes[1].set_ylabel("x axis (mm)")

    axes[1].set_title('Top view')
    axes[1].legend()

    # --------------------------------------------------------------------------
    # Show the TG part in the y-z plane

    # Plot a horizontal reference line
    axes[2].plot(np.arange(-50 - 100, 700) * 15, np.zeros(850) * 10, c='b', linestyle='--')

    # Plot the miniSD trajectory
    axes[2].plot(simulation_summary['TG probe']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG probe']['trajectory'][:, 0] / 1000, 'r', label='probe')

    axes[2].plot(simulation_summary['TG pump a']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump a']['trajectory'][:, 0] / 1000, 'g', label='pump a')

    axes[2].plot(simulation_summary['TG pump b']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump b']['trajectory'][:, 0] / 1000, 'g--', label='pump b')

    axes[2].set_xlabel("z axis (mm)")
    axes[2].set_ylabel("y axis (mm)")
    axes[2].set_title('side view')

    # --------------------------------------------------
    #   Sample region
    # Plot a horizontal reference line
    axes[3].plot(np.arange(-50 - 100, 700) * 15, np.zeros(850) * 10, c='b', linestyle='--')

    # Plot the miniSD trajectory
    axes[3].plot(simulation_summary['TG probe']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG probe']['trajectory'][:, 1] / 1000, 'r', label='probe')

    axes[3].plot(simulation_summary['TG pump a']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump a']['trajectory'][:, 1] / 1000, 'g', label='pump a')

    axes[3].plot(simulation_summary['TG pump b']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump b']['trajectory'][:, 1] / 1000, 'g--', label='pump b')

    axes[3].set_xlabel("z axis (mm)")
    axes[3].set_ylabel("x axis (mm)")
    axes[3].set_xlim([8e3 - 200, 8e3 + 5])
    axes[3].set_ylim([-5, 5])

    axes[3].set_title('Mirror 2 & 3 and sample')

    # --------------------------------------------------------------------------
    # Zoom in to the sample region

    # Plot a horizontal reference line
    axes[4].plot(np.arange(-50 - 100, 700) * 15, np.zeros(850) * 10, c='b', linestyle='--')

    # Plot the miniSD trajectory
    axes[4].plot(simulation_summary['TG probe']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG probe']['trajectory'][:, 0] / 1000, 'r')

    axes[4].plot(simulation_summary['TG pump a']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump a']['trajectory'][:, 0] / 1000, 'g')

    axes[4].plot(simulation_summary['TG pump b']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump b']['trajectory'][:, 0] / 1000, 'g')

    axes[4].set_xlabel("z axis (mm)")
    axes[4].set_ylabel("y axis (mm)")
    axes[4].set_title('Mirror 1')
    axes[4].set_ylim([-1, 1])
    axes[4].set_xlim([simulation_summary['TG probe']['trajectory'][-3][2] / 1000 - 100,
                      simulation_summary['TG probe']['trajectory'][-3][2] / 1000 + 100])

    # --------------------------------------------------------------------------
    # Zoom in to the sample region

    # Plot a horizontal reference line
    axes[5].plot(np.arange(-50 - 100, 700) * 15, np.zeros(850) * 10, c='b', linestyle='--')

    # Plot the miniSD trajectory
    axes[5].plot(simulation_summary['TG probe']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG probe']['trajectory'][:, 0] / 1000, 'r')

    axes[5].plot(simulation_summary['TG pump a']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump a']['trajectory'][:, 0] / 1000, 'g')

    axes[5].plot(simulation_summary['TG pump b']['trajectory'][:, 2] / 1000,
                 simulation_summary['TG pump b']['trajectory'][:, 0] / 1000, 'g')

    axes[5].set_xlabel("z axis (mm)")
    axes[5].set_ylabel("y axis (mm)")
    axes[5].set_title('Mirror 1')
    axes[5].set_ylim([-1, 40])
    axes[5].set_xlim([simulation_summary['TG probe']['trajectory'][-1][2] / 1000 - 30,
                      simulation_summary['TG probe']['trajectory'][-1][2] / 1000 + 10])

    plt.tight_layout()
    plt.show()


def show_field_spectrum_xz_and_z_slice(summary, tag_list, coordinate_container, img_scale=["linear", 'linear']):
    # Show the calcluation result
    # It seems that we can also easily make the following into a single function to reduce the repetition of the code

    fig, axes = plt.subplots(ncols=2, nrows=2)

    fig.set_figheight(6)
    fig.set_figwidth(10)

    if img_scale[0] == "linear":
        axes[0, 0].imshow(summary[tag_list[0]]['yz'], aspect='auto', cmap='jet',
                          extent=[coordinate_container['EzCoor'][0] * 1000,
                                  coordinate_container['EzCoor'][-1] * 1000,
                                  coordinate_container['EyCoor'][0] * 1000,
                                  coordinate_container['EyCoor'][-1] * 1000])
    elif img_scale[0] == 'log':
        axes[0, 0].imshow(summary[tag_list[0]]['yz'], aspect='auto', cmap='jet',
                          extent=[coordinate_container['EzCoor'][0] * 1000,
                                  coordinate_container['EzCoor'][-1] * 1000,
                                  coordinate_container['EyCoor'][0] * 1000,
                                  coordinate_container['EyCoor'][-1] * 1000],
                          norm=LogNorm(vmin=np.max(summary[tag_list[0]]['yz']) / 1e5,
                                       vmax=np.max(summary[tag_list[0]]['yz']), ))
    else:
        print("No such option for img_scale")
        return 0

    axes[0, 0].set_xlabel("Ez (eV)")
    axes[0, 0].set_ylabel("Ex (eV)")
    axes[0, 0].set_title("Spectrum xz projection")

    axes[0, 1].plot(coordinate_container['EzCoor'] * 1e3, summary[tag_list[0]]['z'])
    axes[0, 1].set_xlabel("Ez (eV)")
    axes[0, 1].set_ylabel("I(Q)")
    axes[0, 1].set_title("Spectral Intensity")

    if img_scale[1] == 'linear':
        axes[1, 0].imshow(summary[tag_list[1]]['yz'], aspect='auto', cmap='jet',
                          extent=[coordinate_container['tCoor'][0],
                                  coordinate_container['tCoor'][-1],
                                  coordinate_container['yCoor'][0],
                                  coordinate_container['yCoor'][-1]])
    elif img_scale[1] == 'log':
        axes[1, 0].imshow(summary[tag_list[1]]['yz'], aspect='auto', cmap='jet',
                          extent=[coordinate_container['tCoor'][0],
                                  coordinate_container['tCoor'][-1],
                                  coordinate_container['yCoor'][0],
                                  coordinate_container['yCoor'][-1]],
                          norm=LogNorm(vmin=np.max(summary[tag_list[1]]['yz']) / 1e5,
                                       vmax=np.max(summary[tag_list[1]]['yz']),
                                       ))
    else:
        print("No such option for img_scale")
        return 0

    axes[1, 0].set_xlabel("t (fs)")
    axes[1, 0].set_ylabel("x (um)")
    axes[1, 0].set_title("Intensity xz projection")

    axes[1, 1].plot(coordinate_container['tCoor'], summary[tag_list[1]]['z'])
    axes[1, 1].set_xlabel("t (fs)")
    axes[1, 1].set_ylabel("I(t)")
    axes[1, 1].set_title("Intensity")

    plt.tight_layout()
    plt.show()
