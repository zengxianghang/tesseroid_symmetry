import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os


def draw(altitude, error, run_time, title):
    
    error_y_lim = [5e-13, 5e-2]
    time_y_lim = [7, 280]

    fig, axs = plt.subplots(2, 3)
    for i in range(3):
        axs[0, i].plot(altitude/1000, error[i, :, 0], 'ro--', label=f"truncation order = 2")
        axs[0, i].plot(altitude/1000, error[i, :, 1], 'bs--', label=f"truncation order = 3")
        axs[0, i].plot(altitude/1000, error[i, :, 2], 'g^--', label=f"truncation order = 4")
        axs[0, i].plot(altitude/1000, error[i, :, 3], 'cd--', label=f"truncation order = 5")
        axs[0, i].plot(altitude/1000, error[i, :, 4], 'm>--', label=f"truncation order = 6")
    for i in range(3):
        if i == 1:
            l1, = axs[1, i].plot(altitude/1000, run_time[i, :, 0], 'ro--', label=f"truncation order = 2")
            l2, = axs[1, i].plot(altitude/1000, run_time[i, :, 1], 'bs--', label=f"truncation order = 3")
            l3, = axs[1, i].plot(altitude/1000, run_time[i, :, 2], 'g^--', label=f"truncation order = 4")
            l4, = axs[1, i].plot(altitude/1000, run_time[i, :, 3], 'cd--', label=f"truncation order = 5")
            l5, = axs[1, i].plot(altitude/1000, run_time[i, :, 4], 'm>--', label=f"truncation order = 6")
        else:
            axs[1, i].plot(altitude/1000, run_time[i, :, 0], 'ro--', label=f"truncation order = 2")
            axs[1, i].plot(altitude/1000, run_time[i, :, 1], 'bs--', label=f"truncation order = 3")
            axs[1, i].plot(altitude/1000, run_time[i, :, 2], 'g^--', label=f"truncation order = 4")
            axs[1, i].plot(altitude/1000, run_time[i, :, 3], 'cd--', label=f"truncation order = 5")
            axs[1, i].plot(altitude/1000, run_time[i, :, 4], 'm>--', label=f"truncation order = 6")

    axs[0, 0].annotate("(a)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[0, 1].annotate("(b)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[0, 2].annotate("(c)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[1, 0].annotate("(d)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[1, 1].annotate("(e)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[1, 2].annotate("(f)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[0, 0].text(0.475, 0.825, r'$V$',
        bbox=dict(fc="white", edgecolor='gray', boxstyle='circle'),
        transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.47, 0.825, r'$V_z$',
        bbox=dict(fc="white", edgecolor='gray', boxstyle='circle'),
        transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.455, 0.825, r'$V_{zz}$',
        bbox=dict(fc="white", edgecolor='gray', boxstyle='circle'),
        transform=axs[0, 2].transAxes)
    axs[1, 0].text(0.475, 0.825, r'$V$',
        bbox=dict(fc="white", edgecolor='gray', boxstyle='circle'),
        transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.47, 0.825, r'$V_z$',
        bbox=dict(fc="white", edgecolor='gray', boxstyle='circle'),
        transform=axs[1, 1].transAxes)
    axs[1, 2].text(0.455, 0.825, r'$V_{zz}$',
        bbox=dict(fc="white", edgecolor='gray', boxstyle='circle'),
        transform=axs[1, 2].transAxes)

    for i in range(2):
        for j in range(3):
            if i == 0:
                axs[i, j].set(xlabel='Altitude (km)',\
                    ylabel='Relative error', yscale='log',\
                    ylim=error_y_lim)
            else:
                axs[i, j].set(xlabel='Altitude (km)',\
                    ylabel='Computational time (s)', yscale='log',\
                    ylim=time_y_lim)
            axs[i, j].grid(True)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    fig.tight_layout()

    fig.subplots_adjust(bottom=0.2)
    fig.set_size_inches(9, 5.5)
    axs.flatten()[-2].legend(handles=[l1, l4, l2, l5, l3], \
        bbox_to_anchor=(0.4, -0.48), loc='lower center', \
        ncol=3, borderaxespad=0., columnspacing=1)
    
    plt.show()
    fig.savefig(title, format='pdf')


is_linear_density = False
delta = 5
altitude = (np.array([1, 100, 500, 1000, 5000, 10*1000, 50*1000, 100*1000]))
r_cal = 1761358.0 + altitude
max_order_r_max = [2, 3, 4, 5, 6]

script_dir = os.path.dirname(__file__) 

tag1 = ['V', 'Vz', 'Vzz']
tag2 = ['constant']

parts_num_r_max = 1
# parts_num_r_max = 3
parts_num_r_min = 1

relative_error = np.zeros((len(tag1), len(r_cal), len(max_order_r_max)))
run_time = np.zeros((len(tag1), len(r_cal), len(max_order_r_max)))

for index_tag in range(len(tag1)):
    for index_r in range(len(r_cal)):
        for index_order in range(len(max_order_r_max)):
            relative_dir = f'data/{delta}_{delta}_{tag2[0]}_{tag1[index_tag]}_ice_shell_new_order_{max_order_r_max[index_order]}_0_parts_{parts_num_r_max}_{parts_num_r_min}_{r_cal[index_r]}.pkl'
            filename_new = os.path.join(script_dir, relative_dir)
            relative_dir = f'data/{delta}_{delta}_{tag2[0]}_{tag1[index_tag]}_ice_shell_glq_{r_cal[index_r]}.pkl'
            filename_glq = os.path.join(script_dir, relative_dir)
            with open(filename_glq, 'rb') as f:  
                gf_glq, run_time_glq = pickle.load(f)
            with open(filename_new, 'rb') as f:  
                gf_new, run_time_new = pickle.load(f)

            relative_error[index_tag, index_r, index_order] = np.max(np.abs((gf_new - gf_glq) / gf_glq))
            run_time[index_tag, index_r, index_order] = run_time_new
            
    # for index_r in range(len(r_cal)):
    #     print(f'{altitude[index_r]/1000} km , \
    #         {relative_error[index_r, 0]:.1e} , \
    #         {relative_error[index_r, 1]:.1e} , \
    #         {relative_error[index_r, 2]:.1e} , \
    #         {relative_error[index_r, 3]:.1e} , \
    #         {relative_error[index_r, 4]:.1e}  ')
    # for index_r in range(len(r_cal)):
    #     print(f'{altitude[index_r]/1000} km, \
    #         {run_time[index_r, 0]:.1f} s,\
    #         {run_time[index_r, 1]:.1f} s,\
    #         {run_time[index_r, 2]:.1f} s,\
    #         {run_time[index_r, 3]:.1f} s,\
    #         {run_time[index_r, 4]:.1f} s ')

relative_dir = f'document/figs/{delta}_{delta}_ice_shell_determine_order.pdf'
filename_fig = os.path.join(script_dir, relative_dir)
draw(altitude, relative_error, run_time, filename_fig)
