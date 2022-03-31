import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os
import copy


def draw(altitude, error_new_constant, run_time_new_constant, run_time_glq_constant, \
    error_new_linear, run_time_new_linear, run_time_glq_linear, title):
    
    error_max = max(np.max(error_new_constant), np.max(error_new_linear))
    error_min = min(np.min(error_new_constant), np.min(error_new_linear))
    time_max = max(np.max(time_glq_constant), np.max(time_glq_linear), np.max(time_new_constant), np.max(time_new_linear))
    time_min = min(np.min(time_glq_constant), np.min(time_glq_linear), np.min(time_new_constant), np.min(time_new_linear))
    
    error_y_lim = [2e-10, 11e-5]
    time_y_lim = [8, time_max+1.5e3]

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(altitude, error_new_constant[:, 0], 'ro--', label=f"V_new method")
    axs[0, 0].plot(altitude, error_new_constant[:, 1], 'bs--', label=f"Vz_new method")
    axs[0, 0].plot(altitude, error_new_constant[:, 2], 'g^--', label=f"Vzz_new method")

    axs[1, 0].plot(altitude, run_time_new_constant[:, 0], 'ro--', label=f"V_new method")
    axs[1, 0].plot(altitude, run_time_new_constant[:, 1], 'bs--', label=f"Vz_new method")
    axs[1, 0].plot(altitude, run_time_new_constant[:, 2], 'g^--', label=f"Vzz_new method")
    axs[1, 0].plot(altitude, run_time_glq_constant[:, 0], 'ro-', label=f"V_traditional method")
    axs[1, 0].plot(altitude, run_time_glq_constant[:, 1], 'bs-', label=f"Vz_traditional method")
    axs[1, 0].plot(altitude, run_time_glq_constant[:, 2], 'g^-', label=f"Vzz_traditional method")
    
    axs[0, 1].plot(altitude, error_new_linear[:, 0], 'ro--', label=f"V_new method")
    axs[0, 1].plot(altitude, error_new_linear[:, 1], 'bs--', label=f"Vz_new method")
    axs[0, 1].plot(altitude, error_new_linear[:, 2], 'g^--', label=f"Vzz_new method")

    l1, = axs[1, 1].plot(altitude, run_time_new_linear[:, 0], 'ro--', label=f"V_new method")
    l2, = axs[1, 1].plot(altitude, run_time_new_linear[:, 1], 'bs--', label=f"Vz_new method")
    l3, = axs[1, 1].plot(altitude, run_time_new_linear[:, 2], 'g^--', label=f"Vzz_new method")
    l4, = axs[1, 1].plot(altitude, run_time_glq_linear[:, 0], 'ro-', label=f"V_traditional method")
    l5, = axs[1, 1].plot(altitude, run_time_glq_linear[:, 1], 'bs-', label=f"Vz_traditional method")
    l6, = axs[1, 1].plot(altitude, run_time_glq_linear[:, 2], 'g^-', label=f"Vzz_traditional method")
    
    axs[0, 0].set(xlabel='Altitude (km)', ylabel='Relative error', yscale='log',\
        ylim=error_y_lim)
    axs[0, 1].set(xlabel='Altitude (km)', ylabel='Relative error', yscale='log',\
        ylim=error_y_lim)
    axs[1, 0].set(xlabel='Altitude (km)', ylabel='Computational time (s)', yscale='log',\
        ylim=time_y_lim)
    axs[1, 1].set(xlabel='Altitude (km)', ylabel='Computational time (s)', yscale='log', \
        ylim=time_y_lim)

    axs[0, 0].sharey(axs[0, 1])
    axs[1, 0].sharey(axs[1, 1])

    axs[0, 0].annotate("(a)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[0, 1].annotate("(c)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[1, 0].annotate("(b)", xy=(0.9, 0.9), xycoords="axes fraction")
    axs[1, 1].annotate("(d)", xy=(0.9, 0.9), xycoords="axes fraction")
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    fig.tight_layout()

    fig.subplots_adjust(bottom=0.2)
    axs.flatten()[-2].legend(handles=[l1, l4, l2, l5, l3, l6], \
        bbox_to_anchor=(0.92, -0.55), loc='lower center', \
        ncol=3, borderaxespad=0., columnspacing=1)

    axs[0, 0].grid(True)
    axs[1, 0].grid(True)
    axs[1, 1].grid(True)
    axs[0, 1].grid(True)

    plt.show()
    fig.savefig(title, format='pdf')


delta = 5
altitude = (np.array([1, 100, 500, 1000, 5000, 10*1000, 50*1000, 100*1000]))
r_cal = 1761358.0 + altitude

script_dir = os.path.dirname(__file__) 

tags1 = ['V', 'Vz', 'Vzz']
tags2 = ['constant', 'linear']
tags2 = 'constant'
# tags2 = 'linear'
parts_num_r_max = 1
parts_num_r_min = 1

relative_error = np.zeros((len(r_cal), len(tags1)))
run_time_new = np.zeros((len(r_cal), len(tags1)))
run_time_glq = np.zeros((len(r_cal), len(tags1)))

for index_tag in range(len(tags1)):
    for index_r in range(len(r_cal)):
        relative_dir = f'data/{delta}_{delta}_{tags2}_{tags1[index_tag]}_ice_shell_new_order_4_0_parts_{parts_num_r_max}_{parts_num_r_min}_{r_cal[index_r]}.pkl'
        filename_new = os.path.join(script_dir, relative_dir)
        relative_dir = f'data/{delta}_{delta}_{tags2}_{tags1[index_tag]}_ice_shell_glq_{r_cal[index_r]}.pkl'
        filename_glq = os.path.join(script_dir, relative_dir)
        with open(filename_glq, 'rb') as f:  
            gf_glq, temp_run_time_glq = pickle.load(f)
        with open(filename_new, 'rb') as f:  
            gf_new, temp_run_time_new = pickle.load(f)

        relative_error[index_r, index_tag] = np.max(np.abs((gf_new - gf_glq) / gf_glq))
        run_time_new[index_r, index_tag] = temp_run_time_new
        run_time_glq[index_r, index_tag] = temp_run_time_glq
# for index_r in range(len(r_cal)):
#     for i in range(3):
#         print(f'{altitude[index_r]/1000} km, \
#             {relative_error[index_r, i]:.1e}, \
#             {run_time_glq[index_r, i]:.1f} s,\
#             {run_time_new[index_r, i]:.1f} s')

error_new_constant = copy.deepcopy(relative_error)
time_glq_constant = copy.deepcopy(run_time_glq)
time_new_constant = copy.deepcopy(run_time_new)


tags2 = 'linear'
parts_num_r_max = 1
parts_num_r_min = 1

relative_error = np.zeros((len(r_cal), len(tags1)))
run_time_new = np.zeros((len(r_cal), len(tags1)))
run_time_glq = np.zeros((len(r_cal), len(tags1)))

for index_tag in range(len(tags1)):
    for index_r in range(len(r_cal)):
        relative_dir = f'data/{delta}_{delta}_{tags2}_{tags1[index_tag]}_ice_shell_new_order_4_0_parts_{parts_num_r_max}_{parts_num_r_min}_{r_cal[index_r]}.pkl'
        filename_new = os.path.join(script_dir, relative_dir)
        relative_dir = f'data/{delta}_{delta}_{tags2}_{tags1[index_tag]}_ice_shell_glq_{r_cal[index_r]}.pkl'
        filename_glq = os.path.join(script_dir, relative_dir)
        with open(filename_glq, 'rb') as f:  
            gf_glq, temp_run_time_glq = pickle.load(f)
        with open(filename_new, 'rb') as f:  
            gf_new, temp_run_time_new = pickle.load(f)

        relative_error[index_r, index_tag] = np.max(np.abs((gf_new - gf_glq) / gf_glq))
        run_time_new[index_r, index_tag] = temp_run_time_new
        run_time_glq[index_r, index_tag] = temp_run_time_glq

error_new_linear = copy.deepcopy(relative_error)
time_glq_linear = copy.deepcopy(run_time_glq)
time_new_linear = copy.deepcopy(run_time_new)

relative_dir = f'document/figs/{delta}_{delta}_ice_shell.pdf'
filename_fig = os.path.join(script_dir, relative_dir)
draw(altitude/1000, error_new_constant, time_new_constant, time_glq_constant, \
    error_new_linear, time_new_linear, time_glq_linear, filename_fig)
