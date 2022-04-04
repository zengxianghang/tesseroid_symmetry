import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os


def draw(error, title):
    extent = -180, 180, -90, 90
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(5, 7))

    for i in range(3):
        pcm = axs[i].imshow(error[i], extent=extent, origin = 'lower')
        fig.colorbar(pcm, ax=axs[i])
        axs[i].set(xlabel="Longitude (degree)", ylabel="Latitude (degree)")
    axs[0].annotate("(a)", xy=(0.01, 0.9), xycoords="axes fraction")
    axs[1].annotate("(b)", xy=(0.01, 0.9), xycoords="axes fraction")
    axs[2].annotate("(c)", xy=(0.01, 0.9), xycoords="axes fraction")
    fig.tight_layout()
    
    plt.savefig(title, format='pdf')
 

is_linear_density = False
delta = 5
altitude = 10 * 1000
r_cal = 1761358.0 + altitude

max_order_r_max = 4
parts_num_r_max = 3
# parts_num_r_max = 1
parts_num_r_min = 1

script_dir = os.path.dirname(__file__) 

tag1 = ['V', 'Vz', 'Vzz']
tag2 = ['constant']

relative_error = []
for index_tag in range(len(tag1)):
    relative_dir = f'data/{delta}_{delta}_{tag2[0]}_{tag1[index_tag]}_ice_shell_new_order_{max_order_r_max}_0_parts_{parts_num_r_max}_{parts_num_r_min}_{r_cal}.pkl'
    filename_new = os.path.join(script_dir, relative_dir)
    relative_dir = f'data/{delta}_{delta}_{tag2[0]}_{tag1[index_tag]}_ice_shell_glq_{r_cal}.pkl'
    filename_glq = os.path.join(script_dir, relative_dir)
    with open(filename_glq, 'rb') as f:  
        gf_glq, run_time_glq = pickle.load(f)
    with open(filename_new, 'rb') as f:  
        gf_new, run_time_new = pickle.load(f)

    relative_error.append((gf_new - gf_glq) / gf_glq)
relative_dir = f'document/figs/{delta}_{delta}_ice_shell_order_{max_order_r_max}_0_parts_{parts_num_r_max}_{parts_num_r_min}_{r_cal}.pdf'
filename_fig = os.path.join(script_dir, relative_dir)
draw(relative_error, filename_fig)