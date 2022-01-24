import logging
import pickle
import sys

import fire
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_memory(dict):
    mem_stats = dict['mem_stats']
    mem_stats = [elem/1e6 for elem in mem_stats]
    start_time = min(dict['time_stamps'])
    time_stamps = [elem - start_time for elem in dict['time_stamps']]


    plt.style.use("ggplot")
    plt.plot(time_stamps, mem_stats, label="gpu mem stats")

    # plt.set_xlim([min(time_stamps), max(time_stamps)])
    # axis.set_ylim([0, offset])

    plt.xlabel("time/s")
    plt.ylabel("memory/MB")
    plt.title("gpu mem stats")

    plt.savefig('memstats.png')


def visualize_profile(filename):
    # load profile data
    with open(filename, "rb") as f:
        dict = pickle.load(f)
    visualize_memory(dict)


if __name__ == "__main__":
    fire.Fire(visualize_profile)
