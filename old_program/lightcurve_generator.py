"""Mihir Patankar [mpatankar06@gmail.com]"""
import re
import subprocess
import sys
from datetime import datetime
from glob import glob
from os import path
from subprocess import CalledProcessError
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

# pylint: disable-next=no-name-in-module
from ciao_contrib.runtool import dmcopy, dmextract, dmkeypar, dmlist

matplotlib.use("svg")


def get_light_data(observation_directory, bin_time_input):
    """Parses data product downloads, extracting light curves at energy level specifications and
    converts to a parsable text format. Returns bin time"""
    try:
        subprocess.run(
            f"gunzip {path.join(observation_directory, '*.gz')}",
            shell=True,
            check=True,
            stderr=subprocess.DEVNULL,
        )
    except CalledProcessError as error:
        print("Failed to unzip FITS. See error log for details.")
        log_error(error)
        sys.exit(1)
    print("Successfully unzipped FITS files.")
    source_region_file = glob(path.join(observation_directory, "*reg3.fits"))[0]
    event_list_file = glob(path.join(observation_directory, "*regevt3.fits"))[0]

    # Much easier than reading/writing the parameter to the disk.
    time_delay = float(dmkeypar(infile=event_list_file, keyword="TIMEDEL", echo=True))
    # observation_date = dmkeypar(infile=event_list_file, keyword="DATE-OBS") DATE-END
    # print("observation date", observation_date)
    bin_time = calculate_bintime(bin_time_input, time_delay)

    energies = {
        "broad": "energy=300:7000",
        "soft": "energy=300:1200",
        "medium": "energy=1200:2000",
        "hard": "energy=2000:7000",
    }
    try:
        dmcopy(
            infile=f"{event_list_file}[sky=region({source_region_file})]",
            outfile=f"{event_list_file}.src",
        )
        for light_level, energy_range in energies.items():
            dmextract(
                infile=f"{event_list_file}.src[{energy_range}][bin time=::{bin_time}]",
                outfile=f"{event_list_file}.{light_level}.lc",
                opt="ltc1",
            )
    except OSError as error:
        print("Failed to extract light curves. See error_log for details.")
        log_error(error)
        sys.exit(1)
    print("Successfully extracted light curves.")
    try:
        for light_level in energies:
            dmlist(
                infile=f"{event_list_file}.{light_level}.lc[ \
                    cols time,count_rate,count_rate_err,counts,exposure,area \
                ]",
                opt="data,clean",
                outfile=f"{event_list_file}.{light_level}.lc.txt",
            )
    except OSError as error:
        print("Failed to clean and dump data. See error_log for details.")
        log_error(error)
        sys.exit(1)
    print("Successfully cleaned and dumped data.")
    return bin_time


def parse_light_data(observation_directory):
    """Returns light curve data parsed from txt files, along with a dictionary containing obsid,
    regid, exposure time, total counts, and average count rate."""
    arbitrary_text_file = glob(path.join(observation_directory, "*txt"))[0]
    # Uses regex to search the file name; the region number will be xxxx in _rxxxx_
    region_id = re.search(r"r\d{4}", arbitrary_text_file).group(0)[1:]
    light_level_data = {
        "broad": np.loadtxt(glob(path.join(observation_directory, "*broad*txt"))[0]),
        "soft": np.loadtxt(glob(path.join(observation_directory, "*soft*txt"))[0]),
        "medium": np.loadtxt(glob(path.join(observation_directory, "*medium*txt"))[0]),
        "hard": np.loadtxt(glob(path.join(observation_directory, "*hard*txt"))[0]),
    }
    trim_zero_exposure_points(light_level_data)
    observation_properties = {
        "obsid": path.basename(path.normpath(observation_directory)).split("_")[0],
        "regid": region_id,
        "initial_time": f"{int(light_level_data['broad'][:, 0].min())}",
        "final_time": f"{int(light_level_data['broad'][:, 0].max())}",
        "average_count_rate": f"{light_level_data['broad'][:, 2].mean():.4f}",  # This 2 should be a one
        "total_counts": light_level_data["broad"][:, 3].sum(),
        "total_exposure_time": f"{light_level_data['broad'][:, 4].sum():.2f}",
    }
    return light_level_data, observation_properties


def plot(light_level_data, observation_directory, bin_size):
    """Returns path to the generated matplotlib plot image."""
    initial_time = light_level_data["broad"][:, 0].min()
    zero_shifted_time_kiloseconds = (light_level_data["broad"][:, 0] - initial_time) / 1000
    observation_epoch = max(zero_shifted_time_kiloseconds)
    figure, plots = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 5), constrained_layout=True, sharey=True
    )
    broad_plot = plots[0]
    seperation_plot = plots[1]
    broad_plot.errorbar(
        x=zero_shifted_time_kiloseconds,
        y=light_level_data["broad"][:, 1],
        yerr=light_level_data["broad"][:, 2],
        color="red",
        marker="o",
        markerfacecolor="black",
        markersize=4,
        ecolor="black",
        markeredgecolor="black",
        capsize=3,
    )
    broad_plot.set_xlim([0, observation_epoch])
    seperated_lights = {"hard": "firebrick", "medium": "red", "soft": "lightsalmon"}
    for light_level, color in seperated_lights.items():
        seperation_plot.plot(
            zero_shifted_time_kiloseconds,
            light_level_data[light_level][:, 1],
            color=color,
            label=light_level,
        )
    seperation_plot.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    seperation_plot.set_xlim([0, observation_epoch])
    figure.supylabel("Count Rate (counts/second)")
    figure.supxlabel("Time (kiloseconds)")
    figure.suptitle(
        f"Lightcurve in Broadband and Separated Energy Bands with {bin_size:.3f}s Bin size"
    )
    plot_image_path = path.join(observation_directory, "plot.svg")
    plt.savefig(plot_image_path, bbox_inches="tight")
    plt.close(figure)
    return plot_image_path


def calculate_bintime(bin_time, time_delta):
    """Calculate bin time for light curve histogram data."""
    return int(bin_time / time_delta) * time_delta


def log_error(error):
    """Logs ciao tool errors to a date-stamped and time-stamped text file in the root directory."""
    file_name = f"error-log-{datetime.now()}.txt"
    print("Error log:", file_name)
    with open(file=file_name, mode="a", encoding="utf-8") as log_file:
        log_file.write(str(error))


def trim_zero_exposure_points(light_level_data):
    """Mutate energy level data to remove points where exposure equals zero."""
    broad_light_data = light_level_data["broad"]
    exposure_zero_row_indices = []
    for row_index, row in enumerate(broad_light_data):
        exposure = row[4]
        if exposure == 0:
            exposure_zero_row_indices.append(row_index)
    for level, data in light_level_data.items():
        light_level_data[level] = np.delete(data, exposure_zero_row_indices, axis=0)
