"""Mihir Patankar [mpatankar06@gmail.com]"""
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from glob import glob
from os import path

from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

# pylint: disable=no-name-in-module
from ciao_contrib.runtool import search_csc
from jinja2 import Environment, FileSystemLoader
from pyvo import DALFormatError, dal

from lightcurve_generator import get_light_data, parse_light_data, plot

OBJECT_NAME = input("Enter object name: ")
try:
    SEARCH_RADIUS = float(input("Enter search radius in arcminutes: ")) * units.arcmin
    SIGNIFICANCE_THRESHOLD = float(
        input("Enter signficance threshold (leave blank for default 50): ") or 50.0
    )
    BINTIME = float(input("Enter bin time (leave blank for default 500): ") or 500.0)
except ValueError:
    print("Invalid input.")
    sys.exit(1)
DATA_DIRECTORY = "./data"
OUTPUT_DIRECTORY = "./output"

try:
    sky_coord = SkyCoord.from_name(OBJECT_NAME)
except NameResolveError:
    print(f'No results for object "{OBJECT_NAME}".')
    sys.exit(1)
cone_search = dal.SCSService("http://cda.cfa.harvard.edu/csc2scs/coneSearch")
print("Searching CSC sources...")
start_time = time.time()
try:
    search_results = cone_search.search(sky_coord, SEARCH_RADIUS, verbosity=2)
except DALFormatError:
    print("Failed to connect to search service, check internet connection?")
    sys.exit(1)
print(f"Found {len(search_results)} sources in {(time.time() - start_time):.3f}s.")

significant_results = [
    result
    for result in search_results
    if result["significance"] >= SIGNIFICANCE_THRESHOLD
]
significant_results_count = len(significant_results)
print(f"Found {significant_results_count} sources meeting the significance threshold.")
if significant_results_count == 0:
    sys.exit(0)


def print_data_product_progress():
    """Checks and reports how many files have been downloaded."""
    download_directory = path.join(DATA_DIRECTORY, result["name"].replace(" ", ""))
    download_start_time = time.time()
    while True:
        observation_count = (
            len(os.listdir(download_directory)) if path.isdir(download_directory) else 0
        )
        data_products_count = len(
            glob(path.join(download_directory, "**/*.gz"), recursive=True)
        )
        print(
            f"\rRetrieved {observation_count} observations, {data_products_count} data products...",
            f"({(time.time() - download_start_time):.2f}s)",
            end="",
        )
        if finished_downloading.is_set():
            print("")  # New line
            break
        time.sleep(0.1)


proceed = input("Proceed? (y/n) ")
if proceed != "y":
    sys.exit(0)
# Delete previous data products for this source
shutil.rmtree(DATA_DIRECTORY, ignore_errors=True)
for result_index, result in enumerate(significant_results):
    progress = f"{(result_index + 1)}/{len(significant_results)}"
    print(f"Searching source {result['name']} ({progress})...")
    finished_downloading = threading.Event()
    thread = threading.Thread(target=print_data_product_progress)
    thread.start()
    search_csc(
        pos=f"{result['ra']}, {result['dec']}",
        radius="1.0",
        outfile="search-csc-outfile.tsv",
        radunit="arcsec",
        columns="",
        sensitivity="no",
        download="all",
        root=DATA_DIRECTORY,
        bands="broad, wide",
        filetypes="regevt, reg",
        catalog="csc2",
        verbose="0",
        clobber="1",
    )
    finished_downloading.set()
    thread.join()
print("All data products successfully retrieved.")

render_data = {}
source_directories = os.listdir(DATA_DIRECTORY)
for count, source_directory in enumerate(source_directories):
    if "CXO" not in source_directory or source_directory[-1] == "X":
        continue  # Likely not a valid source folder.
        # Sources ending with X seems to indicate an absence of data products but I'm not sure why.
    print(f"\nProcessing source {source_directory}:")
    render_data[source_directory] = []
    for observation_directory in os.listdir(
        path.join(DATA_DIRECTORY, source_directory)
    ):
        print(f"Processing observation {observation_directory}...")
        observation_full_path = path.join(
            DATA_DIRECTORY, source_directory, observation_directory
        )
        # Regex check if observation directory has a valid name in the format xxxxx_xxx
        is_valid_directory = re.match(r"^\d{5}_\d{3}$", observation_directory)
        # hrcf data products do not work for whatever reason, need to find out why.
        is_hrc_band = re.match(r"^hrcf", os.listdir(observation_full_path)[0])
        if not is_valid_directory or is_hrc_band:
            print("Invalid or unsupported observation, skipping...")
            continue
        bin_time = get_light_data(observation_full_path, BINTIME)
        light_curve_data, observation_properties = parse_light_data(
            observation_full_path
        )
        path_to_plot_image = plot(light_curve_data, observation_full_path, bin_time)
        path_to_plot_image = path.abspath(path_to_plot_image)
        render_data[source_directory].append(
            (observation_properties, path_to_plot_image)
        )
    render_data[source_directory] = sorted(
        render_data[source_directory],
        key=lambda observation: observation[0]["initial_time"],
    )
    print(f"\n{((count + 1) / len(source_directories) * 100):.2f}% sources processed.")

if not path.isdir(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)
file_name = path.join(OUTPUT_DIRECTORY, f"{OBJECT_NAME}-{datetime.now()}.html")
with open(file=file_name, mode="a", encoding="utf-8") as html_file:
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template(str("/data_view.jinja"))
    content = template.render(
        source_count=significant_results_count,
        object_name=OBJECT_NAME,
        search_radius=SEARCH_RADIUS,
        significance_threshold=SIGNIFICANCE_THRESHOLD,
        render_data=render_data,
    )
    html_file.write(content)
print(f"Successfully written to file at {file_name}")
