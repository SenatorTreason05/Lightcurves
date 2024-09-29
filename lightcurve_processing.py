"""Mihir Patankar [mpatankar06@gmail.com]"""
import uuid
import gzip
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
import threading
import os
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from astropy import io, table
from astropy.timeseries import LombScargle
from astropy.stats import bayesian_blocks
from astropy.time import Time

# pylint: disable-next=no-name-in-module
from ciao_contrib.runtool import dmcopy, dmextract, dmkeypar, dmlist, dmstat, new_pfiles_environment, glvary, dmcoords, dither_region, acis_set_ardlib, dmtcalc
from ciao_contrib.cda.data import download_chandra_obsids
from pycrates import *
from matplotlib import pyplot as plt
from pandas import DataFrame

from data_structures import LightcurveParseResults, Message, ObservationData, ObservationHeaderInfo
from postage_stamp_plotter import CropBounds, plot_postagestamps


class ObservationProcessor(ABC):
    """Base class for observation processor implementations for different Chandra instruments."""

    def __init__(self, data_products, binsize, message_collection_queue=None, counts_checker=None):
        self.event_list = Path(data_products.event_list_file)
        self.source_region = Path(data_products.source_region_file)
        self.detector_coords_image, self.sky_coords_image = None, None
        self.message_collection_queue = message_collection_queue
        self.counts_checker = counts_checker
        self.binsize = binsize

    def process(self):
        """Sequence in which all steps of the processing routine are called."""

        message_uuid = uuid.uuid4()
        with new_pfiles_environment():
            observation_id = dmkeypar(infile=f"{self.event_list}", keyword="OBS_ID", echo=True)
            prefix = f"Observation {observation_id}: "

            def status(status):
                self.message_collection_queue.put(Message(f"{prefix}{status}", message_uuid))

            status("Isolating source region...")
            region_event_list = self.isolate_source_region(self.event_list, self.source_region)
            status("Extracting lightcurves...")
            lightcurves = self.extract_lightcurves(region_event_list, self.binsize)
            status("Copying columns...")
            filtered_lightcurves = self.filter_lightcurve_columns(lightcurves)

            status("Checking counts...")
            lightcurve_data = self.get_lightcurve_data(filtered_lightcurves)
            self.counts_checker.queue.put(self.get_lightcurve_counts(lightcurve_data))
            self.counts_checker.queue.join()
            if self.counts_checker.cancel_event.is_set():
                return None
            
            # status("Checking counts...")
            # lightcurve_data = self.get_lightcurve_data(filtered_lightcurves)
            # # Function to check counts
            # def check_counts():
            #     self.counts_checker.queue.put(self.get_lightcurve_counts(lightcurve_data))
            #     self.counts_checker.queue.join()

            # # Create and start the counts checking thread
            # counts_thread = threading.Thread(target=check_counts)
            # counts_thread.start()

            # # Wait for up to 10 seconds for the counts checking to complete
            # counts_thread.join(timeout=30)

            # # Check if the thread is still running after the timeout
            # if counts_thread.is_alive():
            #     status("Counts checking exceeded 30 seconds. Cancelling operation.")
            #     self.counts_checker.cancel_event.set()
            #     return None

            # # Check if the cancel event was set by the counts checker
            # if self.counts_checker.cancel_event.is_set():
            #     return None

            status("Retrieving images...")
            self.get_images(region_event_list)
            status("Plotting lightcurves...")
            results = self.plot(lightcurve_data)
            status("Plotting lightcurves... Done")

        return results

    @abstractmethod
    def extract_lightcurves(self, event_list, binsize):
        """Extract lightcurve(s) from an event list, one should pass in one with a specific source
        region extracted."""

    @staticmethod
    @abstractmethod
    def filter_lightcurve_columns(lightcurves: list[Path]):
        """Filter lightcurve(s) to get the columns we care about and format them."""

    @staticmethod
    @abstractmethod
    def get_lightcurve_data(lightcurves: list[Path]):
        """Return the data from the lightcurve files in a python object."""

    @staticmethod
    def get_lightcurve_counts(lightcurve_data):
        """Return the total counts in a lightcurve so they can be verified to meet the threshold."""

    @abstractmethod
    def plot(self, lightcurve_data) -> LightcurveParseResults:
        """Plots lightcurve(s). Returns what will then be returned by the thread pool future."""

    @staticmethod
    def isolate_source_region(event_list: Path, source_region):
        """Restrict the event list to just the source region of the source we care about."""
        dmcopy(
            infile=f"{event_list}[sky=region({source_region})]",
            outfile=(outfile := f"{event_list.with_suffix('.src.fits')}"),
            clobber="yes",
        )
        return outfile

    def get_images(self, region_event_list):
        """Gets the images from the event list, one in sky coordinates, the other in detector
        coordinates. This is useful to be able to track if a lightcurve dip corresponds with a
        source going over the edge of the detector. The image is cropped otherwise we would be
        dealing with hundreds of thousands of blank pixels. The cropping bounds are written to the
        FITS file for later use in plotting."""

        dmstat(infile=f"{region_event_list}[cols x,y]")
        sky_bounds = CropBounds.from_strings(*dmstat.out_min.split(","), *dmstat.out_max.split(","))
        sky_bounds.double()
        dmcopy(
            infile=f"{self.event_list}"
            f"[bin x={sky_bounds.x_min}:{sky_bounds.x_max}:0.5,"
            f"y={sky_bounds.y_min}:{sky_bounds.y_max}:0.5]",
            outfile=(sky_coords_image := f"{region_event_list}.skyimg.fits"),
        )
        with io.fits.open(sky_coords_image, mode="append") as hdu_list:
            hdu_list.append(sky_bounds.to_hdu())

        dmstat(infile=f"{region_event_list}[cols detx,dety]")
        detector_bounds = CropBounds.from_strings(
            *dmstat.out_min.split(","), *dmstat.out_max.split(",")
        )
        detector_bounds.add_padding(x_padding=5, y_padding=5)
        dmcopy(
            infile=f"{self.event_list}"
            f"[bin detx={detector_bounds.x_min}:{detector_bounds.x_max}:0.5,"
            f"dety={detector_bounds.y_min}:{detector_bounds.y_max}:0.5]",
            outfile=(detector_coords_image := f"{region_event_list}.detimg.fits"),
        )
        with io.fits.open(detector_coords_image, mode="append") as hdu_list:
            hdu_list.append(detector_bounds.to_hdu())
        self.sky_coords_image, self.detector_coords_image = sky_coords_image, detector_coords_image

    def get_observation_details(self):
        """Gets keys from the header block detailing the observation information."""
        dmstat(infile=f"{self.event_list}[cols ra,dec]")
        RA_0=dmstat.out_mean.split(',')[0] 
        dec_0=dmstat.out_mean.split(',')[1]
        dmcoords(infile=f"{self.event_list}", option="cel", ra=RA_0, dec=dec_0)
        theta_0=dmcoords.theta
        phi_0=dmcoords.phi
        return ObservationHeaderInfo( 
            instrument=dmkeypar(infile=f"{self.event_list}", keyword="INSTRUME", echo=True),
            observation_id=dmkeypar(infile=f"{self.event_list}", keyword="OBS_ID", echo=True),
            region_id=dmkeypar(infile=f"{self.event_list}", keyword="REGIONID", echo=True),
            start_time=dmkeypar(infile=f"{self.event_list}", keyword="DATE-OBS", echo=True),
            end_time=dmkeypar(infile=f"{self.event_list}", keyword="DATE-END", echo=True),
            off_axis_offset=round(float(theta_0), 1),
            azimuth=int(phi_0),
            right_ascension=round(float(RA_0), 5),
            declination=round(float(dec_0), 5)
        )


class AcisProcessor(ObservationProcessor):
    """Processes observations produced by the ACIS (Advanced CCD Imaging Spectrometer) instrument
    aboard Chandra."""

    ENERGY_LEVELS = {
        "broad": "energy=150:7000",
        "ultrasoft": "energy=150:300",
        "soft": "energy=300:1200",
        "medium": "energy=1200:2000",
        "hard": "energy=2000:7000",
    }

    @staticmethod
    def adjust_binsize(event_list, binsize):
        """For ACIS, time resolution can be in the seconds in timed exposure mode, as compared to in
        the microseconds for HRC. Thus we must round the binsize to the time resolution."""
        time_resolution = float(dmkeypar(infile=str(event_list), keyword="TIMEDEL", echo=True))
        return binsize // time_resolution * time_resolution

    def extract_lightcurves(self, event_list, binsize):
        outfiles = []
        self.binsize = self.adjust_binsize(event_list, binsize)
        for light_level, energy_range in AcisProcessor.ENERGY_LEVELS.items():
            dmextract(
                infile=f"{event_list}[{energy_range}][bin time=::" f"{self.binsize}]",
                outfile=(outfile := f"{event_list}.{light_level}.lc"),
                opt="ltc1",
                clobber="yes",
            )
            outfiles.append(Path(outfile))
        return outfiles
    
    @staticmethod
    def filter_lightcurve_columns(lightcurves):
        outfiles = []
        for lightcurve in lightcurves:
            dmlist(
                infile=f"{lightcurve}"
                f"[cols time,count_rate,count_rate_err,counts,exposure,area]",
                opt="data,clean",
                outfile=(outfile := f"{lightcurve}.ascii"),
            )
            outfiles.append(Path(outfile))
        return outfiles

    @staticmethod
    def get_lightcurve_data(lightcurves: list[Path]):
        lightcurve_data: dict[str, DataFrame] = {
            energy_level: table.Table.read(lightcurve, format="ascii").to_pandas()
            for energy_level, lightcurve in zip(AcisProcessor.ENERGY_LEVELS.keys(), lightcurves)
        }
        # Trim zero exposure points
        for energy_level, lightcurve_dataframe in lightcurve_data.items():
            lightcurve_data[energy_level] = lightcurve_dataframe[
                lightcurve_dataframe["EXPOSURE"] != 0
            ]
        return lightcurve_data

    @staticmethod
    def get_lightcurve_counts(lightcurve_data):
        return int(lightcurve_data["broad"]["COUNTS"].sum())

    @staticmethod
    def create_csv(lightcurve_data):
        """Create CSV with columns for each energy level."""
        combined_data = DataFrame({
            "time": lightcurve_data["broad"]["TIME"],
            "count_rate": lightcurve_data["broad"]["COUNT_RATE"],
            "counts": lightcurve_data["broad"]["COUNTS"],
            "count_error": lightcurve_data["broad"]["COUNT_RATE_ERR"],
            "ultrasoft_count_rate": lightcurve_data["ultrasoft"]["COUNT_RATE"],
            "soft_count_rate": lightcurve_data["soft"]["COUNT_RATE"],
            "medium_count_rate": lightcurve_data["medium"]["COUNT_RATE"],
            "hard_count_rate": lightcurve_data["hard"]["COUNT_RATE"],
            "ultrasoft_counts": lightcurve_data["ultrasoft"]["COUNTS"],
            "soft_counts": lightcurve_data["soft"]["COUNTS"],
            "medium_counts": lightcurve_data["medium"]["COUNTS"],
            "hard_counts": lightcurve_data["hard"]["COUNTS"],
            "exposure": lightcurve_data["broad"]["EXPOSURE"],
            "area": lightcurve_data["broad"]["AREA"],
        })
        output_csv = StringIO()
        combined_data.to_csv(output_csv, index=False)
        return output_csv

    def plot(self, lightcurve_data):
        # The type casts are important as the data is returned by CIAO as NumPy data types.
        observation_data = ObservationData(
            average_count_rate=float(round(lightcurve_data["broad"]["COUNT_RATE"].mean(), 3)),
            total_counts=self.get_lightcurve_counts(lightcurve_data),
            total_exposure_time=float(round(lightcurve_data["broad"]["EXPOSURE"].sum(), 3)),
            raw_start_time=int(lightcurve_data["broad"]["TIME"].min()),
        )
        # This data is just so people can view the exact numerical data that was plotted.
        # output_plot_data = StringIO(lightcurve_data["broad"].to_string())
        return LightcurveParseResults(
            observation_header_info=self.get_observation_details(),
            observation_data=observation_data,
            plot_csv_data= self.create_csv(lightcurve_data),
            plot_svg_data=self.create_plot(lightcurve_data, self.binsize),
            postagestamp_png_data=plot_postagestamps(
                self.sky_coords_image, self.detector_coords_image
            ),
        )

    def create_plot(self, lightcurve_data: dict[str, DataFrame], binsize):
        """Generate a plt plot to model the lightcurves."""
        matplotlib.use("svg")

        # chandra time initialization
        chandra_mjd_ref = 50814.0
        time_seconds = lightcurve_data["broad"]["TIME"]
        initial_time = time_seconds.min()
        final_time = time_seconds.max()
        initial_time_days = initial_time / 86400.0
        observation_mjd = chandra_mjd_ref + initial_time_days
        observation_date = Time(observation_mjd, format='mjd').to_datetime()
        readable_date = observation_date.strftime('%Y-%m-%d %H:%M:%S')

        # details
        observation_id=dmkeypar(infile=f"{self.event_list}", keyword="OBS_ID", echo=True)
        file_path=self.event_list
        source_name=file_path.parts[1]

        # integer counts
        integer_counts = lightcurve_data["broad"]["COUNTS"].round().astype(int).reset_index(drop=True)
        req_min_counts = 15

        # count rate
        integer_count_rate = lightcurve_data["broad"]["COUNT_RATE"].round().astype(int).reset_index(drop=True)

        # time_kiloseconds = time_seconds / 1000
        zero_shifted_time_kiloseconds = (time_seconds - initial_time) / 1000
        observation_duration = zero_shifted_time_kiloseconds.max()

        # glvary_times, glvary_probs = self.get_glvary_data(glvary_file)
        # width, nrows
        width = 12 * (500 / binsize if binsize < 500 else 1)
        nrows = 14

        # figure 
        figure, (broad_plot, bayesian_blocks_plot_10, bayesian_blocks_plot_5, bayesian_blocks_plot_1, counts_plot, glvary_plot, separation_plot, hr_plot, cumulative_counts_plot, lomb_scargle_plot_freq, lomb_scargle_plot_per, lomb_scargle_plot_win, lomb_scargle_plot_win_cor, fracarea_plot) = plt.subplots(
            nrows=nrows, ncols=1, figsize=(width, nrows*3), constrained_layout=True
        )

        # count rate plot 
        broad_plot.errorbar(
            x=zero_shifted_time_kiloseconds,
            y=lightcurve_data["broad"]["COUNT_RATE"],
            yerr=lightcurve_data["broad"]["COUNT_RATE_ERR"],
            color="red",
            marker="s",
            markerfacecolor="black",
            markersize=4,
            ecolor="black",
            markeredgecolor="black",
            capsize=3,
        )        
        broad_plot.set_xlim([0, observation_duration])
        broad_plot.set_title("Broadband Count Rate", fontsize=14, y=1.05)
        broad_plot.set_ylabel("Count Rate (counts/s)", fontsize=12)
        broad_plot.set_xlabel("Time (kiloseconds)", fontsize=12)
        broad_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        broad_plot.xaxis.set_major_locator(MultipleLocator(5))  
        broad_plot.xaxis.set_minor_locator(MultipleLocator(1))
        broad_plot.tick_params(axis='both', which='major', labelsize=10)
        broad_plot.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=broad_plot.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
        broad_plot.text(0.995, 1.13, f"Start: {readable_date}",
            transform=broad_plot.transAxes,
            fontsize=10, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7))

        # separated light plot
        separated_light_level_colors = {"ultrasoft": "green", "soft": "red", "medium": "gold", "hard": "blue"}

        # loop through light levels
        for light_level, color in separated_light_level_colors.items():
            separation_plot.plot(
                zero_shifted_time_kiloseconds,
                lightcurve_data[light_level]["COUNT_RATE"],
                color=color,
                label=light_level.capitalize(),
            )
        separation_plot.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.19),
            ncol=4,
            frameon=False,
            fontsize=12,
        )
        separation_plot.set_xlim([0, observation_duration])
        separation_plot.set_title("Separated Energy Band Count Rates", fontsize=14, y = 1.29)
        separation_plot.set_ylabel("Count Rate (counts/s)", fontsize=12)
        separation_plot.set_xlabel("Time (kiloseconds)", fontsize=12)
        separation_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        separation_plot.xaxis.set_major_locator(MultipleLocator(5))  
        separation_plot.xaxis.set_minor_locator(MultipleLocator(1))
        separation_plot.tick_params(axis='both', which='major', labelsize=10)
        separation_plot.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2, frameon=False, fontsize=12)
        separation_plot.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=separation_plot.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
        avg_ultra=round(float(lightcurve_data["ultrasoft"]["COUNT_RATE"].mean()), 3)
        avg_soft=round(float(lightcurve_data["soft"]["COUNT_RATE"].mean()), 3)
        avg_medium=round(float(lightcurve_data["medium"]["COUNT_RATE"].mean()), 3)
        avg_hard=round(float(lightcurve_data["hard"]["COUNT_RATE"].mean()), 3)

        left_text_1 = f"Avg Ultra CR: {avg_ultra:.3f}"
        right_text_1 = f"Avg Soft CR: {avg_soft:.3f}"

        left_text_2 = f"Avg Medium CR: {avg_medium:.3f}"
        right_text_2 = f"Avg Hard CR: {avg_hard:.3f}"

        # determine the maximum lengths of the left and right text segments
        max_left_length = max(len(left_text_1), len(left_text_2))
        max_right_length = max(len(right_text_1), len(right_text_2))

        # set the total width with some padding (e.g., 5 characters)
        total_width = max_left_length + max_right_length + 5
        
        sep_str = (
            f"{left_text_1:<{total_width - max_right_length}}{right_text_1}\n"
            f"{left_text_2:<{total_width - max_right_length}}{right_text_2}"
        )
        separation_plot.text(0.995, 1.2, sep_str,
                     transform=separation_plot.transAxes,
                     fontsize=10, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # counts plot
        counts_plot.plot(
            zero_shifted_time_kiloseconds,
            lightcurve_data["broad"]["COUNTS"],
            color="purple",
            label="Broadband Counts",
            marker='o',
            markersize=4,
        )
        counts_plot.set_xlim([0, observation_duration])
        counts_plot.set_title("Counts in Broadband", fontsize=14, y = 1.05)
        counts_plot.set_xlabel("Time (kiloseconds)", fontsize=12)
        counts_plot.set_ylabel("Counts", fontsize=12)
        counts_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        counts_plot.xaxis.set_major_locator(MultipleLocator(5))  
        counts_plot.xaxis.set_minor_locator(MultipleLocator(1))
        counts_plot.tick_params(axis='both', which='major', labelsize=10)
        total_counts=round(float(lightcurve_data["broad"]["COUNTS"].sum()), 3)
        average_count_rate=round(float(lightcurve_data["broad"]["COUNT_RATE"].mean()), 3)
        min_count_rate=round(float(lightcurve_data["broad"]["COUNT_RATE"].min()), 3)
        max_count_rate=round(float(lightcurve_data["broad"]["COUNT_RATE"].max()), 3)

        left_text_1 = f"Total Counts: {total_counts:.3f}"
        right_text_1 = f"Avg Count Rate: {average_count_rate:.3f}"

        left_text_2 = f"Min Count Rate: {min_count_rate:.3f}"
        right_text_2 = f"Max Count Rate: {max_count_rate:.3f}"

        # determine the maximum lengths of the left and right text segments
        max_left_length = max(len(left_text_1), len(left_text_2))
        max_right_length = max(len(right_text_1), len(right_text_2))

        # set the total width with some padding (e.g., 5 characters)
        total_width = max_left_length + max_right_length + 5

        text_str = (
            f"{left_text_1:<{total_width - max_right_length}}{right_text_1}\n"
            f"{left_text_2:<{total_width - max_right_length}}{right_text_2}"
        )
        
        counts_plot.text(0.995, 1.2, text_str,
                     transform=counts_plot.transAxes,
                     fontsize=10, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7))
        counts_plot.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=counts_plot.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # calculate hardness ratios
        hard_counts = lightcurve_data["hard"]["COUNTS"]
        soft_counts = lightcurve_data["soft"]["COUNTS"]
        medium_counts = lightcurve_data["medium"]["COUNTS"]
        ultrasoft_counts = lightcurve_data["ultrasoft"]["COUNTS"]

        total_counts_hs = soft_counts + hard_counts
        hr_hs = -(hard_counts - soft_counts) / total_counts_hs
        total_counts_ms = soft_counts + medium_counts
        hr_ms = -(medium_counts - soft_counts) / total_counts_ms
        total_counts_smh = soft_counts + medium_counts + hard_counts
        hr_smh = (soft_counts - (medium_counts + hard_counts)) / total_counts_smh
        total_counts_mhsu = ultrasoft_counts + soft_counts + medium_counts + hard_counts
        hr_mhsu = -((medium_counts + hard_counts) - (soft_counts + ultrasoft_counts) ) / total_counts_mhsu

        # hardness ratio plot
        hr_plot.plot(
            zero_shifted_time_kiloseconds,
            hr_hs,
            color="blue",
            label="HR_HS (S-H)/(S+H)",
            marker='^',
            markersize=4,
        )

        hr_plot.plot(
            zero_shifted_time_kiloseconds,
            hr_ms,
            color="orange",
            label="HR_MS (S-M)/(S+M)",
            marker='v',
            markersize=4,
        )

        hr_plot.plot(
            zero_shifted_time_kiloseconds,
            hr_smh,
            color="green",
            label="HR_SMH (S-(M+H))/(S+M+H)",
            marker='o',
            markersize=4,
        )

        hr_plot.plot(
            zero_shifted_time_kiloseconds,
            hr_mhsu,
            color="purple",
            label="HR_HMSU ((S+U)-(M+H)))/(S+U+M+H)",
            marker='1',
            markersize=4,
        )

        hr_plot.set_xlim([0, observation_duration])
        hr_plot.set_title("Hardness Ratios", fontsize=14, y = 1.29)
        hr_plot.set_xlabel("Time (kiloseconds)", fontsize=12)
        hr_plot.set_ylabel("Hardness Ratio", fontsize=12)
        hr_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        hr_plot.xaxis.set_major_locator(MultipleLocator(5))
        hr_plot.xaxis.set_minor_locator(MultipleLocator(1))
        hr_plot.tick_params(axis='both', which='major', labelsize=10)
        hr_plot.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2, frameon=False, fontsize=12)
        hr_plot.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=hr_plot.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
    

        region_event_list = self.isolate_source_region(self.event_list, self.source_region)
        dmlist(
            infile=f"{region_event_list}[cols time]",
            outfile=(outfile := f"{region_event_list}.txt"),
            opt="data,raw"
        )
        times = table.Table.read(filename=f"{region_event_list}.txt", format="ascii")
        new_times = np.array(times)
        new_times_array = [x[0] for x in new_times]
        if (new_times_array[0] - initial_time) >= 0:
            new_times_array -= initial_time
        else: 
            new_times_array -= new_times_array[0]
        new_times_array /= 1000.0  # Convert to kiloseconds\


        sorted_times = np.sort(new_times_array)
        sorted_times = np.insert(sorted_times, 0, 0)
        sorted_times = np.append(sorted_times, observation_duration)
        y_values = np.arange(0, len(sorted_times) - 1)
        y_values = np.append(y_values, y_values[-1])

        # jimmy's cumulative counts plot
        cumulative_counts_plot.step(sorted_times, y_values, where='mid', color='magenta')
        cumulative_counts_plot.set_xlim([0, observation_duration])
        cumulative_counts_plot.set_title("Cumulative Counts", fontsize=14, y=1.05)
        cumulative_counts_plot.set_xlabel("Time (kiloseconds)", fontsize=12)
        cumulative_counts_plot.set_ylabel("Cumulative Counts", fontsize=12)
        cumulative_counts_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        cumulative_counts_plot.xaxis.set_major_locator(MultipleLocator(5))
        cumulative_counts_plot.xaxis.set_minor_locator(MultipleLocator(1))
        cumulative_counts_plot.tick_params(axis='both', which='major', labelsize=10)
        cumulative_counts_plot.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=cumulative_counts_plot.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # Compute Bayesian blocks
        if (total_counts < 5):
            hist = np.array([total_counts])
            bin_edges = np.array([0, observation_duration])
            # if fewer than 5 counts: tstart tstop make one big bin with number of counts and skip bb call
        else: 
            bb = bayesian_blocks(new_times_array, fitness="events", p0=5)
            bb10 = bayesian_blocks(new_times_array, fitness="events", p0=10)
            bb1 = bayesian_blocks(new_times_array, fitness="events", p0=1)

            # Compute histogram
            hist, bin_edges = np.histogram(new_times_array, bins=bb)
            hist = np.insert(hist, 0, 0)
            hist = np.append(hist, 0)
            bin_edges = np.insert(bin_edges, 0, 0)
            bin_edges = np.append(bin_edges, observation_duration)

        time_intervals = np.diff(bin_edges)

        min_interval_width = observation_duration / 100.0
        i = 0
        while i < len(time_intervals):
            if time_intervals[i] < min_interval_width:
                if i == 0:
                    hist[i+1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i+1)
                elif i == len(time_intervals) - 1:
                    hist[i-1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i)
                # Compare the counts in hist[i] and hist[i+1]
                elif hist[i-1] < hist[i+1]:
                    # Merge with the previous bin
                    hist[i-1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i)
                else:
                    # Merge with the next bin
                    hist[i] += hist[i+1]
                    hist = np.delete(hist, i+1)
                    bin_edges = np.delete(bin_edges, i+1)
                time_intervals = np.diff(bin_edges)
            else:
                i += 1
            
        count_rates = (hist / time_intervals)/1000
        errors = (np.sqrt(hist) / time_intervals)/1000
        errors = np.abs(errors)

        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        bayesian_blocks_plot_5.step(bin_edges, np.append(count_rates, count_rates[-1]), where='post', color='black', linestyle='-')
        bayesian_blocks_plot_5.errorbar(bin_midpoints, count_rates, yerr=errors, fmt='none', ecolor='red', capsize=3)

        bayesian_blocks_plot_5.set_xlim([0, observation_duration])
        bayesian_blocks_plot_5.set_title("Bayesian Blocks Segmentation: p0 = 5", fontsize=14, y=1.05)
        bayesian_blocks_plot_5.set_xlabel("Time (kiloseconds)", fontsize=12)
        bayesian_blocks_plot_5.set_ylabel("Count Rate", fontsize=12)
        bayesian_blocks_plot_5.grid(True, which='both', linestyle='--', linewidth=0.5)
        bayesian_blocks_plot_5.xaxis.set_major_locator(MultipleLocator(5))
        bayesian_blocks_plot_5.xaxis.set_minor_locator(MultipleLocator(1))
        bayesian_blocks_plot_5.tick_params(axis='both', which='major', labelsize=10)
        bayesian_blocks_plot_5.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=bayesian_blocks_plot_5.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
    
        if (total_counts < 5):
            hist = np.array([total_counts])
            bin_edges = np.array([0, observation_duration])
            # if fewer than 5 counts: tstart tstop make one big bin with number of counts and skip bb call
        else: 
            # Compute histogram
            hist, bin_edges = np.histogram(new_times_array, bins=bb10)
            hist = np.insert(hist, 0, 0)
            hist = np.append(hist, 0)
            bin_edges = np.insert(bin_edges, 0, 0)
            bin_edges = np.append(bin_edges, observation_duration)

        time_intervals = np.diff(bin_edges)

        min_interval_width = observation_duration / 100.0
        i = 0
        while i < len(time_intervals):
            if time_intervals[i] < min_interval_width:
                if i == 0:
                    hist[i+1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i+1)
                elif i == len(time_intervals) - 1:
                    hist[i-1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i)
                # Compare the counts in hist[i] and hist[i+1]
                elif hist[i-1] < hist[i+1]:
                    # Merge with the previous bin
                    hist[i-1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i)
                else:
                    # Merge with the next bin
                    hist[i] += hist[i+1]
                    hist = np.delete(hist, i+1)
                    bin_edges = np.delete(bin_edges, i+1)
                time_intervals = np.diff(bin_edges)
            else:
                i += 1
            
        count_rates = (hist / time_intervals)/1000
        errors = (np.sqrt(hist) / time_intervals)/1000
        errors = np.abs(errors)

        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        bayesian_blocks_plot_10.step(bin_edges, np.append(count_rates, count_rates[-1]), where='post', color='black', linestyle='-')
        bayesian_blocks_plot_10.errorbar(bin_midpoints, count_rates, yerr=errors, fmt='none', ecolor='red', capsize=3)

        bayesian_blocks_plot_10.set_xlim([0, observation_duration])
        bayesian_blocks_plot_10.set_title("Bayesian Blocks Segmentation: p0 = 10", fontsize=14, y=1.05)
        bayesian_blocks_plot_10.set_xlabel("Time (kiloseconds)", fontsize=12)
        bayesian_blocks_plot_10.set_ylabel("Count Rate", fontsize=12)
        bayesian_blocks_plot_10.grid(True, which='both', linestyle='--', linewidth=0.5)
        bayesian_blocks_plot_10.xaxis.set_major_locator(MultipleLocator(5))
        bayesian_blocks_plot_10.xaxis.set_minor_locator(MultipleLocator(1))
        bayesian_blocks_plot_10.tick_params(axis='both', which='major', labelsize=10)
        bayesian_blocks_plot_10.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=bayesian_blocks_plot_10.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

        # again for bb1

        if (total_counts < req_min_counts):
            hist = np.array([total_counts])
            bin_edges = np.array([0, observation_duration])
            # if fewer than 5 counts: tstart tstop make one big bin with number of counts and skip bb call
        else:
            # Compute histogram
            hist, bin_edges = np.histogram(new_times_array, bins=bb1)
            hist = np.insert(hist, 0, 0)
            hist = np.append(hist, 0)
            bin_edges = np.insert(bin_edges, 0, 0)
            bin_edges = np.append(bin_edges, observation_duration)

        time_intervals = np.diff(bin_edges)

        min_interval_width = observation_duration / 100.0
        i = 0
        while i < len(time_intervals):
            if time_intervals[i] < min_interval_width:
                if i == 0:
                    hist[i+1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i+1)
                elif i == len(time_intervals) - 1:
                    hist[i-1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i)
                # Compare the counts in hist[i] and hist[i+1]
                elif hist[i-1] < hist[i+1]:
                    # Merge with the previous bin
                    hist[i-1] += hist[i]
                    hist = np.delete(hist, i)
                    bin_edges = np.delete(bin_edges, i)
                else:
                    # Merge with the next bin
                    hist[i] += hist[i+1]
                    hist = np.delete(hist, i+1)
                    bin_edges = np.delete(bin_edges, i+1)
                time_intervals = np.diff(bin_edges)
            else:
                i += 1
            
        count_rates = (hist / time_intervals)/1000
        errors = (np.sqrt(hist) / time_intervals)/1000
        errors = np.abs(errors)

        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        bayesian_blocks_plot_1.step(bin_edges, np.append(count_rates, count_rates[-1]), where='post', color='black', linestyle='-')
        bayesian_blocks_plot_1.errorbar(bin_midpoints, count_rates, yerr=errors, fmt='none', ecolor='red', capsize=3)

        bayesian_blocks_plot_1.set_xlim([0, observation_duration])
        bayesian_blocks_plot_1.set_title("Bayesian Blocks Segmentation: p0 = 1", fontsize=14, y=1.05)
        bayesian_blocks_plot_1.set_xlabel("Time (kiloseconds)", fontsize=12)
        bayesian_blocks_plot_1.set_ylabel("Count Rate", fontsize=12)
        bayesian_blocks_plot_1.grid(True, which='both', linestyle='--', linewidth=0.5)
        bayesian_blocks_plot_1.xaxis.set_major_locator(MultipleLocator(5))
        bayesian_blocks_plot_1.xaxis.set_minor_locator(MultipleLocator(1))
        bayesian_blocks_plot_1.tick_params(axis='both', which='major', labelsize=10)
        bayesian_blocks_plot_1.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=bayesian_blocks_plot_1.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
    

        # lomb scargle lower bound
        initial_lower_bound = binsize / 1000

        # lomb scargle frequency plot
        frequency, power = LombScargle(zero_shifted_time_kiloseconds, integer_counts).autopower()
        lomb_scargle_plot_freq.set_xlim([0, 1/(binsize/1000)])
        lomb_scargle_plot_freq.plot(frequency, power, color='darkblue')
        lomb_scargle_plot_freq.set_title("Lomb-Scargle Frequency Plot", fontsize=14, y=1.05)
        lomb_scargle_plot_freq.set_xlabel("Frequency (1/kilosecond)", fontsize=12)
        lomb_scargle_plot_freq.set_ylabel("Power", fontsize=12)
        lomb_scargle_plot_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
        lomb_scargle_plot_freq.xaxis.set_major_locator(MultipleLocator(0.2))
        lomb_scargle_plot_freq.xaxis.set_minor_locator(MultipleLocator(0.1))
        lomb_scargle_plot_freq.tick_params(axis='both', which='major', labelsize=10)
        lomb_scargle_plot_freq.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=lomb_scargle_plot_freq.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
    
        # exposure time and data
        exptime = 3.2
        exp = lightcurve_data["broad"]["EXPOSURE"].reset_index(drop=True)

        # lomb scargle period plot
        frequency, power1 = LombScargle(zero_shifted_time_kiloseconds, integer_counts).autopower()
        period = 1/frequency
        lomb_scargle_plot_per.set_xlim([initial_lower_bound, observation_duration])
        lomb_scargle_plot_per.plot(period, power, color='blue')
        lomb_scargle_plot_per.set_title("Lomb-Scargle Periodogram", fontsize=14, y=1.05)
        lomb_scargle_plot_per.set_xlabel("Period (kiloseconds)", fontsize=12)
        lomb_scargle_plot_per.set_ylabel("Power", fontsize=12)
        lomb_scargle_plot_per.grid(True, which='both', linestyle='--', linewidth=0.5)
        lomb_scargle_plot_per.xaxis.set_major_locator(MultipleLocator(5))
        lomb_scargle_plot_per.xaxis.set_minor_locator(MultipleLocator(1))
        lomb_scargle_plot_per.tick_params(axis='both', which='major', labelsize=10)
        lomb_scargle_plot_per.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=lomb_scargle_plot_per.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
    
        # lomb scargle window plot
        frequency, power2 = LombScargle(zero_shifted_time_kiloseconds, exp).autopower()
        period = 1/frequency
        lomb_scargle_plot_win.set_xlim([initial_lower_bound, observation_duration])
        lomb_scargle_plot_win.plot(period, power2, color='lightblue')
        lomb_scargle_plot_win.set_title("Lomb-Scargle Plot of Window Function", fontsize=14, y=1.05)
        lomb_scargle_plot_win.set_xlabel("Period (kiloseconds)", fontsize=12)
        lomb_scargle_plot_win.set_ylabel("Power", fontsize=12)
        lomb_scargle_plot_win.grid(True, which='both', linestyle='--', linewidth=0.5)
        lomb_scargle_plot_win.xaxis.set_major_locator(MultipleLocator(5))
        lomb_scargle_plot_win.xaxis.set_minor_locator(MultipleLocator(1))
        lomb_scargle_plot_win.tick_params(axis='both', which='major', labelsize=10)
        lomb_scargle_plot_win.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=lomb_scargle_plot_win.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
    
        # lomb scargle divided by window plot
        new_power = power1/power2
        lomb_scargle_plot_win_cor.set_xlim([(binsize/1000), observation_duration])
        lomb_scargle_plot_win_cor.plot(period, new_power, color='orange')
        lomb_scargle_plot_win_cor.set_title("Lomb-Scargle Periodogram \nCorrected for Window Function", fontsize=14, y=1.05)
        lomb_scargle_plot_win_cor.set_xlabel("Period (kiloseconds)", fontsize=12)
        lomb_scargle_plot_win_cor.set_ylabel("Ratio", fontsize=12)
        lomb_scargle_plot_win_cor.grid(True, which='both', linestyle='--', linewidth=0.5)
        lomb_scargle_plot_win_cor.xaxis.set_major_locator(MultipleLocator(5))
        lomb_scargle_plot_win_cor.xaxis.set_minor_locator(MultipleLocator(1))
        lomb_scargle_plot_win_cor.tick_params(axis='both', which='major', labelsize=10)
        lomb_scargle_plot_win_cor.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=lomb_scargle_plot_win_cor.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

        observation_directory_event_list = self.event_list
        observation_directory = observation_directory_event_list.parent
        initial_directory = os.getcwd()
        os.chdir(observation_directory)
        download_chandra_obsids([observation_id], filetypes=["bpix", "asol","msk", "evt2"])
        os.chdir(initial_directory)
        obs_path = os.path.join(observation_directory, observation_id)
        obs_path_primary = os.path.join(obs_path, "primary")
        observation_path_primary = Path(obs_path_primary)
        obs_path_secondary = os.path.join(obs_path, "secondary")
        observation_path_secondary = Path(obs_path_secondary)
        gzip_files = observation_path_primary.glob("*.gz")
        for gzip_file in gzip_files:
            with gzip.open(gzip_file, "rb") as gzipped_file:
                unzipped_data = gzipped_file.read()
                with open(gzip_file.with_suffix(""), "wb") as unzipped_file:
                    unzipped_file.write(unzipped_data)
                gzip_file.unlink()
        gzip_files = observation_path_secondary.glob("*.gz")
        for gzip_file in gzip_files:
            with gzip.open(gzip_file, "rb") as gzipped_file:
                unzipped_data = gzipped_file.read()
                with open(gzip_file.with_suffix(""), "wb") as unzipped_file:
                    unzipped_file.write(unzipped_data)
                gzip_file.unlink()
        bpix_files = list(Path(observation_path_primary).rglob('*bpix1.fits'))
        bpixfile=bpix_files[0]
        evt2_files = list(Path(observation_path_primary).rglob('*evt2.fits'))
        evtfile=evt2_files[0]
        asol_files = list(Path(observation_path_primary).rglob('*asol1.fits'))
        asolfile=asol_files[0]
        msk_files = list(Path(observation_path_secondary).rglob('*msk1.fits'))
        mskfile=msk_files[0]
        if len(asol_files) != 1:
            textfile_path = Path(observation_path_primary) / 'asol_files.txt'
    
            with textfile_path.open('w') as textfile:
                for asol_file in asol_files:
                    textfile.write(f"{asol_file}\n")
            asolfile="@"+textfile_path

        region_file = Path(self.source_region)
        fracarea_path = Path(obs_path) / 'fracarea.fits'
        # acis_set_ardlib(badpixfile=bpixfile)
        dither_region(
            infile=asolfile,
            region=f"region({region_file})",
            maskfile=mskfile,
            outfile=fracarea_path
        )
        # Open the FITS file
        with io.fits.open(fracarea_path) as hdul:            
            # Access the desired HDU (assuming the data is in the first extension)
            data_hdu = hdul[1]
            
            # Extract the "time" and "fracarea" columns
            time = data_hdu.data['time']
            fracarea = data_hdu.data['fracarea']
            
            # Convert to NumPy arrays
            xx_array = np.array(time)
            yy_array = np.array(fracarea)

        xx_array -= xx_array[0]
        xx_array /= 1000
        
        
        fracarea_plot.plot(xx_array, yy_array, marker = "None", color = "maroon")
        fracarea_plot.set_xlim(0, observation_duration)
        # fracarea_plot.set_ylim(0, 1)
        fracarea_plot.set_title("Fraction of Region Area", fontsize=14, y=1.05)
        fracarea_plot.set_xlabel("Time (kiloseconds)", fontsize=12)
        fracarea_plot.set_ylabel("Fractional Area", fontsize=12)
        fracarea_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        fracarea_plot.xaxis.set_major_locator(MultipleLocator(5))
        fracarea_plot.xaxis.set_minor_locator(MultipleLocator(1))
        fracarea_plot.tick_params(axis='both', which='major', labelsize=10)
        fracarea_plot.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=fracarea_plot.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
        glvary_check = False
        if (total_counts < req_min_counts):
            glvary_plot.errorbar(x=zero_shifted_time_kiloseconds,
            y=lightcurve_data["broad"]["COUNT_RATE"],
            yerr=lightcurve_data["broad"]["COUNT_RATE_ERR"], ecolor="red", color="olive")
            # if fewer than 15 counts
        
        else: 
            glvary_path = Path(obs_path) / 'gl_prob.fits'
            glvary_lc_path = Path(obs_path) / 'lc_prob.fits'
            dtf_fracarea_path = Path(obs_path) / "dtf_fracarea.fits"
            dtcor = dmkeypar(infile=evtfile, keyword="DTCOR", echo=True)
            glvary_effile = dmtcalc(infile=f"{fracarea_path}[cols time,fracarea]", outfile=dtf_fracarea_path, expression=f"dtf=({dtcor}*fracarea)")
            glvary_result = glvary(infile=f"{evtfile}[sky=region({region_file})]", effile=dtf_fracarea_path, outfile=glvary_path, lcfile=glvary_lc_path)
            
            # Open the FITS file
            with io.fits.open(glvary_lc_path) as hdul:            
                # Access the desired HDU (assuming the data is in the first extension)
                data_hdu = hdul[1]
                
                # Extract the "time" and "fracarea" columns
                glvary_time = data_hdu.data['time']
                glvary_count_rate = data_hdu.data['COUNT_RATE']
                glvary_count_rate_error = data_hdu.data['COUNT_RATE_ERR']
                
                # Convert to NumPy arrays
                glvary_xx_array = np.array(glvary_time)
                glvary_yy_array = np.array(glvary_count_rate)
                glvary_ye_array = np.array(glvary_count_rate_error)

            if glvary_xx_array.size > 0:
                glvary_xx_array -= glvary_xx_array[0]
                glvary_xx_array /= 1000
                glvary_plot.plot(glvary_xx_array, glvary_yy_array, color="olive", label="Data")

                # Create shaded error bars
                glvary_plot.fill_between(
                    glvary_xx_array,
                    glvary_yy_array - (glvary_ye_array/2),  # Lower bound
                    glvary_yy_array + (glvary_ye_array/2),  # Upper bound
                    color='red',
                    alpha=0.3,  # Transparency of the shaded area
                    label='Error Range'
                )
                glvary_check = False
                
            else:
                glvary_check = True

        if glvary_check:
            glvary_plot.errorbar(x=zero_shifted_time_kiloseconds,
            y=lightcurve_data["broad"]["COUNT_RATE"],
            yerr=lightcurve_data["broad"]["COUNT_RATE_ERR"], ecolor="red", color="olive")
            # if fewer than 15 counts

        glvary_plot.set_xlim(0, observation_duration)
        glvary_plot.set_title("Gregory-Loredo Algorithm Lightcurve", fontsize=14, y=1.05)
        glvary_plot.set_xlabel("Time (kiloseconds)", fontsize=12)
        glvary_plot.set_ylabel("Count Rate", fontsize=12)
        glvary_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        glvary_plot.xaxis.set_major_locator(MultipleLocator(5))
        glvary_plot.xaxis.set_minor_locator(MultipleLocator(1))
        glvary_plot.tick_params(axis='both', which='major', labelsize=10)
        glvary_plot.text(0.005, 1.2, f"Source Name: {source_name}\nObsID: {observation_id}",
                    transform=glvary_plot.transAxes, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
        # glvary_plot.text(0.995, 1.13, f"VARINDEX: {VARINDEX}",
        #     transform=broad_plot.transAxes,
        #     fontsize=10, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # Key   22: R              FRAC3SIG     =       1.00000000     / Frac of lc within 3 sigma of avg rate
        # Key   23: R              FRAC5SIG     =       1.00000000     / Frac of lc within 5 sigma of avg rate
        # Key   24: I              VARINDEX     = 1                    / Variability index

        figure.suptitle(
            f"Lightcurve in Broadband and Separated Energy Bands (Binsize of {binsize}s)"
        , fontsize="xx-large")
        plt.savefig(svg_data := StringIO(), bbox_inches="tight")
        plt.close(figure)
        return svg_data


class HrcProcessor(ObservationProcessor):
    """Processes observations produced by the HRC (High Resolution Camera) instrument aboard Chandra."""

    # Define energy level as a constant for one band
    ENERGY_RANGE = "broad=100:10000"  # 0.1 - 10.0 keV

    def adjust_binsize(self, event_list):
        """
        Adjusts the binsize according to the HRC time resolution.

        Parameters:
        event_list (str): Path to the event list file.

        Returns:
        float: Adjusted binsize.
        """
        try:
            time_resolution = float(dmkeypar(infile=event_list, keyword="TIMEDEL", echo=True))
            return max(self.binsize // time_resolution * time_resolution, time_resolution)
        except Exception as e:
            raise ValueError(f"Failed to adjust binsize: {e}")

    def extract_lightcurves(self, event_list):
        """
        Extracts lightcurves from the event list file.

        Parameters:
        event_list (str): Path to the event list file.

        Returns:
        pathlib.Path: Path to the extracted lightcurve file.
        """
        adjusted_binsize = self.adjust_binsize(event_list)
        output_file = Path(f"{event_list}.broad.lc")
        dmextract(
            infile=f"{event_list}[{HrcProcessor.ENERGY_RANGE}][bin time=::{adjusted_binsize}]",
            outfile=str(output_file),
            opt="ltc1",
            clobber="yes",
        )
        return output_file

    def filter_lightcurve_columns(self, lightcurve):
        """
        Filters and extracts specific columns from lightcurve files.

        Parameters:
        lightcurve (pathlib.Path): Lightcurve file path.

        Returns:
        pathlib.Path: Path to the filtered lightcurve file.
        """
        output_file = lightcurve.with_suffix('.ascii')
        dmlist(
            infile=f"{lightcurve}[cols time,count_rate,count_rate_err,counts,exposure,area]",
            opt="data,clean",
            outfile=str(output_file),
        )
        return output_file

    def get_lightcurve_data(self, lightcurve):
        """
        Reads lightcurve data from file and returns as a DataFrame.

        Parameters:
        lightcurve (pathlib.Path): Lightcurve file path.

        Returns:
        pandas.DataFrame: Lightcurve data.
        """
        data = table.Table.read(lightcurve, format="ascii").to_pandas()
        # Filter out zero exposure points
        return data[data["broad"]["EXPOSURE"] != 0]

    def get_lightcurve_counts(self, lightcurve_data):
        """
        Sums the total counts from the lightcurve data.

        Parameters:
        lightcurve_data (pandas.DataFrame): Lightcurve data.

        Returns:
        int: Total counts.
        """
        return int(lightcurve_data["broad"]["COUNTS"].sum())

    @staticmethod
    def create_csv(lightcurve_data):
        """Create CSV with columns for each energy level.

        Parameters
        ----------
        lightcurve_data : dict
            Dictionary of DataFrames containing lightcurve data.

        Returns
        -------
        StringIO
            StringIO object with CSV data.
        """
        combined_data = DataFrame({
            "time": lightcurve_data["broad"]["TIME"],
            "count_rate": lightcurve_data["broad"]["COUNT_RATE"],
            "counts": lightcurve_data["broad"]["COUNTS"],
            "count_error": lightcurve_data["broad"]["COUNT_RATE_ERR"],
            "exposure": lightcurve_data["broad"]["EXPOSURE"],
            "area": lightcurve_data["broad"]["AREA"],
        })
        output_csv = StringIO()
        combined_data.to_csv(output_csv, index=False)
        return output_csv

    def plot(self, lightcurve_data):
        """
        Generates plots for the lightcurve data.

        Parameters:
        lightcurve_data (pandas.DataFrame): Lightcurve data.

        Returns:
        LightcurveParseResults: Results including observation data and plots.
        """
        observation_data = ObservationData(
            average_count_rate=float(lightcurve_data["broad"]["COUNT_RATE"].mean()),
            total_counts=self.get_lightcurve_counts(lightcurve_data),
            total_exposure_time=float(lightcurve_data["broad"]["EXPOSURE"].sum()),
            raw_start_time=int(lightcurve_data["broad"]["TIME"].min()),
        )

        # output_plot_data = StringIO(lightcurve_data.to_string())
        return LightcurveParseResults(
            observation_header_info=self.get_observation_details(),
            observation_data=observation_data,
            plot_csv_data=self.create_csv(lightcurve_data),
            plot_svg_data=self.create_plot(lightcurve_data, self.binsize),
            postagestamp_png_data=plot_postagestamps(
                self.sky_coords_image, self.detector_coords_image
            ),
        )

    @staticmethod
    def create_plot(lightcurve_data: dict[str, DataFrame], binsize):
        """Generate a plt plot to model the lightcurves.

        Parameters
        ----------
        lightcurve_data : dict
            Dictionary of DataFrames containing lightcurve data.
        binsize : float
            Binsize used for the lightcurve.

        Returns
        -------
        StringIO
            StringIO object with SVG data of the generated plot.
        """
        plt.switch_backend("svg")
        initial_time = lightcurve_data["broad"]["TIME"].min()
        zero_shifted_time_ks = (lightcurve_data["broad"]["TIME"] - initial_time) / 1000
        observation_duration = zero_shifted_time_ks.max()

        fig, (count_rate_plot, counts_plot) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), constrained_layout=True)

        # Plot for count rates
        count_rate_plot.errorbar(
            x=zero_shifted_time_ks,
            y=lightcurve_data["broad"]["COUNT_RATE"],
            yerr=lightcurve_data["broad"]["COUNT_RATE_ERR"],
            color="blue",
            marker="o",
            markerfacecolor="black",
            markersize=4,
            ecolor="black",
            markeredgecolor="black",
            capsize=3,
        )
        count_rate_plot.set_xlim([0, observation_duration])
        count_rate_plot.set_title("Broadband Count Rate", fontsize=14)
        count_rate_plot.set_ylabel("Count Rate (counts/s)", fontsize=12)
        count_rate_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        count_rate_plot.xaxis.set_major_locator(MultipleLocator(5))
        count_rate_plot.xaxis.set_minor_locator(MultipleLocator(1))
        count_rate_plot.tick_params(axis='both', which='major', labelsize=10)

        # Plot for counts
        counts_plot.plot(
            zero_shifted_time_ks,
            lightcurve_data["broad"]["COUNTS"],
            color="purple",
            label="Broadband Counts",
            marker='o',
            markersize=4,
        )
        counts_plot.set_xlim([0, observation_duration])
        counts_plot.set_title("Counts in Broadband", fontsize=14)
        counts_plot.set_xlabel("Time (kiloseconds)", fontsize=12)
        counts_plot.set_ylabel("Counts", fontsize=12)
        counts_plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        counts_plot.xaxis.set_major_locator(MultipleLocator(5))
        counts_plot.xaxis.set_minor_locator(MultipleLocator(1))
        counts_plot.tick_params(axis='both', which='major', labelsize=10)

        fig.suptitle(f"Lightcurve in Broadband (Binsize of {binsize}s)", fontsize=16)
        svg_data = StringIO()
        plt.savefig(svg_data, bbox_inches="tight")
        plt.close(fig)
        return svg_data
