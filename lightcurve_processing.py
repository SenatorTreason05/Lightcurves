"""Mihir Patankar [mpatankar06@gmail.com]"""
from io import BytesIO, StringIO
import uuid
from abc import ABC, abstractmethod

from pathlib import Path
from astropy import visualization, table, io

# pylint: disable-next=no-name-in-module
from ciao_contrib.runtool import dmcopy, dmextract, dmkeypar, dmlist, new_pfiles_environment
from pandas import DataFrame
import matplotlib
from matplotlib import pyplot

from data_structures import LightcurveParseResults, Message, ObservationData, ObservationHeaderInfo


class ObservationProcessor(ABC):
    """Base class for observation processor implementations for different Chandra instruments."""

    def __init__(self, data_products, message_collection_queue, config):
        self.event_list = Path(data_products.event_list_file)
        self.source_region = Path(data_products.source_region_file)
        self.source_image = Path(data_products.region_image_file) if config["Plot Image"] else None
        self.message_collection_queue = message_collection_queue
        self.binsize = config["Binsize"]
        self.minimum_counts = config["Minimum Counts"]

    def process(self):
        """Sequence in which all steps of the processing routine are called."""
        message_uuid = uuid.uuid4()
        with new_pfiles_environment():
            observation_id = dmkeypar(infile=f"{self.event_list}", keyword="OBS_ID", echo=True)
            prefix = f"Observation {observation_id}: "
            isolated_event_list = self.extract_source_region()
            self.message_collection_queue.put(Message(f"{prefix}Isolated...", message_uuid))
            lightcurves = self.extract_lightcurves(isolated_event_list, self.binsize)
            self.message_collection_queue.put(Message(f"{prefix}Extracted...", message_uuid))
            filtered_lightcurves = self.filter_lightcurves(lightcurves)
            self.message_collection_queue.put(Message(f"{prefix}Filtered...", message_uuid))
            parse_results = self.parse(filtered_lightcurves)
            self.message_collection_queue.put(Message(f"{prefix}Parsed... Done.", message_uuid))
        return parse_results

    @staticmethod
    @abstractmethod
    def extract_lightcurves(event_list, binsize):
        """Extract lightcurve(s) from an event list, one should pass in one with a specific source
        region extracted."""

    @staticmethod
    @abstractmethod
    def filter_lightcurves(lightcurves: list[Path]):
        """Filter lightcurve(s) to get the columns we care about and format them."""

    @abstractmethod
    def parse(self, lightcurves: list[Path]) -> LightcurveParseResults:
        """Parse lightcurve(s). Returns what will then be returned by the thread pool future."""

    def extract_source_region(self):
        """Extracts the region of the event list that matches the source region we care about."""
        dmcopy(
            infile=f"{self.event_list}[sky=region({self.source_region})]",
            outfile=(outfile := f"{self.event_list}.src"),
        )
        return Path(outfile)

    def get_observation_details(self):
        """Gets keys from the header block detailing the observation information."""
        return ObservationHeaderInfo(
            instrument=dmkeypar(infile=f"{self.event_list}", keyword="INSTRUME", echo=True),
            observation_id=dmkeypar(infile=f"{self.event_list}", keyword="OBS_ID", echo=True),
            region_id=dmkeypar(infile=f"{self.event_list}", keyword="REGIONID", echo=True),
            start_time=dmkeypar(infile=f"{self.event_list}", keyword="DATE-OBS", echo=True),
            end_time=dmkeypar(infile=f"{self.event_list}", keyword="DATE-END", echo=True),
        )


class AcisProcessor(ObservationProcessor):
    """Processes observations produced by the ACIS (Advanced CCD Imaging Spectrometer) instrument
    aboard Chandra."""

    ENERGY_LEVELS = {
        "broad": "energy=300:7000",
        "soft": "energy=300:1200",
        "medium": "energy=1200:2000",
        "hard": "energy=2000:7000",
    }

    @staticmethod
    def adjust_binsize(event_list, binsize):
        """For ACIS, time resolution can be in the seconds in timed exposure mode, as compared to in the microseconds for HRC."""
        # TODO https://cxc.cfa.harvard.edu/ciao/ahelp/times.html find out the difference between interleaved and standard TIME mode
        time_resolution = float(dmkeypar(infile=str(event_list), keyword="TIMEDEL", echo=True))
        return binsize // time_resolution * time_resolution

    @staticmethod
    def extract_lightcurves(event_list, binsize):
        outfiles = []
        for light_level, energy_range in AcisProcessor.ENERGY_LEVELS.items():
            dmextract(
                infile=f"{event_list}[{energy_range}][bin time=::"
                f"{AcisProcessor.adjust_binsize(event_list, binsize)}]",
                outfile=(outfile := f"{event_list}.{light_level}.lc"),
                opt="ltc1",
            )
            outfiles.append(Path(outfile))
        return outfiles

    @staticmethod
    def filter_lightcurves(lightcurves):
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

    def parse(self, lightcurves):
        lightcurve_data: dict[str, DataFrame] = {
            energy_level: table.Table.read(lightcurve, format="ascii").to_pandas()
            for energy_level, lightcurve in zip(AcisProcessor.ENERGY_LEVELS.keys(), lightcurves)
        }
        # Trim zero exposure points
        for energy_level, lightcurve_dataframe in lightcurve_data.items():
            lightcurve_data[energy_level] = lightcurve_dataframe[
                lightcurve_dataframe["EXPOSURE"] != 0
            ]
        # The type casts are important as the data is returned as NumPy data types by CIAO.
        observation_data = ObservationData(
            average_count_rate=float(round(lightcurve_data["broad"]["COUNT_RATE"].mean(), 3)),
            total_counts=int(lightcurve_data["broad"]["COUNTS"].sum()),
            total_exposure_time=float(round(lightcurve_data["broad"]["EXPOSURE"].sum(), 3)),
            raw_start_time=int(lightcurve_data["broad"]["TIME"].min()),
        )
        output_plot_data = StringIO(
            lightcurve_data["broad"][["TIME", "COUNT_RATE", "COUNT_RATE_ERR"]].to_csv()
        )
        return LightcurveParseResults(
            observation_header_info=self.get_observation_details(),
            observation_data=observation_data,
            plot_csv_data=output_plot_data,
            plot_svg_data=self.plot(lightcurve_data),
            postagestamp_png_data=self.plot_postagestamp(self.source_image)
            if self.source_image
            else None,
        )

    @staticmethod
    def plot(lightcurve_data: dict[str, DataFrame]):
        """Generate a pyplot plot to model the lightcurves."""
        matplotlib.use("svg")
        initial_time = lightcurve_data["broad"]["TIME"].min()
        zero_shifted_time_kiloseconds = (lightcurve_data["broad"]["TIME"] - initial_time) / 1000
        observation_duration = zero_shifted_time_kiloseconds.max()
        figure, (broad_plot, seperation_plot) = pyplot.subplots(
            nrows=2, ncols=1, figsize=(10, 5), constrained_layout=True, sharey=True
        )
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
        seperated_light_level_colors = {"soft": "lightsalmon", "medium": "red", "hard": "firebrick"}
        for light_level, color in seperated_light_level_colors.items():
            seperation_plot.plot(
                zero_shifted_time_kiloseconds,
                lightcurve_data[light_level]["COUNT_RATE"],
                color=color,
                label=light_level.capitalize(),
            )
        seperation_plot.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            frameon=False,
            fontsize=14,
        )
        seperation_plot.set_xlim([0, observation_duration])
        figure.supylabel("Count Rate (counts/second)")
        figure.supxlabel("Time (kiloseconds)")
        figure.suptitle("Lightcurve in Broadband and Separated Energy Bands")
        pyplot.savefig(svg_data := StringIO(), bbox_inches="tight")
        pyplot.close(figure)
        return svg_data

    @staticmethod
    def plot_postagestamp(source_image_file):
        """Plots a binned image representing photon position data restricted to the source region"""
        matplotlib.use("agg")
        pyplot.style.use(visualization.astropy_mpl_style)
        image_data = io.fits.getdata(source_image_file, ext=0)
        figure = pyplot.figure()
        plot = figure.add_subplot(111)
        plot.imshow(image_data, cmap="gray")
        pyplot.savefig(png_data := BytesIO(), bbox_inches="tight")
        pyplot.close(figure)
        return png_data


class HrcProcessor(ObservationProcessor):
    """Processes for observations produced by the HRC (High Resolution Camera) instrument aboard
    Chandra."""

    @staticmethod
    def extract_lightcurves(event_list, binsize):
        raise NotImplementedError()

    @staticmethod
    def filter_lightcurves(lightcurves):
        raise NotImplementedError()

    def parse(self, lightcurves):
        raise NotImplementedError()
