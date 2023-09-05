"""Mihir Patankar [mpatankar06@gmail.com]"""
import gzip
import re
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Queue
from threading import Thread
import threading
import uuid
from typing import NamedTuple

# pylint: disable-next=no-name-in-module
from ciao_contrib.runtool import dmcopy, dmextract, dmkeypar, new_pfiles_environment

from message import Message


class LightcurveGenerator:
    def __init__(self, output_directory, binsize, print_queue):
        self.output_directory = output_directory
        self.binsize = binsize
        self.print_queue = print_queue

    @staticmethod
    def unzip_fits_files(observation_path: Path):
        """Files are downloaded in a GNU Zip format. This unzips all files in an observation
        directory into the FITS files we need."""
        gzip_files = observation_path.glob("*.gz")
        try:
            for gzip_file in gzip_files:
                with gzip.open(gzip_file, "rb") as gzipped_file:
                    unzipped_data = gzipped_file.read()
                    with open(gzip_file.with_suffix(""), "wb") as unzipped_file:
                        unzipped_file.write(unzipped_data)
                    gzip_file.unlink()
        except gzip.BadGzipFile as error:
            raise RuntimeError(f"Could not unzip file {gzip_file.name}.") from error

    def process_source(self, path_to_source):
        self.dispatch_observation_processing(path_to_source)

    def dispatch_observation_processing(self, source_directory: Path):
        if not (source_directory.exists() and source_directory.is_dir()):
            raise OSError(f"Source directory {source_directory} not found.")
        message_collection_queue = Queue()

        def transfer_queues():
            while True:
                new_message = message_collection_queue.get()
                message_collection_queue.task_done()
                if not new_message:
                    message_collection_queue.join()
                    break
                self.print_queue.put(new_message)

        transfer_thread = Thread(target=transfer_queues)
        transfer_thread.start()

        observation_directories = [
            observation_directory
            for observation_directory in source_directory.iterdir()
            if observation_directory.is_dir()
        ]
        # The number of max workers is arbitrary right now.
        with ThreadPoolExecutor(max_workers=6) as executor:
            workers = self.assign_workers(observation_directories, message_collection_queue)
            worker_futures = [executor.submit(worker.process) for worker in workers]
            done, not_done = wait(worker_futures, return_when=FIRST_EXCEPTION)
            message_collection_queue.put(None)
            transfer_thread.join()
            for future in done:
                try:
                    future.result()
                except Exception as exception:
                    raise RuntimeError(
                        f"Error while processing {source_directory.name}"
                    ) from exception
            if len(not_done) > 0:
                raise RuntimeError("Some observations were not processed.")

    def assign_workers(self, observation_directories, message_collection_queue):
        return [
            self.get_instrument_processor(
                *self.get_observation_files(observation_directory),
                message_collection_queue,
                self.binsize,
            )
            for observation_directory in observation_directories
        ]

    @staticmethod
    def get_observation_files(observation_directory: Path):
        """Verify and return the observation data product files."""
        LightcurveGenerator.unzip_fits_files(observation_directory)
        valid_contents = tuple(
            data_product
            for data_product in observation_directory.iterdir()
            if data_product.suffix == ".fits"
        )
        if len(valid_contents) != 2:
            raise OSError("Observation contents are invalid.")
        event_list, source_region = None, None
        for data_product in valid_contents:
            if data_product.stem.endswith("regevt3"):
                event_list = data_product
            elif data_product.stem.endswith("reg3"):
                source_region = data_product
        if not event_list or not source_region:
            raise OSError("Missing data products.")
        return event_list, source_region

    @staticmethod
    def get_instrument_processor(event_list, source_region, message_collection_queue, binsize):
        """Determines which instrument on Chandra the observation was obtained through: ACIS
        (Advanced CCD Imaging Spectrometer) or HRC (High Resolution Camera)"""
        acis_pattern, hrc_pattern = r"^acis", r"^hrc"
        if re.match(acis_pattern, event_list.name) and re.match(acis_pattern, source_region.name):
            return AcisProcessor(event_list, source_region, message_collection_queue, binsize)
        if re.match(hrc_pattern, event_list.name) and re.match(hrc_pattern, source_region.name):
            return HrcProcessor(event_list, source_region, message_collection_queue, binsize)
        raise RuntimeError("Unable to resolve observation instrument")


class ObservationProcessor(ABC):
    """"""

    # After 7 hours of debugging race conditions, I have determined you need a lock when using CIAO
    # dm commands or horrific things will happen. CIAO is NOT thread-safe.
    ciao_lock = threading.Lock()

    def __init__(self, event_list: Path, source_region: Path, message_collection_queue, binsize):
        self.event_list = event_list
        self.source_region = source_region
        self.message_collection_queue = message_collection_queue
        self.binsize = binsize
        self.message_uuid = uuid.uuid4()
        self.hit_count = 0

    def process(self):
        with new_pfiles_environment():
            try:
                isolated_event_list = self.extract_source_region()
            except OSError as error:
                raise RuntimeError(
                    f"Could not extract source region {self.source_region.name}"
                ) from error
            self.message_collection_queue.put(
                Message("Extracted source region.", self.message_uuid)
            )
        try:
            self.extract_lightcurve(isolated_event_list, self.binsize)
        except OSError as error:
            raise RuntimeError() from error
        self.message_collection_queue.put(Message("Extracted lightcurve.", self.message_uuid))
        return "Observation successfully"

    @staticmethod
    @abstractmethod
    def extract_lightcurve(event_list, binsize):
        """Extract lightcurve an event list, one should pass in one with a specific source region
        extracted."""

    def extract_source_region(self):
        with ObservationProcessor.ciao_lock:
            dmcopy(
                infile=f"{self.event_list}[sky=region({self.source_region})]",
                outfile=(outfile := f"{self.event_list}.src"),
            )
        return Path(outfile)


class AcisProcessor(ObservationProcessor):
    @staticmethod
    def adjust_binsize(event_list, binsize):
        """For ACIS, time resolution can be in the seconds in timed exposure mode, as compared to in the microseconds for HRC."""
        # TODO https://cxc.cfa.harvard.edu/ciao/ahelp/times.html find out the difference betweeen interleaved and standard TIME mode
        time_resolution = float(dmkeypar(infile=str(event_list), keyword="TIMEDEL", echo=True))
        return binsize // time_resolution * time_resolution

    @staticmethod
    def extract_lightcurve(event_list, binsize):
        energies = {
            "broad": "energy=300:7000",
            "soft": "energy=300:1200",
            "medium": "energy=1200:2000",
            "hard": "energy=2000:7000",
        }
        for light_level, energy_range in energies.items():
            with ObservationProcessor.ciao_lock:
                dmextract(
                    infile=f"{event_list}[{energy_range}]"
                    f"[bin time=::{AcisProcessor.adjust_binsize(event_list, binsize)}]",
                    outfile=f"{event_list}.{light_level}.lc",
                    opt="ltc1",
                )


class HrcProcessor(ObservationProcessor):
    @staticmethod
    def extract_lightcurve(event_list, binsize):
        raise NotImplementedError()
