"""Mihir Patankar [mpatankar06@gmail.com]"""
import gzip
import re
from concurrent import futures
from pathlib import Path
from multiprocessing import Manager, Queue
from threading import Thread
from time import sleep
import time
from data_structures import Message

from lightcurve_processing import AcisProcessor, HrcProcessor


class LightcurveGenerator:
    """Manages the generation and organization of lightcurves from a given source."""

    def __init__(self, binsize, print_queue, increment_sources_processed, exporter):
        self.binsize = binsize
        self.print_queue = print_queue
        self.increment_sources_processed = increment_sources_processed
        self.exporter = exporter

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

    @staticmethod
    def validate_source_directory(source_directory):
        """Makes sure source directory is valid."""
        if not (source_directory.exists() and source_directory.is_dir()):
            raise OSError(f"Source directory {source_directory} not found.")

    def dispatch_source_processing(self, source_queue: Queue):
        """Start the processing of an observation directory, after validating the file structure
        seems correct."""

        message_collection_queue = Manager().Queue()

        def transfer_queues():
            while True:
                new_message = message_collection_queue.get()
                if not new_message:
                    break
                self.print_queue.put(new_message)

        transfer_thread = Thread(target=transfer_queues)
        transfer_thread.start()

        # The number of max workers is arbitrary right now.
        with futures.ProcessPoolExecutor(max_workers=8) as executor:
            while True:
                source_directory = source_queue.get()
                if not source_directory:
                    break
                self.print_queue.put(Message("Processing: " + source_directory.name))
                # TODO Chandra soruces that end with X are invalid, find out why
                self.validate_source_directory(source_directory)
                if source_directory.name.endswith("X"):
                    return
                observation_directories = [
                    observation_directory
                    for observation_directory in source_directory.iterdir()
                    if observation_directory.is_dir()
                ]
                workers = self.assign_workers(observation_directories, message_collection_queue)
                worker_futures = [executor.submit(worker.process) for worker in workers]
                done, not_done = futures.wait(worker_futures, return_when=futures.FIRST_EXCEPTION)

                results = []
                for future in done:
                    try:
                        results.append(future.result())
                    except Exception as exception:
                        raise RuntimeError(
                            f"Error while processing {source_directory.name}"
                        ) from exception
                if len(not_done) > 0:
                    raise RuntimeError("Some observations were not processed.")
                self.print_queue.put(Message("Done with: " + source_directory.name))
                self.increment_sources_processed()
                self.exporter.add_source(source_directory.name, results)

            message_collection_queue.put(None)
            transfer_thread.join()

    def assign_workers(self, observation_directories, message_collection_queue):
        """Map processors to observation directories."""
        aa = [
            self.get_instrument_processor(
                *self.get_observation_files(observation_directory),
                message_collection_queue,
                self.binsize,
            )
            for observation_directory in observation_directories
        ]

        # Temp code!
        for a in aa:
            if isinstance(a, HrcProcessor):
                aa.remove(a)
        return aa

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
    def get_instrument_processor(event_list, source_region, message_collection, binsize):
        """Determines which instrument on Chandra the observation was obtained through: ACIS
        (Advanced CCD Imaging Spectrometer) or HRC (High Resolution Camera)"""
        acis_pattern, hrc_pattern = r"^acis", r"^hrc"
        if re.match(acis_pattern, event_list.name) and re.match(acis_pattern, source_region.name):
            return AcisProcessor(str(event_list), str(source_region), message_collection, binsize)
        if re.match(hrc_pattern, event_list.name) and re.match(hrc_pattern, source_region.name):
            return HrcProcessor(str(event_list), str(source_region), message_collection, binsize)
        raise RuntimeError("Unable to resolve observation instrument")
