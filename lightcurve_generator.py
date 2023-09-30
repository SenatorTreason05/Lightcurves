"""Mihir Patankar [mpatankar06@gmail.com]"""
import gzip
import re
from concurrent import futures
from pathlib import Path
import multiprocessing
from threading import Thread
from data_structures import LightcurveParseResults, Message

from lightcurve_processing import AcisProcessor, HrcProcessor


class LightcurveGenerator:
    """Manages the generation and organization of lightcurves from a given source."""

    def __init__(self, program_config, print_queue, increment_sources_processed_function, exporter):
        self.config = program_config
        self.print_queue = print_queue
        self.increment_sources_processed = increment_sources_processed_function
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

    def dispatch_source_processing(self, source_queue):
        """Start the processing of an observation directory, after validating the file structure
        seems correct."""

        message_collection_queue = multiprocessing.Manager().Queue()

        def transfer_queues():
            while True:
                new_message = message_collection_queue.get()
                if not new_message:
                    break
                self.print_queue.put(new_message)

        transfer_thread = Thread(target=transfer_queues)
        transfer_thread.start()

        with futures.ProcessPoolExecutor(max_workers=14) as executor:
            while True:
                source_directory = source_queue.get()
                if not source_directory:
                    break
                self.print_queue.put(Message("Processing: " + source_directory.name))
                self.validate_source_directory(source_directory)
                # Sources ending in X are "extended sources." They can be ignored.
                if source_directory.name.endswith("X"):
                    self.increment_sources_processed()
                    continue
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
                if self.get_maximum_counts_in_source(results) < self.config["Minimum Counts"]:
                    continue
                self.exporter.add_source(source_directory.name, results)

            message_collection_queue.put(None)
            transfer_thread.join()

    @staticmethod
    def get_maximum_counts_in_source(source_processing_results: list[LightcurveParseResults]):
        """Makes sure that at least one observation in the source meets the user specified total
        counts threshold"""
        return max(
            source_processing_results,
            key=lambda observation: observation.observation_data.total_counts,
        ).observation_data.total_counts

    def assign_workers(self, observation_directories, message_collection_queue):
        """Map processors to observation directories."""
        __ = [
            self.get_instrument_processor(
                *self.get_observation_files(observation_directory),
                message_collection_queue,
                self.config,
            )
            for observation_directory in observation_directories
        ]

        # Temp code!
        for _ in __:
            if isinstance(_, HrcProcessor):
                __.remove(_)
        return __

    @staticmethod
    def get_observation_files(observation_directory: Path):
        """Return the observation data product files."""
        LightcurveGenerator.unzip_fits_files(observation_directory)
        event_list_file, source_region_file = None, None
        for data_product_file in observation_directory.glob("*.fits"):
            stem = data_product_file.stem.lower()
            if stem.endswith("regevt3"):
                event_list_file = data_product_file
            if stem.endswith("reg3"):
                source_region_file = data_product_file
        if not event_list_file or not source_region_file:
            raise OSError("Data product missing.")
        return event_list_file, source_region_file

    @staticmethod
    def get_instrument_processor(event_list_file, source_region_file, message_collection, config):
        """Determines which instrument on Chandra the observation was obtained through: ACIS
        (Advanced CCD Imaging Spectrometer) or HRC (High Resolution Camera)"""
        acis_pattern, hrc_pattern = r"^acis", r"^hrc"
        if re.match(acis_pattern, event_list_file.name):
            return AcisProcessor(event_list_file, source_region_file, message_collection, config)
        if re.match(hrc_pattern, event_list_file.event_list_file.name):
            return HrcProcessor(event_list_file, source_region_file, message_collection, config)
        raise RuntimeError("Unable to resolve observation instrument")
