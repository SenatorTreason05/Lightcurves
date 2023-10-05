"""Mihir Patankar [mpatankar06@gmail.com]"""
import gzip
import multiprocessing
import re
import time
from concurrent import futures
from pathlib import Path
from threading import Thread

from data_structures import CountsChecker, DataProducts, Message
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
        manager = multiprocessing.Manager()
        message_collection_queue = manager.Queue()

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
                counts_checker = CountsChecker(
                    manager.Queue(maxsize=len(observation_directories)),
                    manager.Event(),
                )
                workers = self.assign_workers(
                    observation_directories, message_collection_queue, counts_checker
                )
                worker_futures = [executor.submit(worker.process) for worker in workers]
                count_checker_thread = Thread(
                    target=self.verify_sufficient_counts_in_source,
                    args=(
                        counts_checker,
                        len(worker_futures),
                        self.config["Minimum Counts"],
                    ),
                )
                count_checker_thread.start()
                done, not_done = futures.wait(worker_futures, return_when=futures.FIRST_EXCEPTION)
                count_checker_thread.join()

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
                self.increment_sources_processed()
                if None in results:
                    self.print_queue.put(Message("Insufficient counts in " + source_directory.name))
                    continue
                self.print_queue.put(Message("Done with: " + source_directory.name))
                self.exporter.add_source(source_directory.name, results)

            message_collection_queue.put(None)
            transfer_thread.join()

    @staticmethod
    def verify_sufficient_counts_in_source(
        counts_checker: CountsChecker, queue_max_size, max_counts
    ):
        """Makes sure that all observations contain enough counts to meet the user specified
        threshold, otherwise signal to the processor that it needs to cancel itself."""
        while True:
            if counts_checker.queue.qsize() == queue_max_size:
                queue_members = tuple(counts_checker.queue.get() for _ in range(queue_max_size))
                if any(observation < max_counts for observation in queue_members):
                    counts_checker.cancel_event.set()
                for _ in range(queue_max_size):
                    counts_checker.queue.task_done()
                break
            time.sleep(0.1)

    def assign_workers(self, observation_directories, message_collection_queue, counts_check_queue):
        """Map processors to observation directories."""
        __ = [
            self.get_instrument_processor(
                *self.get_observation_files(observation_directory),
                message_collection_queue,
                counts_check_queue,
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
    def get_instrument_processor(
        event_list_file, source_region_file, message_collection, counts_checker, config
    ):
        """Determines which instrument on Chandra the observation was obtained through: ACIS
        (Advanced CCD Imaging Spectrometer) or HRC (High Resolution Camera)"""
        data_products = DataProducts(event_list_file, source_region_file)
        acis_pattern, hrc_pattern = r"^acis", r"^hrc"
        if re.match(acis_pattern, data_products.event_list_file.name):
            return AcisProcessor(data_products, message_collection, counts_checker, config)
        if re.match(hrc_pattern, data_products.event_list_file.name):
            return HrcProcessor(data_products, message_collection, counts_checker, config)
        raise RuntimeError("Unable to resolve observation instrument")
