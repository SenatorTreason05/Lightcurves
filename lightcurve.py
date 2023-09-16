"""Mihir Patankar [mpatankar06@gmail.com]"""
import gzip
import re
from concurrent import futures
from pathlib import Path
from queue import Queue
from threading import Thread

from lightcurve_processing import AcisProcessor, HrcProcessor


class LightcurveGenerator:
    """Manages the generation and organization of lightcurves from a given source."""

    def __init__(self, binsize, print_queue, exporter):
        self.binsize = binsize
        self.print_queue = print_queue
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

    def dispatch_observation_processing(self, source_directory: Path):
        """Start the processing of an observation directory, after validating the file structure
        seems correct."""
        if not (source_directory.exists() and source_directory.is_dir()):
            raise OSError(f"Source directory {source_directory} not found.")

        # Chandra soruces that end with X are TODO find out
        if source_directory.name.endswith("X"):
            return

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
        with futures.ThreadPoolExecutor(max_workers=6) as executor:
            workers = self.assign_workers(observation_directories, message_collection_queue)
            worker_futures = [executor.submit(worker.process) for worker in workers]
            done, not_done = futures.wait(worker_futures, return_when=futures.FIRST_EXCEPTION)
            message_collection_queue.put(None)
            transfer_thread.join()
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
            self.exporter.add_source(source_directory.name, results)

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
    def get_instrument_processor(event_list, source_region, message_collection_queue, binsize):
        """Determines which instrument on Chandra the observation was obtained through: ACIS
        (Advanced CCD Imaging Spectrometer) or HRC (High Resolution Camera)"""
        acis_pattern, hrc_pattern = r"^acis", r"^hrc"
        if re.match(acis_pattern, event_list.name) and re.match(acis_pattern, source_region.name):
            return AcisProcessor(event_list, source_region, message_collection_queue, binsize)
        if re.match(hrc_pattern, event_list.name) and re.match(hrc_pattern, source_region.name):
            return HrcProcessor(event_list, source_region, message_collection_queue, binsize)
        raise RuntimeError("Unable to resolve observation instrument")
