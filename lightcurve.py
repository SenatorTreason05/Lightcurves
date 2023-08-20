"""Mihir Patankar [mpatankar06@gmail.com]"""
import gzip
from pathlib import Path


class LightcurveGenerator:
    def __init__(self, output_directory, binsize):
        self.output_directory = output_directory
        self.binsize = binsize
        self.error_queue = None

    @staticmethod
    def unzip_fits_files(observation_path: Path):
        """Files are downloaded in a GNU Zip format. This unzips all files in an observation
        directory into the FITS files we need."""
        gzip_files = observation_path.glob("*.gz")
        observation_id = observation_path.name
        try:
            for gzip_file in gzip_files:
                with gzip.open(gzip_file, "rb") as gzipped_file:
                    unzipped_data = gzipped_file.read()
                    with open(gzip_file, "wb") as unzipped_file:
                        unzipped_file.write(unzipped_data)
        except OSError as error:
            raise OSError(f"Could not unzip {observation_id}! Details: {error.args}") from error

    @staticmethod
    def adjust_binsize(binsize, time_resolution):
        """For ACIS, time resolution can be in the seconds in timed exposure mode, as compared to in the microseconds for HRC."""
        # TODO https://cxc.cfa.harvard.edu/ciao/ahelp/times.html find out the difference betweeen interleaved and standard TIME mode
        return binsize // time_resolution * time_resolution

    def process_observation(self, path_to_observation):
        observation_path = Path(path_to_observation)
        if not (observation_path.exists() and observation_path.is_dir()):
            raise OSError(f"Observation directory {path_to_observation} not found.")
