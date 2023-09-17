"""Mihir Patankar [mpatankar06@gmail.com]"""
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from contextlib import suppress
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import psutil
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

# pylint: disable-next=no-name-in-module
from ciao_contrib.runtool import search_csc
from pyvo import DALFormatError, dal
from tqdm import tqdm

from lightcurve_generator import LightcurveGenerator
from data_structures import Message
from exporter import Exporter


class SourceManager:
    """Handles searching, downloading, and processing of sources. Also manages threading for doing
    tasks in parallel and logging."""

    def __init__(self, config):
        self.config = config
        self.sources = []
        self.downloaded_source_queue: Queue[Path] = None
        self.download_message_queue, self.process_message_queue = Queue(), Queue()
        self.sources_downloaded, self.sources_processed = 0, 0

    def search_csc(self):
        """Queries the Chandra Source Catalog for sources matching the search criteria."""
        search_radius = self.config["Search Radius (arcmin)"] * units.arcmin
        object_name = self.config["Object Name"]
        significance_threshold = self.config["Significance Threshold"]
        try:
            sky_coord = SkyCoord.from_name(object_name)
            cone_search = dal.SCSService("http://cda.cfa.harvard.edu/csc2scs/coneSearch")
            print("Searching CSC sources...")
            start_time = time.perf_counter()
            search_results = cone_search.search(sky_coord, search_radius, verbosity=2)
        except NameResolveError:
            print(f'No results for object "{object_name}".')
            sys.exit(1)
        except DALFormatError:
            print("Failed to connect to search service, check internet connection?")
            sys.exit(1)
        print(f"Found {len(search_results)} sources in {(time.perf_counter() - start_time):.3f}s.")
        significant_results = [
            result for result in search_results if result["significance"] >= significance_threshold
        ]
        significant_results_count = len(significant_results)
        print(f"Found {significant_results_count} sources meeting the significance threshold.")
        if significant_results_count == 0:
            sys.exit(0)
        self.downloaded_source_queue = Queue(significant_results_count)
        self.sources = significant_results

    @staticmethod
    def download_data_products(download_directory, right_ascension, declination):
        """Uses a ciao tool to download certain data products from csc2 for a given source using
        the obtained celestial sphere coords for a source name. Currently we are only using the
        event file and the source region map."""
        search_csc(
            pos=f"{right_ascension}, {declination}",
            radius="1.0",
            outfile="search-csc-outfile.tsv",
            radunit="arcsec",
            columns="",
            sensitivity="no",
            download="all",
            root=download_directory,
            bands="broad, wide",
            filetypes="regevt, reg",
            catalog="csc2",
            verbose="0",
            clobber="1",
        )

    def print_data_product_progress(self, source_directory: Path, finished_downloading_source):
        """Checks and reports how many files have been downloaded."""
        complete = False
        last_progress_message = ""
        message_uuid = uuid.uuid4()
        download_start_time = time.perf_counter()
        while True:
            observation_count = len(tuple(source_directory.glob("*")))
            data_products_count = len(tuple(source_directory.rglob("**/*.gz")))
            message = (
                f"Retrieved {observation_count} observations, "
                f"{data_products_count} data products. "
                f"({(time.perf_counter() - download_start_time):.2f}s)"
            )
            if message != last_progress_message:
                self.download_message_queue.put(Message(message, message_uuid))
            last_progress_message = message
            if finished_downloading_source.is_set() and not complete:
                complete = True  # This flag allows the loop to run a final time before finishing.
                continue
            if complete:
                break
            time.sleep(0.2)

    def download_all_data_products(self):
        """Goes through all CSC results, finds their coordinates and requests their data products.
        Spins up another thread as a child to this one to handle counting and reporting the number
        of data products downloaded in the file system."""
        data_directory = Path(self.config["Data Directory"])
        shutil.rmtree(data_directory, ignore_errors=True)
        source_count = 0
        for source_count, source in enumerate(self.sources, 1):
            finished_downloading_source = Event()
            source_directory = data_directory / source["name"].replace(" ", "")
            progress = f"{(source_count)}/{len(self.sources)}"
            self.download_message_queue.put(
                Message(f"Downloading source {source['name']}... {progress}")
            )
            progress_thread = Thread(
                target=self.print_data_product_progress,
                args=(source_directory, finished_downloading_source),
            )
            progress_thread.start()
            self.download_data_products(data_directory, source["ra"], source["dec"])
            finished_downloading_source.set()
            self.sources_downloaded += 1
            progress_thread.join()
            self.downloaded_source_queue.put(source_directory)
        self.downloaded_source_queue.put(None)
        self.download_message_queue.put(None)

    def process(self, process_error_event):
        """Setup and use lightcurve generator class."""

        def increment_sources_processed():
            self.sources_processed += 1

        lightcurve_generator = LightcurveGenerator(
            self.config["Binsize"],
            self.process_message_queue,
            increment_sources_processed,
            Exporter(self.config, len(self.sources)),
        )
        try:
            lightcurve_generator.dispatch_source_processing(self.downloaded_source_queue)
        # This still satisfies PEP8, since it's outputting the error and terminating ASAP.
        # pylint: disable-next=broad-except
        except Exception:
            exception = sys.exc_info()[1]
            exception_traceback = traceback.format_exc().replace("\n", " ")
            self.process_message_queue.put(Message(f"{exception} - {exception_traceback}"))
            self.process_message_queue.put(None)
            self.download_message_queue.put(None)
            process_error_event.set()
            raise
        self.process_message_queue.put(None)
        lightcurve_generator.exporter.export()

    def download_and_process(self):
        """Manages the threads and queues that do the downloading, processing, and terminal output.
        Holds the instance to the output thread class and dependency injects into it. This method
        blocks the program until all threads and queues have finished, at which point the program
        is pretty much done."""
        outputting = self.config["Enable Output"]
        process_error_event = Event()
        # Stop default exception printing behavior
        if outputting:
            threading.excepthook = lambda exception: exception
        download_thread = Thread(target=self.download_all_data_products, daemon=True)
        process_thread = Thread(target=self.process, args=(process_error_event,), daemon=True)
        download_thread.start()
        process_thread.start()
        output_thread = (
            OutputThread(self) if outputting else Thread(target=lambda: None, daemon=True)
        )
        output_thread.start()
        process_thread.join()
        if process_error_event.is_set():
            output_thread.join()
            sys.exit(1)
        download_thread.join()
        if outputting:
            self.download_message_queue.join()
            self.process_message_queue.join()


class Terminal:
    """Static class that handles terminal interaction and state. This class is not meant to be
    instantiated."""

    N_PROGRESS_BAR_ROWS = 3
    current_row_left, current_row_right = 0, 0
    # Immutable stores of the most recent data handed to the terminal to be written.
    left_column_backup, right_column_backup = (), ()

    @staticmethod
    def width():
        """Returns latest terminal width."""
        return shutil.get_terminal_size().columns

    @staticmethod
    def height():
        """Returns latest terminal height."""
        return shutil.get_terminal_size().lines

    @staticmethod
    def clear():
        """Clears the visible terminal with ANSI codes. Data should be buffered somewhere to avoid
        it being lost."""
        lines = Terminal.height()
        for _ in range(lines):
            print("\033[2K\033[1A", end="")

    @classmethod
    def set_scroll(cls, left_column_count, right_column_count):
        """When the terminal window runs out of space to display all messages in either column,
        moves the current row so on the next write call only the latest messages will be written,
        thus creating a scrolling behavior."""
        max_rows = cls.height() - cls.N_PROGRESS_BAR_ROWS
        max_scroll_left = max(left_column_count, max_rows)
        max_scroll_right = max(right_column_count, max_rows)
        if cls.current_row_left < max_scroll_left - max_rows:
            cls.current_row_left += 1
        if cls.current_row_right < max_scroll_right - max_rows:
            cls.current_row_right += 1

    @classmethod
    def write_columns(cls, left_column, right_column, *args):
        """Prints in a two column format, adjusting for escape sequences, with progress bars at the
        bottom. Prints the latest messages from the buffer that can fit on the screen. Backups
        buffer every time so the backup is up-to-date for logging."""
        max_rows = cls.height() - cls.N_PROGRESS_BAR_ROWS
        cls.left_column_backup, cls.right_column_backup = tuple(left_column), tuple(right_column)
        for left_message, right_message in zip_longest(
            left_column[cls.current_row_left : cls.current_row_left + max_rows],
            right_column[cls.current_row_right : cls.current_row_right + max_rows],
            fillvalue=Message(""),
        ):
            column_width = cls.width() // 2
            left_message = left_message.content.ljust(column_width)
            right_message = right_message.content.ljust(column_width)
            # Truncate message if exceeding terminal width, note this doesn't happen in the log.
            if len(left_message) > column_width:
                left_message = left_message[: column_width - 3] + "..."
            if len(right_message) > column_width:
                right_message = right_message[: column_width - 3] + "..."
            print(f"{left_message}{right_message}", end="\r\n")

        for progress_bar in range(cls.N_PROGRESS_BAR_ROWS):
            # Make a new line if not the last row.
            new_line = "\n" if not progress_bar == cls.N_PROGRESS_BAR_ROWS - 1 else ""
            print(str(args[progress_bar]), end=f"\r{new_line}")

    @classmethod
    def dump_to_log(cls, log_directory: Path):
        """Dump the latest backup of column data (updated each write cycle) to a specified log
        file. This is called on program crash in case of any interruptions. The difference between
        this and the terminal writing is it performs no truncation or fitting."""
        log_file_path = log_directory / f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
        with open(log_file_path, mode="w", encoding="utf-8") as file:
            if max((cls.left_column_backup, cls.right_column_backup), key=len) == 0:
                file.write("No output captured.")
            for left_message, right_message in zip_longest(
                cls.left_column_backup, cls.right_column_backup, fillvalue=Message("")
            ):
                spacing = " " * 10
                file.write(f"{left_message.content}{spacing}{right_message.content}\r\n")
        return log_file_path


class OutputThread(Thread):
    """Displays messages from downloading and processing, as well as progress bars."""

    def __init__(self, source_manager):
        Thread.__init__(self, daemon=True)
        self.source_manager = source_manager
        self.done_downloading, self.done_processing = False, False
        self.queues_complete = False
        self.left_column, self.right_column = [Message("- Download -")], [Message("- Process -")]
        self.progress_bars = {}

    def run(self):
        print("...")
        print("\n" * Terminal.height(), end="")  # Make space for output.
        self.init_progress_bars()
        while True:
            if self.queues_complete:
                break
            if not self.check_and_handle_messages():
                time.sleep(0.05)  # Release GIL for a moment
                continue
            self.update_progress_bar_extras()
            Terminal.clear()
            Terminal.set_scroll(len(self.left_column), len(self.right_column))
            Terminal.write_columns(
                self.left_column, self.right_column, *self.progress_bars.values()
            )

    def check_and_handle_messages(self):
        """Polls queues, if there is a new message it is added to the message buffer so it can be
        printed to the console on the next write. Updates progress bar progress. Returns 0 if there
        are no new messages, returns 1 if there are; this way the loop knows when to skip."""
        download_message, process_message = Message(""), Message("")
        with suppress(Empty):
            download_message = self.source_manager.download_message_queue.get_nowait()
        with suppress(Empty):
            process_message = self.source_manager.process_message_queue.get_nowait()

        download_message_present = download_message != Message("")
        process_message_present = process_message != Message("")

        if not download_message_present and not process_message_present:
            return 0  # No new messages recieved to print

        if download_message is None:
            self.done_downloading = True
        elif download_message_present:
            self.update_column(download_message, self.left_column)
            self.source_manager.download_message_queue.task_done()

        if process_message is None:
            self.done_processing = True
        elif process_message_present:
            self.update_column(process_message, self.right_column)
            self.source_manager.process_message_queue.task_done()

        if self.done_downloading and self.done_processing:
            self.source_manager.download_message_queue.task_done()
            self.source_manager.process_message_queue.task_done()
            self.queues_complete = True
            return 0
        self.progress_bars["download"].n = self.source_manager.sources_downloaded
        self.progress_bars["process"].n = self.source_manager.sources_processed
        self.progress_bars["total"].n = (
            self.source_manager.sources_downloaded + self.source_manager.sources_processed
        ) / 2

        return 1

    @staticmethod
    def update_column(incoming_message, column):
        """Either appends a new message to a column, or replaces one if its UUID matches an
        exisiting one. NOTE passing in a novel UUID labeled message will traverse the entire
        column list, this could cause performance issues for large columns."""
        if incoming_uuid := incoming_message.uuid:
            for index, element in enumerate(reversed(column), 1):
                if element.uuid == incoming_uuid:
                    column[-index] = incoming_message
                    return
        column.append(incoming_message)

    def init_progress_bars(self):
        """Creates progress bar objects, and assigns initial properties."""
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            self.progress_bars = {
                "download": tqdm(
                    total=len(self.source_manager.sources),
                    desc="Downloaded Sources:",
                    file=devnull,
                    colour="blue",
                    ncols=Terminal.width(),
                    unit="source",
                    bar_format="{desc} {percentage:3.0f}%|{bar}| "
                    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}    ",
                ),
                "process": tqdm(
                    total=len(self.source_manager.sources),
                    desc="Processed Sources:",
                    file=devnull,
                    colour="green",
                    ncols=Terminal.width(),
                    bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}    ",
                ),
                "total": tqdm(
                    total=len(self.source_manager.sources),
                    desc="Total Progress:",
                    file=devnull,
                    colour="black",
                    ncols=Terminal.width(),
                    bar_format="{desc} {percentage:3.0f}%|{bar}| {postfix}",
                ),
            }
            # Adjust spacing so bars are aligned
            longest_description_length = max(
                len(progress_bar.desc) for progress_bar in self.progress_bars.values()
            )
            for progress_bar in self.progress_bars.values():
                progress_bar.desc = progress_bar.desc.ljust(longest_description_length)

    def update_progress_bar_extras(self):
        """Update dynamic parts of the progress bar other than the progress."""
        for progress_bar in self.progress_bars.values():
            progress_bar.ncols = Terminal.width()

        if self.done_downloading:
            self.progress_bars["process"].bar_format = (
                "{desc} {percentage:3.0f}%|{bar}| "
                + "{n_fmt}/{total_fmt} [{elapsed}<{remaining}    "
            )

        self.progress_bars["total"].postfix = (
            f"\b\b{threading.active_count()} threads, "
            f"RAM used: {psutil.Process().memory_info().rss / 1e6:.2f} MB. PID: {os.getpid()}"
        )


def print_log_location():
    """Call a log dump, and print the file path of the new log file."""
    log_directory = Path("./logs")
    log_directory.mkdir(exist_ok=True)
    print(f"\nLog saved to {Terminal.dump_to_log(log_directory)}.")
