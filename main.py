"""Mihir Patankar [mpatankar06@gmail.com]"""
import atexit
import subprocess
import sys
import time

import search
import search_config


def main():
    """Entry point for the program."""
    subprocess.run(["clear"], check=False)
    source_manager = search.SourceManager(search_config.get_config())
    source_manager.search_csc()
    if input("Proceed? (y/n) ") != "y":
        sys.exit(0)
    start_time = time.perf_counter()
    atexit.register(search.print_log_location)
    source_manager.download_and_process()
    print(f"\nAll sources finished in {time.perf_counter() - start_time:.3f}s")


if __name__ == "__main__":
    print("Starting program.")
    main()
