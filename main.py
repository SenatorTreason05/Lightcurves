"""Mihir Patankar [mpatankar06@gmail.com]"""
import atexit
import os
import sys

import search
import search_config


def main():
    """Entry point for the program."""
    os.system("clear")
    source_manager = search.SourceManager(search_config.get_config())
    source_manager.search_csc()
    if input("Proceed? (y/n) ") != "y":
        sys.exit(0)
    atexit.register(search.print_log_location)
    source_manager.download_and_process()


if __name__ == "__main__":
    print("Starting program.")
    main()
