"""Mihir Patankar [mpatankar06@gmail.com]"""
import time
from pathlib import Path
from queue import Queue

import flask
from bs4 import BeautifulSoup
from pyvo.dal import TAPService

import search_config
from data_structures import DataProducts
from exporter import Exporter
from lightcurve_generator import LightcurveGenerator
from lightcurve_processing import AcisProcessor
from search import SourceManager


def start(html_index_file: Path):
    """Start the web server and set up routes."""
    app = flask.Flask(__name__)
    status_message_queue = Queue()

    @app.route("/", methods=["GET"])
    def index():
        with open(html_index_file, mode="r", encoding="utf-8") as file:
            html_content = file.read()
        html_soup = BeautifulSoup(html_content, "html.parser")
        html_soup.find("script").extract()
        new_script_tag = html_soup.new_tag("script")
        with open(Path("main.js"), mode="r", encoding="utf-8") as file:
            new_script_tag.string = file.read()
        html_soup.find("head").append(new_script_tag)
        return flask.render_template_string(str(html_soup))

    @app.route("/<path:filename>", methods=["GET"])
    def serve_image(filename):
        file_path = Path(filename)
        # "RECALCULATED" is an identifier in the request we are using to know if we should serve the
        # image in a relative or direct manner.
        if file_path.parts[0] == "RECALCULATED":
            file_path = Path(*file_path.parts[1:])
            return flask.send_file(file_path)
        return flask.send_from_directory(html_index_file.parent, file_path)

    def get_status_message():
        while True:
            status_message = status_message_queue.get()
            if status_message is None:
                break
            yield f"data: {status_message}\n\n"
            time.sleep(0.05)

    @app.route("/rebinning_status")
    def rebinning_status():
        return flask.Response(get_status_message(), content_type="text/event-stream")

    @app.route("/recalculate", methods=["POST"])
    def recalculate():
        config = search_config.get_config()
        data_directory = Path(config["Data Directory"])
        request_body = flask.request.get_json()
        source_name = request_body["sourceName"]
        observation_id = request_body["observationID"]

        status_message_queue.put("Downloading Source...")
        redownload_source(source_name, data_directory)

        status_message_queue.put("Reprocessing Observation...")
        processing_results = reprocess_source(
            data_directory / source_name / f"0{observation_id}_000",
            request_body["instrument"],
            float(request_body["newBinsize"]),
        )
        if not processing_results:
            status_message_queue.put("Invalid Binsize!")
            status_message_queue.put(None)
            flask.abort(400)

        status_message_queue.put("Exporting Observation...")
        exporter = Exporter(config)
        plot_directory = exporter.output_directory / "plots" / source_name
        plot_directory.mkdir(parents=True, exist_ok=True)
        path_to_plot_image = exporter.output_directory / Path(
            exporter.write_plot_image(
                plot_directory,
                observation_id,
                processing_results.plot_svg_data,
                processing_results.plot_csv_data,
            )
        )
        status_message_queue.put("Rebinning Complete")
        status_message_queue.put(None)
        return {
            "newData": [*exporter.combine_observation_data(processing_results).values()],
            "newPlotPath": str(Path("RECALCULATED").joinpath(path_to_plot_image)),
        }

    app.run(debug=True, use_reloader=False)


def redownload_source(source_name, download_directory):
    """Query CSC to download the source data products so they can be rebinned."""
    tap_service = TAPService("http://cda.cfa.harvard.edu/csc2tap")
    tap_query = f"""
    SELECT m.ra, m.dec
    FROM csc2.master_source m
    WHERE m.name = '{source_name.replace("2CXO", "2CXO ")}'
    """
    search_results = tap_service.search(tap_query)
    SourceManager.download_data_products(
        download_directory,
        str(search_results["ra"]).strip("[]"),
        str(search_results["dec"]).strip("[]"),
    )


def reprocess_source(observation_directory, instrument, new_binsize):
    """Generate a lightcurve with a different binsize."""
    data_products = DataProducts(*LightcurveGenerator.get_observation_files(observation_directory))
    if instrument == "ACIS":
        processor = AcisProcessor(data_products, new_binsize)
    region_event_list = processor.isolate_source_region(
        processor.event_list, processor.source_region
    )
    try:
        lightcurves = processor.extract_lightcurves(region_event_list, processor.binsize)
    except OSError:
        print("New binsize is invalid. Emitting status 400...")
        return None
    filtered_lightcurves = processor.filter_lightcurve_columns(lightcurves)
    lightcurve_data = processor.get_lightcurve_data(filtered_lightcurves)
    return processor.plot(lightcurve_data)


def main():
    """Main method for use when running the server seperately"""
    html_index_file = input(
        "Provide the absolute path of the html index file from the output you want to be hosted: "
    )
    start(Path(html_index_file))


if __name__ == "__main__":
    main()
