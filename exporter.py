"""Mihir Patankar [mpatankar06@gmail.com"""
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from data_structures import LightcurveParseResults, ExportableObservationData


class Exporter:
    """Exports source and observation data to output documents."""

    def __init__(self, config, source_count):
        self.config = config
        self.source_count = source_count
        self.output_directory = (
            Path(config["Output Directory"]) / f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.master_data = {}

    def add_source(self, source_name, observations: list[LightcurveParseResults]):
        """Append a source and all its data to the master store. Render the plots here so they can
        be released from memory for the remainder of the program."""
        source_data = []
        # Put observations in chronological order.
        sorted_observations = sorted(
            observations, key=lambda observation: observation.observation_data.raw_start_time
        )
        for observation in sorted_observations:
            observation_id = observation.observation_header_info.observation_id
            combined_observation_data = {
                **observation.observation_header_info._asdict(),
                **observation.observation_data._asdict(),
            }
            source_data.append(
                ExportableObservationData(
                    columns={
                        self.format_table_header(key): value
                        for key, value in combined_observation_data.items()
                    },
                    image_path=self.write_svg(source_name, observation_id, observation.svg_data),
                )
            )
        self.master_data[source_name] = source_data

    def write_svg(self, source_name, observation_id, svg_data):
        """Write the plot image to disk and remove it from memory."""
        svg_directory: Path = self.output_directory / "plots" / source_name
        svg_directory.mkdir(parents=True, exist_ok=True)
        file_path = svg_directory / f"{observation_id}.svg"
        with open(file_path, mode="w+", encoding="utf-8") as file:
            file.write(svg_data.getvalue())
            svg_data.close()
        return f"./{file_path.relative_to(self.output_directory)}"

    @staticmethod
    def format_table_header(key):
        """Capitalizes the table header to make it look more presentable."""
        words = key.split("_")
        return " ".join([word.capitalize() for word in words])

    def export(self):
        """Write all master data contents to an HTML file."""
        with open(self.output_directory / "index.html", mode="a", encoding="utf-8") as file:
            environment = Environment(loader=FileSystemLoader("./"))
            template = environment.get_template(str("/output_template.jinja"))
            content = template.render(
                source_count=self.source_count,
                object_name=self.config["Object Name"],
                search_radius=self.config["Search Radius (arcmin)"],
                significance_threshold=self.config["Significance Threshold"],
                master_data=self.master_data,
            )
            file.write(content)
