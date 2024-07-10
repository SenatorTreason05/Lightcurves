"""Mihir Patankar [mpatankar06@gmail.com"""
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from data_structures import LightcurveParseResults, ExportableObservationData


class Exporter:
    """Exports source and observation data to output documents."""

    def __init__(self, config, source_count=0):
        self.config = config
        self.source_count = source_count
        self.output_directory = (
            Path(config["Output Directory"])
            / f"{config['Object Name']}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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
            plot_directory = self.output_directory / "plots" / source_name
            plot_directory.mkdir(parents=True, exist_ok=True)
            observation_id = observation.observation_header_info.observation_id
            combined_observation_data = self.combine_observation_data(observation)
            source_data.append(
                ExportableObservationData(
                    columns={
                        self.format_table_header(key): value
                        for key, value in combined_observation_data.items()
                    },
                    plot_image_path=self.write_plot_image(
                        plot_directory,
                        observation_id,
                        observation.plot_svg_data,
                        observation.plot_csv_data,
                    ),
                    postage_stamp_image_path=self.write_postage_stamp(
                        plot_directory, observation_id, observation.postagestamp_png_data
                    ),
                )
            )
        self.master_data[source_name] = source_data

    @staticmethod
    def combine_observation_data(observation: LightcurveParseResults):
        """Amalgamate all observation data that we want to display."""
        return {
            **observation.observation_header_info._asdict(),
            **observation.observation_data._asdict(),
        }

    def write_plot_image(self, plot_directory, observation_id, svg_data, csv_data):
        """Write the plot svg image to disk and remove it from memory."""
        plot_image_path = plot_directory / f"{observation_id}.svg"
        plot_data_path = plot_directory / f"{observation_id}.csv"
        with open(plot_image_path, mode="w+", encoding="utf-8") as file:
            file.write(svg_data.getvalue())
            svg_data.close()
        with open(plot_data_path, mode="w+", encoding="utf-8") as file:
            file.write(csv_data.getvalue())
            csv_data.close()
        return f"./{plot_image_path.relative_to(self.output_directory)}"

    def write_postage_stamp(self, plot_directory, observation_id, png_data):
        """Write the postage stamp png image to disk and remove it from memory."""
        if not png_data:
            return ""
        file_path = plot_directory / f"{observation_id}.png"
        with open(file_path, mode="wb+") as file:
            file.write(png_data.getvalue())
            png_data.close()
        return f"./{file_path.relative_to(self.output_directory)}"

    @staticmethod
    def format_table_header(key):
        """Capitalizes the table header to make it look more presentable."""
        words = key.split("_")
        return " ".join([word.capitalize() for word in words])

    def export(self):
        """Write all master data contents to an HTML file."""
        with open(
            output := self.output_directory / "index.html", mode="a", encoding="utf-8"
        ) as file:
            environment = Environment(loader=FileSystemLoader("./"))
            template = environment.get_template(("Lightcurves/output_template.jinja"))
            content = template.render(
                source_count=self.source_count,
                object_name=self.config["Object Name"],
                search_radius=self.config["Search Radius (arcmin)"],
                significance_threshold=self.config["Significance Threshold"],
                counts_threshold=self.config["Minimum Counts"],
                zoom=min(1, self.config["Binsize"]/500),
                master_data=self.master_data,
            )
            file.write(content)
        return output
