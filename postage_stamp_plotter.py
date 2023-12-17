"""Mihir Patankar [mpatankar06@gmail.com]"""
from dataclasses import dataclass, fields
from io import BytesIO

import matplotlib
import numpy
from astropy import io, table
from matplotlib import pyplot

ACIS_PIXELS_PER_ARCSECOND = 1 / 0.492


def get_real_ticks_from_real_bounds(real_bounds, image_bounds, tick_interval=5):
    """We cannot use matplotlib.ticker since image pixels and physical pixels have a proportional
    relationship, ticker requires a linear relationship. Real vs image bounds refers to the bounds
    of the image in instrument pixels from Chandra vs screen pixels in which the image data is
    stored. We want our ticks to represent instrument pixels as that information is more useful.
    This returns a tuple of the positions on the image axis in image pixels along with their new
    labels which are now in real pixels."""
    start_tick, end_tick = real_bounds
    real_tick_positions = numpy.arange(
        start_tick + (tick_interval - start_tick % tick_interval),
        (end_tick - end_tick % tick_interval) + 1,
        tick_interval,
    )
    real_tick_positions = numpy.insert(real_tick_positions, 0, start_tick)
    real_tick_positions = numpy.append(real_tick_positions, end_tick)
    image_tick_positions = tuple(
        numpy.interp(tick, real_bounds, image_bounds) for tick in real_tick_positions
    )
    real_tick_labels = tuple(str(tick) for tick in real_tick_positions)
    return image_tick_positions, real_tick_labels


def plot_postagestamps(sky_image, detector_image):
    """Plots the binned sky and detector images which are read as NumPy arrays by astropy. Requires
    the FITS images to have a 'BOUNDS' extension where min and max limits are stored in a binary
    table (BINTABLE). The images are on a square root scale and inverted for visibility purposes.
    The PNG data is returned in a BytesIO object."""
    if sky_image is None or detector_image is None:
        return None
    matplotlib.use("agg")
    sky_image_data = 255 - numpy.sqrt(io.fits.getdata(sky_image, ext=0))
    detector_image_data = 255 - numpy.sqrt(io.fits.getdata(detector_image, ext=0))
    figure, (sky_image_plot, detector_image_plot) = pyplot.subplots(nrows=1, ncols=2)
    figure.set_figwidth(10)

    sky_image_plot.set_title("Sky-Coordinates Postage Stamp", y=1.05)
    sky_image_plot.set_xlabel("Sky-Coordinates X Pixel")
    sky_image_plot.set_ylabel("Sky-Coordinates Y Pixel")
    sky_image_plot.grid(True, linestyle="-", color="gray")
    sky_image_plot.imshow(
        sky_image_data,
        cmap="gray",
        extent=[0, sky_image_data.shape[1], 0, sky_image_data.shape[0]],
    )
    with io.fits.open(sky_image) as hdu_list:
        sky_bounds = CropBounds.from_hdu_list(hdu_list["BOUNDS"])
    sky_y_ticks = get_real_ticks_from_real_bounds(
        (round(sky_bounds.y_min), round(sky_bounds.y_max)),
        sky_image_plot.get_ylim(),
    )
    sky_image_plot.set_yticks(*sky_y_ticks)
    sky_x_ticks = get_real_ticks_from_real_bounds(
        (round(sky_bounds.x_min), round(sky_bounds.x_max)),
        sky_image_plot.get_xlim(),
    )
    sky_image_plot.set_xticks(*sky_x_ticks, rotation=90)

    scale_x_anchor = sky_image_data.shape[1] - 1
    scale_y_anchor = 1
    sky_image_plot.plot(
        [scale_x_anchor, scale_x_anchor + ACIS_PIXELS_PER_ARCSECOND],
        [scale_y_anchor, scale_y_anchor],
        color="red",
        linewidth=2,
    )
    sky_image_plot.text(
        scale_x_anchor, scale_y_anchor, "1 arcsec  ", color="red", ha="right", va="center"
    )

    detector_image_plot.set_title("Detector-Coordinates Postage Stamp", y=1.05)
    detector_image_plot.set_xlabel("Detector-Coordinates X Pixel")
    detector_image_plot.set_ylabel("Detector-Coordinates Y Pixel")
    detector_image_plot.grid(True, linestyle="-", color="gray")
    detector_image_plot.imshow(
        detector_image_data,
        cmap="gray",
        extent=[0, detector_image_data.shape[1], 0, detector_image_data.shape[0]],
    )
    with io.fits.open(detector_image) as hdu_list:
        detector_bounds = CropBounds.from_hdu_list(hdu_list["BOUNDS"])
    detector_y_ticks = get_real_ticks_from_real_bounds(
        (round(detector_bounds.y_min), round(detector_bounds.y_max)),
        detector_image_plot.get_ylim(),
    )
    detector_image_plot.set_yticks(*detector_y_ticks)
    detector_x_ticks = get_real_ticks_from_real_bounds(
        (round(detector_bounds.x_min), round(detector_bounds.x_max)),
        detector_image_plot.get_xlim(),
    )
    detector_image_plot.set_xticks(*detector_x_ticks, rotation=90)
    pyplot.savefig(png_data := BytesIO(), bbox_inches="tight")
    pyplot.close(figure)
    return png_data


@dataclass
class CropBounds:
    """Holds the bounding box of our postage stamp images."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def double(self):
        """Double the bounds, useful for including background in the image."""
        self.add_padding(self.x_max - self.x_min, self.y_max - self.y_min)

    def add_padding(self, x_padding, y_padding):
        """Add padding to the image in pixels, for extending it past tight bounds."""
        self.x_min -= x_padding
        self.x_max += x_padding
        self.y_min -= y_padding
        self.y_max += y_padding

    def to_hdu(self):
        """Convert the dataclass to a Header Data Unit for FITS files."""
        hdu = io.fits.table_to_hdu(
            table.Table({field.name: [(getattr(self, field.name))] for field in fields(self)})
        )
        hdu.header["EXTNAME"] = "BOUNDS"
        return hdu

    @classmethod
    def from_strings(cls, x_min, y_min, x_max, y_max):
        """Casts a set of string values to floats."""
        return cls(float(x_min), float(y_min), float(x_max), float(y_max))

    @classmethod
    def from_hdu_list(cls, hdu_list):
        """Parse our BOUNDS extension HDU list from an image FITS file."""
        bounds_table = table.Table.read(hdu_list)
        return cls(
            float(bounds_table["x_min"][0]),
            float(bounds_table["y_min"][0]),
            float(bounds_table["x_max"][0]),
            float(bounds_table["y_max"][0]),
        )
