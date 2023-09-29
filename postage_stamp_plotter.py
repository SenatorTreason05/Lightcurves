"""Mihir Patankar [mpatankar06@gmail.com]"""
from io import BytesIO

import matplotlib
import numpy
from astropy import io, table
from matplotlib import pyplot


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
    """Plots the binned sky and detector images which are read as NumPy arrays by astropy. The
    detector image is done in a square root scale to give the image more contrast. This requires
    the FITS images to have a 'BOUNDS' extension where min and max limits are stored in a binary
    table (BINTABLE). The PNG data is returned in a BytesIO object."""
    matplotlib.use("agg")
    sky_image_data = io.fits.getdata(sky_image, ext=0)
    detector_image_data = numpy.sqrt(io.fits.getdata(detector_image, ext=0))
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
        sky_bounds = table.Table.read(hdu_list["BOUNDS"])
    sky_y_ticks = get_real_ticks_from_real_bounds(
        (round(float(sky_bounds["y_min"][0])), round(float(sky_bounds["y_max"][0]))),
        sky_image_plot.get_ylim(),
    )
    sky_image_plot.set_yticks(*sky_y_ticks)
    sky_x_ticks = get_real_ticks_from_real_bounds(
        (round(float(sky_bounds["x_min"][0])), round(float(sky_bounds["x_max"][0]))),
        sky_image_plot.get_xlim(),
    )
    sky_image_plot.set_xticks(*sky_x_ticks, rotation=90)
    detector_image_plot.set_title("Detector-Coordinates Postage Stamp (sqrt Scale)", y=1.05)
    detector_image_plot.set_xlabel("Detector-Coordinates X Pixel")
    detector_image_plot.set_ylabel("Detector-Coordinates Y Pixel")
    detector_image_plot.grid(True, linestyle="-", color="gray")
    detector_image_plot.imshow(
        detector_image_data,
        cmap="gray",
        extent=[0, detector_image_data.shape[1], 0, detector_image_data.shape[0]],
    )
    with io.fits.open(detector_image) as hdu_list:
        detector_bounds = table.Table.read(hdu_list["BOUNDS"])
    detector_y_ticks = get_real_ticks_from_real_bounds(
        (round(float(detector_bounds["y_min"][0])), round(float(detector_bounds["y_max"][0]))),
        detector_image_plot.get_ylim(),
    )
    detector_image_plot.set_yticks(*detector_y_ticks)
    detector_x_ticks = get_real_ticks_from_real_bounds(
        (round(float(detector_bounds["x_min"][0])), round(float(detector_bounds["x_max"][0]))),
        detector_image_plot.get_xlim(),
    )
    detector_image_plot.set_xticks(*detector_x_ticks, rotation=90)
    pyplot.savefig(png_data := BytesIO(), bbox_inches="tight")
    pyplot.close(figure)
    return png_data
