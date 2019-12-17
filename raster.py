import rasterio
import rasterio.plot
import pyproj
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from rasterio.plot import show
#from rasterio.plot import show_hist
from rasterio.windows import Window


class Raster:
    """Raster is used to manage importing, manipulating and visualizing raster files."""
    def __init__(self, filepath, coords):
        """
        Args:
            filepath (str): file path to location of .tif files
            coords (list): list of coordinates constraining the size of the raster to be read, beneficial for large rasters where memory is not large enough. 4 args (column_offset, row_offset, width, height).

        Note:
            Highly memory intensive to read large geographic rasters. Avoid importorting unclipped Hansen files.

        """
        self.filepath = filepath
        self.coords = coords
        #: bool
        self.is_Window = False
        if not coords:
            # rasterio object
            self.raster = rasterio.open(self.filepath)
        else:
            self.is_Window = True
            self.raster = Window(*self.coords)
        self.print_details()

    def read(self):
        """Convert raster to array.

        Note:
            Do not read in excessively large rasters, use a window. Reading in rasters requires memory equivalent to the size of the file, uncompressed rasters for this project can exceed 12 gb easily.

        Returns:
            ndarray equivalent of the rasterio or window.

        """
        if self.is_Window:
            with rasterio.open(self.filepath) as src:
                w = src.read(1, window = self.raster)
        else:
            w = self.raster.read(1)
        self.arr_raster = w
        return w

    def print_details(self):
        """Print details of raster file.

        Todo:
            * Return head of file, or sample of contents.

        """
        print(self.raster,"\r")
        if not self.is_Window:
            print("{}\nAttributes: {}".format(self.raster.name,self.raster.count))
        print("\nWidth: {}\nHeight: {}".format(self.raster.width, self.raster.height))

    def stats(self):
        """Prints summary statistics about the raster file.

        Note:
            0's are considered nulls by numpy, so they are not included.

        Todo
            * Fix issue with returning zeros.

        """
        shape = self.arr_raster.shape
        total_array = self.arr_raster.size
        total = np.argwhere(self.arr_raster > -1).shape[0]
        print(np.unique(self.arr_raster))
        print("Number of Elements: {}".format(total_array))
        print("Number of Elements of interest: {}".format(total))
        fmt = '{:<8}{:<20}{:<20}{}'

        unique = np.unique(self.arr_raster)
        print(fmt.format('', 'Element', 'Frequency', 'Percent'))
        for i, x in enumerate(unique):
            if(x != -1):
                Frequency = np.argwhere(self.arr_raster == x).shape[0]
                print(fmt.format(i, x, Frequency, round(100*(Frequency/total), 3)))

    def histogram(self):
        """Generates histogram showing the distribution of elements in the rasterio object. Does not work for numpy arrays.

        Yields: Histogram of distribution

        Note:
            Zero's are again not counted so they will not be represented in the histogram.

        """
        show_hist(self.raster, bins=50, lw=0.0, stacked=False, alpha=0.3,histtype='stepfilled', title="Histogram")

    def destructor(self):
        """Closes raster file.

        Note:
            Note necessary unless encountering memory errors. This should be handled by the python compilier.
        """
        self.raster.close()
