from raster import Raster
import shapefile as shp
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
"""
    **kwargs: res, year, title, bars, norm = None, save = False, error = True, colorbar = False
"""
def graph_fancy(data, ax, res, year, title, bars, norm = None, save = False, error = True, colorbar = False, scalebar = True):
    plt.rcParams['scalebar.location']="lower right"
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma']
    plt.rcParams['text.usetex'] = False
    #load shapefile
    sf = shp.Reader('./shape_files/mich_entidad.shp', encoding="ISO-8859-1")
    dims = data.shape
    ax = ax or plt.gca()
#     plt.subplots(dpi = 500)
    ax.set_ylim([dims[0], 0])
    cmap = ListedColormap(["white", "tan", "springgreen", "darkgreen"])

    ax.set_xlim([0, dims[1]])
    if norm is not None:
        im = ax.imshow(data, cmap = cmap,norm = colors.BoundaryNorm(norm, len(norm)), aspect='auto')

    else:
        im = ax.imshow(data, cmap = cmap, aspect='auto')

    if bars:
        ax.set_xticks(np.arange(-.5, dims[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, dims[0], 1), minor=True)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.grid(which='minor', color='b', linestyle='-', linewidth=.2)
    else:
        ax.axis('off')
    if scalebar:
        scalebar = ScaleBar(res) # 1 pixel = 0.2 meter
        ax.add_artist(scalebar)
    if title == "" and year != "":
        ax.set_title("{}".format(year), fontsize = 8)
    elif title != "":
        ax.set_title("{}, {}".format(title, year), fontsize = 8)

    for shape in sf.shapeRecords():
        # end index of each components of map
        l = shape.shape.parts

        len_l = len(l)  # how many parts of countries i.e. land and islands
        x = [i[0] for i in shape.shape.points[:]] # list of latitude
        y = [i[1] for i in shape.shape.points[:]] # list of longitude
        x = np.asarray(x)
        y = np.asarray(y)
        x = list(np.interp(x, (x.min(), x.max()), (0, dims[1])))
        y = list(np.interp(y, (y.min(), y.max()), (0, dims[0])))
        y = [dims[0]-i for i in y]
        l.append(len(x)) # ensure the closure of the last component
        for k in range(len_l):
            ax.plot(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'k-', linewidth=.4)
    divider = make_axes_locatable(ax)                          # set size of color bar
    if colorbar:
        cax = divider.append_axes("right", size="5%", pad=0.05)    # set size of color bar
        clb = plt.colorbar(im, cax=cax)
        clb.ax.tick_params(labelsize=20)
        if error:
            clb.ax.set_title(r'$\frac{\mathrm{pred}-\mathrm{act}}{10000}*100$', fontsize = 20)
            # clb.ax.tick_params(labelsize=6)
    if save:
        plt.savefig("./figs/{}_bilinear_error.png".format(year), bbox_inches = 'tight', dpi = 400)
    return ax

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)$', text) ]

def get_directories(dir):
    test = [''.join(x[0].split("/")[-1:]) for x in os.walk(os.path.join('test', dir))]
    directories = []
    digits = []
    for i in test:
        if i[0].isdigit():
            digits.append(i[0])
            directories.append(i)
    digits = np.unique(digits)
    sdirects = {}
    for i in digits:
        sdirects["{}km_2017".format(i)] = list(filter(lambda x: x[0] == str(i), directories))
    for i, x in sdirects.items():
        x.sort(key=natural_keys)
    return sdirects
