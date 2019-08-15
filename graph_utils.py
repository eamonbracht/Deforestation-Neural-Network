from raster import Raster
import shapefile as shp
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib_scalebar.scalebar import ScaleBar

def graph_fancy(data, res, year, title, bars):
    plt.rcParams['scalebar.location']="lower right"
    plt.rcParams['font.family'] = "sans-serif"
    #load shapefile
    sf = shp.Reader('../mich/mich_entidad.shp', encoding="ISO-8859-1")
    dims = data.shape
    fig, ax = plt.subplots(dpi = 500)
    ax.set_ylim([dims[0], 0])
    ax.set_xlim([0, dims[1]])
    im = ax.imshow(data, cmap=plt.get_cmap('Reds'))
    if bars:
        ax.set_xticks(np.arange(-.5, dims[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, dims[0], 1), minor=True)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.grid(which='minor', color='b', linestyle='-', linewidth=.2)
    else:
        ax.axis('off')
    fig.colorbar(im)
    scalebar = ScaleBar(res) # 1 pixel = 0.2 meter
    plt.gca().add_artist(scalebar)
    fig.suptitle("{}, {}".format(title, year))
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
            # draw each component of map.
            # l[k] to l[k + 1] is the range of points that make this component
            ax.plot(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'k-', linewidth=1)

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
        sdirects["{}km_2018".format(i)] = list(filter(lambda x: x[0] == str(i), directories))
    for i, x in sdirects.items():
        x.sort(key=natural_keys)
    return sdirects
