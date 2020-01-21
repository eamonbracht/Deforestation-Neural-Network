import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from utils import *
import os, sys
import torch
import json
from PIL import Image
import math
import re


# 0 forested, -1 = deforested
def expand_years(raster, name = "temp", culm = False, save=False):
    """Takes in array of deforestation data and parses it into an set of arrays
    stored in dictionary for the yearly deofrestation progression for each
    pixel then pickles and returns that dictionary. Array contains 0 for no loss
    or 1-n corresponding to the principle year of deforestation

    Args:
        raster (raster obj): raster from raster class containing the
            rasterized data.
        name (str): name of file to be saved.

    Returns:
        dictionary with the status of each year, labeled with a eponymous str
        for that year.

    """
    shape = raster.shape
    print(shape)
    raster_mask = ma.getmask(raster)
    num_years = 17
    years = ma.zeros((num_years, *shape))
    print(years.shape)
    for year in range(1, num_years+1):
        for repl in np.argwhere(raster ==  year):
            years[year-1, repl[0], repl[1]] = 1
        years[year-1] = ma.masked_where(raster_mask, years[year-1])
        years[year-1] = years[year-1].filled(np.nan)

    return years


def load_data(name):
    """Unpickles dictionary containing year-over-year defrestation data.

    Note:
        Pickle must be stored in the data folder.

    Args:
        Name of the .pickle file to import.

    Returns:
        Dictionary containing data in pickled file.

    """
    print("Importing", name, '\r')
    pickleFile = open('data/'+name+'.pickle', 'rb')
    print("Success")
    return(pickle.load(pickleFile))

def print_validation(data):
    """Helper method used for debugging in make_dict function. For each array
    craeted, There should only be a 0 and 1's.

    """
    for i, x in data.items():
        print(i, np.unique(x))

def save_images(data, prefix):
    """Converts an array with T time steps into T images and saves them.

    Args:
        data (array): Deforestation state for an area over a set of
            years. T x m x n
        prefix (str): Unique identifier for saving images to.

    Yields:
        Set of T images capturing each years deforestation state.

    """
    num_years = data.shape[0]
    years = range(2000, 2000+num_years)
    yearslabels = [str(year) for year in years]
    for i, x in zip(yearlabels,data):
        name = 'animation/'+prefix+i+'.png'
        t = plt.figure(figsize(500, 500))
        plt.imshow(x)
        plt.title(i)
        # t.set_cmap("Blues_r")
        plt.savefig(name, bbox_inches = 'tight')
#   TODO: get imagemagick to create gif
#    os.system("cd animation")
#    os.system("convert -delay 10 'frame%d.png[2000-n]' output_full.gif")

def dictionary_to_array(data, reshape, n_years):
    """Takes in dictionary and converts to array.

    Args:
        data (dict): Dictionary contains T m x n matricies where T is the
            number of time steps and m and n are the dimensions of input matrix.
        reshape (bool): Flag to indicate if data needs to be flattened.

    Returns:
        T x X array where X is the number of time steps.

    Note:
        Data must be flattened or linearized so that all data can be captured
        in a 2D array.

    """
    print("Converting dictionary to array")
    tensor = np.stack(data.values(), 0)
    if reshape:
        print("Flattening")
        tensor = tensor.reshape(n_years,-1).T
    print("Array success", tensor.shape)
    return tensor

def save_array_to_csv(data, name):
    """Saves array to dictionary.

    Args:
        data (array): 2D array to be saved.
        name (str): Str specifying the name of the file.

    Yields:
        CSV saved to the data folder with format:
            defor<name>.csv

    """
    np.savetxt("data/defor"+name+".csv", data, delimiter = ",")

def delete_exclude(cords, exclude, data):
    del_val = []
    count = 0
    for pos, i in enumerate(cords[:]):
        if np.any(np.isin(i, exclude[:])):
            del_val.append(pos)
            count+=1
    mod_cords = np.delete(cords, del_val, axis = 0)
    data[mod_cords[:, 0], mod_cords[:, 1]] = 1
    return data

def make_relation(type_rel, shape, exclude, save, combine):
    """Function to create binary relationship tensor for data based on
    cardinality. Creates north, south, east, west, northwest, northeast,
    southwest and southeast realtions.

    Note:
        For an m x n area, the output matrix will have shape X x X where
        X = m*n. Example for north direction:
            Let X have dimensions i, j. i = 1 iff j is north of i.

        Matrix is very sparse.

        Returns tensor not array because pytorch has built in normalization
        that succeeds this step.

    Args:
        type_rel (str): Denoting which type of relation to make.
            TODO: Create a way to dynamically select which is best
        shape (int): num rows, :int: num cols): Array specificying the
            dimensions of the area of interest.
        save (bool): If true, save relational matrix, if False dont.
        combine (str): If true, flatten arrays into single array.

    Returns:
        Pytorch tensor for cadinality relationship.

    """
    i, j = shape
    size = i*j
    directions = type_rel
    if type_rel[0] == "all":
        directions = ["north", "south", "east", "west", "northeast",
            "northwest", "southeast", "southwest"]
    print("Constructing", directions, "relation", end = "\r")
    granular_relations = {}
    for direc in directions:
        granular_relations[direc] = np.zeros((size, size), dtype = "float")

    for key, value in granular_relations.items():
        if key == "north":
            num_cords = j*(i-1)
            i_cords = np.arange(j,num_cords+j)
            j_cords = np.arange(0, num_cords)
            coordinates = np.array(list(zip(i_cords, j_cords)))
            value = delete_exclude(coordinates, exclude, value)
        # south
        elif key == "south":
            num_cords = j*(i-1)
            i_cords = np.arange(0, num_cords)
            j_cords = np.arange(j, num_cords+j)
            coordinates = np.array(list(zip(i_cords, j_cords)))
            value = delete_exclude(coordinates, exclude, value)
        # east
        elif key == "east":
            num_cords = (i*j)-1
            i_cords = np.arange(0, num_cords)
            j_cords = np.arange(1, num_cords+1)
            concat_cords = list(zip(i_cords, j_cords))
            coordinates = [x for x in concat_cords if not(not x[0] == 0 and x[1] % j == 0)]
            value = delete_exclude(coordinates, exclude, value)
        # west
        elif key == "west":
            num_cords = (i*j)-1
            i_cords = np.arange(1, num_cords+1)
            j_cords = np.arange(0, num_cords)
            concat_cords = list(zip(i_cords, j_cords))
            coordinates = [x for x in concat_cords if not(not x[1] == 0 and x[0] % j == 0)]
            value = delete_exclude(coordinates, exclude, value)
        # northeast
        elif key == "northeast":
            num_cords = j*(i-1)+1
            i_cords = np.arange(j-1, num_cords+j-1)
            j_cords = np.arange(0, num_cords)
            concat_cords = list(zip(i_cords, j_cords))
            coordinates = [x for x in concat_cords if not(x[1] == 0 or x[1] % j == 0)]
            value = delete_exclude(coordinates, exclude, value)
        # northwest
        elif key == "northwest":
            num_cords = j*(i-1)-1
            i_cords = np.arange(j+1, num_cords+j+1)
            j_cords = np.arange(0, num_cords)
            concat_cords = list(zip(i_cords, j_cords))
            coordinates = [x for x in concat_cords if not(x[1] == 0 and x[0] % j == 0)]
            value = delete_exclude(coordinates, exclude, value)
        # southeast
        elif key == "southeast":
            num_cords = j*(i-1)-1
            i_cords = np.arange(0, num_cords)
            j_cords = np.arange(j+1, num_cords+j+1)
            concat_cords = list(zip(i_cords, j_cords))
            coordinates = [x for x in concat_cords if not(not x[0] == 0 and x[1] % j == 0)]
            value = delete_exclude(coordinates, exclude, value)
        # southwest
        else:
            num_cords = j*(i-1)+1
            i_cords = np.arange(0, num_cords)
            j_cords = np.arange(j-1, num_cords+j)
            concat_cords = list(zip(i_cords, j_cords))
            coordinates = [x for x in concat_cords if (not x[0] == 0 and x[0] % j == 0)]
            value = delete_exclude(coordinates, exclude, value)
    tensor = np.stack(granular_relations.values(), 1)
    if(combine == True):
        out = np.zeros((size, size), dtype = "bool")
        for i in granular_relations.values():
            # print(i.shape)
            out = out + i
        tensor = np.expand_dims(out, axis = 1)
    tensor = torch.from_numpy(tensor)
    print("{} Construction Successful".format(type_rel))
    if save:
        print("saving as csv")
        np.savetxt("data/"+type_rel+".csv", relation, delimiter = ",")
    return tensor

def show_frame(data, frame):
    """Simple function that takes in a 3D array and returns a plot of one of
    the arrrays.

    Args:
        data (array): 3D array representing year-over-year deforestaiton loss.
        frame: Specific year (index in array, {0, T}, to show.

    Yields:
        Matplotlib imshow visualization.

    Example:
        data = [15, 200, 200]
        imshow(data, 14)
        Show the final year of deforestation.

    """
    try:
        t = plt.imshow(data[frame])
        plt.title(str(frame))
        plt.show()
    except:
        print("Invalid Frame")

def reduce_dimensions(data, i_size):
    """Reduce dimension of input vector to make training easier. The input size
    i, will not be output exactly in_order to maintain dimensionality.

    Args:
        data (array): 3D array consisting of T 2D arrays to be reduced.
        i_size (float): Float point specifying the reduction size.

    Returns:
        3D array consisting of T reduced dimenion arrays with values
        between 0 and __.
    TODO: Specify largest value

    """
    print("Reducing size")
    _, y, x = data.shape
    size = (x*30/i_size, y*30/i_size)
    reduced_array = []
    for i in data:
        im = Image.fromarray(i)
        im.thumbnail(size, Image.NEAREST)
        reduced_array.append(np.array(im))
    reduced_array = np.asarray(reduced_array)
    # show_frame(reduced_array, reduced_array.shape[0]-1)
    return(reduced_array)

def print_stats(data):
    """Utility function to print summary statistics about the composition
    of a 2D array.

    Args:
        data (array): 2D array containing raw deforestation data.

    Note:
        This only effective after creating raster object but before make_dict
        but is compatiable with any 2D array.

    Example:
                Element             Frequency           Percent
        0       0                   3965899             99.147
        1       1                   10460               0.262
        2       2                   2381                0.06
        3       3                   586                 0.015
        4       4                   2230                0.056
        5       5                   1767                0.044
        6       6                   1150                0.029
        7       7                   2874                0.072
        8       8                   1171                0.029
        9       9                   4876                0.122
        10      10                  3005                0.075
        11      11                  1122                0.028
        12      12                  2479                0.062

    """
    shp = data.shape
    max_year = shp[0]-1
    total = np.argwhere(data[max_year] != -2).shape[0]
    unique = np.unique(data[max_year])
    fmt = '{:<8}{:<20}{:<20}{}'
    print(fmt.format('', 'Element', 'Frequency', 'Percent'))
    for i, x in enumerate(unique):
        if(x != -2):
            Frequency = np.argwhere(data[max_year] == x).shape[0]
            print(fmt.format(i, x, Frequency, round(100*(Frequency/total), 3)))


def compile_files(directory):
    """Given a directory, it will create a json file containing the paths for
    every csv in the "data" subfolder of that directory

    Args:
        directory (str): directory of csv files.

    Yields:
        .json file containing all the .csv files saved to the argument
        directory containing list of files saved in the subfolder data.

    """
    csv_files = []
    try:
        for root, dirs, files in os.walk(os.path.join(directory, "data")):
            for name in files:
                if name.endswith(".csv"):
                    csv_files.append(os.path.join(root, name))
    except FileNotFoundError:
        print("Directory does not exits")
    with open(os.path.join(directory, 'files.json'), 'w') as f:
        json.dump({"files": csv_files}, f)


def import_json(directory):
    """Imports json file and returns the list of strings in the "files" section
    of the json

    Args:
        directory (str): Directory location of json file.

    Returns:
        List, can be empty, of the files section of the .json.

    Note:
        .json file must be named files.json.

    """
    try:
        with open(os.path.join(directory, 'test.json')) as json_file:
            data = json.load(json_file)
            return(data['files'])
    except:
        print("File does not exist")

def roundup(x, ks):
    x_new = int(math.ceil(x / float(ks))) * ks
    print("converting {}   ->   {}".format(x, x_new))
    return x_new




def grid_area(years, ks, suffix, mean = False, save = False, keep_nan = False):
    """Function that does a convolutional interpolation of 3D array

    Uses a classic DNN convolutional summation with a stide length equal to the
    kernel size. Pads the years array, with np.nan's so that each y x dimension
    so it is a multiple of the kernel.

    Args:
        years (np array): 3d array of deforestation (years * y * x)
        ks (int): kernel size - 30 * ks / 1000 = end spatial representation of
            each pixel.
        suffix (string): sufffix for csv name, <pixel size>km_<suffix>
        save (bool): Flag for whether or not to save the return value.

    Returns:
        An resized representation of the years arguement array. Each pixel
        represents the sum of all pixels in the years array.

    Example:
        ks = 2
        years =     |   1   1   2   2   |
                    |   1   1   2   2   |
                    |   3   3   4   4   |
                    |   3   3   4   4   |

        returns ->  |   4   8    |
                    |   12  16   |

    """
    data_shape = years.shape
    print(data_shape)
    res = str(round(30*ks/1000, 1)).replace(".", "_")
    print("pixel resolution: {}km".format(res))
    num_years = data_shape[0]
    new_dims = [roundup(data_shape[1], ks), roundup(data_shape[2], ks)]
    pad_widths = [new_dims[0]-data_shape[1], new_dims[1] - data_shape[2]]
    new_array = np.pad(years, ((0, 0), (0, pad_widths[0]), (0, pad_widths[1])),
        'constant', constant_values = (np.nan,))
    trans_shape = [int(i/ks) for i in new_dims]
    resized = np.full((num_years, *trans_shape), np.nan)
    for i in range(trans_shape[0]):
        for j in range(trans_shape[1]):
            temp = new_array[:, i*ks:(i+1)*ks,
                j*ks:(j+1)*ks].reshape(num_years, -1)
            if np.isnan(temp).all():
                resized[:, i, j] = np.nan
            else:
                if mean:
                    resized[:, i, j] = np.nanmean(temp, axis = 1)
                else:
                    resized[:, i, j] = np.nansum(temp, axis = 1)
    if keep_nan:
        resized = np.nan_to_num(resized, 0)
    if save:
        np.savetxt("data/{}km_{}.csv".format(res, suffix),
            resized.reshape(num_years, -1), delimiter = ",")
    return resized


def recombine_crop(predictions, mp):
    """Greates a compositional prediction from a set np arrays.

    Args:
        predictions (dict): A dictionary of 3d numpy arrays representing the
             forecasted deforestation for each subsected area.
        opt (DotDict): Dictionary of parameters describing the current model
            and the prediction space.

    Returns:
        An array unpad that presents the composed forecast for each year in the
        predicton range. Visualizes the final position in the array

    """
    comp = np.zeros((mp.n_pred, *mp.new_dims))
    quad_dims = [int(x/mp.tsize) for x in mp.new_dims]
    count = 0
    for key, i in predictions.items():
        curval = int(re.findall(r'\d+$', key)[-1])
        while count+1 != curval:
            print(" no matching file adding zeros")
            quadrant = np.zeros((mp.n_pred, mp.tsize, mp.tsize))
            xpos = int((count)/(quad_dims[1]-1))
            ypos = int((count)%quad_dims[0])
            print("appending zeros to y: {}, x: {}".format(ypos, xpos))
            comp[:, ypos*mp.tsize:(ypos+1)*mp.tsize,
            xpos*tsize:(xpos+1)*mp.tsize] += quadrant
            count +=1
        quadrant = np.asarray(i).reshape(mp.n_pred, mp.tsize, mp.tsize)
        xpos = int((count)/(quad_dims[1]-1))
        ypos = int((count)%quad_dims[0])
        comp[:, ypos*mp.tsize:(ypos+1)*mp.tsize,
            xpos*mp.tsize:(xpos+1)*mp.tsize] += quadrant
        count += 1
    xmin = int((mp.new_dims[1] - mp.shape[1])/2)
    xmax = mp.new_dims[1]-(mp.new_dims[1]-mp.shape[1]-xmin)
    ymin = int((mp.new_dims[0] - mp.shape[0])/2)
    ymax = mp.new_dims[0]-(mp.new_dims[0]-mp.shape[0]-ymin)
    print("{}:{} \t {}:{}".format(xmin, xmax, ymin, ymax))
    unpad = comp[:, ymin:ymax, xmin:xmax]
    plt.imshow(unpad[-1])
    plt.colorbar()
    plt.show()
    return unpad

def exclude_values(mp, data, other_data = None):
    """Function that sets the values outside of the geographic area to null.

    Args:
        mp (DotDict): Dictionary of parameters describing the current model
            and the prediction space. The most important being the exclude_dir
            parameter used to fetch the values to be excluded
        data (np array): Array to be modified and excluded.
        other_data (np array): Optional array of exclude values. If passed,
            it will override the exclude_dir keep_values

    Returns:
        Returns array with the positions in the exlcude_dir set to np.nan

    Example:
        data =          |   1   1   1   1   |
                        |   1   1   1   1   |

        exclude_dir =   [[0, 1], 0, 3]]

        return =        |   nan 1   1   nan |
                        |   1   1   1   1   |

    """
    if other_data is not None:
        exclude = other_data
    else:
        exclude = np.loadtxt(mp.exclude_dir, delimiter = ",", dtype = np.intc)
    print(exclude.shape)
    for pos in exclude:
            data[:, pos[0], pos[1]] = np.nan
    return data

def lasso_window(x, y, lasso, opt):
    """
    """
    x1 = x - lasso if x > lasso else 0
    x2 = x + lasso if x + lasso < opt.shape[1] else opt.shape[1]-1
    y1 = y - lasso if y > lasso else 0
    y2 = y + lasso if y + lasso < opt.shape[0] else opt.shape[0]-1
    cx = lasso-1 if x > lasso else x-1
    cy = lasso-1 if y > lasso else y-1
    return x1, x2, y1, y2, cx, cy

def data_to_lasso(input_data, opt, lasso1, lasso2, start_year, out_dir, file_name,
    save = False):
    keep_values = np.argwhere(~np.isnan(input_data[10]))
    output_data = np.zeros((keep_values.shape[0]*opt.years, lasso2-lasso1+5))
    idn = 10
    for pos, (y, x) in enumerate(keep_values):
        for i in range(opt.years):
            output_data[pos*opt.years+i, 0:5] = [idn, y, x, start_year+i, input_data[i, y, x]]
        for lasso in range(lasso1, lasso2):
            x1, x2, y1, y2, cx, cy = lasso_window(x, y, lasso, opt)
            las = np.copy(input_data[:, y1:y2, x1:x2])
            try:
                las[:, cy, cx] = np.nan
            except IndexError:
                print(lasso, x1, x2, y1, y2, cx, cy, x, y)
            if np.all(np.isnan(las)):
                mean = [0]*opt.years
            else:
                mean = np.nanmean(las.reshape(opt.years, -1) , axis = 1)
            for i in range(opt.years):
                output_data[pos*opt.years+i, lasso-lasso1+5] = mean[i]
        idn+=1
        print('\r', "{:>5}/{} \t {}%".format(pos, keep_values.shape[0],
            round(pos/(keep_values.shape[0])*100, 3)), end = "")
    df = pd.DataFrame(data=output_data)
    cols = [str(i) for i in range(lasso1, lasso2)]
    df.columns = ['id2', 'y_c', 'x_c', 'year','gt', *cols]
    df.fillna(0)
    df = df.astype({"id2": int,
                    "y_c": int,
                    'x_c': int,
                    'year': int,})
    if save:
        df.to_csv(os.path.join(out_dir, file_name+".csv"))
    return df
