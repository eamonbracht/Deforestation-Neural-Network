import numpy as np
#from raster import Raster
import matplotlib.pyplot as plt
from scipy.misc import imresize
import pickle
from utils import *
import os, sys
import torch
import json
from PIL import Image
import math

# 0 forested, -1 = deforested
def make_dict(raster, name, years, shape = False, save=False):
    """Takes in array of deforestation data and parses it into an set of arrays stored in dictionary for the yearly deofrestation progression for each pixel then pickles and returns that dictionary.

    Args:
        raster (raster obj): raster from raster class containing the rasterized data.
        name (str): name of file to be saved.

    Returns:
        dictionary with the status of each year, labeled with a eponymous str for that year.

    """
    year_dict = {}
    num_years = years+1
    years = range(2000, 2000+num_years)
    yearlabels = [str(year) for year in years]
    input_array = np.copy(raster.arr_raster)
    size = input_array.shape
    print(input_array[0, 0])
    if shape:
        print("maintaining -1's")
        input_array[input_array == 0] = -2
        input_array[input_array == -2] = -1
    else:
        print("removing -1's")
        input_array[input_array == -1] = 0
    print(input_array[0, 0])
    for num, val in enumerate(yearlabels):
        temp = np.copy(input_array)
        temp[(temp <= num) & (temp>0)] = 1
        temp[(temp > num)] = 0
        year_dict[val] = temp
        progress(num, str(int(val)), num_years)
# uncommnet for trouble shooting
#    print_validation(year_dict)
    if save:
        print("Saving data...")
        name = 'data/'+name+'.pickle'
        with open(name, 'wb') as handle:
            pickle.dump(year_dict, handle)
        print("Saved to", name)
    return year_dict

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
    """Helper method used for debugging in make_dict function. For each array craeted, There should only be a 0 and 1's.

    """
    for i, x in data.items():
        print(i, np.unique(x))

def save_images(data, prefix):
    """Converts an array with T time steps into T images and saves them.

    Args:
        data (array): Deforestation state for an area over a set of years. T x m x n
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

def dictionary_to_array(data, reshape):
    """Takes in dictionary and converts to array.

    Args:
        data (dict): Dictionary contains T m x n matricies where T is the number of time steps and m and n are the dimensions of input matrix.
        reshape (bool): Flag to indicate if data needs to be flattened.

    Returns:
        T x X array where X is the number of time steps.

    Note:
        Data must be flattened or linearized so that all data can be captured in a 2D array.

    """
    print("Converting dictionary to array")
    tensor = np.stack(data.values(), 0)
    if reshape:
        print("Flattening")
        tensor = tensor.reshape(15,-1).T
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

def make_relation(type_rel, data, save, combine):
    """Function to create binary relationship tensor for data based on cardinality. Creates north, south, east, west, northwest, northeast, southwest and southeast realtions.

    Note:
        For an m x n area, the output matrix will have shape X x X where X = m*n. Example for north direction:
            Let X have dimensions i, j. i = 1 iff j is north of i.

        Matrix is very sparse.

        Returns tensor not array because pytorch has built in normalization that succeeds this step.

    Args:
        type_rel (str): Denoting which type of relation to make.
            TODO: Create a way to dynamically select which is best
        data (:int: num rows, :int: num cols): Array specificying the dimensions of the area of interest.
        save (bool): If true, save relational matrix, if False dont.
        combine (str): If true, flatten arrays into single array.

    Returns:
        Pytorch tensor for cadinality relationship.

    """
    i, j = data
    size = i*j
    directions = type_rel
    if type_rel[0] == "all":
        directions = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
    print("Constructing", directions, "relation", end = "\r")
    granular_relations = {}
    for direc in directions:
        granular_relations[direc] = np.zeros((size, size), dtype = "uint8")
    # print("i {} j {}".format(i, j))
    for key, value in granular_relations.items():
        if key == "north":
            num_cords = j*(i-1)
            i_cords = np.arange(j,num_cords+j)
            j_cords = np.arange(0, num_cords)
            coordinates = list(zip(i_cords, j_cords))
            for cord in coordinates:
                value[cord] = 1
        # south
        if key == "south":
            num_cords = j*(i-1)
            i_cords = np.arange(0, num_cords)
            j_cords = np.arange(j, num_cords+j)
            coordinates = list(zip(i_cords, j_cords))
            for cord in coordinates:
                value[cord] = 1
        # east
        if key == "east":
            num_cords = (i*j)-1
            i_cords = np.arange(0, num_cords)
            j_cords = np.arange(1, num_cords+1)
            coordinates = list(zip(i_cords, j_cords))
            for cord in coordinates:
                if (not cord[0] == 0 and cord[1] % j == 0):
                    pass
                else:
                    value[cord] = 1
        # west
        if key == "west":
            num_cords = (i*j)-1
            i_cords = np.arange(1, num_cords+1)
            j_cords = np.arange(0, num_cords)
            coordinates = list(zip(i_cords, j_cords))
            for cord in coordinates:
                if (not cord[1] == 0 and cord[0] % j == 0):
                    pass
                else:
                    value[cord] = 1
        # northeast
        if key == "northeast":
            num_cords = j*(i-1)+1
            i_cords = np.arange(j-1, num_cords+j-1)
            j_cords = np.arange(0, num_cords)
            coordinates = list(zip(i_cords, j_cords))
            for cord in coordinates:
                if(cord[1] == 0 or cord[1] % j == 0):
                    pass
                else:
                    value[cord] = 1
        # northwest
        if key == "northwest":
            num_cords = j*(i-1)-1
            i_cords = np.arange(j+1, num_cords+j+1)
            j_cords = np.arange(0, num_cords)
            coordinates = list(zip(i_cords, j_cords))
            for cord in coordinates:
                if(not cord[1] == 0 and (cord[0]) % j == 0):
                    pass
                else:
                    value[cord] = 1
        # southeast
        if key == "southeast":
            num_cords = j*(i-1)-1
            i_cords = np.arange(0, num_cords)
            j_cords = np.arange(j+1, num_cords+j+1)
            coordinates = list(zip(i_cords, j_cords))
            for cord in coordinates:
                if(not cord[0] == 0 and (cord[1]) % j == 0):
                    pass
                else:
                    value[cord] = 1
        # southwest
        if key == "southwest":
            num_cords = j*(i-1)+1
            i_cords = np.arange(0, num_cords)
            j_cords = np.arange(j-1, num_cords+j)
            coordinates = list(zip(i_cords, j_cords))
            for cord in coordinates:
                if(cord[0] == 0 or (cord[0]) % j == 0):
                    pass
                else:
                    value[cord] = 1
#    tensor = dictionary_to_array(granular_relations, False)
    tensor = np.stack(granular_relations.values(), 1)
    if(combine == True):
        out = np.zeros((size, size), dtype = "uint8")
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
    """Simple function that takes in a 3D array and returns a plot of one of the arrrays.

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
    """Reduce dimension of input vector to make training easier. The input size i, will not be output exactly in_order to maintain dimensionality.

    Args:
        data (array): 3D array consisting of T 2D arrays to be reduced.
        i_size (float): Float point specifying the reduction size.

    Returns:
        3D array consisting of T reduced dimenion arrays with values between 0 and __.
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
    """Utility function to print summary statistics about the composition of a 2D array.

    Args:
        data (array): 2D array containing raw deforestation data.

    Note:
        This only effective after creating raster object but before make_dict but is compatiable with any 2D array.

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
    """Given a directory, it will create a json file containing the paths for every csv in the "data" subfolder of that directory

    Args:
        directory (str): directory of csv files.

    Yields:
        .json file containing all the .csv files saved to the argument directory containing list of files saved in the subfolder data.

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
    """Imports json file and returns the list of strings in the "files" section of the json

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

def grid_area(years, ks, save = False, sum = True):
    data_shape = years.shape
    res = int(30*ks/1000)
    print("pixel resolution: {}km".format(res))
    num_years = data_shape[0]
    new_dims = [roundup(data_shape[1], ks), roundup(data_shape[2], ks)]
    new_array = np.zeros((num_years, *new_dims))
    print(new_array.shape)
    for year in range(num_years):
        new_array[year, :data_shape[1], :data_shape[2]] = years[year]
    filt = np.ones((ks, ks))
    trans_shape = [int(i/ks) for i in new_dims]
    resized = np.zeros((num_years, *trans_shape))
    for year in range(num_years):
        for i in range(trans_shape[0]-1):
            for j in range(trans_shape[1]-1):
                inter = np.multiply(new_array[year, i*ks:(i+1)*ks, j*ks:(j+1)*ks], filt)
                if sum:
                    resized[year, i, j] = np.sum(inter)
                else:
                    resized[year, i, j] = np.mean(inter)
    print(resized.shape)
    if not np.isnan(resized).any() and save:

        np.savetxt("data/{}km_2018.csv".format(res), resized.reshape(19, -1), delimiter = ",")
    else:
        if save:
            pass
        else:
            print("array contains Nan's")
    return resized
