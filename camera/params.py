import cv
import os
import pickle

# global params/config
class Params:
    params_file = os.path.join(os.path.dirname(__file__), "params.cfg")

    hue_bins = 30
    sat_bins = 32

    matrix_code_size = 12
    edge_cell_threshold = 8 # number of edge cells that have to be correct
    min_pixels_per_cell = 3 # each pixel in the code must be at least 3x3
    min_code_size_pixels = min_pixels_per_cell * min_pixels_per_cell * matrix_code_size * matrix_code_size
    code_edge_min_pixels = 0.3 # must be < 0.5 as two edges will have half on/off
    code_squareness_deviation = 10 # number of pixels width/height can be different
    cell_pixel_threshold = 0.4 # fraction of pixels that have to be on for cell to be considered on
    
    num_rows = 8
    num_cols = 12

    flood_fill_hue = 50
    flood_fill_sat = 50

    box_min_area = 0.20
    box_max_area = 0.70
    tab_min_area = 0.08

    min_pixels_per_well = 100
    
    # no longer using box histogram to detect the box
    #box_histogram = None
    tab_histogram = None
    
    @staticmethod
    def save():
        f = open(Params.params_file, 'w')
        # histograms can't currently be pickled/unpickled so we convert to array
        #pickle.dump([[cv.QueryHistValue_2D(Params.box_histogram, i, j) for i in range(Params.hue_bins)] for j in range(Params.sat_bins)], f)
        pickle.dump([[cv.QueryHistValue_2D(Params.tab_histogram, i, j) for i in range(Params.hue_bins)] for j in range(Params.sat_bins)], f)

    @staticmethod
    def load():
        f = open(Params.params_file, 'r')
        #b = pickle.load(f)
        t = pickle.load(f)
        # hue varies from 0 to 179
        # saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
        #Params.box_histogram = cv.CreateHist([Params.hue_bins, Params.sat_bins], cv.CV_HIST_ARRAY, [[0, 179], [0, 255]])
        Params.tab_histogram = cv.CreateHist([Params.hue_bins, Params.sat_bins], cv.CV_HIST_ARRAY, [[0, 179], [0, 255]])
        for i in range(Params.hue_bins):
            for j in range(Params.sat_bins):
                #cv.SetND(Params.box_histogram.bins, [i, j], b[j][i])
                cv.SetND(Params.tab_histogram.bins, [i, j], t[j][i])
