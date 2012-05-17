import cv
import os
import pickle

# global params/config
class Params:
    params_file = os.path.join(os.path.dirname(__file__), "params.cfg")

    max_box_scan_attempts = 1000
    
    hue_bins = 30
    sat_bins = 32

    white_threshold = (150, 150, 150) # in RGB space, this should select both the box and codes on tube

    code_decoded_length = 10
    matrix_code_size = 12
    min_pixels_per_cell = 3 # each pixel in the code must be at least 3x3
    min_code_size_pixels = min_pixels_per_cell * min_pixels_per_cell * matrix_code_size * matrix_code_size # well with < this number of on pixels will be considered empty
    code_squareness_deviation = 5 # number of pixels width/height can be different
    cell_pixel_threshold = 0.5 # fraction of pixels that have to be on for cell to be considered on

    edge_min_pixels_solid = min_pixels_per_cell * matrix_code_size # min pixels for "solid" line of code
    code_min_pixels_slice = min_pixels_per_cell # min pixels on for any row/col slice through the code. row/col could theoretically only have 1 on pixel

    num_rows = 8
    num_cols = 12

    # note these are BGR colors
    annotate_empty_color = (255, 0, 0) # blue
    annotate_present_color = (0, 255, 0) # green
    annotate_not_decoded = (0, 0, 255) # red
    
    box_max_area = 0.80 # box should not take up more than this fraction of total image
    box_fill_threshold = (20, 20, 20)
    
    tab_histogram = None
    tab_pixel_cutoff = 0.02
    histogram_threshold = 15

    min_pixels_per_well = 100
    
    # no longer using box histogram to detect the box
    #box_histogram = None
    
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
