import cv
import os
import pickle

# global params/config
class Params:
    params_file = os.path.join(os.path.dirname(__file__), "params.cfg")

    max_box_scan_attempts = 100
    camera_sleep_missing_codes = 0.1 # sleep between frames when there are undetected codes
    camera_sleep_no_box = 0.5 # sleep between frames when box is not detected

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
    annotate_outside = (255,255,0)

    box_fill_threshold = (20, 20, 20)

    tab_color_low = (5, 75, 120) # HSV low threshold for tabs (orange)
    tab_color_high = (30, 255, 255) # HSV high threshold for tabs (orange)
    min_pixels_per_well = 100

