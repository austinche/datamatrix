import cv
import sys
import math
import csv
import numpy as np
import itertools
import re

import datamatrix

class Params:
    # error prone params
    tube_well_factor = 5 # >= 4. closer to 4 gets more of the well which may be slower, larger may miss code
    canny_low_high_ratio = 2
    canny_high_thresholds = [1000, 500, 250]
    lines_giveup_threshold = 3
    hough_lines_distance_resolution = 1 # pixels
    hough_lines_angle_resolution = 0.01 # radians
    hough_lines_threshold = 37
    dotted_pixel_range = 4
    box_fill_threshold = 20
    code_pixel_tolerance = 4

    # Box specific (will only work with orange tabbed boxes)
    tab_color_low = (10, 100, 100) # HSV low threshold for tabs (orange)
    tab_color_high = (20, 255, 255) # HSV high threshold for tabs (orange)

    # Fixed params
    code_regex = re.compile("^\d{10}$")
    code_decoded_length = 10
    matrix_code_size = 12
    num_rows = 8
    num_cols = 12
    num_wells = num_rows * num_cols
    half_pi = math.pi / 2

    # params unlikely to impact accuracy
    min_pixels_per_cell = 4 # each code must have at least this number of pixels on each side
    min_pixels_code_side = min_pixels_per_cell * matrix_code_size
    min_box_pixels = min_pixels_code_side * num_wells * 4

    min_tube_radius = min_pixels_per_cell * matrix_code_size / 2 # min pixels for tube radius

    # Preferences
    # note these are BGR colors
    annotate_empty_color = (255, 0, 0) # blue
    annotate_present_color = (0, 255, 0) # green
    annotate_not_decoded = (0, 0, 255) # red
    annotate_outside = (255,255,0)
    annotate_tabs = (0,255,255)
    annotate_error = (0, 0, 255) # red

class TwoLines:
    # Represent two lines that form right angles in an image
    def __init__(self, image, min_pixels, line1, line2):
        # lines should be in (rho, theta) form
        self.min_pixels = min_pixels
        self.line1 = line1
        self.line2 = line2
        self.image = image

        (self.line1p1, self.line1p2) = self.polar2cart(line1)
        (self.line2p1, self.line2p2) = self.polar2cart(line2)
        self.corner = self.intersection()

        (self.line1end, self.line1dist, self.line1score) = self.find_line_end(self.image, self.corner, self.line1p1, self.line1p2)
        (self.line2end, self.line2dist, self.line2score) = self.find_line_end(self.image, self.corner, self.line2p1, self.line2p2)

        self.score = self.calc_score()

    def __repr__(self):
        return repr(((self.line1p1, self.line1p2), (self.line2p1, self.line2p2)))

    def __lt__(self, other):
        return self.score > other.score

    def calc_score(self):
        # returns a score for ranking lines on an image
        # higher is better, 0 is worst
        # currently score by largest image enclosed by lines
        if not self.line1end or not self.line2end:
            return 0
        return self.line1dist * self.line2dist

    def annotate(self, image):
        if self.corner == None or self.line1end == None or self.line2end == None:
            print "Points are None"
            return
        cv.Line(image, self.int_point(self.line1end), self.int_point(self.corner), (0,0,255), 1)
        cv.Line(image, self.int_point(self.line2end), self.int_point(self.corner), (0,0,255), 1)
        cv.Circle(image, self.int_point(self.corner), 1, (0, 255, 0), 1)
        cv.ShowImage("lines", image)
        cv.WaitKey()

    def int_point(self, point):
        return (cv.Round(point[0]), cv.Round(point[1]))

    @staticmethod
    def best_lines(image, min_pixels, line1s, line2s):
        # given a list of lines in line1s and line2s
        # returns the TwoLines object that is the best right angle or None
        best = min(itertools.product(line1s, line2s), key=TwoLines.right_angledness)
        return TwoLines(image, min_pixels, best[0], best[1])

    @staticmethod
    def right_angledness(x):
        angle1 = x[0][1]
        angle2 = x[1][1]
        diff = angle1 - angle2
        # get it into [0..pi/2) range
        while diff < 0:
            diff += math.pi
        while diff >= Params.half_pi:
            diff -= Params.half_pi
        return diff

    def polar2cart(self, line):
        # convert line from (rho, theta) to (x1, y1), (x2, y2) form
        rho = line[0]
        theta = line[1]
        a = math.cos(theta)
        b = math.sin(theta)
        x1 = a * rho
        y1 = b * rho
        x2 = x1 - 1000 * b
        y2 = y1 + 1000 * a
        return ((x1, y1), (x2, y2))

    def intersection(self):
        # returns the intersection of the 2 lines
        # http://en.wikipedia.org/wiki/Line-line_intersection
        (x1, y1) = self.line1p1
        (x2, y2) = self.line1p2
        (x3, y3) = self.line2p1
        (x4, y4) = self.line2p2
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if d == 0:
            return None

        a = (x1 * y2 - y1 * x2)
        b = (x3 * y4 - y3 * x4)
        return ((a * (x3 - x4) - (x1 - x2) * b) / d,
                (a * (y3 - y4) - (y1 - y2) * b) / d)

    def sum_pixels(self, image, start_point, end_point):
        # goes from start_point to end_point
        # summing pixels intensities while it's greater than half the moving average
        threshold = 0

        li = cv.InitLineIterator(image, start_point, end_point)
        total = 0
        count = 0
        for p in li:
            if p < threshold:
                return total
            total += p
            count += 1
            threshold = total / count / 2
        return total

    def find_line_end(self, image, corner, point1, point2):
        # find the end of line from corner for the line between 2 points
        # that intersects a circle enclosed in the image
        # (x-r)^2+(y-r)^2=r^2

        # http://mathworld.wolfram.com/Circle-LineIntersection.html

        r = image.width / 2
        r2 = r * r

        # shift origin to center of circle
        x1 = point1[0] - r
        y1 = point1[1] - r
        x2 = point2[0] - r
        y2 = point2[1] - r
        dx = x2 - x1
        dy = y2 - y1
        dr = math.sqrt(dx * dx + dy * dy)
        D = x1 * y2 - x2 * y1
        dr2 = dr * dr
        delta = r2 * dr2 - D * D

        if delta <= 0:
            # if line doesn't intersect circle at 2 points, it's definitely wrong
            return (None, 0, 0)

        sgn_dy = -1 if dy < 0 else 1
        sqrt_delta = math.sqrt(delta)
        a1 = sgn_dy * dx * sqrt_delta
        b1 = abs(dy) * sqrt_delta
        points = [(r+(D * dy + a1) / dr2, r+(-D * dx + b1) / dr2), (r + (D * dy - a1) / dr2, r + (-D * dx - b1) / dr2)]

        # calculate the distances for each point
        # if any are too short, we eliminate them
        dist1 = math.hypot(points[0][0]-corner[0], points[0][1]-corner[1])
        dist2 = math.hypot(points[1][0]-corner[0], points[1][1]-corner[1])

        if dist1 < self.min_pixels and dist2 < self.min_pixels:
            return (None, 0, 0)

        score1 = self.sum_pixels(image, self.int_point(corner), self.int_point(points[0]))
        score2 = self.sum_pixels(image, self.int_point(corner), self.int_point(points[1]))

        if dist1 < self.min_pixels:
            return (points[1], dist2, score2)
        else:
            if dist2 < self.min_pixels:
                return (points[0], dist1, score1)
            else:
                # both points could be correct
                # pick the direction with the higher intensity pixels
                if score1 > score2:
                    return (points[0], dist1, score1)
                else:
                    return (points[1], dist2, score2)

    def crop_and_rotate(self):
        # returns new cropped and rotated image

        if not self.line1end or not self.line2end:
            return None
        # figure out which is top left and which is bottom right corner
        # the intersection corner goes in the bottom left
        # returns (top-left, bottom-left, bottom-right)
        if self.on_left(self.line1end, self.corner, self.line2end):
            tl_corner = self.line2end
            height = self.line2dist
            br_corner = self.line1end
            width = self.line1dist
        else:
            tl_corner = self.line1end
            height = self.line1dist
            br_corner = self.line2end
            width = self.line2dist

        img_width = cv.Ceil(width)
        img_height = cv.Ceil(height)

        # rotate the code
        rotation = cv.CreateMat(2, 3, cv.CV_32FC1)
        cv.GetAffineTransform((tl_corner, self.corner, br_corner), ((0,0), (0, height-1), (width-1, height-1)), rotation)
        rotated = cv.CreateMat(img_height, img_width, cv.CV_8UC1)
        cv.WarpAffine(self.image, rotated, rotation)

        # removing a pixel from the left/bottom edges helps in forcing
        # those pixels to be more on
        return cv.GetSubRect(rotated, (1, 0, img_width-1, img_height-1))

    def on_left(self, point1, point2, target):
        # return true if target is on left of line made by point1..point2
        return (point2[0] - point1[0]) * (target[1] - point1[1]) - (point2[1]- point1[1]) * (target[0] - point1[0]) > 0

class BoxScanner:
    FONT = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1.0, 1.0)

    def __init__(self):
        self.codes = [[None for j in range(Params.num_cols)] for i in range(Params.num_rows)]
        self.image = None
        self.info_image = None
        self.decoded_codes = 0
        self.decoded_empty = 0
        self.error_message = None
        self.code_min_pixels = Params.min_pixels_code_side
        self.code_size_range = None

    #
    # Public methods
    #

    def write_box_info(self, output):
        output.write(str(self.codes))
        output.write("\n")
        if self.error_message:
            output.write("Error: %s\n" % self.error_message)
        total = self.decoded_codes + self.decoded_empty
        output.write("Wells done: %d empty: %d codes: %d unknown: %d\n" % (total, self.decoded_empty, self.decoded_codes, Params.num_wells - total))

    def write_code_csv(self, output):
        # write code in csv form with well (e.g. A1-H12) followed by barcode and then status
        # status column is OK for code present and decoded, EMPTY for no tube, or UNKNOWN for undecoded
        writer = csv.writer(output, delimiter=',')
        writer.writerow(["TUBE", "BARCODE", "STATUS"])
        for c in range(Params.num_cols):
            for r in range(Params.num_rows):
                well = "%s%02d" % (chr(ord('A') + r), c+1)
                if self.codes[r][c] == None:
                    writer.writerow([well, "", "UNKNOWN"])
                elif not self.codes[r][c]:
                    writer.writerow([well, "", "EMPTY"])
                else:
                    writer.writerow([well, self.codes[r][c], "OK"])

    def annotated_image(self, width, height):
        # returns an annotated image of the current box
        # resizes to the given size
        # uses part of the image to add a caption

        image = self.info_image
        if image == None:
            return None

        # shrink image
        caption_height = 40

        # image can be different sizes depending on if it's been cropped yet
        # we shrink to fixed size and add a caption
        shrink = cv.CreateImage((width, height), image.depth, image.nChannels)
        cv.SetImageROI(shrink, (0, 0, width, height  - caption_height))
        cv.Resize(image, shrink)

        # add caption text
        cv.SetImageROI(shrink, (0, height - caption_height, width, caption_height))
        cv.Set(shrink, (255, 255, 255))
        if self.error_message:
            cv.PutText(shrink, self.error_message, (0, 20), BoxScanner.FONT, Params.annotate_error)
        else:
            total = self.decoded_codes + self.decoded_empty
            cv.PutText(shrink, "Unknown: %d " % (Params.num_wells - total), (0, 20), BoxScanner.FONT, Params.annotate_not_decoded)
            cv.PutText(shrink, "Empty: %d " % self.decoded_empty, (120, 20), BoxScanner.FONT, Params.annotate_empty_color)
            cv.PutText(shrink, "Codes: %d " % (self.decoded_codes), (220, 20), BoxScanner.FONT, Params.annotate_present_color)
        cv.ResetImageROI(shrink)
        return shrink

    def scan(self, image):
        # returns count of wells decoded

        self.image = image
        self.error_message = None

        if not self.find_box_and_rotate():
            self.info_image = image
            self.error_message = "Box not found!"
            return 0

        # we crop and rotate image above
        # after here, we don't change it
        # we make a copy of the image for info_image so we
        # can annotate it without worry about affecting the decoding
        self.info_image = cv.CloneImage(self.image)
        if not self.find_orientation():
            self.error_message = "Box orientation detection failed!"
            return 0

        if not self.find_wells():
            self.error_message = "Finding wells failed!"
            return 0

        self.decode_codes()

        return self.decoded_empty + self.decoded_codes

    #
    # Decoding methods
    #

    def find_box_and_rotate(self):
        # finds the box and rotate it to a standard orientation

        # We assume some part of the box exists at the center of the image
        # For speed, we crop the image and only search that region for some portion of the box
        # this size should be > than maximum possible size of one square in the box
        size = int(max(self.image.width, self.image.height) / Params.num_cols)
        x = int((self.image.width - size) / 2)
        y = int((self.image.height - size) / 2)
        rect = (x, y, size, size)

        gray = cv.CreateImage(cv.GetSize(self.image), 8, 1)
        cv.CvtColor(self.image, gray, cv.CV_BGR2GRAY)

        mask = cv.CreateImage((self.image.width + 2, self.image.height + 2), 8, 1)
        cv.SetZero(mask)
        area = 0
        for i in range(x, x+size):
            for j in range(y, y+size):
                if cv.Get2D(mask, j, i)[0] == 0:
                    (area, _, _) = cv.FloodFill(gray, (i, j), 0, Params.box_fill_threshold, Params.box_fill_threshold, cv.CV_FLOODFILL_MASK_ONLY + cv.CV_FLOODFILL_FIXED_RANGE, mask)
                    if area > Params.min_box_pixels:
                        break
            if area > Params.min_box_pixels:
                break

        mask = cv.GetSubRect(mask, (1, 1, self.image.width, self.image.height))

        # find the bounding rectangle for the on pixels which should be the box outline
        contours = cv.FindContours(mask, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL)

        if len(contours) == 0:
            return False
        (max_contour, max_contour_area) = self.max_contour_area(contours)
        rect = cv.MinAreaRect2(max_contour)

        #center = rect[0]
        (height, width) = rect[1]
        #angle = rect[2]

        # if we selected entire area, then there's likely no box
        if height * width >= self.image.width * self.image.height:
            return False

        if height > width:
            (width, height) = rect[1]

        if height < Params.min_pixels_code_side * Params.num_rows or width < Params.min_pixels_code_side * Params.num_cols:
            return False

        self.image = self.crop_and_rotate(self.image, rect)

        #self.debug_resize()
        return True

    def find_orientation(self):
        # assume the box has been rotated so that tabs are on left/right
        # find the box tabs to determine orientation

        # this is maximum size of a single well as it doesn't include the border
        width = self.image.width / Params.num_cols + 50

        right_border = self.image.width - width - 1

        size = (width, self.image.height)

        # if the image includes the full box, the extra horizontal width is about equal to one well width on each side
        # so the following image extracts will definitely include the tabs if they exist
        # call the tabs location based on which quadrants have the maximal pixel count with tab color
        # do the left and right sides separately
        # then we split into top and bottom

        left_image = cv.CreateImage(size, 8, 3)
        right_image = cv.CreateImage(size, 8, 3)
        left_tabs = cv.CreateImage(size, 8, 1)
        right_tabs = cv.CreateImage(size, 8, 1)

        # left
        cv.SetImageROI(self.image, (0, 0, width, self.image.height))
        cv.CvtColor(self.image, left_image, cv.CV_BGR2HSV)
        cv.InRangeS(left_image, Params.tab_color_low, Params.tab_color_high, left_tabs)
        cv.Erode(left_tabs, left_tabs)

        # right
        cv.SetImageROI(self.image, (right_border, 0, width, self.image.height))
        cv.CvtColor(self.image, right_image, cv.CV_BGR2HSV)
        cv.InRangeS(right_image, Params.tab_color_low, Params.tab_color_high, right_tabs)
        cv.Erode(right_tabs, right_tabs)

        middle = self.image.height / 2

        cv.SetImageROI(left_tabs, (0, 0, width, middle))
        tl = cv.CreateMat(1, width, cv.CV_32FC1)
        cv.Reduce(left_tabs, tl, 0, cv.CV_REDUCE_SUM)

        cv.SetImageROI(left_tabs, (0, middle, width, middle))
        bl = cv.CreateMat(1, width, cv.CV_32FC1)
        cv.Reduce(left_tabs, bl, 0, cv.CV_REDUCE_SUM)

        cv.SetImageROI(right_tabs, (0, 0, width, middle))
        tr = cv.CreateMat(1, width, cv.CV_32FC1)
        cv.Reduce(right_tabs, tr, 0, cv.CV_REDUCE_SUM)

        cv.SetImageROI(right_tabs, (0, middle, width, middle))
        br = cv.CreateMat(1, width, cv.CV_32FC1)
        cv.Reduce(right_tabs, br, 0, cv.CV_REDUCE_SUM)

        left_diff = cv.CreateMat(1, width, cv.CV_32FC1)
        cv.Sub(tl, bl, left_diff)

        right_diff = cv.CreateMat(1, width, cv.CV_32FC1)
        cv.Sub(tr, br, right_diff)

        # positive value means more pixels on top, negative means more on bottom
        left_sum = cv.Sum(left_diff)[0]
        right_sum = cv.Sum(right_diff)[0]

        # cv.ResetImageROI(left_tabs)
        # cv.ResetImageROI(right_tabs)
        # cv.ShowImage("left tab", left_image)
        # cv.ShowImage("right tab", right_image)
        # cv.ShowImage("left tab processed", left_tabs)
        # cv.ShowImage("right tab processed", right_tabs)
        # cv.WaitKey()

        # find edge of tab and use as edge of wells
        # we consider the widest/brightest thing to be the tab

        # This can help reduce holes
        cv.Dilate(left_diff, left_diff)
        cv.Erode(left_diff, left_diff)
        cv.Dilate(right_diff, right_diff)
        cv.Erode(right_diff, right_diff)

        if left_sum < 0 and right_sum < 0:
            self.tabs_on_bottom = True
            (left_start, left_end) = self.longest_contiguous(np.asarray(left_diff)[0] < 0)
            (right_start, right_end) = self.longest_contiguous(np.asarray(right_diff)[0] < 0)
            top = middle
            bottom = self.image.height - 1
        elif left_sum > 0 and right_sum > 0:
            self.tabs_on_bottom = False
            (left_start, left_end) = self.longest_contiguous(np.asarray(left_diff)[0] > 0)
            (right_start, right_end) = self.longest_contiguous(np.asarray(right_diff)[0] > 0)
            top = 0
            bottom = middle - 1
        else:
            return False # no tabs found, fail as we cannot determine box orientation

        cv.ResetImageROI(self.image)

        self.tab_locations = (left_start, right_border+right_end)

        # annotate the box tabs
        cv.Rectangle(self.info_image, (left_start, top), (left_end, bottom), Params.annotate_tabs, 15)
        cv.Rectangle(self.info_image, (right_border+right_start, top), (right_border+right_end, bottom), Params.annotate_tabs, 15)

        return True

    def find_wells(self):
        # finds position of all wells

        left_tab_end = self.tab_locations[0]
        right_tab_start = self.tab_locations[1]
        width = right_tab_start - left_tab_end + 1

        # convert to grayscale
        gray = cv.CreateImage(cv.GetSize(self.image), 8, 1)
        cv.CvtColor(self.image, gray, cv.CV_BGR2GRAY)

        cv.SetImageROI(gray, (left_tab_end, 0, width, self.image.height))
        #self.debug_resize(gray)
        # find the edges of the wells
        # we expect the cols/rows that contain the well edges to be the brightest
        # we first do the cols since those edges go all the way from top to bottom
        # the left/right edges don't go all the way to the box edge

        avg_cols = cv.CreateMat(1, width, cv.CV_8UC1)
        cv.Reduce(gray, avg_cols, 0, cv.CV_REDUCE_AVG)

        # do a binary search for the threshold for finding box edges in the picture
        (c_min, c_max, _, _) = cv.MinMaxLoc(avg_cols)
        # the left and right of the box should be removed due to truncating image at tabs
        col_edges = self.find_threshold(np.asarray(avg_cols)[0], c_min, c_max, Params.num_cols + 1)

        if col_edges == None:
            print "Unable to find well columns"
            return False

        left_x = left_tab_end + col_edges[0][0]
        right_x = left_tab_end + col_edges[-1][1]

        # approximate where the rows should be based on well size calculated from the column edges
        well_size = (right_x - left_x + 1) / Params.num_cols
        # well_size/2 is extra fudge factor to make sure top and bottom rows are in picture
        # but should not be enough that it will get the top and bottom edges of the box
        height = Params.num_rows * well_size + well_size / 2
        top = (self.image.height-height) / 2

        # now find the rows
        cv.SetImageROI(gray, (left_x, top, right_x - left_x + 1, height))
        #self.debug_resize(gray)

        avg_rows = cv.CreateMat(height, 1, cv.CV_8UC1)
        cv.Reduce(gray, avg_rows, 1, cv.CV_REDUCE_AVG)

        row_edges = self.find_threshold(np.asarray(avg_rows)[:,0], c_min, c_max, Params.num_rows + 1)
        if row_edges == None:
            print "Unable to find well rows"
            return False

        # now store the coordinates of every well

        well_x = []
        cell_width = 0
        for i in range(Params.num_cols):
            well_x.append(left_tab_end + (col_edges[i+1][0] + col_edges[i][1]) / 2)
            cell_width += col_edges[i+1][0] - col_edges[i][1]
            #cv.Rectangle(self.image, (col_edges[i][1], 0), (col_edges[i+1][0], self.image.height), (255, 0, 0), 5)
        cell_width /= Params.num_cols

        well_y = []
        cell_height = 0
        for i in range(Params.num_rows):
            well_y.append(top + (row_edges[i+1][0] + row_edges[i][1]) / 2)
            cell_height += row_edges[i+1][0] - row_edges[i][1]
            #cv.Rectangle(self.image, (0, row_edges[i][1]), (self.image.width, row_edges[i+1][0]), (255, 0, 0), 5)
        cell_height /= Params.num_rows

        self.tube_radius = int((cell_width + cell_height) / Params.tube_well_factor)

        if self.tube_radius < Params.min_tube_radius:
            print "Tubes are too small:", self.tube_radius
            return False

        cv.ResetImageROI(gray)

        #self.debug_resize()

        self.well_x = well_x
        self.well_y = well_y
        self.gray = gray
        return True

    def decode_codes(self):
        # decode every well in the box

        # make images that we can reuse for every well
        self.well_size = 2 * self.tube_radius
        self.well = cv.CreateImage((self.well_size, self.well_size), 8, 1)
        self.edges = cv.CreateImage((self.well_size, self.well_size), 8, 1)
        self.well_mask = cv.CreateImage((self.well_size, self.well_size), 8, 1)
        cv.SetZero(self.well)
        cv.SetZero(self.well_mask)
        cv.Circle(self.well_mask, (self.tube_radius, self.tube_radius), self.tube_radius, 255, -1)

        self.decoded_codes = 0
        self.decoded_empty = 0
        for c in range(Params.num_cols):
            for r in range(Params.num_rows):
                # map to box location depending on its orientation
                if self.tabs_on_bottom:
                    box_c = Params.num_cols - c - 1
                    box_r = r
                else:
                    box_c = c
                    box_r = Params.num_rows - r - 1

                # check if already have the code and don't try decoding again
                # we assume if we decoded a code, it's impossible that it's incorrect
                # however, if called something empty before, we could be wrong
                # so we still repeat the empty well test
                if self.codes[box_r][box_c]:
                    self.annotate_well(c, r, Params.annotate_present_color)
                    self.decoded_codes += 1
                    continue

                result = self.decode_code(c, r)
                if result == False:
                    self.codes[box_r][box_c] = False
                    self.decoded_empty += 1
                    self.annotate_well(c, r, Params.annotate_empty_color)
                elif result == None:
                    self.annotate_well(c, r, Params.annotate_not_decoded)
                else:
                    self.codes[box_r][box_c] = result
                    self.decoded_codes += 1
                    self.annotate_well(c, r, Params.annotate_present_color)

    def decode_code(self, col, row):
        # Decode a single code at the given location
        # returns False if empty, None if unknown, or the code if decoded

        offset_x = self.well_x[col] - self.tube_radius
        offset_y = self.well_y[row] - self.tube_radius
        well_rect = (offset_x, offset_y, self.well_size, self.well_size)

        cv.SetImageROI(self.gray, well_rect)

        # use grayscale image and mask out everything but the tube
        cv.Copy(self.gray, self.well, self.well_mask)

        # Get edges
        has_lines = False
        for high_threshold in Params.canny_high_thresholds:
            cv.Canny(self.well, self.edges, high_threshold / Params.canny_low_high_ratio, high_threshold)
            lines = self.find_lines()
            if len(lines) > 0:
                has_lines = True
                for twolines in lines:
                    code = self.decode_code_with_lines(twolines)
                    if code != None:
                        return code
            if len(lines) > Params.lines_giveup_threshold:
                break

        if not has_lines:
            return False # declare well empty
        return None

    def find_lines(self, threshold=Params.hough_lines_threshold):
        # processes the self.well image to find the two solid edge lines of the code

        # find the 2 solid lines of the code
        lines = cv.HoughLines2(self.edges, cv.CreateMemStorage(), cv.CV_HOUGH_STANDARD, Params.hough_lines_distance_resolution, Params.hough_lines_angle_resolution, threshold)

        if len(lines) == 0:
            return []

        angles = {} # bin the result lines together by similarity

        for line in lines:
            rho = int(line[0])
            theta = line[1]

            # bin angles, make between -pi/2..pi/2
            # store perpendicular angles together
            angle = theta
            if angle > Params.half_pi:
                angle -= math.pi
            angle = round(angle, 1)

            if angle < 0:
                nangle = round(angle + Params.half_pi, 1)
            else:
                nangle = angle

            # combine similar angles within 0.1 radians together
            if not nangle in angles:
                higher = round(nangle + 0.1, 1)
                lower = round(nangle - 0.1, 1)
                if higher in angles:
                    nangle = higher
                elif lower in angles:
                    nangle = lower
                else:
                    angles[nangle] = [{}, {}]

            if angle >= 0:
                theta_bin = angles[nangle][0]
            else:
                theta_bin = angles[nangle][1]

            if not rho in theta_bin:
                theta_bin[rho] = []
            theta_bin[rho].append(line)

        #self.debug(self.edges)

        # remove angles that don't have lines going in both directions perpendicularly
        angles = [angles[a] for a in angles if len(angles[a][0]) > 0 and len(angles[a][1]) > 0]

        if len(angles) == 0:
            return []

        # calculate product of every combination for each angle and flatten
        angles = map(lambda x: list(itertools.product(x[0].values(), x[1].values())), angles)

        # flatten and get best angle pairs for each group
        lines = [TwoLines.best_lines(self.well, self.code_min_pixels, pair[0], pair[1]) for a in angles for pair in a]

        lines.sort()

        return lines

    def decode_code_with_lines(self, twolines):
        # Use the solid L edges given by twolines
        # Find the upper right corner

        self.well_code = twolines.crop_and_rotate()
        if self.well_code == None:
            return None

        # image now has solid edges found and rotated with corner in bottom left

        # find the top/right corner

        if self.code_size_range == None:
            size_range = range(self.code_min_pixels, min(self.well_code.cols, self.well_code.rows))
        else:
            size_range = self.code_size_range

        top_dotted = None
        for height in size_range:
            top_dotted = self.find_row_edge(height)
            if top_dotted != None:
                break

        if top_dotted == None:
            return None

        right_dotted = None
        for width in size_range:
            right_dotted = self.find_col_edge(width)
            if right_dotted != None:
                break
        if right_dotted == None:
            return None

        code = self.decode_code_with_coordinates(top_dotted, right_dotted)

        if code != None and self.code_size_range == None:
            lower = min(height, width)
            upper = max(height, width)
            self.code_min_pixels = lower - Params.code_pixel_tolerance
            self.code_size_range = range(lower, upper+1)
            for i in range(1,Params.code_pixel_tolerance):
                self.code_size_range.append(upper + i)
                self.code_size_range.append(lower - i)

        #twolines.annotate(cv.CloneImage(self.image))

        #cv.ShowImage("well", self.well)
        #self.debug(self.well_code)
        return code

    def find_row_edge(self, height):
        if height <= 0 or height > self.well_code.rows:
            return None

        width = min(self.well_code.cols, height + Params.code_pixel_tolerance)
        row = cv.GetRow(self.well_code, self.well_code.rows - height)
        row = cv.GetSubRect(row, (0, 0, width, 1))

        # use Otsu auto-threshold to convert to binary data
        # and search for the dotted pattern
        tmp = cv.CreateMat(1, row.cols, cv.CV_8UC1)
        cv.Threshold(row, tmp, 0, 255, cv.CV_THRESH_BINARY + cv.CV_THRESH_OTSU)
        data = np.asarray(tmp)[0] > 0
        return self.calc_dottedness(data)

    def find_col_edge(self, width):
        if width < 0 or width >= self.well_code.cols:
            return None

        height = min(self.well_code.rows, width + Params.code_pixel_tolerance)
        col = cv.GetCol(self.well_code, width)
        col = cv.GetSubRect(col, (0, col.rows-height, 1, height))

        tmp = cv.CreateMat(col.rows, 1, cv.CV_8UC1)
        cv.Threshold(col, tmp, 0, 255, cv.CV_THRESH_BINARY + cv.CV_THRESH_OTSU)
        data = np.fliplr(np.rot90(np.asarray(tmp)))[0] > 0
        return self.calc_dottedness(data)

    def decode_code_with_coordinates(self, top_dotted, right_dotted):
        # Note coordinates for right dotted are from the bottom of the image

        height = right_dotted[-1] + 5 # need to add at least 3, add a bit more to aid debugging
        width = top_dotted[-1] + 5
        data = cv.GetSubRect(self.well_code, (0, self.well_code.rows-height, width, height))

        tmp = cv.CreateMat(data.rows, data.cols, cv.CV_8UC1)
        cv.Threshold(data, tmp, 0, 255, cv.CV_THRESH_BINARY + cv.CV_THRESH_OTSU)

        # convert to boolean matrix
        data = np.asarray(tmp) > 0

        # calculate value for every cell
        # we take the average of a 3x3 square around each pixel
        bits = []
        for r in range(Params.matrix_code_size-3,-1,-1):
            rowbits = []
            for c in range(Params.matrix_code_size-2):
                y = height - right_dotted[r] - 1
                x = top_dotted[c]
                cell = data[y-1:y+2,x-1:x+2]

                rowbits.append(np.count_nonzero(cell) > 4)
            bits.append(rowbits)

        # print bits
        # self.debug(tmp)
        return self.datamatrix_decode(bits)

    def calc_dottedness(self, data):
        # expects row to start with True (solid edge)
        # returns means indicating center coordinates

        idx = self.find_runs(data)
        if len(idx) < Params.matrix_code_size / 2:
            # not a dotted row
            return None

        # we skip the first true range which should be the solid edge
        means = []
        ranges = []
        min_range = idx[0][1] - idx[0][0]
        max_range = idx[0][1] - idx[0][0]
        for i in range(Params.matrix_code_size/2-1):
            false_begin = idx[i][1]
            false_end = idx[i+1][0] - 1
            true_begin = idx[i+1][0]
            true_end = idx[i+1][1] - 1
            means.append((false_begin + false_end) / 2)
            means.append((true_begin + true_end) / 2)
            false_range = false_end - false_begin + 1
            true_range = true_end - true_begin + 1
            min_range = min(min_range, true_range, false_range)
            max_range = max(max_range, true_range, false_range)
            if max_range > min_range + Params.dotted_pixel_range:
                return None
        if means[-1] < self.code_min_pixels - Params.code_pixel_tolerance:
            return None # too small to be the code
        return means

    def annotate_well(self, x, y, color):
        # annotate a particular well with some color circle
        cv.Circle(self.info_image, (self.well_x[x], self.well_y[y]), self.tube_radius + 15, color, 15)

    def datamatrix_decode(self, bits):
        try:
            code = datamatrix.decode(bits)
        except Exception, e:
            #print e
            return None

        if not Params.code_regex.match(code):
            #print "datamatrix code returned is incorrect", code
            return None
        return code

    #
    # Utility routines
    #

    def find_threshold(self, data, low, high, target):
        # data is numpy array
        # finds a threshold between low and high
        # such that find_runs returns target
        while (high - low) > 2:
            threshold = (low + high) / 2
            condition = data > threshold
            grouped = self.find_runs(condition)
            if len(grouped) == target:
                return grouped
            elif len(grouped) < target:
                # we make assumption here about direction of threshold with number of runs
                high = threshold
            else:
                low = threshold

        return None

    def find_runs(self, condition):
        # returns the start index and end index+1 of all runs
        # in condition (numpy array) that are true
        d = np.diff(condition)
        idx, = d.nonzero()

        idx += 1
        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size]
        idx.shape = (-1,2)
        return idx

    def longest_contiguous(self, condition):
        # returns the start and end indices of the longest contiguous block
        # in condition (numpy array) that is true
        idx = self.find_runs(condition)
        block_size = idx[:,1] - idx[:,0]
        if len(block_size) == 0:
            return (None, None)
        i = np.argmax(block_size)
        return tuple(idx[i])

    def max_contour_area(self, contours):
        max_contour = None
        max_contour_area = -1
        while contours != None and len(contours) != 0:
            area = cv.ContourArea(contours)
            if area > max_contour_area:
                max_contour = contours
                max_contour_area = area
            contours = contours.h_next()
        return (max_contour, max_contour_area)

    def crop_and_rotate(self, image, rect):
        # rect should be a RotatedRect as returned by MinAreaRect2
        # we make the width of the new image be the longer side

        (height, width) = rect[1]
        if height > width:
            (width, height) = rect[1]
        width = cv.Round(width)
        height = cv.Round(height)

        points = cv.BoxPoints(rect)
        # the points are in clockwise order around rectangle
        # figure out which is top-left, bottom-left, top-right
        # check which is the longer side
        if math.hypot(points[0][0]-points[1][0], points[0][1]-points[1][1]) > math.hypot(points[1][0]-points[2][0], points[1][1]-points[2][1]):
            tl = points[0]
            tr = points[1]
            bl = points[3]
        else:
            tl = points[1]
            tr = points[2]
            bl = points[0]

        # rotate/deskew the box
        trans = cv.CreateMat(2, 3, cv.CV_32FC1)
        cv.GetAffineTransform((bl, tl, tr), ((0, height-1), (0,0), (width-1, 0)), trans)
        rotated = cv.CreateImage((width, height), image.depth, image.nChannels)
        cv.WarpAffine(image, rotated, trans)
        return rotated

    def debug_resize(self, image=None):
        if image == None:
            image = self.image
        resized = cv.CreateImage((image.width / 4, image.height / 4), image.depth, image.nChannels)
        cv.Resize(image, resized)
        self.debug(resized)

    def debug(self, image=None):
        if image == None:
            image = self.image
        cv.ShowImage("debug", image)
        cv.WaitKey()

