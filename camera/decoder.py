import sys
import cv
import math
import csv

import datamatrix
from params import Params

class BoxScanner:
    def __init__(self):
        self.codes = [[None for j in range(Params.num_cols)] for i in range(Params.num_rows)]
        self.image = None
        self.info_image = None

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

    def last_image(self):
        return self.info_image

    def scan(self, image):
        self.image = image
        self.info_image = image # leave the info image as the original until we know we've detected a box

        if not self.find_box_and_rotate():
            print "Box not found!"
            return False

        if not self.find_orientation():
            print "Box orientation detection failed!"
            return False

        self.info_image = self.image

        (count, empty) = self.decode_codes()
        print self.codes
        print "Wells done:", count, "empty:", empty, "codes:", count - empty, "unknown:", Params.num_cols * Params.num_rows - count

        if count == Params.num_cols * Params.num_rows:
            return True
        else:
            return False
        
    def decode_code(self, image):
        # image should contain a single code
        # The code should be rotated so the edges are horizontal/vertical
        # Image should be mostly cropped but there may still be some extra margins

        # find the L corner by finding the intersection of the brightest column and row
        
        col_proj = cv.CreateMat(1, image.width, cv.CV_32FC1)
        cv.Reduce(image, col_proj, 0, cv.CV_REDUCE_AVG)
        row_proj = cv.CreateMat(image.height, 1, cv.CV_32FC1)
        cv.Reduce(image, row_proj, 1, cv.CV_REDUCE_AVG)
        (_, _, _, max_col) = cv.MinMaxLoc(col_proj)
        (_, _, _, max_row) = cv.MinMaxLoc(row_proj)

        l_corner_x = max_col[0]
        l_corner_y = max_row[1]

        # try to find the edges of the code
        # We start from the L-corner and expand the two L edges out as long as the row/col is bright
        # We then expand the other two edges while the brightness of the row/col is good
        
        if l_corner_x < image.width / 2:
            if l_corner_y < image.height / 2:
                # L-corner in top left
                corner_top = True
                corner_left = True
                
                top = self.find_horizontal_edge(image, (l_corner_x, image.width-1), (l_corner_y, 0), Params.edge_min_pixels_solid)
                left = self.find_vertical_edge(image, (l_corner_x, 0), (top, image.height-1), Params.edge_min_pixels_solid)

                size = min(image.width - left, image.height - top)

                bottom = self.find_horizontal_edge(image, (left, left+size-1), (top, image.height-1), Params.code_min_pixels_slice)
                right = self.find_vertical_edge(image, (left, image.width-1), (top, bottom), Params.code_min_pixels_slice)
            else:
                # L-corner in bottom left
                corner_top = False
                corner_left = True
                
                bottom = self.find_horizontal_edge(image, (l_corner_x, image.width-1), (l_corner_y, image.height-1), Params.edge_min_pixels_solid)
                left = self.find_vertical_edge(image, (l_corner_x, 0), (0, bottom), Params.edge_min_pixels_solid)

                size = min(image.width - left, bottom + 1)

                top = self.find_horizontal_edge(image, (left, left+size-1), (bottom, 0), Params.code_min_pixels_slice)
                right = self.find_vertical_edge(image, (left, image.width-1), (top, bottom), Params.code_min_pixels_slice)

        else:
            if l_corner_y < image.height / 2:
                # L-corner in top right
                corner_top = True
                corner_left = False

                top = self.find_horizontal_edge(image, (0, l_corner_x), (l_corner_y, 0), Params.edge_min_pixels_solid)
                right = self.find_vertical_edge(image, (l_corner_x, image.width-1), (top, image.height-1), Params.edge_min_pixels_solid)

                size = min(right + 1, image.height - top)

                bottom = self.find_horizontal_edge(image, (right-size+1, right), (top, image.height-1), Params.code_min_pixels_slice)
                left = self.find_vertical_edge(image, (right, 0), (top, bottom), Params.code_min_pixels_slice)
                
            else:
                # L-corner in bottom right
                corner_top = False
                corner_left = False
                
                bottom = self.find_horizontal_edge(image, (0, l_corner_x), (l_corner_y, image.height-1), Params.edge_min_pixels_solid)
                right = self.find_vertical_edge(image, (l_corner_x, image.width-1), (0, bottom), Params.edge_min_pixels_solid)

                size = min(right + 1, bottom + 1)

                top = self.find_horizontal_edge(image, (right-size+1, right), (bottom, 0), Params.code_min_pixels_slice)
                left = self.find_vertical_edge(image, (right, 0), (top, bottom), Params.code_min_pixels_slice)


        width = right - left + 1
        height = bottom - top + 1

        # check if width/height are reasonable
        if width < 0 or height < 0 or abs(width - height) > Params.code_squareness_deviation:
            return None

        pixel_width = 1.0 * width / Params.matrix_code_size
        pixel_height = 1.0 * height / Params.matrix_code_size

        if min(pixel_width, pixel_height) < Params.min_pixels_per_cell:
            return None

        ipixel_width = int(pixel_width)
        ipixel_height = int(pixel_height)

        # threshold of whether a cell is on or off
        threshold = ipixel_width * ipixel_height * 255 * Params.cell_pixel_threshold

        # calculate value for every cell
        # we skip the edges and assume they are correct
        bits = []
        for r in range(1, Params.matrix_code_size-1):
            rowbits = []
            for c in range(1, Params.matrix_code_size-1):
                # map to correct orientation. The L-corner should be in bottom left
                if corner_top and corner_left:
                    mapped_c = Params.matrix_code_size - r - 1
                    mapped_r = c
                elif corner_top and not corner_left:
                    mapped_c = Params.matrix_code_size - c - 1
                    mapped_r = Params.matrix_code_size - r - 1
                elif not corner_top and not corner_left:
                    mapped_c = r
                    mapped_r = Params.matrix_code_size - c - 1
                else:
                    mapped_c = c
                    mapped_r = r
                    
                rect = (int(math.ceil(left + mapped_c * pixel_width)), int(math.ceil(top + mapped_r * pixel_height)), ipixel_width, ipixel_height)
                intensity = self.cell_intensity(image, rect)
                rowbits.append(intensity > threshold)
            bits.append(rowbits)

        try:
            code = datamatrix.decode(bits)
        except:
            return None

        if len(code) != Params.code_decoded_length:
            print "datamatrix code returned is of the wrong length", code
            return None

        return code

    def find_horizontal_edge(self, image, (x_start, x_end), (y_start, y_end), threshold):
        # goes from [y_start..y_end] and counts the num of pixels on between [x_start..x_end] for each row
        # returns the last row that has a count > the threshold
        direction = 1 if y_start < y_end else -1
        width = x_end - x_start + 1
        for y in range(y_start, y_end + direction, direction):
            cv.SetImageROI(image, (x_start, y, width, 1))
            if cv.CountNonZero(image) < threshold:
                return y - direction # return the previous index
        return y_end

    def find_vertical_edge(self, image, (x_start, x_end), (y_start, y_end), threshold):
        # same as find_horizontal_edge but in vertical direction
        direction = 1 if x_start < x_end else -1
        height = y_end - y_start + 1
        cv.SetImageROI(image, (x_start, y_start, 1, height))
        for x in range(x_start, x_end + direction, direction):
            cv.SetImageROI(image, (x, y_start, 1, height))
            if cv.CountNonZero(image) < threshold:
                return x - direction # return the previous index
        return x_end
    
    def cell_intensity(self, image, rect):
        cv.SetImageROI(image, rect)
        return cv.Sum(image)[0]

    def fill_from_all_points(self, image, x_range, y_range):
        for i in x_range:
            for j in y_range:
                if cv.Get2D(image, j, i)[0] > 0:
                    cv.FloodFill(image, (i, j), (0, 0, 0), Params.box_fill_threshold, Params.box_fill_threshold)

    def annotate_image(self, rect, color):
        # annotate a particular well with some color rectangle
        # we add some margin around it so they don't overlap with nearby well
        cv.SetImageROI(self.image, self.inner_rect)
        cv.Rectangle(self.image, (rect[0]+25, rect[1]+25), (rect[0]+rect[2]-25, rect[1]+rect[3]-25), color, 25)
                     
    def decode_codes(self):
        cv.SetImageROI(self.image, self.inner_rect)

        # threshold/convert to black/white
        image = self.threshold(self.image)

        # get rid of the box
        # we assume box is located on all 4 edges and we flood fill from all points
        self.fill_from_all_points(image, range(image.width), [0]) # top
        self.fill_from_all_points(image, range(image.width), [image.height-1]) # bottom
        self.fill_from_all_points(image, [0], range(image.height)) # left
        self.fill_from_all_points(image, [image.height-1], range(image.height)) # right

        # process it a bit to help contour finding
        search_img = cv.CloneImage(image)
            
        cv.Erode(search_img, search_img) # remove background noise
        cv.Dilate(search_img, search_img, iterations=3) # make the code image brighter and edges bigger

        count = 0
        empty = 0
        for c in range(Params.num_cols):
            for r in range(Params.num_rows):
                # map to box location depending on its orientation
                if self.tabs_on_bottom:
                    box_c = Params.num_cols - c - 1
                    box_r = r
                else:
                    box_c = c
                    box_r = Params.num_rows - r - 1

                offset_x = c * self.well_size
                offset_y = r * self.well_size
                well_rect = (offset_x, offset_y, self.well_size, self.well_size)

                # check if already have the code
                if self.codes[box_r][box_c] != None:
                    if not self.codes[box_r][box_c]:
                        self.annotate_image(well_rect, Params.annotate_empty_color)
                        empty += 1
                    else:
                        self.annotate_image(well_rect, Params.annotate_present_color)                        
                    count += 1
                    continue
                
                
                well = cv.GetSubRect(search_img, well_rect)
                if cv.CountNonZero(well) < Params.min_code_size_pixels:
                    # call the well empty
                    self.codes[box_r][box_c] = False
                    empty += 1
                    count += 1
                    self.annotate_image(well_rect, Params.annotate_empty_color)
                    continue

                # find the code in the well
                contours = cv.FindContours(well, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL, offset=(offset_x, offset_y))
                success = False
                while contours != None and len(contours) != 0:
                    rect = cv.MinAreaRect2(contours)

                    (height, width) = rect[1]
                    area = height * width
                    if area > Params.min_code_size_pixels:
                        # due to previous dilation operation, we expect the rectangle to be larger than the code
                        well_img = self.crop_and_rotate(image, rect)
                        code = self.decode_code(well_img)
                        if code != None:
                            self.codes[box_r][box_c] = code
                            count += 1
                            self.annotate_image(well_rect, Params.annotate_present_color)
                            success = True
                            break

                    contours = contours.h_next()
                if not success:
                    self.annotate_image(well_rect, Params.annotate_not_decoded)

        return (count, empty)
    
    def find_box_and_rotate(self):
        # finds the box and rotate it to a standard orientation
        
        bwimg = self.threshold(self.image)
        cv.Erode(bwimg, bwimg) # this helps to reduce noise a bit
        
        # find the bounding rectangle for the on pixels which should be the box outline
        contours = cv.FindContours(bwimg, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL)
        if len(contours) == 0:
            return False
        (max_contour, max_contour_area) = self.max_contour_area(contours)
        rect = cv.MinAreaRect2(max_contour)

        #center = rect[0]
        (height, width) = rect[1]
        #angle = rect[2]

        # if area is too large, then there's likely no box and we selected the entire image
        if height * width > Params.box_max_area * bwimg.width * bwimg.height:
            return False
            
        if height > width:
            (width, height) = rect[1]

        if height < Params.min_pixels_per_well * Params.num_rows or width < Params.min_pixels_per_well * Params.num_cols:
            return False

        self.image = self.crop_and_rotate(self.image, rect)
        return True
    
    def find_orientation(self):
        # find the box tabs to determine orientation

        hsv = cv.CreateImage(cv.GetSize(self.image), 8, 3)
        cv.CvtColor(self.image, hsv, cv.CV_BGR2HSV)

        # this is maximum size of a single well as it doesn't include the border
        width = hsv.width / Params.num_cols

        right_border = hsv.width - width
        
        # if the image includes the full box, the extra horizontal width is about equal to one well width on each side
        # so the following image extracts will definitely include the tabs if they exist
        middle = self.image.height / 2
        tl = self.histogram_search(hsv, (0, 0, width, middle), Params.tab_histogram)
        bl = self.histogram_search(hsv, (0, middle, width, middle), Params.tab_histogram)
        tr = self.histogram_search(hsv, (right_border, 0, width, middle), Params.tab_histogram)
        br = self.histogram_search(hsv, (right_border, middle, width, middle), Params.tab_histogram)

        cutoff = width * middle * Params.tab_pixel_cutoff
        tl_count = cv.CountNonZero(tl)
        bl_count = cv.CountNonZero(bl)
        tr_count = cv.CountNonZero(tr)
        br_count = cv.CountNonZero(br)
        if tl_count < cutoff and tr_count < cutoff and bl_count > cutoff and br_count > cutoff:
            self.tabs_on_bottom = True
            left = bl
            right = br
        elif tl_count > cutoff and tr_count > cutoff and bl_count < cutoff and br_count < cutoff:
            self.tabs_on_bottom = False            
            left = tl
            right = tr
        else:
            return False # no tabs found, fail as we cannot determine box orientation

        # find edge of tab and use as edge of wells
        left_border = 0
        dst = cv.CreateMat(1, left.width, cv.CV_32FC1)
        cv.Reduce(left, dst, 0, cv.CV_REDUCE_SUM)
        avg = cv.Avg(dst)
        for i in xrange(dst.width-1, -1, -1):
            if cv.Get1D(dst, i) > avg:
                left_border = i
                break
        cv.Reduce(right, dst, 0, cv.CV_REDUCE_SUM)
        avg = cv.Avg(dst)        
        for i in xrange(dst.width):
            if cv.Get1D(dst, i) > avg:
                right_border = right_border + i
                break

        # figure out well size
        # this is a crude estimate
        self.well_size = (right_border - left_border) / Params.num_cols

        if self.well_size < Params.min_pixels_per_well:
            return False

        # to remove the top and bottom borders, we assume that the entirety of the box there is displayed
        # and they are equal sizes and that all wells are perfectly square
        box_height = self.well_size * Params.num_rows
        if box_height > self.image.height:
            return False

        top_border = (self.image.height - box_height) / 2
        bottom_border = box_height + top_border

        self.inner_rect = (left_border, top_border, right_border-left_border, bottom_border-top_border)
        return True

    def histogram_search(self, hsv, rect, histogram):
        cv.SetImageROI(hsv, rect)
        hue = cv.CreateImage(cv.GetSize(hsv), cv.IPL_DEPTH_8U, 1)
        sat = cv.CreateImage(cv.GetSize(hsv), cv.IPL_DEPTH_8U, 1)

        cv.SetImageCOI(hsv, 1)
        cv.Copy(hsv, hue)
        cv.SetImageCOI(hsv, 2)
        cv.Copy(hsv, sat)

        cv.SetImageCOI(hsv, 0)
        cv.ResetImageROI(hsv)
        
        match = cv.CreateImage(cv.GetSize(hue), 8, 1)
        cv.CalcBackProject([hue, sat], match, histogram)
        cv.Threshold(match, match, Params.histogram_threshold, 255, cv.CV_THRESH_BINARY)        
        return match

    def max_contour_area(self, contours):
        max_contour = None
        max_contour_area = -1
        while contours != None:
            area = cv.ContourArea(contours)
            if area > max_contour_area:
                max_contour = contours
                max_contour_area = area
            contours = contours.h_next()
        return (max_contour, max_contour_area)


    def threshold(self, image):
        # threshold/convert color image to black/white
        bwimg = cv.CreateImage(cv.GetSize(image), 8, 1)
        cv.InRangeS(image, Params.white_threshold, (255,255,255), bwimg)
        return bwimg
                
    def crop_and_rotate(self, image, rect):
        # rect should be a RotatedRect as returned by MinAreaRect2
        # we make the width of the new image be the longer side

        (height, width) = rect[1]
        if height > width:
            (width, height) = rect[1]
        width = int(width)
        height = int(height)
        
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
        cv.GetAffineTransform((bl, tl, tr), ((0, height), (0,0), (width, 0)), trans)
        rotated = cv.CreateImage((width, height), image.depth, image.nChannels)
        cv.WarpAffine(image, rotated, trans)
        return rotated
        
