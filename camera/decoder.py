import cv
import math
import numpy

import datamatrix
from params import Params

class BoxScanner:
    def __init__(self, img):
        self.image = img

    def scan(self):
        if not (self.find_box_and_rotate() and self.find_orientation()):
            return False
        
        self.decode_codes()
        return True

    def decode_code(self, image):
        # image should be a rotated, cropped image
        # but the L edge has not been located yet

        # sometimes we get some extra rows/cols
        # truncate the image on all four sides so that all edges have at least > half of pixels on
        width = 0
        height = 0
        x = 0
        y = 0

        # truncate top
        for r in range(image.height):
            cv.SetImageROI(image, (0, r, image.width, 1))
            if cv.CountNonZero(image) > Params.code_edge_min_pixels * image.width:
                y = r
                break

        # truncate bottom
        for r in range(image.height-1, -1, -1):
            cv.SetImageROI(image, (0, r, image.width, 1))
            if cv.CountNonZero(image) > Params.code_edge_min_pixels * image.width:
                height = r - y
                break

        # truncate left
        for c in range(image.width):
            cv.SetImageROI(image, (c, 0, 1, image.height))
            if cv.CountNonZero(image) > Params.code_edge_min_pixels * image.height:
                x = c
                break

        # truncate right
        for c in range(image.width-1, -1, -1):
            cv.SetImageROI(image, (c, 0, 1, image.height))
            if cv.CountNonZero(image) > Params.code_edge_min_pixels * image.height:
                width = c - x
                break

        # check if width/height are reasonable
        if abs(width - height) > Params.code_squareness_deviation:
            return None
        
        pixel_width = 1.0 * width / Params.matrix_code_size
        pixel_height = 1.0 * height / Params.matrix_code_size
        
        if pixel_width < Params.min_pixels_per_cell or pixel_height < Params.min_pixels_per_cell:
            return None

        ipixel_width = int(pixel_width)
        ipixel_height = int(pixel_height)

        # threshold of whether a cell is on or off
        threshold = ipixel_width * ipixel_height * 255 * Params.cell_pixel_threshold

        # calculate value for every cell
        bits = []
        for r in range(Params.matrix_code_size):
            rowbits = []
            for c in range(Params.matrix_code_size):
                rect = (int(math.ceil(x + c * pixel_width)), int(math.ceil(y + r * pixel_height)), ipixel_width, ipixel_height)
                intensity = self.cell_intensity(image, rect)
                rowbits.append(intensity > threshold)
            bits.append(rowbits)

        # determine which edges are solid/every other
        top_type = self.edge_type(bits, [0], range(Params.matrix_code_size))
        bottom_type = self.edge_type(bits, [Params.matrix_code_size-1], range(Params.matrix_code_size))
        left_type = self.edge_type(bits, range(Params.matrix_code_size), [0])
        right_type = self.edge_type(bits, range(Params.matrix_code_size), [Params.matrix_code_size-1])

        if top_type == None or bottom_type == None or left_type == None or right_type == None:
            return None

        # remove edges (get internal data) bits and convert to numpy aray
        bits = numpy.array(bits)[1:Params.matrix_code_size-1,1:Params.matrix_code_size-1]

        # find the L corner
        # rotate counter clockwise so the L corner is at bottom left
        if top_type == 1: # top is solid
            if left_type == 1:
                bits = numpy.rot90(bits, 1)
            elif right_type == 1:
                bits = numpy.rot90(bits, 2)
            else:
                return None
        else: # top is alternating
            if left_type == 0:
                bits = numpy.rot90(bits, 3)                
            elif right_type == 0:
                pass # no rotation needed
            else:
                return None

        bits = bits.tolist() # back to list to call datamatrix decoder
        try:
            code = datamatrix.decode(bits)
        except:
            return None

        return code
    
    def edge_type(self, bits, row_range, col_range):
        # returns 1 if all on, 0 if alternating and None if neither
        last = None
        num_on = 0
        num_alternating = 0
        for r in row_range:
            for c in col_range:
                if bits[r][c]:
                    num_on += 1
                if last != None and bits[r][c] != last:
                    num_alternating += 1
                last = bits[r][c]

        if num_alternating > Params.edge_cell_threshold:
            return 0
        if num_on > Params.edge_cell_threshold:
            return 1
        return None
        
    def cell_intensity(self, image, rect):
        cv.SetImageROI(image, rect)
        return cv.Sum(image)[0]
    
    def decode_codes(self):
        cv.SetImageROI(self.image, self.inner_rect)

        # threshold/convert to black/white
        bwimg = cv.CreateImage(cv.GetSize(self.image), 8, 1)
        cv.InRangeS(self.image, (200, 200, 200), (255,255,255), bwimg)
        
        # get rid of the box
        # we assume box is located on all 4 edges and we flood fill from all
        # from the maximum point
        cv.SetImageROI(bwimg, (0, 0, bwimg.width, 1))
        (_, _, _, top_loc) = cv.MinMaxLoc(bwimg)
        cv.SetImageROI(bwimg, (0, bwimg.height-1, bwimg.width, 1))
        (_, _, _, bottom_loc) = cv.MinMaxLoc(bwimg)
        cv.SetImageROI(bwimg, (0, 0, 1, bwimg.height))
        (_, _, _, left_loc) = cv.MinMaxLoc(bwimg)
        cv.SetImageROI(bwimg, (bwimg.height-1, 0, 1, bwimg.height))
        (_, _, _, right_loc) = cv.MinMaxLoc(bwimg)
        
        cv.ResetImageROI(bwimg)
        cv.FloodFill(bwimg, top_loc, (0, 0, 0), (128, 128, 128), (128, 128, 128))
        cv.FloodFill(bwimg, (bottom_loc[0], bwimg.height-1), (0, 0, 0), (128, 128, 128), (128, 128, 128))
        cv.FloodFill(bwimg, left_loc, (0, 0, 0), (128, 128, 128), (128, 128, 128))
        cv.FloodFill(bwimg, (bwimg.height-1, right_loc[1]), (0, 0, 0), (128, 128, 128), (128, 128, 128))

        # blur it a bit to help contour finding
        smoothed = cv.CreateImage(cv.GetSize(bwimg), 8, 1)
        cv.Smooth(bwimg, smoothed)

        count = 0
        for c in range(Params.num_cols):
            for r in range(Params.num_rows):
                offset_x = c * self.well_size
                offset_y = r * self.well_size
                
                well = cv.GetSubRect(smoothed, (offset_x, offset_y, self.well_size, self.well_size))
                contours = cv.FindContours(well, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL, offset=(offset_x, offset_y))
                
                code = None
                while contours != None and len(contours) != 0:
                    rect = cv.MinAreaRect2(contours)

                    (height, width) = rect[1]
                    area = height * width
                    if area > Params.min_code_size_pixels:
                        well_img = self.crop_and_rotate(bwimg, rect)
                        code = self.decode_code(well_img)
                        if code != None:
                            count += 1
                            break

                    contours = contours.h_next()

        print "Decoded images: ", count
        #cv.ResetImageROI(self.image)
        #cv.ShowImage("processed", self.image)
        #cv.WaitKey(0)
    
    def find_box_and_rotate(self):
        # finds the box and rotate it to a standard orientation
        
        # We assume some part of the box exists at the center of the image
        # For speed, we crop the image and only search that region for some portion of the box
        # this size should be > than maximum possible size of one square in the box
        size = int(max(self.image.width, self.image.height) / Params.num_cols)
        x = int((self.image.width - size) / 2)
        y = int((self.image.height - size) / 2)
        rect = (x, y, size, size)
        
        hsv = cv.CreateImage(cv.GetSize(self.image), 8, 3)
        cv.CvtColor(self.image, hsv, cv.CV_BGR2HSV)

        (mask, area) = self.fill_find(hsv, rect, Params.box_min_area)
        # if area is too large, then there's likely no box and we selected the entire image
        if mask == None or area > Params.box_max_area * hsv.width * hsv.height:
            return False

        cv.Erode(mask, mask, iterations=5) # this helps to reduce noise
        
        # find the bounding rectangle for the filled pixels which should be the box outline
        contours = cv.FindContours(mask, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL)
        (max_contour, max_contour_area) = self.max_contour_area(contours)
        rect = cv.MinAreaRect2(max_contour)

        #center = rect[0]
        (height, width) = rect[1]
        if height > width:
            (width, height) = rect[1]
        #angle = rect[2]

        if height < Params.min_pixels_per_well * Params.num_rows or width < Params.min_pixels_per_well * Params.num_cols:
            return False

        self.image = self.crop_and_rotate(self.image, rect)
        self.hsv = cv.CreateImage(cv.GetSize(self.image), 8, 3)
        cv.CvtColor(self.image, self.hsv, cv.CV_BGR2HSV)

        return True
    
    def find_orientation(self):
        # find the box tabs to determine orientation

        # this is maximum size of a single well as it doesn't include the border
        width = self.hsv.width / Params.num_cols

        right_border = self.hsv.width - width
        
        # if the image includes the full box, the extra horizontal width is about equal to one well width on each side
        # so the following image extracts will definitely include the tabs if they exist
        middle = self.image.height / 2
        tl = self.histogram_search(self.hsv, (0, 0, width, middle), Params.tab_histogram)
        bl = self.histogram_search(self.hsv, (0, middle, width, middle), Params.tab_histogram)
        tr = self.histogram_search(self.hsv, (right_border, 0, width, middle), Params.tab_histogram)
        br = self.histogram_search(self.hsv, (right_border, middle, width, middle), Params.tab_histogram)

        cutoff = width * middle * Params.tab_min_area
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
        return match

    def fill_find(self, hsv, rect, min_area_fraction):
        # flood fill every pixel in turn until the area is greater than the min
        mask = cv.CreateMat(hsv.height + 2, hsv.width + 2, cv.CV_8UC1)

        min_area = min_area_fraction * hsv.width * hsv.height
        for x in xrange(rect[2]):
            for y in xrange(rect[3]):
                (area, average_color, rect) = cv.FloodFill(hsv, (rect[0] + x, rect[1] + y), (0, 0, 0), (Params.flood_fill_hue, Params.flood_fill_sat, 255), (Params.flood_fill_hue, Params.flood_fill_sat, 255), cv.CV_FLOODFILL_FIXED_RANGE + cv.CV_FLOODFILL_MASK_ONLY, mask)

                # check if we likely selected what we wanted
                if area > min_area:
                    mask = cv.GetSubRect(mask, (1, 1, hsv.width, hsv.height))
                    return (mask, area)

        return (None, None)

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
        trans = cv.CreateMat(2, 3, cv.CV_32FC1 )
        cv.GetAffineTransform((bl, tl, tr), ((0, height), (0,0), (width, 0)), trans)
        rotated = cv.CreateImage((width, height), image.depth, image.nChannels)
        cv.WarpAffine(image, rotated, trans)
        return rotated
        
