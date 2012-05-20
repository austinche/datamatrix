import cv
import math
import csv
import numpy as np

import datamatrix
from params import Params

class BoxScanner:
    FONT = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1.0, 1.0)

    def __init__(self):
        self.codes = [[None for j in range(Params.num_cols)] for i in range(Params.num_rows)]
        self.image = None
        self.info_image = None
        self.decode_info = (0, 0)

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
        cv.ResetImageROI(image)

        caption_height = 40

        # image can be different sizes depending on if it's been cropped yet
        # we shrink to fixed size and add a caption
        shrink = cv.CreateImage((width, height), image.depth, image.nChannels)
        cv.SetImageROI(shrink, (0, 0, width, height  - caption_height))
        cv.Resize(image, shrink)

        # add caption text
        cv.SetImageROI(shrink, (0, height - caption_height, width, caption_height))
        cv.Set(shrink, (255, 255, 255))
        (count, empty) = self.decode_info
        cv.PutText(shrink, "Unknown: %d " % (96 - count), (0, 20), BoxScanner.FONT, Params.annotate_not_decoded)
        cv.PutText(shrink, "Empty: %d " % empty, (120, 20), BoxScanner.FONT, Params.annotate_empty_color)
        cv.PutText(shrink, "Codes: %d " % (count - empty), (220, 20), BoxScanner.FONT, Params.annotate_present_color)
        cv.ResetImageROI(shrink)
        return shrink

    def scan(self, image):
        # returns count of wells decoded

        self.image = image

        if not self.find_box_and_rotate():
            print "Box not found!"
            self.info_image = image
            return 0

        if not self.find_orientation():
            print "Box orientation detection failed!"
            self.info_image = image
            return 0

        self.decode_info = self.decode_codes()
        (count, empty) = self.decode_info
        self.info_image = self.image # for consistency, we need to set this image after the above decoding annotates everything

        print self.codes
        print "Wells done:", count, "empty:", empty, "codes:", count - empty, "unknown:", Params.num_cols * Params.num_rows - count

        return count

    def datamatrix_decode(self, bits):
        try:
            code = datamatrix.decode(bits)
        except Exception, e:
            #print e
            return None

        if len(code) != Params.code_decoded_length:
            print "datamatrix code returned is of the wrong length", code
            return None
        return code

    def decode_code_2(self, image):
        # This method seems to decode some tubes that the other method doesn't
        # but in general seems to do worse than the other one

        # image should be black/white and contain a single code
        # The code should be rotated so the edges are horizontal/vertical
        # Image should be mostly cropped but there may still be some extra margins

        # we use numpy to do the processing
        # convert to boolean matrix
        # use GetSubRect to convert to Mat
        cv.ResetImageROI(image)
        data = np.asarray(cv.GetSubRect(image, (0, 0, image.width, image.height))) > 0

        # find longest contiguous vertical and horizontal block
        # that forms the largest square

        solid_rows = self.find_solid_edge(data)
        solid_cols = self.find_solid_edge(data.transpose())

        # find the best combination of row/col to make the largest code
        best = None
        for r in solid_rows:
            for c in solid_cols:
                # the row/col must intersect with each other
                if c[0] < r[1][0] or c[0] > r[1][1]:
                    continue
                if r[0] < c[1][0] or r[0] > c[1][1]:
                    continue
                w = max(c[0] - r[1][0], r[1][1] - c[0])
                h = max(r[0] - c[1][0], c[1][1] - r[0])
                if w < Params.edge_min_pixels_solid or h < Params.edge_min_pixels_solid:
                    continue
                size = w * h
                if best == None or size > best[0]:
                    best = (size, r, c)

        if best == None:
            return None

        # crop and rotate the L corner to the bottom left
        row = best[1]
        col = best[2]
        if col[0] - row[1][0] > row[1][1] - col[0]:
            # l_corner on right
            col_low = row[1][0]
            col_high = col[0]
            on_left = False
        else:
            # l_corner on left
            col_low = col[0]
            col_high = row[1][1]
            on_left = True
        if row[0] - col[1][0] > col[1][1] - row[0]:
            # l_corner on bottom
            row_low = col[1][0]
            row_high = row[0]
            if on_left:
                rotation = 0
            else:
                rotation = 3
        else:
            # l_corner on top
            row_low = row[0]
            row_high = col[1][1]
            if on_left:
                rotation = 1
            else:
                rotation = 2

        data = data[row_low:row_high+1,col_low:col_high+1]
        if rotation > 0:
            data = np.rot90(data, rotation)

        # find the dotted edges
        dotted_row = self.find_dotted_edge(data)
        dotted_col = self.find_dotted_edge(np.rot90(data))
        if dotted_row == None or dotted_col == None:
            return None

        # calculate value for every cell
        # we take the average of a 3x3 square around each pixel
        bits = []
        for r in range(len(dotted_row)):
            rowbits = []
            for c in range(len(dotted_col)):
                # note: the dotted_row array gives column coordinates
                # and vice-versa for the dotted_col array
                y = dotted_col[r]
                x = dotted_row[c]
                cell = data[y-1:y+2,x-1:x+2]

                rowbits.append(np.count_nonzero(cell) > 4)
            bits.append(rowbits)

        return self.datamatrix_decode(bits)


    def find_solid_edge(self, data):
        # finds the longest solid edge along the rows
        (height, width) = data.shape
        longest = []
        for i in range(height):
            row = data[i,:]
            (start, end) = self.longest_contiguous(row)
            if start == None or end == None:
                continue
            length = end - start
            if length > Params.edge_min_pixels_solid:
                longest.append((i, [start, end-1], length))
        return longest

    def calc_dottedness(self, idx):
        # add the False ranges
        means = []
        ranges = []
        # the row/col start with different True/False so we have to handle both
        # for simplicity, we add additional True blocks to beginning or end
        if idx[0][0] == 0:
            last = idx[-1][1]
            idx = np.r_[idx, np.array([[last, last+1]])]
        else:
            idx = np.r_[np.array([[-1, 0]]), idx]
        for i in range(Params.matrix_code_size/2):
            false_begin = idx[i][1]
            false_end = idx[i+1][0] - 1
            true_begin = idx[i+1][0]
            true_end = idx[i+1][1] - 1
            means.append((false_begin + false_end) / 2)
            means.append((true_begin + true_end) / 2)
            ranges.append(false_end - false_begin + 1)
            ranges.append(true_end - true_begin + 1)

        # we remove the pseudo square we added at the beginning
        if idx[0][0] == 0:
            means = means[:-2]
            ranges = ranges[:-2]
        else:
            means = means[1:-1]
            ranges = ranges[1:-1]

        return (means, np.std(np.array(ranges)))

    def find_dotted_edge(self, data):
        # searches for dotted edge along the top rows of data

        (height, width) = data.shape
        dotted_row = None
        for r in xrange(height):
            # row should alternate between True/False
            row = data[r,:]
            idx = self.find_runs(row)
            if len(idx) < Params.matrix_code_size / 2:
                # not a dotted row
                # if we already have a dotted row, then break
                # otherwise keep searching for first dotted row
                if dotted_row != None:
                    break
            else:
                (mean, std) = self.calc_dottedness(idx)
                # we only save the one with the lowest std
                if dotted_row == None or std < dotted_row[1]:
                    dotted_row = (mean, std)

        if dotted_row == None:
            return None

        return dotted_row[0]

    def decode_code(self, image):
        # image should be black/white and contain a single code
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
                #bottom = self.find_horizontal_edge(image, (left, right), (top, bottom), Params.code_min_pixels_slice)

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

        return self.datamatrix_decode(bits)

    def find_horizontal_edge(self, image, (x_start, x_end), (y_start, y_end), threshold):
        # goes from [y_start..y_end] and counts the num of pixels on between [x_start..x_end] for each row
        # returns the last row that has a count > the threshold
        # this has to be careful to work when y_start == y_end
        direction = 1 if y_start < y_end else -1
        width = x_end - x_start + 1
        last = y_start
        for y in range(y_start, y_end + direction, direction):
            cv.SetImageROI(image, (x_start, y, width, 1))
            if cv.CountNonZero(image) < threshold:
                return last
            last = y
        return last

    def find_vertical_edge(self, image, (x_start, x_end), (y_start, y_end), threshold):
        # same as find_horizontal_edge but in vertical direction
        direction = 1 if x_start < x_end else -1
        height = y_end - y_start + 1
        cv.SetImageROI(image, (x_start, y_start, 1, height))
        last = x_start
        for x in range(x_start, x_end + direction, direction):
            cv.SetImageROI(image, (x, y_start, 1, height))
            if cv.CountNonZero(image) < threshold:
                return last
            last = x
        return last

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
        cv.Rectangle(self.image, (rect[0]+15, rect[1]+15), (rect[0]+rect[2]-15, rect[1]+rect[3]-15), color, 25)

    def decode_codes(self):
        # use black white thresholded image

        cv.SetImageROI(self.image, self.inner_rect)

        image = self.threshold(self.image)

        #self.debug()

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

                # check if already have the code and don't try decoding again
                # we assume if we decoded a code, it's impossible that it's incorrect
                # however, if called something empty before, we could be wrong
                # so we still repeat the empty well test
                if self.codes[box_r][box_c]:
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
                        # image can be a not perfect black/white due to interpolation above
                        # This puts it back to black/white and gives better results
                        cv.Threshold(well_img, well_img, 128, 255, cv.CV_THRESH_BINARY)

                        code = self.decode_code(well_img)
                        if code == None:
                            code = self.decode_code_2(well_img)
                        if code != None:
                            self.codes[box_r][box_c] = code
                            count += 1
                            self.annotate_image(well_rect, Params.annotate_present_color)
                            success = True
                            break
                        #cv.ResetImageROI(well_img)
                        #self.debug(well_img)

                    contours = contours.h_next()
                if not success:
                    self.annotate_image(well_rect, Params.annotate_not_decoded)

        # annotate the inner rect
        cv.ResetImageROI(self.image)
        cv.Rectangle(self.image, (self.inner_rect[0], self.inner_rect[1]), (self.inner_rect[0]+self.inner_rect[2], self.inner_rect[1]+self.inner_rect[3]), Params.annotate_outside, 10)

        return (count, empty)

    def find_box_and_rotate(self):
        # finds the box and rotate it to a standard orientation

        bwimg = self.threshold(self.image)

        cv.Dilate(bwimg, bwimg, iterations=3) # this helps findcontours work better

        # find the bounding rectangle for the on pixels which should be the box outline
        contours = cv.FindContours(bwimg, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL)

        if len(contours) == 0:
            return False
        (max_contour, max_contour_area) = self.max_contour_area(contours)
        rect = cv.MinAreaRect2(max_contour)

        #center = rect[0]
        (height, width) = rect[1]
        #angle = rect[2]

        # if we selected entire area, then there's likely no box
        if height * width >= bwimg.width * bwimg.height:
            return False

        if height > width:
            (width, height) = rect[1]

        if height < Params.min_pixels_per_well * Params.num_rows or width < Params.min_pixels_per_well * Params.num_cols:
            return False

        self.image = self.crop_and_rotate(self.image, rect)

        #self.debug_resize()
        return True

    def find_orientation(self):
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

        # This can help reduce holes
        cv.Dilate(left_diff, left_diff)
        cv.Erode(left_diff, left_diff)
        cv.Dilate(right_diff, right_diff)
        cv.Erode(right_diff, right_diff)

        # find edge of tab and use as edge of wells
        # we consider the widest/brightest thing to be the tab

        # cv.ResetImageROI(left_tabs)
        # cv.ResetImageROI(right_tabs)
        # cv.ShowImage("left tab", left_image)
        # cv.ShowImage("right tab", right_image)
        # cv.ShowImage("left tab processed", left_tabs)
        # cv.ShowImage("right tab processed", right_tabs)
        # cv.WaitKey()

        if left_sum < 0 and right_sum < 0:
            self.tabs_on_bottom = True
            (_, left_border) = self.longest_contiguous(np.asarray(left_diff)[0] < 0)
            (right_border_offset, _) = self.longest_contiguous(np.asarray(right_diff)[0] < 0)
        elif left_sum > 0 and right_sum > 0:
            self.tabs_on_bottom = False
            (_, left_border) = self.longest_contiguous(np.asarray(left_diff)[0] > 0)
            (right_border_offset, _) = self.longest_contiguous(np.asarray(right_diff)[0] > 0)
        else:
            return False # no tabs found, fail as we cannot determine box orientation

        if left_border == None or right_border_offset == None:
            return False

        right_border = right_border + right_border_offset

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

        width = right_border - left_border + 1
        height = bottom_border - top_border + 1
        self.inner_rect = (left_border, top_border, width, height)

        return True

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
        while contours != None:
            area = cv.ContourArea(contours)
            if area > max_contour_area:
                max_contour = contours
                max_contour_area = area
            contours = contours.h_next()
        return (max_contour, max_contour_area)

    def threshold(self, image, threshold=Params.white_threshold):
        # threshold/convert color image to black/white
        bwimg = cv.CreateImage(cv.GetSize(image), 8, 1)
        cv.InRangeS(image, threshold, (255,255,255), bwimg)
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
        cv.SaveImage("/tmp/debug.tif", image)
        cv.WaitKey()

