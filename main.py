# Import Required Modules
import os
import cv2
import numpy as np

# Load all image path
PATH = "image_data\input"
data_folder_path = [os.path.join(PATH, image) for image in os.listdir(PATH) if '.jpg' in image]

# Read an image
image = cv2.imread(data_folder_path[1])


def get_canny(image):
    """
    Input: take a raw image
    Output: detect edges of img using canny
    """
    # Convert the image to gray image
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blurs an image using a Gaussian filter
    image_blur = cv2.GaussianBlur(image_gray, (11, 11), 0)

    # Canny edge detection
    image_canny = cv2.Canny(image_blur, 50, 150)
    return image_canny


canny = get_canny(image)


def region_of_interest(image):
    """
    Only keeps the region of the image defined by a polygon
    -----------------------------------------------------
    Input: get a edge image, vertices as the limit of ROI
    Output: A mask of image which only contains ROI
    """
    height = image.shape[0]
    width = image.shape[1]
    polygon = np.array([[(width, height), (int(width / 2), int(height / 2)), (0, height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    image_mask = cv2.bitwise_and(image, mask)
    return image_mask


image_mask = region_of_interest(canny)


def draw_lines(image, lines):
    image_line = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4, 1)
            cv2.line(image_line, (x1, y1), (x2, y2), (255, 0, 0), 10)
    image_combine_line = cv2.addWeighted(image, 0.8, image_line, 1, 1)
    return image_combine_line


# Use function HoughLinesP() to detect lines in an image
lines = cv2.HoughLinesP(image_mask, 2, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=5)

# Draw detected lines in the image
image_line = draw_lines(image, lines)


def find_intersection_point(left, right):
    """
    The intersection point between 2 vectors satisfies:
        y = left_slope * x + left_intercept
        y = right_slope * x + right_intercept
    => Intersection point:
        x = (right_intercept - left_intercept) / (left_slope - right_slope)
        y = left_slope * x + left_intercept
    """

    try:
        left_slope, left_intercept = left
    except TypeError:
        left_slope, left_intercept = 0, 0

    try:
        right_slope, right_intercept = right
    except TypeError:
        right_slope, right_intercept = 0, 0
    x = (right_intercept - left_intercept) / (left_slope - right_slope)
    y = left_slope * x + left_intercept
    return np.array([x, y])

def make_coordinates(intersection_point, line_params):
    try:
        slope, intercept = line_params
    except TypeError:
        slope, intercept = 0, 0
    y1 = image.shape[0]
    y2 = int(intersection_point[1] * 100 / 90)
    if (slope == 0):
        x1 = image.shape[0]
        x2 = int(intersection_point[0] * 100 / 90)
    else:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    """
    Merge detected lines which on the same lane
    And cover the full length of the lanes.
    """
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        """
        Line that goes through 2 points (x1, y1), (x2, y2):
        y = slope * x + intercept
        => If (slope < 0), this line belongs to the left border
           Else it belongs to the right border
        """
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Find the line goes through all lines
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    # Find the intersection point between left line vector and right line vector
    intersection_point = find_intersection_point(left_fit_avg, right_fit_avg)

    """
    Extend lines' length from bottom of the original 
    image to the 90% of intersection point's length
    """
    left_line = make_coordinates(intersection_point, left_fit_avg)
    right_line = make_coordinates(intersection_point, right_fit_avg)

    return np.array([left_line, right_line])


avg_line = average_slope_intercept(image, lines)
image_line = draw_lines(image, avg_line)


def draw_lane(image, lines):
    image_lane = np.zeros_like(image)
    points_list = np.array(avg_line, np.int32).reshape((4, 2))
    # Insert offset width
    points_list[0][0] += 10
    points_list[1][0] += 10
    points_list[2][0] -= 10
    points_list[3][0] -= 10
    points_list = np.array([points_list[0], points_list[1], points_list[3], points_list[2]])
    cv2.fillPoly(image_lane, [points_list], (0, 255, 0))
    image_combine_lane = cv2.addWeighted(image_line, 1, image_lane, 0.5, 1)
    return image_combine_lane


image_lane = draw_lane(image_line, avg_line)

cv2.imshow('Matched Template', image_lane)
cv2.waitKey(0)
cv2.destroyAllWindows()
