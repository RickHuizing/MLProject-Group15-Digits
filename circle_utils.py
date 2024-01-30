import cv2
import numpy as np


def threshold_image(img, threshold=0):
    img = np.float32(img)
    [thresh, img] = cv2.threshold(img, threshold, 6, cv2.THRESH_BINARY)
    return img


def count_circles(img, threshold=0):
    img = threshold_image(img, threshold)
    img = cv2.convertScaleAbs(img)

    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = contours[1] if len(contours) == 2 else contours[2]
    contours = contours[0] if len(contours) == 2 else contours[1]

    hierarchy = hierarchy[0]

    count = 0
    result = img.copy()
    result = cv2.merge([result, result, result])

    for component in zip(contours, hierarchy):
        contour_component = component[0]
        hierarchy_component = component[1]

        if (hierarchy_component[3] > -1) & (hierarchy_component[2] < 0):
            count += 1
            cv2.drawContours(result, [contour_component], 0, (0, 0, 255), 2)

    return count


def get_circle_data(data, threshold=3):
    counter = 0
    circle_counts = []

    for digit in data:
        img = digit.reshape(16, 15)
        circle_count = count_circles(img, threshold)
        circle_counts.append(circle_count)
        # print("Circles in digit ", counter, ": ", circle_count)
        counter += 1

    circle_counts = np.array(circle_counts)
    return circle_counts
