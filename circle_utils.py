import cv2
import numpy as np


def threshold_image(img, threshold: float = 0.5):
    img = np.float32(img)
    [thresh, img] = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    return img


def count_circles(img, threshold: float = 0.5):
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


def get_circle_data(data, threshold=0.5):
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


def find_optimal_threshold(data, thresholds_to_try=10, verbose=False):
    circle_counts = np.zeros((thresholds_to_try, data.shape[0]), dtype=np.int8)

    for threshold in range(thresholds_to_try):
        circle_counts[threshold] = get_circle_data(data, threshold=threshold / 10)

    expected_circle_counts_per_class = [1, 0, 0, 0, 0, 0, 1, 0, 2, 1]

    as_expected_per_threshold = np.zeros(circle_counts.shape[0])
    for threshold, circle_data in enumerate(circle_counts):
        as_expected = 0
        for digit in range(10):
            for i in range(100):
                if circle_data[digit * 100 + i] == expected_circle_counts_per_class[digit]:
                    as_expected += 1
        if verbose:
            print(f'threshold {threshold / 10}: {as_expected / 10}% expected number of circles ')
        as_expected_per_threshold[threshold] = as_expected
    best_threshold = np.argmax(as_expected_per_threshold) / 10
    return best_threshold
