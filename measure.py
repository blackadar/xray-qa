"""
Algorithmically measures the distance between bones in a hand joint.
"""

import pathlib
import numpy as np
import numpy.polynomial.polynomial as poly
from PIL import Image

read_from = pathlib.Path('data/out/')


def find_horizontal_range(image, show_plots=True):
    """
    Algorithmically discovers the approximate horizontal range of a joint.
    :param show_plots: Display intermediate plots for the algorithm
    :param image: np.ndarray Image to analyze
    :return: (start, stop) Approximation of starting and stopping columns
    """
    # Variables in the operation which can be tuned to the data
    tb_rows = 5  # Number of rows on the top and bottom of the image to consider in the row average
    polyfit_degree = 6  # Degree of the polynomial fit to the averaged rows
    tb_gradient_poll_rate = 3  # Polling rate of the gradient of the row average
    ignore_cols = 25  # Number of columns to ignore on the left and right when finding the max rate of change

    # Pre-compute some stats to make things easier
    num_cols = image.shape[1]
    cols_range = np.arange(0, num_cols)

    # Trim the top and bottom of the joint
    top = image[0:tb_rows, :]
    bottom = image[-tb_rows:, :]
    tb = np.vstack([top, bottom])

    # Compute stats on the compiled top and bottom rows...
    # Average the rows together to get a single average row of values:
    tb_avg = np.mean(tb, axis=(0, ))
    # Find a Polynomial to fit that average row:
    tb_poly = poly.Polynomial(poly.polyfit(cols_range, tb_avg, deg=polyfit_degree))(cols_range)
    # Find the derivative of the average (with a sampling rate to reduce amplitude from noise):
    tb_prime = np.abs(np.gradient(tb_avg, tb_gradient_poll_rate))
    # Find the derivative of the polynomial:
    poly_prime = np.abs(np.gradient(tb_poly))

    # Ignore the edge maxes as they're not what we're looking for.
    denoise_poly_prime = poly_prime[ignore_cols:-ignore_cols]

    # If the image follows the observed pattern, there will be two inflections to find on either side of the joint,
    # for the start and end of bone in the image. We'll split the image in half (we can assume it's centered post-QA)
    # to find the inflection points with argmax.
    dnpp_1 = denoise_poly_prime[:len(denoise_poly_prime)//2]
    dnpp_2 = denoise_poly_prime[len(denoise_poly_prime)//2:]

    # Finally, find the max of the arrays and offset them to match the real image column indices.
    bone_start = np.argmax(dnpp_1) + ignore_cols  # We took some columns off the edge earlier
    bone_end = np.argmax(dnpp_2) + ignore_cols + len(denoise_poly_prime)//2  # Same as above, also offset

    if show_plots:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.plot(tb_avg, label="Image Column Average")
        plt.plot(tb_poly, label="Polyfit")
        plt.vlines(bone_start, 0, np.max(tb_poly), linestyles="--", colors='red')
        plt.vlines(bone_end, 0, np.max(tb_poly), linestyles="--", colors='red')
        plt.xlabel('Image Row (x)')
        plt.ylabel('Average Value')
        plt.title('Image Average Column')
        plt.legend()
        plt.show()

        plt.plot(tb_prime, label="Image Gradient")
        plt.plot(poly_prime, label="Polyfit Gradient")
        plt.vlines(bone_start, 0, np.max(poly_prime), linestyles="--", colors='red')
        plt.vlines(bone_end, 0, np.max(poly_prime), linestyles="--", colors='red')
        plt.xlabel('Image Row (x)')
        plt.ylabel('Absolute Value, Gradient of Column')
        plt.title('Image Rate of Change')
        plt.legend()
        plt.show()

        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gist_gray')
        rect = patches.Rectangle((bone_start, 0), bone_end-bone_start, image.shape[1], alpha=0.2)
        ax.add_patch(rect)
        plt.show()

    return bone_start, bone_end


def measure_gaps(image, horizontal_range, show_plots=True):
    """
    Algorithmically measures the average gap distance between joint bones over a horizontal range.
    :param show_plots: Display intermediate plots for the algorithm
    :param image: np.ndarray Image to analyze
    :param horizontal_range: (start, stop) Columns to run over, can be estimated by find_horizontal_range()
    :return: (start, end) Algorithm approximation of start and end rows of the joint gap
    """
    # Parameters for the algorithm
    threshold = 0.6
    tolerance = 5  # Number of pixels that can be below the threshold while maintaining the region
    max_length = 20  # TODO: Determine max #pixels a joint space could be
    min_length = 10  # TODO: Determine min #pixels a joint space could be
    valid_range = (60, 90)  # TODO: Determine valid range of pixels a joint space could be in
    polyfit_degree = 5

    num_cols = horizontal_range[1] - horizontal_range[0]
    cols_range = np.arange(0, num_cols)

    trim = image[:, horizontal_range[0]:horizontal_range[1]]
    col_grads = np.array([np.abs(np.gradient(col)) for col in trim.T]).T
    # col_grads_polyfits = np.array([poly.Polynomial(poly.polyfit(cols_range, col, deg=polyfit_degree))(cols_range) for col in col_grads])
    grad_avg = np.mean(col_grads, axis=1)
    thresh_indices = np.argwhere(grad_avg >= np.max(grad_avg) * threshold)

    runs = []  # List of tuples (start, stop)
    prev = thresh_indices[0][0]
    start = thresh_indices[0][0]
    for idx in thresh_indices[1:]:
        diff = idx - prev
        if diff > tolerance:
            # Break the run
            runs.append((start, prev))
            start = idx[0]
            prev = idx[0]
            continue
        prev = idx[0]

    if len(runs) == 0:
        print("No runs found!")
        return None

    # Ideally, this should result in a single run. But sometimes it won't so we'll need to pick.
    # The safest bet is the one that encompasses the center of the image, since the joint gap should be very close.
    # TODO: Investigate Longest Run Plausibility
    result = None
    if len(runs) > 1:
        for start, end in runs:
            # If we never pass there's no result that encompassed the center of the image. We'll just choose the first.
            result = runs[0]
            if image.shape[0]//2 in range(start, end):
                result = (start, end)
                break
    else:
        result = runs[0]

    if show_plots:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.plot(grad_avg, label="Avg Gradient")
        plt.hlines(np.max(grad_avg) * threshold, 0, image.shape[0],
                   linestyles="--", colors='orange', label="Threshold")
        plt.vlines(result[0], np.min(grad_avg), np.max(grad_avg), linestyles="--", colors='red', label="Gap Start")
        plt.vlines(result[1], np.min(grad_avg), np.max(grad_avg), linestyles="--", colors='red', label="Gap End")
        plt.legend()
        plt.xlabel("Image Rows (x)")
        plt.ylabel("Average Gradient Amplitude")
        plt.title("Region Gradient Analysis")
        plt.show()

        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gist_gray')
        rect = patches.Rectangle((horizontal_range[0], 0), horizontal_range[1] - horizontal_range[0],
                                 image.shape[0] - 1, alpha=0.2)
        ax.add_patch(rect)
        ax.hlines(result[0], 0, image.shape[1] - 1, linestyles="--", colors='red')
        ax.hlines(result[1], 0, image.shape[1] - 1, linestyles="--", colors='red')
        plt.show()

    return result


def main():
    """
    Run measurement across the input folder, and output to TODO
    :return: None
    """
    image = Image.open('data/out/9000099_v06_dip2.png')
    i = np.array(image)
    h = find_horizontal_range(i, show_plots=True)
    measure_gaps(i, h, show_plots=True)

    # TODO: Read entire directory
    # TODO: Output measurement results


if __name__ == "__main__":
    main()
