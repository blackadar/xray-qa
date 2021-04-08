"""
Crops joints out of PNGs based on their .txt file and the ROI config size.
"""
import configparser
import pathlib
import math
import sys
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

config = configparser.ConfigParser()
config.read('xray-qa.cfg')
WIDTH = int(config['roi-size']['width'])
HEIGHT = int(config['roi-size']['height'])

read_from = pathlib.Path('data/hands_test/')
output_to = pathlib.Path('data/out/')


def main():
    """
    Runs the cropping tool on read_from
    :return: None
    """
    from scan import Scan

    scans = []
    images = list(read_from.glob(f"*.png"))
    infos = list(read_from.glob(f"*.txt"))
    output_to.mkdir(exist_ok=True, parents=True)

    print(f"=== Xray-QA Crop Tool ===\n"
          f"{len(images)} images found.\n"
          f"{len(infos)} txt files found.\n"
          f"{read_from} -> {output_to}\n"
          f"Crop joints to {WIDTH} x {HEIGHT}.\n"
          f"Press [ENTER] to continue, Ctrl+C to cancel.\n")

    input()

    # Read Scans from files, only those with a PNG and a txt
    intersection = set([path.stem for path in images]) & set(path.stem for path in infos)
    images = [image for image in images if image.stem in intersection]
    infos = [info for info in infos if info.stem in intersection]
    for image, info in zip(images, infos):
        scans.append(Scan.from_files(image, info))
    scans.sort(key=lambda x: x.patient)  # Sort based on patient number

    print(f"{len(intersection)} image, txt pairs will be considered.")

    # Produce the Joint image for each
    for idx, scan in enumerate(scans):
        sys.stdout.write(f"\rCropping joints for patient {idx} of {len(scans)}. ({idx / len(scans) * 100: .2f}%)")
        sys.stdout.flush()
        image = np.array(scan.image)
        # Note: Post-Processing could go here
        for joint in scan.joints:
            joint_image = angled_center_crop(image, joint.x, joint.y, WIDTH, HEIGHT, joint.angle)
            im = Image.fromarray(joint_image)
            im.save(output_to / f"{scan.patient}_{scan.visit}_{joint.label}.png")
    sys.stdout.write(f"\rCropped joints for {len(scans)} patients. (100.00%)\n")
    sys.stdout.flush()


def angled_center_crop(image, x, y, width, height, angle):
    """
    This is from Carmine's process.py script.
    Accepts an image and produces a crop of the angled rectangle centered on (x, y).
    :param image: np.ndarray Image to crop
    :param x: X-position of the Rectangle center
    :param y: Y-position of the Rectangle center
    :param width: Width of the Rectangle
    :param height: Height of the Rectangle
    :param angle: Angle of rotation of the rectangle to crop, in radians
    :return: width x height (np.ndarray) cropped from image
    """

    def points4(start_point, end_point):
        points = []
        points.append(start_point)
        points.append((end_point[0], start_point[1]))
        points.append(end_point)
        points.append((start_point[0], end_point[1]))
        return points

    def z_rotate(origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return int(qx), int(qy)

    def rotate_points(points, center, angle):
        rotated = []
        n = len(points)
        for i in range(n):
            rotated.append(z_rotate(center, points[i], angle))
        return rotated

    def crop(image, x, y, width, height):
        return image[y:y + height, x:x + width]

    def center_crop(image, x, y, width, height):
        return crop(image, int(x - (width / 2)), int(y - (height / 2)), width, height)

    start_point = (int(x - (width * 0.5)), int(y - (height * 0.5)))
    end_point = (start_point[0] + width, start_point[1] + height)
    points = points4(start_point, end_point)

    rotated = rotate_points(points, (x, y), angle)

    min_x = min(rotated[0][0], rotated[1][0], rotated[2][0], rotated[3][0])
    max_x = max(rotated[0][0], rotated[1][0], rotated[2][0], rotated[3][0])

    min_y = min(rotated[0][1], rotated[1][1], rotated[2][1], rotated[3][1])
    max_y = max(rotated[0][1], rotated[1][1], rotated[2][1], rotated[3][1])

    # crop the image around the super bounds
    temp = crop(image, min_x, min_y, max_x - min_x, max_y - min_y)

    # cv2_imshow(temp)

    degrees = math.degrees(angle)
    temp = ndimage.rotate(temp, degrees)

    rotated_height = temp.shape[0]
    rotated_width = temp.shape[1]

    center_x = int(rotated_width * 0.5)
    center_y = int(rotated_height * 0.5)

    return center_crop(temp, center_x, center_y, width, height)


if __name__ == '__main__':
    main()
