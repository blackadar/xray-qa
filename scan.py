"""
Objects to represent the data structures of the project.
"""

import matplotlib.patches
from PIL import Image
import math
import pathlib
import os
import configparser
import numpy as np
import crop
import cv2
import matplotlib.pyplot as plt
from tools import dice

# Attempt to change working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = configparser.ConfigParser()
config.read('xray-qa.cfg')
WIDTH = int(config['roi-size']['width'])
HEIGHT = int(config['roi-size']['height'])


class Scan:
    """
    Container to represent a Scan in the context of the X-Ray Hand GUI.
    Holds the data, paths, and properties of displaying the scan to the user.
    """

    def __init__(self, image, joints, attribs, patient='', image_path=None, info_path=None, visit=''):
        self.image = image
        self.attribs = attribs
        self.joints = joints
        self.patient = patient
        self.visit = visit
        self.image_path = pathlib.Path(image_path) if image_path is not None else None
        self.info_path = pathlib.Path(info_path) if info_path is not None else None
        self.contrast_enhancement = 1.0
        self.axlimits = [0, self.image.size[0], self.image.size[1], 0] if self.image is not None else None
        self.selected_joint = None
        self.backup_image = None
        self.modified = False

        try:  # Needs to be an int to facilitate sorting
            self.patient = int(self.patient)
        except Exception as e:
            print(f"Unable to cast Patient ID '{self.patient}' to int.\n {e}")

    def __str__(self):
        st = f"<{self.patient}>\n" \
               f"{self.image_path} {self.info_path}\n" \
               f"attributes: {self.attribs}\n"
        for joint in self.joints:
            st += f"{joint}\n"
        return st

    @staticmethod
    def from_files(image_path, info_path):
        """
        Creates a Scan object from an Image and Info file path.
        :param image_path: str, Path-Like Path to Image
        :param info_path: str, Path-Like Path to Info (or None)
        :return: Scan object
        """
        image_path = pathlib.Path(image_path) if image_path is not None else None
        info_path = pathlib.Path(info_path) if info_path is not None else None
        image = Image.open(image_path) if image_path is not None else None
        joints = []
        attribs = ''
        name_details = str(image_path.stem).split('_') if image_path is not None else str(info_path.stem).split('_')
        patient = name_details[0]
        visit = name_details[1] if len(name_details) > 1 else ''
        if info_path is not None:
            info_lines = open(info_path, mode='r').readlines()
            attribs = info_lines[0].strip()
            for line in info_lines[1:]:
                joints.append(Joint.from_line(line))
        return Scan(image, joints, attribs, patient=patient, image_path=image_path, info_path=info_path, visit=visit)

    def set_axlimits_from_joints(self):
        """
        Sets axlimits object property automatically based on the contained joints.
        """
        assert len(self.joints) > 0
        xs = []
        ys = []
        for joint in self.joints:
            xs.append(joint.x)
            ys.append(joint.y)
        min_x = np.min(xs)
        min_y = np.min(ys)
        max_x = np.max(xs)
        max_y = np.max(ys)

        self.axlimits = [min_x - WIDTH, max_x + WIDTH,  max_y + HEIGHT, min_y - HEIGHT]

    def save(self):
        """
        Saves modifications to the file path stored.
        """
        if self.modified:
            if 'q' not in self.attribs:
                self.attribs += 'q'
        if self.info_path is None:
            inf = self.image_path.parent / (self.image_path.stem + '.txt')
            print(f"Creating new info file: {inf}")
            self.info_path = inf
        with open(self.info_path, 'w') as f:
            f.write(f"{self.attribs}\n")
            for joint in self.joints:
                f.write(f"{joint.save_format()}\n")

    def euclidean_distance(self, other):
        """
        Finds Euclidean distance to other Scan
        Sum of Euclidean Distance of Joints.
        :param other: other Scan to find Distance to
        :return: float Distance
        """
        assert type(other) is Scan
        assert len(self.joints) == len(other.joints)
        dists = []
        for s, o in zip(self.joints, other.joints):
            dists.append(s.euclidean_distance(o))
        return np.sum(np.array(dists))

    def dice_similarity(self, other):
        """
        Finds DICE similarity to other scan.
        Average of DICE similarity of Joints.
        :param other: other Scan to find Distance to
        :return: float Distance
        """
        assert type(other) is Scan
        assert len(self.joints) == len(other.joints)
        dists = []
        for s, o in zip(self.joints, other.joints):
            dists.append(s.dice_similarity(o, size=self.image.size))
        return np.divide(np.sum(np.array(dists)), len(self.joints))


class Joint:
    """
    Object to represent a Joint ROI and its display properties.
    """

    def __init__(self, x, y, angle=0.0, label=None):
        self.x = x
        self.y = y
        self.angle = angle
        self.label = label
        self.patch = self._get_patch()
        self.marker_patch = self._get_patch(marker=True)

    def __str__(self):
        return f"    [{self.label}]" \
               f"    {self.x}, {self.y}  {self.angle}"

    def save_format(self):
        """
        Returns a string in the format which it appears in the txt file.
        :return: str Save-formatted string
        """
        return f"{self.label} {self.x} {self.y} {self.angle}"

    def reload_patch(self):
        """
        Sets the patch property based on internal methods.
        """
        self.patch = self._get_patch()
        self.marker_patch = self._get_patch(marker=True)

    @staticmethod
    def from_line(txt):
        """
        Creates a Joint object based on a save-formatted String.
        :param txt: str, String formatted in the save format.
        :return: Joint object
        """
        spl = txt.split(' ')
        label = spl[0].strip()
        x = int(spl[1])
        y = int(spl[2])
        angle = float(spl[3])
        return Joint(x, y, angle, label)

    def _get_patch(self, marker=False):

        def convert_angle(radians):
            """
            Converts an angle in radians to an angle in degrees.
            :param radians: float Angle in Radians
            :return: float Angle in Degrees
            """
            return math.degrees(radians)

        def convert_coordinates(center_x, center_y, width, height, degrees):
            """
            Converts the center point of a rectangle to the top-left point of the rectangle, with rotation
            kept in consideration.
            :param center_x: int, center x coordinate
            :param center_y: int, center y coordinate
            :param width: int, width of rectangle
            :param height: int, height of rectangle
            :param degrees: float, degrees of rotation in degrees (NOT radians)
            :return: tuple (x, y)
            """
            # Corner point, without rotation
            x = center_x - (width / 2)
            y = center_y - (height / 2)
            # Translate to the origin
            o_x = x - center_x
            o_y = y - center_y
            # Apply rotation with trig
            rot_x = o_x * math.cos(degrees) - o_y * math.sin(degrees)
            rot_y = o_x * math.sin(degrees) + o_y * math.cos(degrees)
            # Translate back
            c_x = rot_x + center_x
            c_y = rot_y + center_y

            return c_x, c_y

        if marker:
            t_x, t_y = convert_coordinates(self.x, self.y, WIDTH, 0, self.angle)
            angle = convert_angle(self.angle)

            return matplotlib.patches.Rectangle((t_x, t_y), WIDTH, 0, angle,
                                                linewidth=0.5, edgecolor='r', facecolor='none',
                                                linestyle=(0, (5, 10)), alpha=0.5)
        else:
            t_x, t_y = convert_coordinates(self.x, self.y, WIDTH, HEIGHT, self.angle)
            angle = convert_angle(self.angle)

            return matplotlib.patches.Rectangle((t_x, t_y), WIDTH, HEIGHT, angle,
                                                linewidth=0.5, edgecolor='r', facecolor='none')

    def euclidean_distance(self, other):
        """
        Finds Euclidean Distance to another Joint
        :param other: Joint to find distance to
        :return: float distance
        """
        assert type(other) is Joint
        a = np.array([self.x, self.y, self.angle])
        b = np.array([other.x, other.y, other.angle])
        return np.linalg.norm(a - b)

    def build_mask(self, size):
        """
        Builds an image mask for the Joint based on a size.
        Note that size will scale and change position based on the image's coordinate system.
        :param size: tuple (image.size[0], image.size[1])
        :return: np array 0/1 mask
        """

        def convert_coordinates(center_x, center_y, width, height, degrees):
            """
            Converts center, angle rectangle to 4 points
            :param center_x: int, center x coordinate
            :param center_y: int, center y coordinate
            :param width: int, width of rectangle
            :param height: int, height of rectangle
            :param degrees: float, degrees of rotation in degrees (NOT radians)
            :return: tuple (x, y)
            """
            def _convert(x, y):
                # Translate to the origin
                o_x = x - center_x
                o_y = y - center_y
                # Apply rotation with trig
                rot_x = o_x * math.cos(degrees) - o_y * math.sin(degrees)
                rot_y = o_x * math.sin(degrees) + o_y * math.cos(degrees)
                # Translate back
                c_x = rot_x + center_x
                c_y = rot_y + center_y
                return int(c_x), int(c_y)

            top_left = (center_x - (width / 2), center_y - (height / 2))
            top_right = (center_x + (width / 2), center_y - (height / 2))
            bottom_left = (center_x - (width / 2), center_y + (height / 2))
            bottom_right = (center_x + (width / 2), center_y + (height / 2))

            return [_convert(*bottom_right), _convert(*top_right), _convert(*top_left), _convert(*bottom_left), ]

        # This generates a box outline. May be useful elsewhere...
        # t_l, t_r, b_l, b_r = convert_coordinates(self.x, self.y, WIDTH, HEIGHT, self.angle)
        # im = np.zeros((size[0], size[1]))
        # color = (255, 255, 255)
        # thickness = 1
        # im = cv2.line(im, t_l, t_r, color, thickness)
        # im = cv2.line(im, t_r, b_r, color, thickness)
        # im = cv2.line(im, b_r, b_l, color, thickness)
        # im = cv2.line(im, b_l, t_l, color, thickness)

        contours = convert_coordinates(self.x, self.y, WIDTH, HEIGHT, self.angle)
        contours = np.array([ [contours[0][0], contours[0][1]], [contours[1][0], contours[1][1]],
                              [contours[2][0], contours[2][1]], [contours[3][0], contours[3][1]] ], dtype=np.int32)
        im = np.zeros((size[1], size[0]), dtype=np.int32)
        color = (1, 1, 1)
        cv2.fillPoly(im, [contours], color=color)

        return im

    def dice_similarity(self, other, size):
        """
        Calculates DICE distance between one joint and another.
        Gets a mask for both based on the parent image and finds the DICE diff.
        :param size: (int, int) size of image
        :param other: Joint to find distance to
        :return: float distance
        """
        assert type(other) is Joint
        mask_a = self.build_mask(size)
        mask_b = other.build_mask(size)

        return dice(mask_a, mask_b, empty_score=1.0)
