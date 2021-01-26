import matplotlib.patches
from PIL import Image
import math
import pathlib
import configparser
import numpy as np

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
        self.image_path = pathlib.Path(image_path)
        self.info_path = pathlib.Path(info_path) if info_path is not None else None
        self.contrast_enhancement = 1.0
        self.axlimits = [0, self.image.size[0], self.image.size[1], 0]
        self.selected_joint = None
        self.backup_image = None
        self.modified = False

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
        image = Image.open(image_path)
        joints = []
        attribs = ''
        name_details = str(image_path.stem).split('_')
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
                self.attribs = self.attribs + 'q'
        if self.info_path is None:
            inf = self.image_path.parent / (self.image_path.stem + '.txt')
            print(f"Creating new info file: {inf}")
            self.info_path = inf
        with open(self.info_path, 'w') as f:
            f.write(f"{self.attribs}\n")
            for joint in self.joints:
                f.write(f"{joint.save_format()}\n")


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

    def _get_patch(self):

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

        t_x, t_y = convert_coordinates(self.x, self.y, WIDTH, HEIGHT, self.angle)
        angle = convert_angle(self.angle)

        return matplotlib.patches.Rectangle((t_x, t_y), WIDTH, HEIGHT, angle,
                                            linewidth=0.5, edgecolor='r', facecolor='none')
