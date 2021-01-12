import matplotlib.patches
from PIL import Image
import math


class Scan:

    def __init__(self, image, joints, attribs, patient='', image_path=None, info_path=None):
        self.image = image
        self.attribs = attribs
        self.joints = joints
        self.patient = patient
        self.image_path = image_path
        self.info_path = info_path
        self.contrast_enhancement = 1.0
        self.selected_joint = None
        self.backup_image = None

    @staticmethod
    def from_files(image_path, info_path):
        image = Image.open(image_path)
        info_lines = open(info_path, mode='r').readlines()
        attribs = info_lines[0]
        joints = []
        for line in info_lines[1:]:
            joints.append(Joint.from_line(line))
        return Scan(image, joints, attribs, patient=image_path.stem, image_path=image_path, info_path=info_path)

    def save(self, to_file):
        pass


class Joint:

    def __init__(self, x, y, angle=0.0, label=None):
        self.x = x
        self.y = y
        self.angle = angle
        self.label = label
        self.patch = self._get_patch()

    def reload_patch(self):
        self.patch = self._get_patch()

    @staticmethod
    def from_line(txt):
        spl = txt.split(' ')
        label = spl[0].upper()
        x = int(spl[1])
        y = int(spl[2])
        angle = float(spl[3])
        return Joint(x, y, angle, label)

    def _get_patch(self):
        # TODO: Dynamic w/h from file
        WIDTH = 200
        HEIGHT = 100

        def convert_angle(radians):
            return math.degrees(radians)

        def convert_coordinates(center_x, center_y, width, height, degrees):
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
