"""
Handles Inputs/Outputs from BoneFinder (Manchester)
"""
import pathlib

from PIL import Image

from scan import Joint, Scan

labels = ['mcp2', 'pip2', 'dip2',
          'mcp3', 'pip3', 'dip3',
          'mcp4', 'pip4', 'dip4',
          'mcp5', 'pip5', 'dip5']


def parse_pts(path):
    """
    Parses a .pts file from BoneFinder, expecting (x, y) points.
    :param path: Str or Path-Like path to .pts file
    :return: List of Points [(x, y), ...]
    """
    with open(path) as f:
        lines = f.readlines()
        version = lines[0]
        n_points = lines[1]
        points = []
        for line in lines[2:]:
            if line.strip() != '{' and line.strip() != '}':
                x = int(float(line.strip().split(' ')[0]))
                y = int(float(line.strip().split(' ')[1]))
                points.append((x, y))
    return points


def get_joints(pts, top_extend=False):
    """
    Converts a 37 point output from BoneFinder to the set of points we use.
    Also configurable to include the extra point at the finger tip for angle calculation.
    :param pts: List of points parsed from pts file
    :param top_extend: Boolean, whether or not to include the extra point at the finger tip
    :return: List of joint points, subset of pts
    """
    if not top_extend:
        joint_pts = [
                pts[18], pts[19], pts[20],  # Index
                pts[13], pts[14], pts[15],  # Middle
                pts[8], pts[9], pts[10],  # Ring
                pts[3], pts[4], pts[5]  # Pinky
        ]
    else:
        joint_pts = [
                pts[18], pts[19], pts[20], pts[21],  # Index
                pts[13], pts[14], pts[15], pts[16],  # Middle
                pts[8], pts[9], pts[10], pts[11],  # Ring
                pts[3], pts[4], pts[5], pts[6]  # Pinky
        ]
    return joint_pts


def get_angles(joint_pts):
    """
    Requires top_extend=True from get_joints
    :param joint_pts: Output from extraction of our Joint pts from BoneFinder pts
    :return: List of angles
    """
    import math
    angles = []
    for i in range(len(joint_pts) // 4):  # For each finger
        j_start = i * 4
        for pt in range(j_start, j_start + 3):  # For each point in finger
            this_pt = joint_pts[pt]
            next_pt = joint_pts[pt + 1]
            angles.append(math.atan2(next_pt[1] - this_pt[1], next_pt[0] - this_pt[0]))
    return angles


def pts_to_Joints(path):
    """
    Converts a pts file to a Joint object.
    Calls above functions to do so.
    :param path: Path to pts file
    :return: Joint object
    """
    joints = []
    labels_idx = 0
    pts = parse_pts(path)
    joint_pts = get_joints(pts)
    angles = get_angles(get_joints(pts, top_extend=True))
    for loc, angle in zip(joint_pts, angles):
        joints.append(Joint(loc[0], loc[1], angle, labels[labels_idx]))
        labels_idx += 1
    return joints


def pts_image_to_Scan(pts_path, image_path):
    """
    Converts a pts file and Image file pair to a Scan object.
    :param pts_path: Path to pts file
    :param image_path: Path to patient image
    :return: Scan object
    """
    image = Image.open(image_path)
    pts_path = pathlib.Path(pts_path)
    patient = pts_path.stem.split('.')[0]
    joints = pts_to_Joints(pts_path)
    return Scan(image, joints, 'b', patient, image_path=str(image_path))


def convert_pts_directory(pts_dir, images_dir):
    """
    Converts a directory full of .pts files to .txt files alongside patient images.
    This allows us to convert the output of BoneFinder to our project's format.
    :param pts_dir: Str or Path-Like to the directory holding pts files
    :param images_dir: Str or Path-Like to the directory holding image files
    :return: None, .save() is called on Scan objects to save the output.
    """
    pts_dir = pathlib.Path(pts_dir)
    images_dir = pathlib.Path(images_dir)
    pts_list = list(pts_dir.glob(f"*.pts"))
    images_list = list(images_dir.glob(f"*.png"))

    for pts in pts_list:
        match = [img for img in images_list if str(pts.stem) in str(img.name)]
        im = match[0]
        s = pts_image_to_Scan(pts, im)
        s.save()
