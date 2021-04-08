"""
Handles Inputs/Outputs from BoneFinder (Manchester)
"""
import pathlib
import shutil

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
    pvinf = pts_path.stem.split('.')[0].split('_')
    patient = pvinf[0]
    if len(pvinf) == 2:
        visit = pvinf[1]
    else:
        visit = None
    joints = pts_to_Joints(pts_path)
    return Scan(image, joints, 'b', patient, image_path=str(image_path), visit=visit)


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
        pts_name = pts.stem.split('.')[0]
        match = [img for img in images_list if pts_name in str(img.name)]
        im = match[0]
        s = pts_image_to_Scan(pts, im)
        s.save()


##################################################################################
# Functions to handle Finding Best Fit from QoF Summary, and Moving the PTS file #
##################################################################################

def select_qof(qof_path, pts_dir, selection_out_dir):
    """
    Reads a QOF txt and decides on which .pts file to use.
    Copies the pts file to one which has a simpler name.
    :param selection_out_dir:
    :param qof_path:
    :param pts_dir:
    :return:
    """
    pts_dir = pathlib.Path(pts_dir)
    qof_path = pathlib.Path(qof_path)
    selection_out_dir = pathlib.Path(selection_out_dir)

    with open(qof_path) as f:
        sets = []
        lines = f.readlines()
        lbls = lines[0]
        for line in lines[1:]:
            cols = line.split()
            pts_path = cols[0].split(":")[0]
            pts_file = pts_path.split("/")[-1]
            patient_visit = pts_file.split('.')[0]
            qof_sum = cols[1]
            sets.append((patient_visit, pts_file, qof_sum))

    it_patient_visit = sets[0][0]
    it_max_qof_sum = sets[0][2]
    it_max_qof_sum_idx = 0

    for idx, s in enumerate(sets):
        if s[0] == it_patient_visit:
            # Same Patient/Visit, another QOF entry to compare
            if s[2] > it_max_qof_sum:
                # Update the Max
                it_max_qof_sum = s[2]
                it_max_qof_sum_idx = idx
            else:
                # Leave the Max
                continue
        else:
            # New Patient/Visit, first write the best to the out dir
            shutil.copy(pts_dir / sets[it_max_qof_sum_idx][1], selection_out_dir / (it_patient_visit + ".pts"))
            # Then set up for the group of patient/visit
            it_patient_visit = s[0]
            it_max_qof_sum = s[2]
            it_max_qof_sum_idx = idx

    # Handle the last one
    shutil.copy(pts_dir / sets[it_max_qof_sum_idx][1], selection_out_dir / (it_patient_visit + ".pts"))


