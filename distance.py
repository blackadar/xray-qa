"""
Functions to find distances between two sets of measurements.
"""
import pathlib
import numpy as np
from scan import Scan
from tools import progress
from multiprocessing import Pool, cpu_count

set_a = "C:\\Users\\Jordan\\PycharmProjects\\xray-qa\\data\\qa-all"  # QA
set_b = "C:\\Users\\Jordan\\PycharmProjects\\xray-qa\\data\\processed_v2_bonefinder"
images = "C:\\Users\\Jordan\\PycharmProjects\\xray-qa\\data\\processed_v2"
ignore_visit = False
mp = True
save = False

top = 500


def _mp_work(x, y):
    return x.dice_similarity(y)


def main():
    """Main"""
    print("============ Prep ===========")

    print("Reading Directories")

    a = pathlib.Path(set_a)
    b = pathlib.Path(set_b)
    i = pathlib.Path(images)

    a_infos = list(a.glob(f"*.txt"))
    b_infos = list(b.glob(f"*.txt"))
    all_imgs = list(i.glob(f"*.png"))

    print("Finding Intersection...")

    a_stems = [path.stem for path in a_infos] if not ignore_visit else [path.stem.split("_")[0] for path in a_infos]
    b_stems = [path.stem for path in b_infos] if not ignore_visit else [path.stem.split("_")[0] for path in b_infos]
    im_stems = [path.stem for path in all_imgs] if not ignore_visit else [path.stem.split("_")[0] for path in b_infos]

    intersection = (set(a_stems) & set(b_stems)) & set(im_stems)
    a_infos = [inf for inf in a_infos if (inf.stem if not ignore_visit else inf.stem.split("_")[0]) in intersection]
    b_infos = [inf for inf in b_infos if (inf.stem if not ignore_visit else inf.stem.split("_")[0]) in intersection]
    all_imgs = [im for im in all_imgs if (im.stem if not ignore_visit else im.stem.split("_")[0]) in intersection]

    print("Building Scan Objects...")

    pairs = []

    for a, b, im in zip(a_infos, b_infos, all_imgs):
        a_scan = Scan.from_files(im, a)
        b_scan = Scan.from_files(im, b)
        pairs.append((a_scan, b_scan))

    print("====== Euclidean Distance ======")

    print("Finding Distances Between Scans...")

    e_dists = []

    for idx, pair in enumerate(pairs):
        progress(idx, len(pairs))
        dist = pair[0].euclidean_distance(pair[1])
        e_dists.append(dist)

    print("Statistics...")

    e_dists = np.array(list(zip(e_dists, pairs)), dtype=object)
    srt = e_dists[e_dists[:, 0].argsort()]

    print(f"Top {top} Farthest:")

    for idx, p in enumerate(srt[-top:][::-1]):
        dist = p[0]
        a = p[1][0]
        b = p[1][1]
        print(f"- {dist: 0.2f} : {a.patient} ({a.info_path} <> {b.info_path})")

    print(f"Average: {np.average(srt[:, 0])}")
    print(f"Min: {np.min(srt[:, 0])}")
    print(f"Max: {np.max(srt[:, 0])}")
    print(f"Median: {np.median(srt[:, 0])}")


    print("======== Sørensen–Dice =========")

    print("Finding Distances Between Scans...")

    if mp:
        with Pool(cpu_count() - 1) as pool:
            d_dists = pool.starmap(_mp_work, pairs)
    else:
        d_dists = []
        for idx, pair in enumerate(pairs):
            progress(idx, len(pairs))
            dist = pair[0].dice_similarity(pair[1])
            d_dists.append(dist)

    print("Statistics...")

    d_dists = np.array(list(zip(d_dists, pairs)), dtype=object)
    srt = d_dists[d_dists[:, 0].argsort()]

    print(f"Top {top} Dissimilar:")

    for idx, p in enumerate(srt[:top]):
        dist = p[0]
        a = p[1][0]
        b = p[1][1]
        print(f"- {dist: 0.2f} : {a.patient} ({a.info_path} <> {b.info_path})")

    print(f"Average: {np.average(srt[:, 0])}")
    print(f"Min: {np.min(srt[:, 0])}")
    print(f"Max: {np.max(srt[:, 0])}")
    print(f"Median: {np.median(srt[:, 0])}")

    print("====== Meta Statistics =========")
    e_list = list(e_dists[:, 0])
    d_list = list(d_dists[:, 0])

    print(f"Coef. Corr. Dice & Euclidean: {np.corrcoef(e_list, d_list)[0][1]}")

    if save:
        print("Saving arrays...")
        with open('euclidean.npy', 'wb') as f:
            np.save(f, e_dists)
        with open('dice.npy', 'wb') as f:
            np.save(f, d_dists)


if __name__ == '__main__':
    main()
