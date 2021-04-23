"""
Functions to find distances between two sets of measurements.
"""
import pathlib
import numpy as np
from scan import Scan
from tools import progress
from multiprocessing import Pool, cpu_count

set_a = "C:\\Users\\Jordan\\PycharmProjects\\xray-qa\\data\\qa-all"  # QA
set_b = "C:\\Users\\Jordan\\PycharmProjects\\xray-qa\\data\\processed_v2_bonefinder_release"  # Comparison
images = "C:\\Users\\Jordan\\PycharmProjects\\xray-qa\\data\\processed_v2"  # Images (used for Size)

ignore_visit = True  # Ignore visit # in parsing
mp = True  # Turn on Multiprocessing for Supported Operations
save = False  # Save supported results to npy files

euclidean = False  # Find Euclidean Distance
dice = False  # Find DICE Distance
meta = False  # Calculate correlation stats
joint_stats = True  # Find "problem" joints by < TPR Threshold

top = 100
tpr_threshold = 0.5


def _mp_work(x, y):
    return x.dice_similarity(y, count_over=tpr_threshold)


def _joint_mp_work(x, y, size):
    return x.dice_similarity(y, size, return_name=True)


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

    if euclidean:

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

    if dice:
        print("======== Sørensen–Dice =========")

        print("Finding Distances Between Scans...")

        if mp:
            with Pool(cpu_count() - 1) as pool:
                results = pool.starmap(_mp_work, pairs)
            d_dists = [r[0] for r in results]
            joint_threshold_count = [r[1] for r in results]
        else:
            d_dists = []
            joint_threshold_count = []
            for idx, pair in enumerate(pairs):
                progress(idx, len(pairs))
                dist, tpr_count = pair[0].dice_similarity(pair[1], count_over=tpr_threshold)
                d_dists.append(dist)
                joint_threshold_count.append(tpr_count)

        print("Statistics...")

        d_dists = np.array(list(zip(d_dists, pairs)), dtype=object)
        srt = d_dists[d_dists[:, 0].argsort()]

        joint_threshold_sum = np.sum(np.array(joint_threshold_count))
        joint_tpr = joint_threshold_sum / (len(pairs) * 12)  # 12 joints per hand
        print(f"Joint TPR: {joint_tpr}")

        hand_threshold_sum = np.count_nonzero(np.array(joint_threshold_count) == 12)  # All joints over threshold
        hand_tpr = hand_threshold_sum / len(pairs)
        print(f"Hand TPR: {hand_tpr}")

        print()

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

    if meta:
        print("====== Meta Statistics =========")
        e_list = list(e_dists[:, 0])
        d_list = list(d_dists[:, 0])

        print(f"Coef. Corr. Dice & Euclidean: {np.corrcoef(e_list, d_list)[0][1]}")

    if joint_stats:
        print("====== Joint < TPR Threshold Statistics ======")

        joint_pairs = []
        for pair in pairs:
            size = pair[0].image.size
            for a, b in zip(pair[0].joints, pair[1].joints):
                joint_pairs.append((a, b, size))

        if mp:
            with Pool(cpu_count() - 1) as pool:
                results = pool.starmap(_joint_mp_work, joint_pairs)
            d_j_dists = [r[0] for r in results]
            d_j_labels = [r[1] for r in results]
        else:
            d_j_dists = []
            d_j_labels = []
            for idx, pair in enumerate(joint_pairs):
                progress(idx, len(joint_pairs))
                dist, name = pair[0].dice_similarity(pair[1], pair[2], return_name=True)
                d_j_dists.append(dist)
                d_j_labels.append(name)

        miss_counts = {}
        for dist, label in zip(d_j_dists, d_j_labels):
            if dist < tpr_threshold:
                if label not in miss_counts.keys():
                    miss_counts[label] = 1
                else:
                    miss_counts[label] += 1
        print("Miss Counts Below TPR Threshold: ")
        print(str(miss_counts))

    if save:
        print("Saving arrays...")
        with open('euclidean.npy', 'wb') as f:
            np.save(f, e_dists)
        with open('dice.npy', 'wb') as f:
            np.save(f, d_dists)


if __name__ == '__main__':
    main()
