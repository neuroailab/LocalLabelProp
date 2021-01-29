import numpy as np
import os
import argparse

DATA_DIR = os.environ.get('DATA_DIR', '/data5/chengxuz/Dataset/llp_pub')
FULL_LABEL_NPY = os.path.join(DATA_DIR, 'all_label.npy')
NUM_CAT = 1000


def main():
    percents = [1, 2, 3, 5, 6, 10, 20, 50]
    all_labels = np.load(FULL_LABEL_NPY)
    save_dir = os.path.join(DATA_DIR, 'metas')
    os.system('mkdir -p ' + save_dir)

    for percent in percents:
        per_cat_img = percent * 13
        part_label = []
        part_index = []
        curr_idx = 0
        no_img_cat = {cat_idx: 0 for cat_idx in range(NUM_CAT)}

        while (len(part_label) < per_cat_img * NUM_CAT) \
                and (len(part_label) < len(all_labels)):
            curr_label = all_labels[curr_idx]
            if no_img_cat[curr_label] < per_cat_img:
                no_img_cat[curr_label] += 1
                part_label.append(curr_label)
                part_index.append(curr_idx)
            curr_idx += 1

        lbl_path = os.path.join(save_dir, 'p%02i_label.npy' % percent)
        idx_path = os.path.join(save_dir, 'p%02i_index.npy' % percent)
        np.save(lbl_path, np.asarray(part_label))
        np.save(idx_path, np.asarray(part_index))


if __name__ == '__main__':
    main()
