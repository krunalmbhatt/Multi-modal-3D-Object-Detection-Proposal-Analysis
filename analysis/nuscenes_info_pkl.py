# analysis/inspect_nuscenes_infos.py
import pickle
import numpy as np
from pprint import pprint

INFOS_PATH = 'data/nuscenes/nuscenes_infos_val.pkl'

def main():
    infos = pickle.load(open(INFOS_PATH, 'rb'))
    print(f"type(infos) = {type(infos)}")
    # usually infos is a dict with key 'infos'
    raw_list = infos.get('infos', infos.get('data_infos'))
    print(f"Number of entries in raw_list = {len(raw_list)}\n")

    # look at the first entry
    first = raw_list[0]
    print("first.keys() =")
    pprint(list(first.keys()))
    print()

    # for each key, print out a summary of its type/shape
    for k, v in first.items():
        if isinstance(v, np.ndarray):
            print(f"{k:20s}: ndarray, shape = {v.shape}, dtype = {v.dtype}")
        elif isinstance(v, list):
            print(f"{k:20s}: list, length = {len(v)}, sample = {v[:2]}")
        else:
            print(f"{k:20s}: {type(v)} – {v}")

    # if 'gt_boxes' is present, show a few rows
    if 'gt_boxes' in first:
        print("\nFirst few GT boxes (gt_boxes):")
        print(first['gt_boxes'][:5])

if __name__ == '__main__':
    main()


