import sys
sys.path.append('bop_toolkit')
import pickle
import argparse

def main(dataset_name, mode):
    if dataset_name == 'lm':
        from LINEMOD.data_preprocess_lm import Preprocessor
    elif dataset_name == 'lmo':
        from LINEMOD.data_preprocess_lmo import Preprocessor
    elif dataset_name == 'pbr':
        from LINEMOD.data_preprocess_pbr import Preprocessor
    elif dataset_name == 'syn':
        from LINEMOD.data_preprocess_syn import Preprocessor
    loader = Preprocessor(mode=mode).build()
    dataset = []
    for i, sample in enumerate(loader):
        batch = {
            'image' : sample['image'],
            'depth' : sample['depth'],
            'depth_scale' : sample['depth_scale'],
            'obj_id' : sample['obj_id'],
            'mask' : sample['mask'],
            'bbox_obj' : sample['bbox_obj'],
            'RT' : sample['TCO'],
            'visib_fract': sample['visib_fract']
        }
        if mode == 'test':
            batch['bbox_faster'] = sample['bbox_faster']      
        dataset.append(batch)
        print(i, '/', len(loader))
    if mode == 'train' and dataset_name == 'pbr':
        file = f"./train_pbr.pickle"
    elif mode == 'train' and dataset_name == 'syn':
        file = f"./train_syn.pickle"
    else:
        file = f"./{mode}.pickle"
    with open(file, "wb") as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-m', '--mode', default='test', type=str, help='train or test')
    args.add_argument('-d', '--dataset', default='lm', type=str, help='lm, lmo, pbr, syn')
    args = args.parse_args()
    main(args.dataset, args.mode)
