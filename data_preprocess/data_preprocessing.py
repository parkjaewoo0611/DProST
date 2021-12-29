import sys
sys.path.append('bop_toolkit')
import pickle
import argparse

def main(dataset, mode):
    if dataset == 'lm':
        from LINEMOD.data_preprocess_lm import Preprocessor
    elif dataset == 'lmo':
        from LINEMOD.data_preprocess_lmo import Preprocessor
    elif dataset == 'pbr':
        from LINEMOD.data_preprocess_pbr import Preprocessor
    elif dataset == 'syn':
        from LINEMOD.data_preprocess_syn import Preprocessor
    loader = Preprocessor(mode=mode).build()
    dataset = []
    for i, sample in enumerate(loader):
        batch = {
            'image' : sample['image'],
            'obj_id' : sample['obj_id']
        }
        target = {
            'mask' : sample['mask'],
            'bbox_obj' : sample['bbox_obj'],
            'RT' : sample['TCO'],
            'visib_fract': sample['visib_fract']
        }
        if mode == 'test':
            target['bbox_faster'] = sample['bbox_faster']      
        dataset.append((batch, target))
        print(i, '/', len(loader))
    if mode == 'train' and dataset == 'pbr':
        file = f"./train_pbr.pickle"
    if mode == 'train' and dataset == 'syn':
        file = f"./train_syn.pickle"
    else:
        file = f"./{mode}.pickle"
    with open(file, "wb") as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-m', '--mode', default='train', type=str, help='train or test')
    args.add_argument('-d', '--dataset', default='syn', type=str, help='lm, lmo, pbr, syn')
    args = args.parse_args()
    main(args.dataset, args.mode)
