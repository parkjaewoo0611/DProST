import sys
sys.path.append('bop_toolkit')
import pickle
import argparse

def main(dataset_name, mode):
    if dataset_name == 'lm':
        data_root_path = 'data/LINEMOD'
    if dataset_name == 'lmo':
        data_root_path = 'data/OCCLUSION'

    if dataset_name == 'lm' and (mode == 'train' or mode == 'test'):
        from LINEMOD.data_preprocess_lm import Preprocessor

    elif dataset_name == 'lmo' and (mode == 'train' or mode == 'test'):
        from LINEMOD.data_preprocess_lmo import Preprocessor

    elif dataset_name == 'lm' and mode == 'train_syn':
        from LINEMOD.data_preprocess_syn_deepim import Preprocessor
    
    elif dataset_name == 'lmo' and mode == 'train_syn':
        from LINEMOD.data_preprocess_syn_deepim import Preprocessor
        # from data.data_preprocess.LINEMOD.data_preprocess_syn_pvnet import Preprocessor

    elif mode == 'train_pbr':
        from LINEMOD.data_preprocess_pbr import Preprocessor
    

    loader = Preprocessor(mode=mode, name=dataset_name).build()
    dataset = []
    for i, sample in enumerate(loader):
        batch = {
            'image' : sample['image'],
            'depth' : sample['depth'],
            'depth_scale' : sample['depth_scale'],
            'obj_id' : sample['obj_id'],
            'mask' : sample['mask'],
            'bbox_obj' : sample['bbox_obj'],
            'RT' : sample['RT'],
            'visib_fract': sample['visib_fract']
        }
        if mode == 'test':
            batch['bbox_faster'] = sample['bbox_faster']      
        dataset.append(batch)
        print(i, '/', len(loader))
        

    file = f"{data_root_path}/{mode}.pickle"
    with open(file, "wb") as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-m', '--mode', default='train_syn', type=str, help='train, test, train_pbr, train_syn')
    args.add_argument('-d', '--dataset', default='lmo', type=str, help='lm, lmo')
    args = args.parse_args()
    main(args.dataset, args.mode)
