import numpy as np

import sys
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
from utils.bop_toolkit.bop_toolkit_lib.inout import load_json
from utils.bop_toolkit.bop_toolkit_lib.misc import get_symmetry_transformations

idx2class = {
    1 : "002_master_chef_can",
    2 : "003_cracker_box",
    3 : "004_sugar_box",
    4 : "005_tomato_soup_can",
    5 : "006_mustard_bottle",
    6 : "007_tuna_fish_can",
    7 : "008_pudding_box",
    8 : "009_gelatin_box",
    9 : "010_potted_meat_can",
    10 : "011_banana",
    11 : "019_pitcher_base",
    12 : "021_bleach_cleanser",
    13 : "024_bowl",
    14 : "025_mug",
    15 : "035_power_drill",
    16 : "036_wood_block",
    17 : "037_scissors",
    18 : "040_large_marker",
    19 : "051_large_clamp",
    20 : "052_extra_large_clamp",
    21 : "061_foam_brick"
}

class2idx = {
    "002_master_chef_can" : 1,
    "003_cracker_box" : 2,
    '004_sugar_box' : 3,
    "005_tomato_soup_can" : 4,
    "006_mustard_bottle" : 5,
    "007_tuna_fish_can": 6,
    "008_pudding_box" : 7,
    "009_gelatin_box" : 8,
    "010_potted_meat_can" : 9,
    "011_banana" : 10,
    "019_pitcher_base" : 11,
    "021_bleach_cleanser" : 12,
    "024_bowl" : 13,
    "025_mug" : 14,
    "035_power_drill" : 15,
    "036_wood_block" : 16,
    "037_scissors" : 17,
    "040_large_marker" : 18,
    "051_large_clamp" : 19,
    "052_extra_large_clamp" : 20,
    "061_foam_brick" : 21
}

idx2symmetry = {
    1 : "sym_dis",
    2 : "none",
    3 : "none",
    4 : "none",
    5 : "none",
    6 : "none",
    7 : "none",
    8 : "none",
    9 : "none",
    10 : "none",
    11 : "none",
    12 : "none",
    13 : "sym_con",
    14 : "none",
    15 : "none",
    16 : "sym_dis",
    17 : "none",
    18 : "sym_con",
    19 : "sym_dis",
    20 : "sym_dis",
    21 : "sym_dis"
}

idx2radius = {
    1 : 86.1740,
    2 : 135.6663,
    3 : 99.6637,
    4 : 60.8266,
    5 : 103.6818,
    6 : 45.8036,
    7 : 72.0897,
    8 : 57.4040,
    9 : 66.3194,
    10 : 103.9851,
    11 : 136.4676,
    12 : 133.7827,
    13 : 84.7634,
    14 : 70.4044,
    15 : 122.1458,
    16 : 119.1654,
    17 : 110.5286,
    18 : 60.9245,
    19 : 99.6208,
    20 : 128.6310,
    21 : 52.1546,
}


obj_path = '../Dataset/YCBV/models'
mesh_info_path = f'{obj_path}/models_info.json'
mesh_infos = load_json(mesh_info_path)

idx2syms = {}
idx2diameter = {}
for idx in list(mesh_infos.keys()):
      mesh_info = mesh_infos[idx]
      idx2syms[int(idx)] = get_symmetry_transformations(mesh_info, max_sym_disc_step = 1/8)
      idx2diameter[int(idx)] = mesh_info['diameter']

# cmu camera intrinsic
CMU = {
      'fx': 1077.836,
      'fy': 1078.189,
      'px': 323.7872,
      'py': 279.6921,
}
CMU['K'] = np.array([[CMU['fx'],  0,         CMU['px']],
                     [0,          CMU['fy'], CMU['py']],
                     [0,          0,         1]])
                     
# uw camera intrinsic
UW = {
      'fx': 1066.778,
      'fy': 1067.487,
      'px': 312.9869,
      'py': 241.3109,
}
UW['K'] = np.array([[UW['fx'],  0,        UW['px']],
                    [0,         UW['fy'], UW['py']],
                    [0,         0,        1]])

# parameters for metric function
from utils.bop_toolkit.bop_toolkit_lib import renderer
import os
TAUS = list(np.arange(0.05, 0.51, 0.05))
WIDTH = 640
HEIGHT = 480

# vsd parameters from bop_toolkit.bop_toolkit_lib.eval_calc_scores
VSD_DELTA = 15
VSD_THRESHOLD = np.arange(0.05, 0.51, 0.05)[:, np.newaxis]
VSD_NORMALIZED_BY_DIAMETER = True
VSD_REN = renderer.create_renderer(WIDTH, HEIGHT, 'vispy', mode='depth')

for obj_id in idx2class.keys():
    VSD_REN.add_object(obj_id, os.path.join(obj_path, f'obj_{obj_id:06d}.ply'))

MSSD_THRESHOLD = np.arange(0.05, 0.51, 0.05)[:, np.newaxis]
MSPD_THRESHOLD = np.arange(5, 51, 5)[:, np.newaxis] * WIDTH/640