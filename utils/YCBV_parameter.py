import numpy as np

YCBV_idx2class = {
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

YCBV_class2idx = {
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

YCBV_idx2symmetry = {
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

YCBV_idx2syms= {
    1 : [{
          "R": np.array([[-1, -4.31991e-014, 0], 
                         [4.31991e-014,  -1, 0],
                         [0,              0, 1]]),
          "t": np.array([[0], [0], [0]])
    }],
    2 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    3 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    4 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    5 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    6 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    7 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    8 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    9 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    10 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    11 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    12 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    13 : [{
          "R": np.array([[-1, -4.31991e-014, 0], 
                         [4.31991e-014,  -1, 0],
                         [0,              0, 1]]),
          "t": np.array([[0], [0], [0]])
    }],
    14 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    15 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
}

LM_idx2diameter = {
    1 : 102.099,
    2 : 247.506,
    #3 : 167.355,
    4 : 172.492,
    5 : 201.404,
    6 : 154.546,
    #7 : 124.264,
    8 : 261.472,
    9 : 108.999,
    10 : 164.628,
    11 : 175.889,
    12 : 145.543,
    13 : 278.078,
    14 : 282.601,
    15 : 212.358,
}

LM_idx2radius = {
    1 : 59.5355,
    2 : 140.3643,
    4 : 99.6404,
    5 : 110.6762,
    6 : 84.6778,
    8 : 145.8172,
    9 : 63.9980,
    10 : 82.9911,
    11 : 91.8091,
    12 : 75.5216,
    13 : 145.0773,
    14 : 148.1368,
    15 : 109.9537,
}

## radius of DeepIM/data/LINEMOD_6D/LM6d_converted/LM6d_refine/models/**/textured.obj
LM_idx2synradius = {
    1 : 0.0595,
    2 : 0.1404,
    4 : 0.0996,        
    5 : 0.1107,         
    6 : 0.0847,         
    8 : 0.1458,
    9 : 0.0640,         
    10 : 0.0830,
    11 : 0.0918,        
    12 : 0.0755,        
    13 : 0.1451,        
    14 : 0.1481,
    15 : 0.1100         
}


FX = 572.4114
FY = 573.57043
PX = 325.2611
PY = 242.04899

K = np.array([[FX,  0, PX],
              [ 0, FY, PY],
              [ 0,  0,  1]])

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
obj_path = '../Dataset/LINEMOD/models'
for obj_id in LM_idx2class.keys():
    VSD_REN.add_object(obj_id, os.path.join(obj_path, f'obj_{obj_id:06d}.ply'))

MSSD_THRESHOLD = np.arange(0.05, 0.51, 0.05)[:, np.newaxis]
MSPD_THRESHOLD = np.arange(5, 51, 5)[:, np.newaxis] * WIDTH/640