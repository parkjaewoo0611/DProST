import numpy as np

LM_idx2class = {
    1: "ape",
    2: "benchvise",
    #3: 'bowl',
    4: "camera",
    5: "can",
    6: "cat",
    #7: 'cup',
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

LM_class2idx = {
    "ape" : 1,
    "benchvise" : 2,
    #'bowl' : 3,
    "camera" : 4,
    "can" : 5,
    "cat": 6,
    #"cup" : 7,
    "driller" : 8,
    "duck" : 9,
    "eggbox" : 10,
    "glue" : 11,
    "holepuncher" : 12,
    "iron" : 13,
    "lamp" : 14,
    "phone" : 15,
}

LM_idx2symmetry = {
    1 : 'none',
    2 : 'none',
    #3 : 'sym_con',
    4 : 'none',
    5 : 'none',
    6 : 'none',
    #7 : 'none',
    8 : 'none',
    9 : 'none',
    10 : 'sym_dis',
    11 : 'sym_dis',
    12 : 'none',
    13 : 'none',
    14 : 'none',
    15 : 'none',
}

LM_idx2syms= {
    1 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    2 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    #3 : 'sym_con',
    4 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    5 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    6 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    #7 : 'none',
    8 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    9 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    10 : [{"R": np.array([[-0.999964, -0.00333777, -0.0077452], 
                         [0.00321462, -0.999869, 0.0158593  ],
                         [-0.00779712, 0.0158338, 0.999844 ]]),
          "t": np.array([[0.232611], [0.694388], [-0.0792063]])
    }],
    11 : [{"R": np.array([[-0.999633, 0.026679, 0.00479336], 
                         [-0.0266744, -0.999644, 0.00100504  ],
                         [0.00481847, 0.000876815, 0.999988 ]]),
          "t": np.array([[-0.262139], [-0.197966], [0.0321652]])
    }],
    12 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
    }],
    13 : [{"R": np.eye(3, 3),
          "t": np.zeros([3, 1])
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
#from utils.bop_toolkit.bop_toolkit_lib import renderer
#import os
#TAUS = list(np.arange(0.05, 0.51, 0.05))
WIDTH = 640
HEIGHT = 480

# vsd parameters from bop_toolkit.bop_toolkit_lib.eval_calc_scores
#VSD_DELTA = 15
#VSD_THRESHOLD = np.arange(0.05, 0.51, 0.05)[:, np.newaxis]
#VSD_NORMALIZED_BY_DIAMETER = True
#VSD_REN = renderer.create_renderer(WIDTH, HEIGHT, 'vispy', mode='depth')
#obj_path = '../Dataset/LINEMOD/models'
#for obj_id in LM_idx2class.keys():
#    VSD_REN.add_object(obj_id, os.path.join(obj_path, f'obj_{obj_id:06d}.ply'))

#MSSD_THRESHOLD = np.arange(0.05, 0.51, 0.05)[:, np.newaxis]
#MSPD_THRESHOLD = np.arange(5, 51, 5)[:, np.newaxis] * WIDTH/640
