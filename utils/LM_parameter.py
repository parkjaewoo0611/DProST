import numpy as np

idx2class = {
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

class2idx = {
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

idx2symmetry = {
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

idx2syms= {
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

idx2diameter = {
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

idx2radius = {
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


FX = 572.4114
FY = 573.57043
PX = 325.2611
PY = 242.04899

K = np.array([[FX,  0, PX],
              [ 0, FY, PY],
              [ 0,  0,  1]])
WIDTH = 640
HEIGHT = 480

# # parameters for metric function
# from utils.bop_toolkit.bop_toolkit_lib import renderer
# import os
# TAUS = list(np.arange(0.05, 0.51, 0.05))
# # vsd parameters from bop_toolkit.bop_toolkit_lib.eval_calc_scores
# VSD_DELTA = 15
# VSD_THRESHOLD = np.arange(0.05, 0.51, 0.05)[:, np.newaxis]
# VSD_NORMALIZED_BY_DIAMETER = True
# VSD_REN = renderer.create_renderer(WIDTH, HEIGHT, 'vispy', mode='depth')
# obj_path = '../Dataset/LINEMOD/models'
# for obj_id in idx2class.keys():
#    VSD_REN.add_object(obj_id, os.path.join(obj_path, f'obj_{obj_id:06d}.ply'))
# MSSD_THRESHOLD = np.arange(0.05, 0.51, 0.05)[:, np.newaxis]
# MSPD_THRESHOLD = np.arange(5, 51, 5)[:, np.newaxis] * WIDTH/640

DATA_PARAM = {
        'idx2symmetry' : idx2symmetry,
        'idx2diameter' : idx2diameter,
        'idx2radius' : idx2radius,
        'idx2syms' : idx2syms,
        'idx2class' : idx2class,
      #   'taus' : TAUS,
      #   'vsd_delta' : VSD_DELTA,
      #   'vsd_threshold' : VSD_THRESHOLD,
      #   'vsd_normalized_by_diameter' : VSD_NORMALIZED_BY_DIAMETER,
      #   'vsd_ren' : VSD_REN,
      #   'mssd_threshold' : MSSD_THRESHOLD,
      #   'mspd_threshold' : MSPD_THRESHOLD
    }