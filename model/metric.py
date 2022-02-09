from utils.bop_toolkit.bop_toolkit_lib.pose_error import vsd, mssd, mspd, re, te, proj, add, adi
import numpy as np
from scipy.integrate import simps

# unit of distance -> mm, rotation -> degree
def RE_TE_02(RE, TE, **kwargs):
    check = [(re < 2) and (te < 20) for re, te in zip(RE, TE)]
    score = np.mean(check) * 100
    return score

def RE_TE_05(RE, TE, **kwargs):
    check = [(re < 5) and (te < 50) for re, te in zip(RE, TE)]
    score = np.mean(check) * 100
    return score

# unit of distance -> pixel
def PROJ_02(PROJ, **kwargs):
    check = [proj < 2 for proj in PROJ]
    score = np.mean(check) * 100
    return score

def PROJ_05(PROJ, **kwargs):
    check = [proj < 5 for proj in PROJ]
    score = np.mean(check) * 100
    return score

def PROJ_10(PROJ, **kwargs):
    check = [proj < 10 for proj in PROJ]
    score = np.mean(check) * 100
    return score

# unit of distance -> mm
def ADD_S_02(ADD_S, diameter, **kwargs):
    THR = [d * 0.02 for d in diameter]
    check = [add_s < thr for add_s, thr in zip(ADD_S, THR)]
    score = np.mean(check) * 100
    return score

def ADD_S_05(ADD_S, diameter, **kwargs):
    THR = [d * 0.05 for d in diameter]
    check = [add_s < thr for add_s, thr in zip(ADD_S, THR)]
    score = np.mean(check) * 100
    return score

def ADD_S_10(ADD_S, diameter, **kwargs):
    THR = [d * 0.10 for d in diameter]
    check = [add_s < thr for add_s, thr in zip(ADD_S, THR)]
    score = np.mean(check) * 100
    return score

# threshold from 0 ~ 10cm 
# from https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi/blob/master/lib/utils/eval.py
# Our RT values follow mm 
dx = 1
THR = np.arange(0, 100, dx).astype(np.float32)
N_THR = THR.shape[0]
def ADD_AUC(ADD, **kwargs):
    N = len(ADD)
    count_correct = np.zeros(N_THR, dtype=np.float32)
    for s in ADD:
        correct = s < THR
        count_correct += correct 
    area = simps(count_correct / float(N), dx=dx)
    return area

def ADD_S_AUC(ADD_S, **kwargs):
    N = len(ADD_S)
    count_correct = np.zeros(N_THR, dtype=np.float32)
    for s in ADD_S:
        correct = s < THR
        count_correct += correct 
    area = simps(count_correct / float(N), dx=dx)
    return area

def ADD_SS_AUC(ADD_SS, **kwargs):
    N = len(ADD_SS)
    count_correct = np.zeros(N_THR, dtype=np.float32)
    for s in ADD_SS:
        correct = s < THR
        count_correct += correct 
    area = simps(count_correct / float(N), dx=dx)
    return area
