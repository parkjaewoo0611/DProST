import numpy as np
import matplotlib.pyplot as plt 
import json

obj_name = ['ape', 'benchwise', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
def graph_value(error, max_dis=10):
    D = np.array(error)
    D[np.where(D > max_dis)] = np.inf;
    D = np.sort(D)
    n = len(error)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n * 100
    x = np.array([0.0]+list(D)+[max_dis])
    y = np.array([0.0]+list(acc)+[acc[-1]])
    return x, y

error_type = 'TXE'
max_dis = 5
title = "translation x"
x_label = "threshold(mm)"
y_label = "accuracy(%)"


colors=['orange', 'purple', 'green', 'red', 'blue', 'black', 'darkgreen', 'gold', 'olive', 'pink', 'teal', 'grey', 'brown']
plt.gca().set_prop_cycle(color=colors)
gm_files = [
    'ECCV2022/LM_pbr_each/1/log/error_1.json',
    'ECCV2022/LM_pbr_each/2/log/error_2.json',
    'ECCV2022/LM_pbr_each/4/log/error_4.json',
    'ECCV2022/LM_pbr_each/5/log/error_5.json',
    'ECCV2022/LM_pbr_each/6/log/error_6.json',
    'ECCV2022/LM_pbr_each/8/log/error_8.json',
    'ECCV2022/LM_pbr_each/9/log/error_9.json',
    'ECCV2022/LM_pbr_each/10/log/error_10.json',
    'ECCV2022/LM_pbr_each/11/log/error_11.json',
    'ECCV2022/LM_pbr_each/12/log/error_12.json',
    'ECCV2022/LM_pbr_each/13/log/error_13.json',
    'ECCV2022/LM_pbr_each/14/log/error_14.json',
    'ECCV2022/LM_pbr_each/15/log/error_15.json',
    ]
errors = []

for i, path in enumerate(gm_files):
    obj = path.split('_')[-1][:-5]
    with open(path, 'r') as f:
        error = json.load(f)[error_type]
        # errors.extend([error_type])
    x, y = graph_value(error, max_dis)
    plt.plot(x,y,label=f"{obj_name[i]}")

plt.title(title,fontsize=17)
plt.xlabel(x_label,fontsize=17)
plt.ylabel(y_label,fontsize=17)
plt.legend(fontsize=11, loc='lower right')
plt.savefig('gm.png',dpi=300)
plt.clf()
plt.cla()
plt.close()



colors=['orange', 'purple', 'green', 'red', 'blue', 'black', 'darkgreen', 'gold', 'olive', 'pink', 'teal', 'grey', 'brown']
plt.gca().set_prop_cycle(color=colors)
pm_files = [
    'ECCV2022/LM_PM/1/log/error_1.json',
    'ECCV2022/LM_PM/2/log/error_2.json',
    'ECCV2022/LM_PM/4/log/error_4.json',
    'ECCV2022/LM_PM/5/log/error_5.json',
    'ECCV2022/LM_PM/6/log/error_6.json',
    'ECCV2022/LM_PM/8/log/error_8.json',
    'ECCV2022/LM_PM/9/log/error_9.json',
    'ECCV2022/LM_PM/10/log/error_10.json',
    'ECCV2022/LM_PM/11/log/error_11.json',
    'ECCV2022/LM_PM/12/log/error_12.json',
    'ECCV2022/LM_PM/13/log/error_13.json',
    'ECCV2022/LM_PM/14/log/error_14.json',
    'ECCV2022/LM_PM/15/log/error_15.json',
    ]
errors = []

for i, path in enumerate(pm_files):
    obj = path.split('_')[-1][:-5]
    with open(path, 'r') as f:
        error = json.load(f)[error_type]
        # errors.extend([error_type])
    x, y = graph_value(error, max_dis)
    plt.plot(x,y,label=f"{obj_name[i]}")

plt.title(title,fontsize=17)
plt.xlabel(x_label,fontsize=17)
plt.ylabel(y_label,fontsize=17)
plt.legend(fontsize=11, loc='lower right')
plt.savefig('pm.png',dpi=300)
plt.clf()
plt.cla()
plt.close()



colors=['orange', 'purple', 'green', 'red', 'blue', 'black', 'darkgreen', 'gold', 'olive', 'pink', 'teal', 'grey', 'brown']
plt.gca().set_prop_cycle(color=colors)
im_files = [
    'ECCV2022/LM_IM/1/log/error_1.json',
    'ECCV2022/LM_IM/2/log/error_2.json',
    'ECCV2022/LM_IM/4/log/error_4.json',
    'ECCV2022/LM_IM/5/log/error_5.json',
    'ECCV2022/LM_IM/6/log/error_6.json',
    'ECCV2022/LM_IM/8/log/error_8.json',
    'ECCV2022/LM_IM/9/log/error_9.json',
    'ECCV2022/LM_IM/10/log/error_10.json',
    'ECCV2022/LM_IM/11/log/error_11.json',
    'ECCV2022/LM_IM/12/log/error_12.json',
    'ECCV2022/LM_IM/13/log/error_13.json',
    'ECCV2022/LM_IM/14/log/error_14.json',
    'ECCV2022/LM_IM/15/log/error_15.json',
    ]
errors = []
for i, path in enumerate(im_files):
    obj = path.split('_')[-1][:-5]
    with open(path, 'r') as f:
        error = json.load(f)[error_type]
        # errors.extend([error_type])
    x, y = graph_value(error, max_dis)
    plt.plot(x,y,label=f"{obj_name[i]}")

plt.title(title,fontsize=17)
plt.xlabel(x_label,fontsize=17)
plt.ylabel(y_label,fontsize=17)
plt.legend(fontsize=11, loc='lower right')
plt.savefig('im.png',dpi=300)
plt.clf()
plt.cla()
plt.close()