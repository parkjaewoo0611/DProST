#!/bin/bash
# LINEMOD
server=( 
147.46.66.98    # 1   2200~3000 -->3000
147.46.66.98    # 2   1400~3000
147.46.66.98    # 4   1700~3000
147.46.66.98    # 5   1100~3000
147.46.242.23   # 6   100~3000
147.46.241.106  # 8   300~3000
147.46.66.98    # 9   1400~3000
147.46.242.23   # 10  3000
147.46.241.106  # 11  300~3000
147.46.241.106  # 12  300~3000
147.46.242.23   # 13  100~3000
147.46.241.93   # 14  3000
147.46.241.93   # 15  1100~3000
) 
folder=(
1108_230155 # 1 ape         done  91.4
1111_020525 # 2 benchwise   done  100.0   2700
1111_020633 # 4 cam         done  98.3  
1111_020415 # 5 can         done  99.4
1106_143924 # 6 cat         done  97.1
1111_161858 # 8 driller     done  99.8 
1113_125136 # 9 duck        done  86.9
1110_213641 # 10  eggbox    done  95.6
1111_162046 # 11 glue       done  100.0
1111_163027 # 12 holep      done  95.9
1106_143927 # 13 iron       done  99.4
1113_140754 # 14 lamp       done  99.7
1111_165841 # 15 phone      done  94.5
)
early_stop_epoch=(
3000
2700
2500
2700
2600
2900
2700
2700
2800
2700
2500
2700
3000
)
obj=(
1
2
4
5
6
8
9
10
11
12
13
14
15
)
dataset=(
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
"LINEMOD"
)
iter=(
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
"iter2"
)
start_level=( 
2
2
2
2
2
2
2
2
2
2
2
2
2
)

end_level=( 
1
1
1
1
1
1
1
1
1
1
1
1
1
)

for i in "${!obj[@]}"
do
  s=${server[i]}
  f=${folder[i]}
  e=${early_stop_epoch[i]}
  o=${obj[i]}
  d=${dataset[i]}
  it=${iter[i]}
  en=${end_level[i]}
  st=${start_level[i]}

  # mkdir -p saved/models/${d}_${e}/${o}
  # mkdir -p saved/log/${d}_${e}/${o}
  # echo server: ${s} folder: ${f} obj: ${o}
  # echo "moving to destination: saved/models/${d}_${e}/${o}"
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/models/ProjectivePose/${f}/checkpoint-epoch${e}.pth saved/models/${d}_${e}/${o}/checkpoint-epoch${e}.pth
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/models/ProjectivePose/${f}/config.json saved/models/${d}_${e}/${o}/config.json
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/log/ProjectivePose/${f}/info.log saved/log/${d}_${e}/${o}/info.log

  echo "testing model: ${d}_${e}--> result: ${d}_${e}_${it} testing obj: ${o}..."
  mkdir -p saved/results/${d}_${e}_${it}/${o}
  python test.py -c saved/models/${d}_${e}/${o}/config.json -r saved/models/${d}_${e}/${o}/checkpoint-epoch${e}.pth -d 0 --result_path saved/results/${d}_${e}_${it}/${o} --start_level ${st} --end_level ${en}
done
