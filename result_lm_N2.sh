#!/bin/bash
# LINEMOD
server=( 
147.46.111.101 # 1 
147.46.111.101 # 2
147.46.111.101 # 4
147.46.111.101 # 5
147.46.112.21 # 6 
147.46.112.21 # 8
147.46.241.106 # 9 
147.46.66.98 # 10
147.46.111.101 # 11
147.46.241.93 # 12
147.46.111.101 # 13 
147.46.241.93 # 14
147.46.241.93 # 15
) 
folder=(
1122_142430 # 1 ape         3000
1122_142436 # 2 benchwise   3000
1122_142500 # 4 cam         3000
1122_142443 # 5 can         3000
1122_143046 # 6 cat         3000
1122_143345 # 8 driller     3000
1122_144654 # 9 duck        3000
1122_144918 # 10  eggbox    3000
1124_165624 # 11 glue       2800
1124_164624 # 12 holep      3000
1124_165645 # 13 iron       3000
1122_145225 # 14 lamp       3000
1122_145235 # 15 phone      3000
)
early_stop_epoch=(
2700
2700
2500
2700
2600
2900
2700
2700
2500
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
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
"iter3"
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
0
0
0
0
0
0
0
0
0
0
0
0
0
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

  # mkdir -p saved/models/${d}_${e}_N2/${o}
  # mkdir -p saved/log/${d}_${e}_N2/${o}
  # echo server: ${s} folder: ${f} obj: ${o}
  # echo "moving to destination: saved/models/${d}_${e}_N2/${o}"
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/models/ProjectivePose/${f}/checkpoint-epoch${e}.pth saved/models/${d}_${e}_N2/${o}/checkpoint-epoch${e}.pth
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/models/ProjectivePose/${f}/config.json saved/models/${d}_${e}_N2/${o}/config.json
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/log/ProjectivePose/${f}/info.log saved/log/${d}_${e}_N2/${o}/info.log

  echo "testing model: ${d}_${e}_N2 --> result: ${d}_${e}_${it}_N2 testing obj: ${o}..."
  mkdir -p saved/results/${d}_${e}_N2/${o}
  python test.py -c saved/models/${d}_${e}_N2/${o}/config.json -r saved/models/${d}_${e}_N2/${o}/checkpoint-epoch${e}.pth -d 0 --result_path saved/results/${d}_${e}_${it}_N2/${o} --start_level ${st} --end_level ${en}
done