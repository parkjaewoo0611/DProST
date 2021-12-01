#!/bin/bash
# LINEMOD
server=( 
147.46.111.101 # 1 
147.46.111.101 # 2
147.46.111.101 # 4
147.46.111.101 # 5
147.46.241.93 # 6 
147.46.241.93 # 8
147.46.112.21 # 9 
147.46.112.21 # 10
147.46.66.98 # 11
147.46.66.98 # 12
147.46.66.98 # 13 
147.46.241.106 # 14
147.46.241.106 # 15
) 
folder=(
1119_165552 # 1 ape         3000
1119_165559 # 2 benchwise   3000
1119_165505 # 4 cam         3000
1119_165457 # 5 can         3000
1118_192242 # 6 cat         3000
1118_192247 # 8 driller     3000
1118_192525 # 9 duck        3000
1118_192517 # 10  eggbox    3000
1122_004505 # 11 glue       2800
1122_004527 # 12 holep      3000
1122_004546 # 13 iron       3000
1122_003420 # 14 lamp       3000
1122_003422 # 15 phone      3000
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

  # mkdir -p saved/models/${d}_${e}_N4/${o}
  # mkdir -p saved/log/${d}_${e}_N4/${o}
  # echo server: ${s} folder: ${f} obj: ${o}
  # echo "moving to destination: saved/models/${d}_${e}_N4/${o}"
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/models/ProjectivePose/${f}/checkpoint-epoch${e}.pth saved/models/${d}_${e}_N4/${o}/checkpoint-epoch${e}.pth
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/models/ProjectivePose/${f}/config.json saved/models/${d}_${e}_N4/${o}/config.json
  # sshpass -p "Jaewoo0611." scp -P 416 bjw0611@${s}:/data/bjw0611/Pose/ProjectivePose/saved/log/ProjectivePose/${f}/info.log saved/log/${d}_${e}_N4/${o}/info.log

  echo "testing model: ${d}_${e}_N4 --> result: ${d}_${e}_${it}_N4 testing obj: ${o}..."
  mkdir -p saved/results/${d}_${e}_N4/${o}
  python test.py -c saved/models/${d}_${e}_N4/${o}/config.json -r saved/models/${d}_${e}_N4/${o}/checkpoint-epoch${e}.pth -d 0 --result_path saved/results/${d}_${e}_${it}_N4/${o} --start_level ${st} --end_level ${en}
done