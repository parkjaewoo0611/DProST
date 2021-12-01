#!/bin/bash
# OCCLUSION
server=( 
147.46.111.101 # 1
147.46.111.101 # 5
147.46.121.38 # 6
147.46.111.101 # 8
147.46.111.101 # 9
147.46.111.101 # 10
147.46.121.38 # 11
147.46.111.101 # 12
) 
folder=(
1106_143824 # 1   ~1200
1109_234909 # 5   ~1100
1110_234218 # 6   ~1200
1111_183232 # 8   ~1000
1106_143826 # 9   ~1100
1109_234949 # 10  ~1000
1110_231702 # 11  ~1200
1111_183234 # 12  ~1000
)
early_stop_epoch=(
1200
1100
1200
900
800
1000
700
1000
)
obj=(
1
5
6
8
9
10
11
12
)
dataset=(
"OCCLUSION"
"OCCLUSION"
"OCCLUSION"
"OCCLUSION"
"OCCLUSION"
"OCCLUSION"
"OCCLUSION"
"OCCLUSION"
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
  # sshpass -p "Isac930419!" scp isaackang@${s}:Pose/ProjectivePose/saved/models/ProjectivePose/${f}/checkpoint-epoch${e}.pth saved/models/${d}_${e}/${o}/checkpoint-epoch${e}.pth
  # sshpass -p "Isac930419!" scp isaackang@${s}:Pose/ProjectivePose/saved/models/ProjectivePose/${f}/config.json saved/models/${d}_${e}/${o}/config.json
  # sshpass -p "Isac930419!" scp isaackang@${s}:Pose/ProjectivePose/saved/log/ProjectivePose/${f}/info.log saved/log/${d}_${e}/${o}/info.log

  echo "testing model: ${d}_${e}--> result: ${d}_${e}_${it} testing obj: ${o}..."
  mkdir -p saved/results/${d}_${e}_${it}/${o}
  python test.py -c saved/models/${d}_${e}/${o}/config.json -r saved/models/${d}_${e}/${o}/checkpoint-epoch${e}.pth -d 0 --result_path saved/results/${d}_${e}_${it}/${o} --start_level ${st} --end_level ${en}
done
