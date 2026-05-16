#!/bin/bash
set -euo pipefail

# Jobs to run
games=("BattleZone" "NameThisGame" "Phoenix")
repeats=(0)

# Partitions to target (edit as needed)
partitions=("a100" "swarm_a100" "swarm_h100" "swarm_l4" "quad_h200" "dual_h200")

pick_resources() {
  case "$1" in
    a100)                       echo "24 60:00:00"   ;; # 24 CPUs, 60h
    swarm_a100|swarm_h100)      echo "24 120:00:00"  ;; # 24 CPUs, 120h
    swarm_l4)                   echo "12 120:00:00"  ;; # 12 CPUs, 120h
    quad_h200)                  echo "12 60:00:00"   ;; # 12 CPUs, 60h
    dual_h200)                  echo "24 60:00:00"   ;; # 24 CPUs, 60h
    *)                          echo "24 60:00:00"   ;; # default
  esac
}

for part in "${partitions[@]}"; do
  read -r CPUS WALLTIME <<<"$(pick_resources "$part")"

  for game in "${games[@]}"; do
    for rep in "${repeats[@]}"; do

      job_name="${game:0:3}${rep}"

      sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
#SBATCH --job-name=${job_name}
#SBATCH --time=${WALLTIME}
#SBATCH -p ${part}
#SBATCH --account=ecs

module load conda/python3
source activate Rainbow

cd /home/\$USER/Rainbow
pip install -e .
export WANDB_MODE=offline

python Rainbow.py --game ${game} --repeat ${rep}
EOF

    done
  done
done
