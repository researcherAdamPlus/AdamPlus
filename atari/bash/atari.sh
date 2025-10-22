#!/bin/bash
trap "echo 'Killing all child processes...'; kill 0" EXIT
output_dir="output"
# env="BreakoutNoFrameskip-v4"
# lr=0.0001


#!/bin/bash
pids=()
trap 'echo "Ctrl-C detected. Killing all child processes..."; kill "${pids[@]}" 2>/dev/null' SIGINT

output_dir="output"
max_jobs=20
CONFIG_FILE="./bash/atari_config.csv"

read -r header < "$CONFIG_FILE"

while IFS=, read -r id opt beta1 beta2 seed lr batch_size total_timesteps learning_starts end_e gpu env run; do
    run=$(echo "$run" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ "$run" != "TRUE" ]; then
        # echo "skipping: $opt $beta1 $beta2 (run=$run)"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge $max_jobs ]; do
        sleep 1
    done
    echo $id $opt $beta1 $beta2 $seed $lr $batch_size $total_timesteps $learning_starts $end_e $gpu $env
    CUDA_VISIBLE_DEVICES=${gpu}\
    python dqn_atari.py --seed=${seed}\
                    --optimizer=${opt}\
                    --beta_1=${beta1}\
                    --beta_2=${beta2}\
                    --seed=${seed}\
                    --learning_rate=${lr}\
                    --batch_size=${batch_size}\
                    --total_timesteps=${total_timesteps}\
                    --learning_starts=${learning_starts}\
                    --end_e=${end_e}\
                    --env_id=${env} &
    
        pids+=($!)
done < <(tail -n +2 "$CONFIG_FILE")
wait


# agent_name="Rainbow"

# max_jobs=36

# for lin_layers in 2; do
#     for db in -6 ; do
#         for seed in 34 ; do
#             for beta2 in 0.9 0.999; do
#                 # for opt in "Adan" "AdaShift" "AMSGrad" "RMSprop" "PIDAOSI" ; do
#                 for opt in "Adam" "AdamPlus" ; do
#                     outname="${env}_${agent_name}_${opt}_${beta2}_seed${seed}_${db}"
#                     echo ${outname}
#                     while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
#                         sleep 1
#                     done
#                     python rainbow_atari.py --seed=${seed}\
#                                         --env_id=${env}\
#                                         --optimizer=${opt}\
#                                          --beta_2=${beta2}  &
#                 done
#             done
#         done
#     done
# done
# wait
#                                         #
#                                         # --lin_layers=${lin_layers}\