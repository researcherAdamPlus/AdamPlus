#!/bin/bash
pids=()
trap 'echo "Ctrl-C detected. Killing all child processes..."; kill "${pids[@]}" 2>/dev/null' SIGINT

output_dir="output"
max_jobs=35
CONFIG_FILE="./bash/mnist_config.csv"

read -r header < "$CONFIG_FILE"

while IFS=, read -r id opt beta1 beta2 seed epoch lr weight_decay decay_steps decay_rate batch_size run gpu env; do
    env=$(echo "$env" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ "$run" != "TRUE" ]; then
        echo "skipping: $opt $beta1 $beta2 (run=$run)"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge $max_jobs ]; do
        sleep 1
    done
    echo $id $opt $beta1 $beta2 $seed $epoch $lr $weight_decay $decay_steps $decay_rate $batch_size $run $gpu $env
    CUDA_VISIBLE_DEVICES=$gpu \
    python main.py --seed=${seed} \
                    --learning_rate=${lr} \
                    --optimizer=${opt} \
                    --beta_1=${beta1} \
                    --beta_2=${beta2} \
                    --model=${env} \
                    --weight_decay=${weight_decay} \
                    --batch_size=${batch_size} \
                    --epoch=${epoch} \
                    --decay_steps=${decay_steps} \
                    --decay_rate=${decay_rate} \
                    --db=-30 &
    
        pids+=($!)
done < <(tail -n +2 "$CONFIG_FILE")
wait


                # running_jobs=$(jobs -p | wc -l)
                # if [ $running_jobs -ge 40 ]; then
                #     wait
                # fi
# for lr in 0.0001 0.0005; do
#     for beta2 in 0.999; do
#         for seed in 51 ; do
#             for opt in "Adopt"; do
#                 # read -r beta2 <<<"${optimizer_params[$opt]}"
#                 outname="${opt}_${lr}_${beta2}_${seed}_${wd}"
#                 echo ${outname}
#                 python main.py --seed=${seed}\
#                                 --learning_rate=${lr}\
#                                 --optimizer=${opt}\
#                                 --beta_2=${beta2}\
#                                 --model=${model}\
#                                 --weight_decay=${wd}\
#                                 --batch_size=${batch_size}\
#                                 --epoch=${epoch}\
#                                 --db=-30 &

#             done
#         done
#     done
# done
# wait
# --beta_1=0.9\
