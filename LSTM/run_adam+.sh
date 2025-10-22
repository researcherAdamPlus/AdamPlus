run=0
nlayer=1
# lr=0.003
for lr in 0.004 0.005 ; do
  for seed in {141..145}; do
    echo "=== starting run $run === $(date)"
    CUDA_VISIBLE_DEVICES=0 python -u main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 \
      --seed "$seed" --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 \
      --beta2 0.999 --optimizer adam+ --lr "$lr" --eps 1e-16 --eps_sqrt 0.0 \
      --nlayer ${nlayer} --run "$run" 2>&1 | tee -a "logs/run_0_seed_${seed}_nlayer_${nlayer}_lr${lr}_adam+.log"
  done
done