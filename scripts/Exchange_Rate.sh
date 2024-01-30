# Random Seeds
seeds=(1 2022 2023 2024 2025 2026)
lr=5e-5
model=dozerformer
patch_size=48
# Dozer attention parameters
local_window=3
stride=0
vary_len=1
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data Exchange --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 32 --dropout 0.2 \
    --learning_rate $lr --patch_size $patch_size --loss L1 \
    --local_window $local_window --stride $stride --vary_len $vary_len

    #----------------------------------predict length 192---------------------------------------
    python run.py --seed $seed --data Exchange --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 48 --pred_len 192 --embed_dim 32 --dropout 0.2 \
    --learning_rate $lr --patch_size 24 --loss L1 \
    --local_window 3 --stride $stride --vary_len 1

    #----------------------------------predict length 336---------------------------------------
    python run.py --seed $seed --data Exchange --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 32 --dropout 0.2 \
    --learning_rate $lr --patch_size 48 --loss L1 \
    --local_window 3 --stride $stride --vary_len 1

    #----------------------------------predict length 720---------------------------------------
    python run.py --seed $seed --data Exchange --model $model --moving_avg '13, 17' \
    --seq_len 336 --label_len 96 --pred_len 720 --embed_dim 32 --dropout 0.2 \
    --learning_rate $lr --patch_size 48 --loss L1 \
    --local_window 3 --stride $stride --vary_len 1
done








