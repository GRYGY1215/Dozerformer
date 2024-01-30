# 10 mins
# Random Seeds
seeds=(1 2022 2023 2024 2025 2026)
lr=1e-3
model=dozerformer_Linear
patch_size=96
# Dozer attention parameters
local_window=1
stride=2
vary_len=1
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data electricity --model $model --moving_avg '13, 17' \
    --seq_len 720 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate 1e-3 --patch_size $patch_size \
    --local_window $local_window --stride $stride --vary_len $vary_len

    #----------------------------------predict length 192---------------------------------------
    python run.py --seed $seed --data electricity --model $model --moving_avg '13, 17' \
    --seq_len 720 --label_len 96 --pred_len 192 --embed_dim 8 \
    --learning_rate 1e-3 --patch_size $patch_size \
    --local_window $local_window --stride $stride --vary_len $vary_len

    #----------------------------------predict length 336---------------------------------------
    python run.py --seed $seed --data electricity --model $model --moving_avg '13, 17' \
    --seq_len 720 --label_len 96 --pred_len 336 --embed_dim 8 \
    --learning_rate 1e-3 --patch_size $patch_size \
    --local_window $local_window --stride $stride --vary_len $vary_len

    #----------------------------------predict length 720---------------------------------------
    python run.py --seed $seed --data electricity --model $model --batch_size 16 --moving_avg '13, 17' \
    --seq_len 720 --label_len 96 --pred_len 720 --embed_dim 8 \
    --learning_rate 1e-3 --patch_size $patch_size \
    --local_window $local_window --stride $stride --vary_len $vary_len
done
