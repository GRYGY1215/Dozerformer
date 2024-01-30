# Random Seeds
seeds=(1 2022 2023 2024 2025 2026)

lr=5e-4
model=dozerformer
patch_size=24
# Dozer attention parameters
local_window=3
stride=0
vary_len=1
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 24---------------------------------------
    python run.py --seed $seed --data ILI --model $model --moving_avg '13, 17' \
    --seq_len 120 --label_len 24 --pred_len 24 --embed_dim 64 --dropout 0.4 \
    --learning_rate $lr --patch_size $patch_size \
    --local_window $local_window --stride $stride --vary_len $vary_len

    #----------------------------------predict length 36---------------------------------------
    python run.py --seed $seed --data ILI --model $model --moving_avg '13, 17' \
    --seq_len 120 --label_len 24 --pred_len 36 --embed_dim 64 --dropout 0.4 \
    --learning_rate $lr --patch_size $patch_size \
    --local_window $local_window --stride $stride --vary_len $vary_len

    #----------------------------------predict length 48---------------------------------------
    python run.py --seed $seed --data ILI --model $model --moving_avg '13, 17' \
    --seq_len 120 --label_len 24 --pred_len 48 --embed_dim 64 --dropout 0.4 \
    --learning_rate 5e-4 --patch_size 24 \
    --local_window 3 --stride $stride --vary_len $vary_len

    #----------------------------------predict length 60---------------------------------------
    python run.py --seed $seed --data ILI --model $model --moving_avg '13, 17' \
    --seq_len 120 --label_len 24 --pred_len 60 --embed_dim 64 --dropout 0.2 \
    --learning_rate $lr --patch_size $patch_size \
    --local_window $local_window --stride $stride --vary_len $vary_len
done
