#!/bin/zsh

# for seed in {0..9}; do
for seed in 4; do

    # python ../../core/run/ur5_ppo_continuous_action.py --env_id "ur_reacher_2d" --optim "adam_hesscale" --learning_rate "0.0003" --delta "0.00000001" --total_timesteps "300000" --exp_name "rl_ppo_ur5" --seed $seed
    # killall -u homayoon python

    # python ../../core/run/ur5_ppo_continuous_action.py --env_id "ur_reacher_2d" --optim "adahesscale_adamstyle_scaled" --learning_rate "0.0003" --delta "0.00000001" --total_timesteps "300000" --exp_name "rl_ppo_ur5" --seed $seed
    # killall -u homayoon python

    python ../../core/run/ur5_ppo_continuous_action.py --env_id "ur_reacher_2d" --optim "adam" --learning_rate "0.0003" --total_timesteps "300000" --exp_name "rl_ppo_ur5" --seed $seed
    killall -u homayoon python

    python ../../core/run/ur5_ppo_continuous_action.py --env_id "ur_reacher_2d" --optim "adahesscale_adamstyle" --learning_rate "0.0003" --total_timesteps "300000" --exp_name "rl_ppo_ur5" --seed $seed
    killall -u homayoon python

done
