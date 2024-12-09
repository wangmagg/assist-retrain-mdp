#!/bin/bash

declare -a reward_config_paths=(
    "configs/reward/reward_low_retrain_cost.json"
    "configs/reward/reward_medium_retrain_cost.json"
)
declare -a switch_config_paths=(
    "configs/switch/switch_ambig.json"
    "configs/switch/switch_high_risk.json"
    "configs/switch/switch_uniform.json"
)

for reward_config_path in "${reward_config_paths[@]}"
do  
    for switch_config_path in "${switch_config_paths[@]}"
    do
        python3 -m src.simulate_dqn \
            --reward-config-path $reward_config_path \
            --switch-config-path $switch_config_path 
    done
done