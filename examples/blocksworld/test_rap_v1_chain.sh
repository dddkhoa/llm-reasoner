# prompt with the same length
# set the depth with the max length

export VAL=/home/kaiser/llm-reasoners/LLMs-Planning/planner_tools/VAL

CUDA_VISIBLE_DEVICES=0 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_2_data.json' --depth_limit 2 --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/chat_v1_step2  --n_iters 3

# CUDA_VISIBLE_DEVICES=0 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_4_data.json' --depth_limit 4 --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/chat_v1_step4  --n_iters 3

# CUDA_VISIBLE_DEVICES=0 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_6_data.json' --depth_limit 6  --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/chat_v1_step6  --n_iters 3

# CUDA_VISIBLE_DEVICES=0 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_8_data.json' --depth_limit 8 --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step8  --n_iters 3

# CUDA_VISIBLE_DEVICES=0 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_10_data.json' --depth_limit 10 --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step10  --n_iters 3

# CUDA_VISIBLE_DEVICES=0 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_12_data.json' --depth_limit 12 --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step12  --n_iters 3