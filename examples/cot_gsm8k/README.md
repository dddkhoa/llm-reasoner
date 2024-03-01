```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/cot_gsm8k/inference.py --exllama_model_dir $LLAMA2_CKPTS --exllama_lora_dir None --exllama_mem_map '[16,22]' | tee cot_log.log
```
accuracy: 0.461

CUDA_VISIBLE_DEVICES=0 python examples/cot_gsm8k/inference.py --base_lm hf --hf_path meta-llama/Llama-2-13b-chat-hf --log_dir logs/gsm8k_cot_v1.1 --hf_quantized 'nf4' | tee cot_log.log