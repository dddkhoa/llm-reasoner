from reasoners.lm import ExLlamaModel
import json
from reasoners.benchmark import BWEvaluator
import fire
import torch

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature
    def __call__(self, example, prompt=None):
        inputs = prompt["icl"].replace("<init_state>", example["init"])\
            .replace("<goals>", example["goal"]).replace("<action>", "")
        output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=True,
                                          temperature=self.temperature,
                                          eos_token_id='\n[').text[0][:-1].strip()
        return output

def main(llama_path='meta-llama/Llama-2-13b-chat-hf', 
         peft_path=None,
         data_path='examples/blocksworld/data/step_4.json', 
         prompt_path='examples/blocksworld/prompts/prompt.json',
         disable_log=False, 
         batch_size=1, 
         config_file: str = "examples/blocksworld/data/bw_config.yaml",
         domain_file: str = "examples/blocksworld/data/generated_domain.pddl", 
         quantized = "nf4", 
         load_awq_pth = None,
         resume=0, log_dir=None, temperature=0.8):

    # base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir,
    #                       mem_map=exllama_mem_map, max_batch_size=batch_size,
    #                       max_new_tokens=300, max_seq_length=2048)
    from reasoners.lm import HFModel
        

    with open(prompt_path) as f:
        prompt = json.load(f)

    device = torch.device("cuda:0")
    base_model = HFModel(llama_path, llama_path, device=device, max_batch_size=1, max_new_tokens=512, quantized=quantized, peft_pth=peft_path, load_awq_pth=load_awq_pth)

    reasoner = CoTReasoner(base_model, temperature=temperature)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log, output_extractor=lambda x:x, sample_prompt_type="rap") # rap prompt includes cot
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)