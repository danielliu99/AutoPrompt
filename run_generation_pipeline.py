from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml, modify_input_for_ranker, validate_generation_config, override_config
import argparse
import os
from estimator.estimator_llm import LLMEstimator
# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--generation_config_path', default='config/config_diff/config_generation.yml', type=str, help='Configuration file path')
parser.add_argument('--ranker_config_path', default='config/config_diff/config_ranking.yml', type=str, help='Configuration file path')

parser.add_argument('--task_description',
                    default='',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--prompt',
                    default='',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--load_dump', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_ranker_steps', default=20, type=int, help='Number of iterations')
parser.add_argument('--num_generation_steps', default=20, type=int, help='Number of iterations')

opt = parser.parse_args()

# print(opt.prompt)

ranker_config_params = override_config(opt.ranker_config_path)
generation_config_params = override_config(opt.generation_config_path)
validate_generation_config(ranker_config_params, generation_config_params)

if opt.task_description == '':
    task_description = input("Describe the task: ")
else:
    # task_description = opt.task_description
    with open(opt.task_description, "r", encoding="utf-8") as file:
      task_description = file.read()

if opt.prompt == '':
    initial_prompt = input("Initial prompt: ")
else:
    # initial_prompt = opt.prompt
    with open(opt.prompt, "r", encoding="utf-8") as file:
      initial_prompt = file.read()


"""prompts/meta_prompts_ranking"""
ranker_pipeline = OptimizationPipeline(ranker_config_params, output_path=os.path.join(opt.output_dump, 'ranker'))
if opt.load_dump != '':
    ranker_pipeline.load_state(os.path.join(opt.load_dump, 'ranker'))
    ranker_pipeline.predictor.init_chain(ranker_config_params.dataset.label_schema)

if (ranker_pipeline.cur_prompt is None) or (ranker_pipeline.task_description is None):
    """根据初始prompt和description，修改用于优化ranker的prompt和task description (prompts/modifiers)"""
    ranker_mod_prompt, ranker_mod_task_desc = modify_input_for_ranker(ranker_config_params, task_description,
                                                                      initial_prompt)
    ranker_pipeline.cur_prompt = ranker_mod_prompt
    ranker_pipeline.task_description = ranker_mod_task_desc

# best_prompt = ranker_pipeline.run_pipeline(opt.num_ranker_steps)
generation_config_params.eval.function_params = ranker_config_params.predictor.config
# generation_config_params.eval.function_params.instruction = best_prompt['prompt']
generation_config_params.eval.function_params.instruction = ranker_config_params.annotator.config.instruction
generation_config_params.eval.function_params.label_schema = ranker_config_params.dataset.label_schema

print(generation_config_params.eval.function_params.instruction)

generation_pipeline = OptimizationPipeline(generation_config_params, task_description, initial_prompt,
                                           output_path=os.path.join(opt.output_dump, 'generator'))
if opt.load_dump != '':
    generation_pipeline.load_state(os.path.join(opt.load_dump, 'generator'))
best_generation_prompt = generation_pipeline.run_pipeline(opt.num_generation_steps)
print('\033[92m' + 'Calibrated prompt score:', str(best_generation_prompt['score']) + '\033[0m')
print('\033[92m' + 'Calibrated prompt:', best_generation_prompt['prompt'] + '\033[0m')
