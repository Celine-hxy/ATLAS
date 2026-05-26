import argparse
import json
import os
import pickle
from math import comb
from time import sleep

import vllm.envs as envs
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.parser import *
from utils.scorer import *
from utils.utils import set_seed
from vllm import LLM, SamplingParams


def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--result_filename', type=str, default="result.json")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy 
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def infer(args, settings):
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")

    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    llm = LLM(model=model_name_or_path, 
              tensor_parallel_size=len(available_gpus), 
              trust_remote_code=True, 
              gpu_memory_utilization=0.96,
              )

    results = []
    for data_name, n_sampling, k in settings:
        factor = 1
        for i in range(2, 65):
            if n_sampling % i == 0:
                factor = i
        generation_epoch = n_sampling // factor
        print(f"use n = {factor}, generation epoch is: {generation_epoch}")
        sampling_params = SamplingParams(temperature=args.temperature, 
                                        max_tokens=args.max_tokens, 
                                        n=factor,
                                        top_p=args.top_p,
                                        top_k=args.top_k,
                                        )
        
        examples = load_data(data_name, args.split, args.data_dir)

        prompt_batch = []
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
        for example in tqdm(examples, total=len(examples)):
            # parse question and answer
            question = parse_question(example, data_name)
            cur_prompt = question + " " + instruction_following

            messages = [
                {"role": "user", "content": cur_prompt}
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
            prompt_batch.append(cur_prompt)
        # print(prompt_batch[0])
        
        model_name = "/".join(args.model_name_or_path.split("/")[-3:])
        out_file_prefix = f'{args.split}_t{args.temperature}'
        out_file = f'{args.output_dir}/{model_name}/{data_name}/{out_file_prefix}_k{n_sampling}.jsonl'

        os.makedirs(f'{args.output_dir}/{model_name}/{data_name}', exist_ok=True)
        os.makedirs(f'{args.completions_save_dir}/{model_name}/{data_name}', exist_ok=True)

        file_outputs = []
        correct_cnt = 0
        multi_correct_cnt = 0
        for cur_generation_epoch in range(generation_epoch):
            completions_save_file = f'{args.completions_save_dir}/{model_name}/{data_name}/{out_file_prefix}_k{n_sampling}_gen_round{cur_generation_epoch}.pkl'
            
            completions = llm.generate(prompt_batch, sampling_params)
            
            save_completions(completions, completions_save_file)
            for i in range(len(examples)):
                d = examples[i]
                question = parse_question(d, data_name)
                generated_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]
                if cur_generation_epoch == 0:
                    file_outputs.append({
                        "question": question,
                        "generated_responses": generated_responses,
                    })
                    if "id" in d:
                        file_outputs[i]["id"] = d["id"]
                    if "source" in d:
                        file_outputs[i]["source"] = d["source"]
                else:
                    file_outputs[i]['generated_responses'] += generated_responses
        print("llm generate done")
        print(len(file_outputs))
        
        pass_at_k_list = []

        for i in tqdm(range(len(examples)), "check correct..."):
            d = examples[i]
            gt_cot, gt_ans = parse_ground_truth(d, data_name)
            generated_responses = file_outputs[i]['generated_responses']
            # generated_answers = [extract_answer(generated_response, data_name) for generated_response in generated_responses]
            # is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            is_correct_list = [compute_score(generated_response, gt_ans) for generated_response in generated_responses]
            is_correct = any(is_correct_list)
            multi_correct_cnt += sum(is_correct_list)
            if is_correct:
                correct_cnt += 1
            # file_outputs[i]['generated_answers'] = generated_answers
            file_outputs[i]['gold_answer'] = gt_ans
            file_outputs[i]['is_correct'] = is_correct
            file_outputs[i]['answers_correctness'] = is_correct_list
            
            if len(is_correct_list) > 1:
                correct_answers = sum(is_correct_list)
                n = len(generated_responses)
                if correct_answers > 0:
                    if n - correct_answers < k:
                        pass_at_k = 1
                    else:
                        pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                    pass_at_k_list.append(pass_at_k)
                else:
                    pass_at_k_list.append(0)
                
        temp_out_file = out_file + ".tmp"
        with open(temp_out_file, 'w', encoding='utf-8') as f:
            count = 0
            for d in tqdm(file_outputs, "writing generation to jsonl file..."):
                f.write(json.dumps(d, ensure_ascii=False))
                f.write("\n")
                count += 1
                if count % 100 == 0:
                    f.flush()
            f.flush()
        os.rename(temp_out_file, out_file)

        mean_metric_name = f"Mean@{n_sampling}"
        acc_metric_name = f"Acc (Pass@{n_sampling})"
        pass_metric_name = f"Pass@{k}"

        print(f"========== {data_name} ==========")
        print(f"Multi-correct cnt / total cnt: {multi_correct_cnt}/{len(examples) * n_sampling}")
        mean_metric_val = 100 * multi_correct_cnt / (len(examples)*n_sampling)
        print(f"{mean_metric_name}: {mean_metric_val:.2f}")
        print("-" * 100)
        
        print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
        acc_metric_val = 100 * correct_cnt / len(examples)
        print(f"{acc_metric_name}: {acc_metric_val:.2f}")
        print("-" * 100)
        
        if pass_at_k_list:
            pass_metric_val = 100 * sum(pass_at_k_list) / len(pass_at_k_list)
        else:
            pass_metric_val = 100 * correct_cnt / len(examples)
        print(f"{pass_metric_name}: {pass_metric_val:.2f}")

        results.append({
            "data_name": data_name,
            mean_metric_name: mean_metric_val,
            pass_metric_name: pass_metric_val,
        })

    return results


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    settings = [
        ("hle_math", 4, 4),
        ("math500", 4, 4),
        ("minerva", 4, 4),
        ("olympiad", 4, 4),
        ("amc23", 32, 32),
        ("aime24", 32, 32),
        ("aime25", 32, 32),
        ("amo_bench", 32, 32),
        ("gpqa", 4, 4),
    ]

    results = []
    results = infer(args, settings)

    for result in results:
        print(f"========== {result['data_name']} ==========")
        for metric_name, metric_val in result.items():
            if metric_name == "data_name":
                continue
            print(f"{metric_name}: {metric_val:.2f}")
    
    result_filename = args.result_filename.replace("/", "_")
    result_path = f'results/{result_filename}'
    with open(result_path, 'w') as f:
        json.dump(results, f)