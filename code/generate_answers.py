""" Utility classes and functions related to FREB-TQA (NAACL 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
from transformers import TapexTokenizer, BartForConditionalGeneration, TapasTokenizer, TapasForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM
from perturbation import *
import json
import pandas as pd
from utils import execute, to_float32s, normalize_answers, reformat_table
import torch
from tqdm import tqdm
import os


def generate_answer(model, tokenizer, ids, query, table, model_type, device, output_path):
    with open(output_path, 'w') as f:
        for i, (id, q, t) in tqdm(enumerate(zip(ids, query, table))):
            if model_type == 'tapex':
                t = pd.DataFrame(t['rows'], columns=t['header'])
                encoding = tokenizer(
                    table=t, query=q, return_tensors="pt").to(device)
                truncate = False
                if len(encoding['input_ids'][0]) > 1024:
                    truncate = True
                    input_ids = encoding['input_ids'][0][:1024]
                    attention_mask = encoding['attention_mask'][0][:1024]
                    encoding = {'input_ids': input_ids.unsqueeze(
                        dim=0), 'attention_mask': attention_mask.unsqueeze(dim=0)}
                try:
                    outputs = model.generate(**encoding)
                    pred = tokenizer.batch_decode(
                        outputs, skip_special_tokens=True)
                except:
                    pred = "none"
            elif model_type == 'tapas':
                t = pd.DataFrame(t['rows'], columns=t['header'])
                truncate = False
                try:
                    inputs = tokenizer(table=t, queries=q,
                                       return_tensors="pt").to(device)
                except ValueError:
                    inputs = tokenizer(
                        table=t, queries=q, truncation=True, return_tensors="pt").to(device)
                if len(inputs['input_ids'][0]) > 512:
                    truncate = True
                    input_ids = inputs['input_ids'][0][:512]
                    attention_mask = inputs['attention_mask'][0][:512]
                    token_type_ids = inputs['token_type_ids'][0][:512]
                    inputs = {'input_ids': input_ids.unsqueeze(dim=0), 'attention_mask': attention_mask.unsqueeze(
                        dim=0), 'token_type_ids': token_type_ids.unsqueeze(dim=0)}
                try:
                    outputs = model(**inputs)
                    cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
                    coordinates, aggregation_inds = tokenizer.convert_logits_to_predictions(
                        cpu_inputs, outputs.logits.detach().cpu(), outputs.logits_aggregation.detach().cpu())
                    coordinates, aggregation_inds = coordinates[0], aggregation_inds[0]
                    id2aggregation = {0: "NONE", 1: "SUM",
                                      2: "AVERAGE", 3: "COUNT"}
                    predicted_agg = id2aggregation[aggregation_inds]
                    denotation, values = execute(predicted_agg, coordinates, t)
                    denotation = to_float32s(denotation)
                    denotation = normalize_answers(denotation)
                    pred = [denotation[0] if len(denotation) == 1 else "none"]
                except:
                    pred = "none"
            else:
                table_string = reformat_table(table)
                prompt = f"Answer the question according to the table and. Return the answer following Answer:[final answer]. Table: {table_string} Question:{query}\nAnswer:"
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer(
                    [text], return_tensors="pt").to(device)
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                pred = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)[0]
                truncate = False
            instance_new = {
                "id": id, "truncate": truncate, "pred_answer": pred}
            f.write(json.dumps(instance_new)+"\n")
    return


def main(dataset_path: str, dataset_name: str, model_path: str, perturb: str, seed: int, model_type: str, part: str, device: str):
    system = model_path.split("/")[-1].strip()
    perturb_map = {"row_shuffle": row_shuffling, "col_shuffle": col_shuffling, "target_row_random": target_row_random, "target_row_front": target_row_front,
                   "target_row_middle": target_row_middle, "target_row_bottom": target_row_bottom, "target_col_front": target_col_front, "target_col_random": target_col_random,
                   "target_col_back": target_col_back, "transpose": transposed, "mask_cell_random_wtq": wtq_mask_cell_all_random, "mask_cell_all_wtq": wtq_mask_cell_all,
                   "mask_cell_all_tat": tat_cell_mask_all, "mask_cell_random_tat": tat_cell_mask_random, "relevant_row_shuffle_wtq": shuffling_relevant_rows_wtq,
                   "relevant_row_shuffle_tat": shuffling_relevant_row_tat, "escape_table": escape_table}

    if model_type == 'tapex':
        tokenizer = TapexTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)

    elif model_type == 'tapas':
        model = TapasForQuestionAnswering.from_pretrained(model_path)
        tokenizer = TapasTokenizer.from_pretrained(
            model_path, drop_rows_to_fit=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,  torch_dtype=torch.bfloat16, device_map="auto", use_safetensors=True, trust_remote_code=True).eval()

    device = torch.device(device)
    model.to(device)

    # load dataset
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    if not os.path.isdir('../results'):
        os.makedirs('../results/1')
        os.makedirs('../results/2')
        os.makedirs('../results/3')

    ids = [item['id'] for item in data]
    query = [item['question'] for item in data]
    answer = [item['answers'] for item in data]
    table = [item['table'] for item in data]

    if perturb in ["no_shuffle", "no_shuffle_short_table", "no_shuffle_ans_change",
                   "no_shuffle_ans_no_change", "no_shuffle_ori_table"]:
        generate_answer(model, tokenizer, ids, query, table, model_type, device,
                        output_path=f"../results/{part}/{system}_{dataset_name}_{seed}_{perturb}.json")

    elif perturb == "escape_table":
        perturb_fn = perturb_map[perturb]
        table_shuffle = [perturb_fn(t, seed, '') for t in table]
        generate_answer(model, tokenizer, ids, query, table_shuffle, model_type, device,
                        output_path=f"../results/2/{system}_{dataset_name}_{seed}_{perturb}.json")

    elif perturb in ["mask_cell_random_wtq", "mask_cell_all_wtq", "relevant_row_shuffle_wtq",
                     "mask_cell_random_tat", "mask_cell_all_tat", "relevant_row_shuffle_tat"]:
        perturb_fn = perturb_map[perturb]
        table_shuffle = [perturb_fn(instance, seed) for instance in data]
        generate_answer(model, tokenizer, ids, query, table_shuffle, model_type, device,
                        output_path=f"../results/2/{system}_{dataset_name}_{seed}_{perturb}.json")

    else:
        perturb_fn = perturb_map[perturb]
        table_shuffle = [perturb_fn(t, seed, a) for t, a in zip(table, answer)]
        generate_answer(model, tokenizer, ids, query, table_shuffle, model_type, device,
                        output_path=f"../results/{part}/{system}_{dataset_name}_{seed}_{perturb}.json")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str,
                    help="The path to the dataset")
parser.add_argument("--dataset_name", type=str,
                    choices=['wtq', 'tat', 'sqa', 'wikisql'],
                    help="The name of the dataset")
parser.add_argument("--model_path", type=str,
                    help="The path to the model")
parser.add_argument("--perturb_method", type=str,
                    choices=['row_shuffle', 'col_shuffle', 'target_row_random', 'target_col_random',
                             'target_row_front', 'target_row_middle', 'target_row_bottom',
                             'target_col_front', 'target_col_back', 'no_shuffle',
                             'transpose', 'mask_cell_all_wtq', 'mask_cell_random_wtq',
                             'relevant_row_shuffle_wtq', 'mask_cell_all_tat', 'mask_cell_random_tat',
                             'relevant_row_shuffle_tat', 'escape_table', 'no_shuffle_short_table',
                             'no_shuffle_ans_change', 'no_shuffle_ans_no_change', 'no_shuffle_ori_table'],
                    help="The methods used to perturb the table")
parser.add_argument("--seed", type=int, choices=[33, 16, 77, 69, 42],
                    default=33)
parser.add_argument("--part", type=int, choices=[1, 2, 3],
                    default=1)
parser.add_argument("--model_type", type=str, choices=['tapex', 'tapas', 'llm'],
                    default='tapex')
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

print(f"dataset_path: {args.dataset_path},dataset_name: {args.dataset_name}, \
      model_path: {args.model_path}, part: {args.part}\
      perturb_method: {args.perturb_method},seed: {args.seed}, model_type:{args.model_type}")

if __name__ == "__main__":
    main(args.dataset_path, args.dataset_name, args.model_path,
         args.perturb_method, args.seed, args.model_type, args.part, args.device)
