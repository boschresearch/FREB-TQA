import json
from perturbation import *
perturb_map = {"row_shuffle": row_shuffling, "col_shuffle": col_shuffling, "target_row_random": target_row_random, "target_row_front": target_row_front,
               "target_row_middle": target_row_middle, "target_row_bottom": target_row_bottom, "target_col_front": target_col_front, "target_col_random": target_col_random,
               "target_col_back": target_col_back, "transpose": transposed, "mask_cell_random_wtq": wtq_mask_cell_all_random, "mask_cell_all_wtq": wtq_mask_cell_all,
               "mask_cell_all_tat": tat_cell_mask_all, "mask_cell_random_tat": tat_cell_mask_random, "relevant_row_shuffle_wtq": shuffling_relevant_rows_wtq,
               "relevant_row_shuffle_tat": shuffling_relevant_row_tat, "escape_table": escape_table}


def merge_part1():
    datasets = ["wtq", "tat", "sqa", "wikisql"]
    METHOD_STRUCTURE_PERTURBATION = ['row_shuffle', 'col_shuffle', 'target_row_random', 'target_row_front',
                                     'target_row_middle', 'target_row_bottom', 'target_col_random', 'target_col_front', 'target_col_back', 'transpose']

    seeds = [33]
    all_new = []
    for dataset in datasets:
        with open(f"../dataset/01_str/{dataset}_factoid_dev.json", "r") as f:
            data = [json.loads(line) for line in f]
        for seed in seeds:
            for item in data:
                for method in METHOD_STRUCTURE_PERTURBATION:
                    print(dataset, seed, method)
                    perturb_fn = perturb_map[method]
                    table_shuffle = perturb_fn(
                        item["table"], seed, item["answers"])
                    new_item = {k: v for k, v in item.items()}
                    new_item["table"] = table_shuffle
                    new_item.update(
                        {"seed": seed, "dataset": dataset, "perturb": method})

                    all_new.append(new_item)
        for item in data:
            item.update({"seed": 33, "dataset": dataset,
                        "perturb": "no_shuffle"})
            all_new.append(item)

    print(len(all_new))
    with open("../dataset/01_str/structure_perturb.json", "w") as f:
        for item in all_new:
            f.write(json.dumps(item)+"\n")


def merge_part2():
    datasets = ["wtq", "tat"]
    ATTENTION_TO_REL = ['escape_table', 'mask_cell_all',
                        'mask_cell_random', 'relevant_row_shuffle']

    seeds = [33]
    all_new = []
    for dataset in datasets:
        with open(f"../dataset/02_rel/{dataset}_nonfactoid_dev.json", "r") as f:
            data = [json.loads(line) for line in f]
        for method in ATTENTION_TO_REL:
            if method != "escape_table":
                method = method+f"_{dataset}"
            perturb_fn = perturb_map[method]
            for item in data:
                if dataset == "tat":
                    table = item["table"]
                else:
                    table = item["table"]
                if 'relevant_row_shuffle' in method or 'mask_cell_random' in method:
                    for seed in seeds:
                        table_shuffle = perturb_fn(item, seed)
                        new_item = {k: v for k, v in item.items()}
                        new_item["table"] = table_shuffle
                        new_item.update(
                            {"seed": seed, "dataset": dataset, "perturb": method})
                        all_new.append(new_item)
                else:
                    if method == "escape_table":
                        table_shuffle = perturb_fn(table, 33, '')
                    else:
                        table_shuffle = perturb_fn(item, 33)
                    new_item = {k: v for k, v in item.items()}
                    new_item["table"] = table_shuffle
                    new_item.update(
                        {"seed": 33, "dataset": dataset, "perturb": method})
                    all_new.append(new_item)
        for item in data:
            item.update({"seed": 33, "dataset": dataset,
                        "perturb": "no_shuffle"})
            all_new.append(item)

    print(len(all_new))
    with open("../dataset/02_rel/attention_rel_cells.json", "w") as f:
        for item in all_new:
            f.write(json.dumps(item)+"\n")


def merge_part3():
    all_data = []
    method_lists = ['no_shuffle_short_table', 'no_shuffle_ans_change',
                    'no_shuffle_ans_no_change', 'no_shuffle_ori_table']
    datasets = ["wtq", "tat"]
    for dataset in datasets:
        for method in method_lists:
            with open(f"../dataset/03_agg_com/{dataset}_nonfactoid_dev_{method}.json", "r") as f:
                data = [json.loads(line) for line in f]
                for item in data:
                    item.update(
                        {"dataset": dataset, "perturb": method, "seed": 33})
                    all_data.append(item)

    with open(f"../dataset/03_agg_com/aggregation_compare.json", "w") as f:
        for item in all_data:
            f.write(json.dumps(item)+"\n")


merge_part3()
