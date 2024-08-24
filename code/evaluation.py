import re
from codecs import open
from math import isnan, isinf
from abc import ABCMeta, abstractmethod
import json
import argparse
import os

################ String Normalization ################


def normalize(x):
    # if not isinstance(x, unicode):
    #    x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    # x = ''.join(c for c in unicodedata.normalize('NFKD', x)
    #            if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


################ Value Types ################

class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' + str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


################ Value Instantiation ################

def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                        in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


def exact_match_score(pred_path, gold_path):
    """
    calculate em 
    """
    results = {'non-truncate': 0, 'all': 0}
    with open(pred_path, "r") as f:
        pred = [json.loads(line) for line in f]
    with open(gold_path, "r") as f:
        factoid = [json.loads(line) for line in f]
        answers = {item['id']: item['answers'] for item in factoid}
    # align ids
    gold = [answers[item["id"]] for item in pred]
    for p, g in zip(pred, gold):
        p_ = p["pred_answer"]
        if not isinstance(p_, list):
            p_ = [p_]
        if not isinstance(g, list):
            g = [g]
        if check_denotation(to_value_list(g), to_value_list(p_)):
            if "truncate" in p.keys() and not p["truncate"]:
                results["non-truncate"] += 1
            results["all"] += 1
    if "truncate" in pred[0].keys():
        non_truncate_total = len(
            [item for item in pred if not item["truncate"]])
        return results["all"]/len(pred), results["non-truncate"]/non_truncate_total
    return results["all"]/len(pred), 0


def vp(shuffle_path: str, no_shuffle_path: str, gold_path: str):
    """
    Calculate variation percentage.
    Args:
        shuffle_path: path to the file to calculate vp for.
        no_shuffle_path: path to the file for original prediction without perturbation.
        gold_path: path to the gold prediction
    """
    with open(gold_path, "r") as f:
        data = [json.loads(line) for line in f]
        gold_answers = [item['answers'] for item in data]

    with open(shuffle_path, "r") as f:
        change_predictions = [json.loads(line) for line in f]

    with open(no_shuffle_path, "r") as f:
        original_predictions = [json.loads(line) for line in f]

    results_non_truncate = {'co_inco': [], 'co_co': [],
                            'inco_co': [], 'inco_inco': [], 'change': []}
    results_all = {'co_inco': [], 'co_co': [],
                   'inco_co': [], 'inco_inco': [], 'change': []}

    if not len(change_predictions) == len(original_predictions) == len(gold_answers):
        print(
            "Provided files length mismatch! Please check the file and rerun the results!")
        print(len(change_predictions), len(
            original_predictions), len(gold_answers))
    if "truncate" not in change_predictions[0].keys() and \
            "truncate" not in original_predictions[0].keys():
        truncate = [True for i in len(change_predictions)]
    else:
        truncate = [item["truncate"] for item in change_predictions]
    for i, (change, original, gold, t) in enumerate(zip(change_predictions,
                                                        original_predictions, gold_answers, truncate)):
        change_ans = change["pred_answer"]
        original_ans = original["pred_answer"]
        if not isinstance(gold, list):
            gold = [gold]
        if not isinstance(change_ans, list):
            change_ans = [change_ans]
        if not isinstance(original_ans, list):
            original_ans = [original_ans]
        target_values = to_value_list(gold)
        predicted_values_change = to_value_list(change_ans)
        predicted_values_original = to_value_list(original_ans)
        correct_original = check_denotation(
            target_values, predicted_values_original)
        correct_change = check_denotation(
            target_values, predicted_values_change)
        original_change = check_denotation(
            predicted_values_original, predicted_values_change)
        if correct_original and correct_change:
            if not t:
                results_non_truncate['co_co'].append(i)
            results_all['co_co'].append(i)
        elif correct_original and not correct_change:
            if not i:
                results_non_truncate['co_inco'].append(i)
            results_all['co_inco'].append(i)
        elif not correct_original and correct_change:
            if not t:
                results_non_truncate['inco_co'].append(i)
            results_all['inco_co'].append(i)
        elif not correct_original and not correct_change:
            if not t:
                results_non_truncate['inco_inco'].append(i)
            results_all['inco_inco'].append(i)
        if correct_original and not original_change:
            if not t:
                results_non_truncate['change'].append(i)
            results_all['change'].append(i)
    try:
        variation_percentage_t = (len(results_non_truncate['co_inco']) +
                                  len(results_non_truncate['inco_co'])) / \
            (len(results_non_truncate['co_inco']) + len(results_non_truncate['inco_co']) + len(
                results_non_truncate['co_co']) + len(results_non_truncate['inco_inco']))
    except:
        variation_percentage_t = 0
    variation_percentage_a = (len(results_all['co_inco']) + len(results_all['inco_co'])) / \
        (len(results_all['co_inco']) + len(results_all['inco_co']) +
         len(results_all['co_co']) + len(results_all['inco_inco']))
    return variation_percentage_a, variation_percentage_t


# =====================================================
# PART1: VP and Emd for structure perturbations

METHOD_STRUCTURE_PERTURBATION = ['no_shuffle', 'row_shuffle', 'col_shuffle', 'target_row_random', 'target_row_front',
                                 'target_row_middle', 'target_row_bottom', 'target_col_random', 'target_col_front', 'target_col_back', 'transpose']


def calculate_emd_structure(gold_path, system, dataset, seed):
    """
    Generate emd results for structure perturbations
    """
    em_all = {method: None for method in METHOD_STRUCTURE_PERTURBATION}
    em_non_truncate = {
        method: None for method in METHOD_STRUCTURE_PERTURBATION}
    emd_all = {method: None for method in METHOD_STRUCTURE_PERTURBATION}
    emd_non_truncate = {
        method: None for method in METHOD_STRUCTURE_PERTURBATION}
    for method in METHOD_STRUCTURE_PERTURBATION:
        print(method)
        results_a, results_t = exact_match_score(
            f"../results/1/{system}_{dataset}_{seed}_{method}.json", gold_path)
        em_all[method] = results_a
        em_non_truncate[method] = results_t
    for k, v in em_all.items():
        emd_all[k] = v - em_all["no_shuffle"]
    for k, v in em_non_truncate.items():
        emd_non_truncate[k] = v - em_non_truncate["no_shuffle"]
    print("===================================")
    with open(f"../results/1/{system}_{dataset}_{seed}_structure_perturb_emd.json", "w") as f:
        json.dump({"em_all": em_all, "emd_all": emd_all, "em_non_truncate": em_non_truncate,
                  "emd_non_truncate": em_non_truncate}, f, indent=4)
    return


def calculate_vp_structure(gold_path, system, dataset, seed):
    """
    Generate vp results for structure perturbations
    """
    vp_all = {method: None for method in METHOD_STRUCTURE_PERTURBATION}
    vp_non_truncate = {
        method: None for method in METHOD_STRUCTURE_PERTURBATION}
    no_shuffle_path = f"../results/1/{system}_{dataset}_{seed}_no_shuffle.json"
    for method in METHOD_STRUCTURE_PERTURBATION:
        shuffle_path = f"../results/1/{system}_{dataset}_{seed}_{method}.json"
        vp_a, vp_nt = vp(shuffle_path, no_shuffle_path, gold_path)
        vp_all[method] = vp_a
        vp_non_truncate[method] = vp_nt
    with open(f"../results/1/{system}_{dataset}_{seed}_structure_perturb_vp.json", "w") as f:
        json.dump(
            {"vp_all": vp_all, "vp_non_truncate": vp_non_truncate}, f, indent=4)
    return


# ===============================================================
# PART2: VP and Emd for structure perturbations
ATTENTION_TO_REL = ['no_shuffle', 'escape_table', 'mask_cell_all',
                    'mask_cell_random', 'relevant_row_shuffle']


def calculate_emd_attention(system, gold_path, dataset, seed):
    """
    Generate emd results for attention to relevant cells
    """
    em_all = {m: [] for m in ATTENTION_TO_REL}
    em_non_truncate = {m: [] for m in ATTENTION_TO_REL}
    for method in ATTENTION_TO_REL:
        print(method)
        if method in ['no_shuffle', 'escape_table']:
            perturb = f"{dataset}_{seed}_{method}"
        else:
            perturb = f"{dataset}_{seed}_{method}_{dataset}"
        results_a, results_t = exact_match_score(
            f"../results/2/{system}_{perturb}.json", gold_path)
        em_all[method].append(results_a)
        em_non_truncate[method].append(results_t)
    with open(f"../results/2/{system}_{dataset}_{seed}_attention_relevant_cell_em.json", "w") as f:
        json.dump(
            {"em_all": em_all, "em_non_truncate": em_non_truncate}, f, indent=4)
    return


def calculate_vp_attention(system, gold_path, dataset, seed):
    """
    Generate vp results for attention to relevant cells.
    """
    vp_all = {m: [] for m in ATTENTION_TO_REL}
    vp_non_truncate = {m: [] for m in ATTENTION_TO_REL}
    no_shuffle_path = f"../results/2/{system}_{dataset}_{seed}_no_shuffle.json"
    for method in ATTENTION_TO_REL:
        print(method)
        if method in ['no_shuffle', 'escape_table']:
            perturb = f"{dataset}_{seed}_{method}"
        else:
            perturb = f"{dataset}_{seed}_{method}_{dataset}"
        shuffle_path = f"../results/2/{system}_{perturb}.json"
        results_all, results_nt = vp(shuffle_path, no_shuffle_path, gold_path)
        vp_all[method] = results_all
        vp_non_truncate[method] = results_nt
    with open(f"../results/2/{system}_{dataset}_{seed}_attention_relevant_cell_vp.json", "w") as f:
        json.dump(
            {"vp_all": vp_all, "vp_non_truncate": vp_non_truncate}, f, indent=4)
    return

# ===============================================================
# PART3: VP and Emd for aggregation/comparison


def calculate_emd_aggregation(system, dataset, seed):
    """
    Generate emd results for aggregation/comparison robustness
    """
    method_lists = ['no_shuffle_short_table', 'no_shuffle_ans_change',
                    'no_shuffle_ans_no_change', 'no_shuffle_ori_table']

    em = {m: None for m in method_lists}
    for method in method_lists:
        gold_path = f"../dataset/03_agg_com/{dataset}_nonfactoid_dev_{method}.json"
        results, _ = exact_match_score(
            f"../results/3/{system}_{dataset}_{seed}_{method}.json", gold_path)
        em[method] = results
    with open(f"../results/3/{system}_{dataset}_agg_com_em.json", "w") as f:
        json.dump(
            {"em_all": em}, f, indent=4)
    return


def break_into_separate(merge_path, part):
    if not os.path.isdir('../results'):
        os.makedirs('../results/1')
        os.makedirs('../results/2')
        os.makedirs('../results/3')
    with open(merge_path, "r") as f:
        data = [json.loads(line) for line in f]
    datasets = list(set(item["dataset"] for item in data))
    perturbations = list(set(item["perturb"] for item in data))
    seeds = list(set(item["seed"] for item in data))
    dic = {f"{dataset}_{seed}_{p}": []
           for dataset in datasets for p in perturbations for seed in seeds}
    for item in data:
        ds = item["dataset"]
        p = item["perturb"]
        s = item["seed"]
        key = f"{ds}_{s}_{p}"
        dic[key].append(item)
    for k, v in dic.items():
        output_path = f"../results/{part}/{system}_{k}.json"
        with open(output_path, "w") as f:
            for item in v:
                f.write(json.dumps(item)+"\n")
    return


def main(system, part, seed, merge, merge_path):
    if merge:
        break_into_separate(merge_path, part)
    if part == 1:
        for dataset in ["wtq", "tat", "sqa", "wikisql"]:
            calculate_emd_structure(
                gold_path=f"../dataset/01_str/{dataset}_factoid_dev.json", system=system, dataset=dataset, seed=seed)
            calculate_vp_structure(
                gold_path=f"../dataset/01_str/{dataset}_factoid_dev.json", system=system, dataset=dataset, seed=seed)
    elif part == 2:
        for dataset in ["wtq", "tat"]:
            calculate_emd_attention(
                gold_path=f"../dataset/02_rel/{dataset}_nonfactoid_dev.json", system=system, dataset=dataset, seed=seed)
            calculate_vp_attention(
                gold_path=f"../dataset/02_rel/{dataset}_nonfactoid_dev.json", system=system, dataset=dataset, seed=seed)

    elif part == 3:
        for dataset in ["wtq", "tat"]:
            calculate_emd_aggregation(
                system=system, dataset=dataset, seed=args.seed)
    return


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,
                    help="The path to the model")
parser.add_argument("--seed", type=int, choices=[33, 16, 77, 69, 42],
                    default=33)
parser.add_argument("--part", type=int, choices=[1, 2, 3],
                    default=1)
parser.add_argument("--merge", action="store_true",
                    help="whether the evaluated file is merged or not.")
parser.add_argument("--merge_path", type=str, default="")
args = parser.parse_args()
system = args.model_path.split("/")[-1].strip()

if __name__ == "__main__":
    main(system, args.part, args.seed, args.merge, args.merge_path)
