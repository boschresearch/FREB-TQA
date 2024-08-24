from perturbation import *

import re
import pandas as pd
import numpy as np
import json
import math
import six
import struct
import argparse


def reformat_table(table):
    if isinstance(table, list):
        header = table[0]
    elif isinstance(table, dict):
        header = table['header']
    output = ' | '.join(header)
    output += '\n'
    if isinstance(table, list):
        rows = table[1:]
    elif isinstance(table, dict):
        rows = table['rows']
    for row in rows:
        output += ' | '.join(row)
        output += '\n'
    return output


def generate_demostration(demo):
    with open(demo, 'r') as f:
        data = json.load(f)
    prompt = ""
    for i, item in enumerate(data):
        question = item['question']
        answers = item['answers']
        table = reformat_table(item['table'])
        prompt += f"Example {i}: Question: {question} Table: {table} Answer: {answers}"
    return prompt


def find_answer_float(ans):
    if not isinstance(ans, str):  # list object
        output_ = [re.findall(r"[-+]?(?:\d*\.*\d+)", item) for item in ans]
        output = []
        for opt in output_:
            if not opt == []:
                output += opt
    else:
        output = re.findall(r"[-+]?(?:\d*\.*\d+)", ans)
    if output == []:
        return np.nan
    else:
        output = [float(item) for item in output]
        return output[0]


# ========================================================
# from robut paper eval_utils.py


def _split_thousands(delimiter, value):
    split = value.split(delimiter)
    return len(split) > 1 and any(map(lambda x: len(x) == 3, split))


def convert_to_float(value):
    """Converts value to a float using a series of increasingly complex heuristics.

    Args:
    value: object that needs to be converted. Allowed types include
        float/int/strings.

    Returns:
    A float interpretation of value.

    Raises:
    ValueError if the float conversion of value fails.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, six.string_types):
        raise ValueError(
            "Argument value is not a string. Can't parse it as float")
    sanitized = value

    try:
        # Example: 1,000.7
        if "." in sanitized and "," in sanitized:
            return float(sanitized.replace(",", ""))
        # 1,000
        if "," in sanitized and _split_thousands(",", sanitized):
            return float(sanitized.replace(",", ""))
        # 5,5556
        if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
                ",", sanitized):
            return float(sanitized.replace(",", "."))
        # 0.0.0.1
        if sanitized.count(".") > 1:
            return float(sanitized.replace(".", ""))
        # 0,0,0,1
        if sanitized.count(",") > 1:
            return float(sanitized.replace(",", ""))
        return float(sanitized)
    except ValueError:
        # Avoid adding the sanitized value in the error message.
        raise ValueError("Unable to convert value to float")


def _safe_convert_to_float(value):
    float_value = convert_to_float(value)
    if math.isnan(float_value):
        raise ValueError('Value is NaN %s' % value)
    return float_value


def _parse_value(value):
    """Parses a cell value to a number or lowercased string."""
    try:
        return _safe_convert_to_float(value)
    except ValueError:
        try:
            return value.lower()
        except ValueError:
            return value


def _collect_cells_from_table(cell_coos,
                              table):
    cell_values = []
    for cell in cell_coos:
        value = str(table.iat[cell[0], cell[1]])
        cell_values.append(value)
    return cell_values


def execute(aggregation_type, cell_coos,
            table):
    """Executes predicted structure against a table to produce the denotation."""
    values = _collect_cells_from_table(cell_coos, table)
    values_parsed = [_parse_value(value) for value in values]
    values_parsed = tuple(values_parsed)
    if aggregation_type == "NONE":
      # In this case there is no aggregation
        return values_parsed, values
    else:  # Should perform aggregation.
        if not values and (aggregation_type == "AVERAGE" or
                           aggregation_type == "SUM" or
                           aggregation_type == "SUBTRACT" or
                           aggregation_type == "DIVIDE" or
                           aggregation_type == "PROPORTION"):
            # Summing or averaging an empty set results in an empty set.
            # NB: SQL returns null for sum over an empty set.
            return tuple(), values
        if aggregation_type == "COUNT":
            denotation = len(values)
        else:
            # In this case all values must be numbers (to be summed or averaged).
            try:
                values_num = [convert_to_float(value) for value in values]
            except ValueError:
                return values_parsed, values
            if aggregation_type == "SUM":
                denotation = sum(values_num)
            elif aggregation_type == "AVERAGE":
                denotation = sum(values_num) / len(values_num)
            elif aggregation_type == "SUBTRACT":
                if len(values_num) == 1:
                    denotation = values_num[0]
                else:  # in order subtraction
                    denotation = values_num[0]
                    for item in values_num[1:]:
                        denotation = denotation - item
            elif aggregation_type == "DIVIDE":
                if len(values_num) == 1:
                    denotation = values_num[0]
                else:  # in order subtraction
                    denotation = values_num[0]
                    for item in values_num[1:]:
                        denotation = denotation / item
            elif aggregation_type == "PROPORTION":
                if len(values_num) == 1:
                    denotation = values_num[0]
                elif len(values_num) == 2:
                    denotation = (values_num[0] - values_num[1])/values_num[0]
                elif len(values_num) == 3:
                    denotation = (values_num[0] - values_num[1])/values_num[2]
                else:
                    return values_parsed, values
            elif aggregation_type == "COMPARE":
                return max(values_num)
            else:
                raise ValueError('Unknwon aggregation type: %s' %
                                 aggregation_type)
    return tuple([float(denotation)]), values


def to_float32(v):
    """If v is a float reduce precision to that of a 32 bit float."""
    if not isinstance(v, float):
        return v
    return struct.unpack("!f", struct.pack("!f", v))[0]


def to_float32s(elements):
    return tuple(to_float32(v) for v in elements)


def _normalize_float(answer):
    if answer is None:
        return None
    try:
        value = convert_to_float(answer)
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    except ValueError:
        return answer.lower()


def normalize_answers(answers):
    normalized_answers = (_normalize_float(a) for a in answers)
    normalized_answers = (a for a in normalized_answers if a is not None)
    normalized_answers = (str(a) for a in normalized_answers)
    normalized_answers = list(normalized_answers)
    normalized_answers.sort()
    return normalized_answers


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
