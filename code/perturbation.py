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

from utils import normalize
import math

# ===========================structural perturbations=====================================


def row_shuffling(table: dict, seed: int, answer: list) -> dict:
    """
    randomly shuffling all rows in the table.
    """
    import random
    random.seed(seed)
    random.shuffle(table['rows'])
    return table


def col_shuffling(table: dict, seed: int, answer: list) -> dict:
    """
    randomly shuffling all columns in the table.
    """
    import random
    random.seed(seed)
    order = [i for i in range(len(table['header']))]
    random.shuffle(order)
    header = [table['header'][od] for od in order]
    rows = [[row[od] for od in order] for row in table['rows']]
    return {'header': header, 'rows': rows}


def target_row_random(table: dict, seed: int, answer: list) -> dict:
    """
    randomly shuffling only the rows containing the answers.
    """
    import random
    random.seed(seed)
    # identify possible rows
    pos_idx = {}
    sample_list = [i for i in range(len(table['rows']))]
    position_move, position_move_abs = 0, 0
    for i, row in enumerate(table['rows']):
        if i not in pos_idx.values():
            if answer_in_table(row, answer):
                now_pos = random.sample(sample_list, 1)[0]
                pos_idx[i] = now_pos
                pos_idx[now_pos] = i
                position_move_abs += abs(now_pos - i)
                # positive value indicate move latter than original index
                position_move += (now_pos - i)
                sample_list.remove(now_pos)
            else:
                pos_idx[i] = i
    ordered_pos_idx = {k: v for k, v in sorted(
        pos_idx.items(), key=lambda item: item[1])}
    new_rows = [table['rows'][pre] for pre, now in ordered_pos_idx.items()]
    return {'header': table['header'], 'rows': new_rows, 'position_move': position_move, 'position_move_abs': position_move_abs}


def target_row_front(table: dict, seed: int, answer: list) -> dict:
    """
    shifting rows containing the answers to the front part of the table.
    """
    import random
    random.seed(seed)
    pos_idx = {}
    position_move, position_move_abs = 0, 0
    end = math.ceil(len(table['rows'])/3)
    sample_list = [i for i in range(end)]
    len_sample = len(sample_list)
    for i, row in enumerate(table['rows']):
        if i not in pos_idx.values():
            if answer_in_table(row, answer):
                if i > end:
                    # not in the front, move to front
                    if not sample_list == []:
                        now_pos = random.sample(sample_list, 1)[0]
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        sample_list.remove(now_pos)
                    else:
                        # more target rows than sample list, just put them in order
                        now_pos = len_sample
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        len_sample += 1

                else:
                    # answer already in front, do not change
                    pos_idx[i] = i
            else:
                # non-target row
                pos_idx[i] = i
    ordered_pos_idx = {k: v for k, v in sorted(
        pos_idx.items(), key=lambda item: item[1])}
    new_rows = [table['rows'][pre] for pre, now in ordered_pos_idx.items()]
    return {'header': table['header'], 'rows': new_rows, 'position_move': position_move, 'position_move_abs': position_move_abs}


def target_row_middle(table: dict, seed: int, answer: list) -> dict:
    """
    shifting rows containing the answers to the middle part of the table.
    """
    import random
    random.seed(seed)
    pos_idx = {}
    position_move, position_move_abs = 0, 0
    start = math.floor(len(table['rows'])/3)
    bottom = len(table['rows']) - start
    bottom_ = len(table['rows']) - start
    sample_list = [i for i in range(start, bottom)]
    len_sample = len(sample_list)
    for i, row in enumerate(table['rows']):
        if i not in pos_idx.values():
            if answer_in_table(row, answer):
                # target row
                if i < start or i > bottom:
                    # not in middle. move to middle
                    if not sample_list == []:
                        now_pos = random.sample(sample_list, 1)[0]
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        sample_list.remove(now_pos)
                    else:
                        # more target rows than sample list, just put them in order
                        now_pos = bottom_
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        bottom_ += 1

                else:
                    # answer already in the middle, do not change
                    pos_idx[i] = i
            else:
                # non-target row
                pos_idx[i] = i
    ordered_pos_idx = {k: v for k, v in sorted(
        pos_idx.items(), key=lambda item: item[1])}
    new_rows = [table['rows'][pre] for pre, now in ordered_pos_idx.items()]
    return {'header': table['header'], 'rows': new_rows, 'position_move': position_move, 'position_move_abs': position_move_abs}


def target_row_bottom(table: dict, seed: int, answer: list) -> dict:
    """
    shifting rows containing the answers to the bottom part of the table.
    """
    import random
    random.seed(seed)
    pos_idx = {}
    position_move, position_move_abs = 0, 0
    bottom = math.ceil(len(table['rows'])/3)
    start = len(table['rows']) - bottom
    start_ = len(table['rows']) - bottom
    end = len(table['rows'])
    sample_list = [i for i in range(start, end)]
    len_sample = len(sample_list)
    for i, row in enumerate(table['rows']):
        if i not in pos_idx.values():
            if answer_in_table(row, answer):
                # target row
                if i < start:
                    # not in bottom, move to bottom
                    if not sample_list == []:
                        now_pos = random.sample(sample_list, 1)[0]
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        sample_list.remove(now_pos)
                    else:
                        # more target rows than sample list, just put them in order
                        now_pos = start_ - 1
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        start_ -= 1

                else:
                    # answer already in the bottom, do not change
                    pos_idx[i] = i
            else:
                # non-target row
                pos_idx[i] = i
    ordered_pos_idx = {k: v for k, v in sorted(
        pos_idx.items(), key=lambda item: item[1])}
    new_rows = [table['rows'][pre] for pre, now in ordered_pos_idx.items()]
    return {'header': table['header'], 'rows': new_rows, 'position_move': position_move, 'position_move_abs': position_move_abs}


def row_to_col(table: dict) -> list:
    """
    transpose row-wise table to column-wise table.
    """
    header = table['header']
    fields = len(header)
    rows = table['rows']
    col_list = [[row[i] for row in rows] for i in range(fields)]
    assert len(col_list) == len(header)
    col_list = [[h] + r for h, r in zip(header, col_list)]
    return col_list


def col_to_row(columns: list):
    """
    transpose col-wise table to row-wise table.
    """
    header = [item[0] for item in columns]
    len_rows = len(columns[0])
    rows = [[col[i] for col in columns] for i in range(1, len_rows)]

    return header, rows


def answer_in_table(row: list, answer: list, return_id=False):
    """
    detect if answer is the given row or not. 
    return answer_in_table, answer_position_id.
    """
    # normalize the row
    row = [normalize(item).lower() for item in row]
    # normalize the answer
    answer = [normalize(ans).lower() for ans in answer]
    if len(answer) == 1:
        answer = answer[0]
        if not return_id:
            return answer in row
        else:
            col_id = [j for j, cell in enumerate(row) if cell == answer]
            return answer in row, col_id
    else:
        if not return_id:
            if any([True for item in answer if item in row]):
                return True
            else:
                return False
        else:
            col_id = []
            if any([True for item in answer if item in row]):
                for item in answer:
                    if item in row:
                        col_id += [j for j,
                                   cell in enumerate(row) if cell == item]
                return True, col_id
            else:
                return False, []


def target_col_random(table: dict, seed: int, answer: list) -> dict:
    """
    randomly shuffling only the columns containing the answers.
    """
    import random
    random.seed(seed)
    pos_idx = {}
    sample_list = [i for i in range(len(table['header']))]
    position_move = 0
    position_move_abs = 0
    table_col = row_to_col(table)
    for i, col in enumerate(table_col):
        if i not in pos_idx.values():
            if answer_in_table(col, answer):
                now_pos = random.sample(sample_list, 1)[0]
                pos_idx[i] = now_pos
                pos_idx[now_pos] = i
                position_move_abs += abs(now_pos - i)
                # positive value indicate move latter then original index
                position_move += (now_pos - i)
                sample_list.remove(now_pos)
            else:
                pos_idx[i] = i
    ordered_pos_idx = {k: v for k, v in sorted(
        pos_idx.items(), key=lambda item: item[1])}
    new_cols = [table_col[pre] for pre, now in ordered_pos_idx.items()]
    header, rows = col_to_row(new_cols)
    return {'header': header, 'rows': rows, 'position_move': position_move, 'position_move_abs': position_move_abs}


def target_col_front(table: dict, seed: int, answer: list) -> dict:
    """
    shifting the columns containing the answers to the front part of the table.
    """
    import random
    random.seed(seed)
    pos_idx = {}
    position_move, position_move_abs = 0, 0
    end = math.ceil(len(table['header'])/3)
    sample_list = [i for i in range(end)]
    len_sample = len(sample_list)
    table_col = row_to_col(table)
    for i, col in enumerate(table_col):
        if i not in pos_idx.values():
            if answer_in_table(col, answer):
                if i >= end:
                    # not in front. move to front
                    if not sample_list == []:
                        now_pos = random.sample(sample_list, 1)[0]
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        sample_list.remove(now_pos)
                    else:
                        # more target cols than sample list, just put them in order
                        now_pos = len_sample
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        len_sample += 1

                else:
                    # answer already in front, do not change
                    pos_idx[i] = i
            else:
                # non-target row
                pos_idx[i] = i
    ordered_pos_idx = {k: v for k, v in sorted(
        pos_idx.items(), key=lambda item: item[1])}
    new_cols = [table_col[pre] for pre, now in ordered_pos_idx.items()]
    header, rows = col_to_row(new_cols)
    return {'header': header, 'rows': rows, 'position_move': position_move, 'position_move_abs': position_move_abs}


def target_col_back(table: dict, seed: int, answer: list) -> dict:
    """
    shifting the columns containing the answers to the back part of the table.
    """
    import random
    random.seed(seed)
    pos_idx = {}
    position_move, position_move_abs = 0, 0
    bottom = math.ceil(len(table['header'])/3)
    start = len(table['header']) - bottom
    start_ = len(table['header']) - bottom
    end = len(table['header'])
    sample_list = [i for i in range(start, end)]
    len_sample = len(sample_list)
    table_col = row_to_col(table)
    for i, col in enumerate(table_col):
        if i not in pos_idx.values():
            if answer_in_table(col, answer):
                if i < start:
                    if not sample_list == []:
                        now_pos = random.sample(sample_list, 1)[0]
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        sample_list.remove(now_pos)
                    else:
                        now_pos = start_ - 1
                        pos_idx[i] = now_pos
                        pos_idx[now_pos] = i
                        position_move += (now_pos - i)
                        position_move_abs += abs(now_pos-i)
                        start_ -= 1
                else:
                    pos_idx[i] = i
            else:
                pos_idx[i] = i
    ordered_pos_idx = {k: v for k, v in sorted(
        pos_idx.items(), key=lambda item: item[1])}
    new_cols = [table_col[pre] for pre, now in ordered_pos_idx.items()]
    header, rows = col_to_row(new_cols)
    return {'header': header, 'rows': rows, 'position_move': position_move, 'position_move_abs': position_move_abs}


def transposed(table: dict, seed: int, answer: list) -> dict:
    """
    transposing the table.
    """
    # do not set new header, treat header as new rows
    rows = table['rows']
    header = table['header']
    fields = len(header)
    added_rows = [header] + rows
    # transpose
    transposed = [[row[i] for row in added_rows] for i in range(fields)]
    return {'rows': transposed[1:], 'header': [str(i) for i in range(len(transposed[0]))]}


# ===========================attention to relevant cells=====================================
def escape_table(instance: dict, seed: int, mask_token="''") -> dict:
    """
    return a null table.
    """
    return {'header': ['none'], 'rows': [['none']]}


def check_subset(target_row: list, original_row: list, n: int):
    """
    check if target row is subset of the original row
    n: relevant cells number in a row
    """
    return len([True for item in target_row if item in original_row]) == n


def wtq_mask_cell_all(instance: dict, seed: int, mask_token="''") -> dict:
    """
    masking out all relevant and answer cells. 
    """
    table = instance['table']
    target_header = instance["short_table"][0]
    target_col_idx = list(set([i for item in target_header for i, h in enumerate(
        table['header']) if item.lower() == h.lower()]))
    # if not len(target_col_idx)==len(target_header):
    #     print('col mismatch', len(target_col_idx), len(target_header))
    target_row = instance["short_table"][1:]
    target_row_idx = list(set([i for row in target_row for i, r in enumerate(
        table['rows']) if check_subset(row, r, len(target_col_idx))]))
    # if not len(target_row_idx)==len(target_row):
    #     print(target_row_idx)
    #     print('row mismatch', len(target_row_idx), len(target_row))
    new_table = []
    for i, row in enumerate(table['rows']):
        new_row = []
        for j, cell in enumerate(row):
            if i in target_row_idx and j in target_col_idx:
                new_row.append(mask_token)
            else:
                if instance['relevant_rule'] != [] and [i, j] in instance['relevant_rule']:
                    new_row.append(mask_token)
                else:
                    new_row.append(cell)
        new_table.append(new_row)
    return {'header': table['header'], 'rows': new_table}


def wtq_mask_cell_all_random(instance: dict, seed: int, mask_token="''") -> dict:
    """
    masking out irrelevant cells randomly. 
    """
    import random
    random.seed(seed)
    table = instance['table']
    target_header = instance["short_table"][0]
    target_col_idx = list(set([i for item in target_header for i, h in enumerate(
        table['header']) if item.lower() == h.lower()]))
    # if not len(target_col_idx)==len(target_header):
    #     print('col mismatch', len(target_col_idx), len(target_header))
    target_row = instance["short_table"][1:]
    target_row_idx = list(set([i for row in target_row for i, r in enumerate(
        table['rows']) if check_subset(row, r, len(target_col_idx))]))
    # if not len(target_row_idx)==len(target_row):
    #     print(target_row_idx)
    #     print('row mismatch', len(target_row_idx), len(target_row))
    new_table = []
    try:
        # sample from columns not in relevant cells
        target_col_idx_ = random.sample([i for i in range(len(
            instance['table']['header'])) if i not in target_col_idx], k=len(instance['short_table'][0]))
    except ValueError:
        col_not_in = [i for i in range(
            len(instance['table']['header'])) if i not in target_col_idx]
        target_col_idx_ = col_not_in + \
            random.sample(target_col_idx, k=len(
                instance['short_table'][0])-len(col_not_in))
    try:
        # sample from rows not in relevant cells
        target_row_idx_ = random.sample([i for i in range(len(
            instance['table']['rows'])) if i not in target_row_idx], k=len(instance['short_table'][1:]))
    except ValueError:
        row_not_in = [i for i in range(
            len(instance['table']['rows'])) if i not in target_row_idx]
        target_row_idx_ = row_not_in + \
            random.sample(target_row_idx, k=len(
                instance['short_table'][1:])-len(row_not_in))
    for i, row in enumerate(table['rows']):
        new_row = []
        for j, cell in enumerate(row):
            if i in target_row_idx_ and j in target_col_idx_:
                if instance['relevant_rule'] != [] and [i, j] in instance['relevant_rule']:
                    new_row.append(cell)
                else:
                    new_row.append(mask_token)
            else:
                new_row.append(cell)
        new_table.append(new_row)
    return {'header': table['header'], 'rows': new_table}


def shuffling_relevant_rows_wtq(instance: dict, seed: int) -> dict:
    """
    randomly shuffling rows containing relevant cells.
    """
    import random
    random.seed(seed)
    table = instance['table']
    target_header = instance["short_table"][0]
    target_col_idx = list(set([i for item in target_header for i, h in enumerate(
        table['header']) if item.lower() == h.lower()]))
    # if not len(target_col_idx)==len(target_header):
    #     print('col mismatch', len(target_col_idx), len(target_header))
    target_row = instance["short_table"][1:]
    target_row_idx = list(set([i for row in target_row for i, r in enumerate(
        table['rows']) if check_subset(row, r, len(target_col_idx))]))
    # if not len(target_row_idx)==len(target_row):
    #     print(target_row_idx)
    #     print('row mismatch', len(target_row_idx), len(target_row))
    if instance['relevant_rule'] != []:
        for item in instance['relevant_rule']:
            if item[0] not in target_row_idx:
                target_row_idx.append(item[0])
            if item[0] not in target_col_idx:
                target_col_idx.append(item[1])
    target_row_idx = list(set(target_row_idx))
    new_target_row_idx = target_row_idx
    if not len(target_row_idx) < 2:
        while new_target_row_idx == target_row_idx:
            new_target_row_idx = random.sample(
                target_row_idx, k=len(target_row_idx))
    else:
        # randomly move to someplace
        to_change = random.sample([i for i in range(
            len(instance['table']['rows'])) if i != target_row_idx[0]], k=1)[0]
        tv = target_row_idx[0]
        target_row_idx = [tv, to_change]
        new_target_row_idx = [to_change, tv]
    row_change = {ori: now for ori, now in zip(
        target_row_idx, new_target_row_idx)}
    new_table = []
    for i, row in enumerate(table['rows']):
        if i in target_row_idx:
            new_table.append(table['rows'][row_change[i]])
        else:
            new_table.append(row)
    return {'header': table['header'], 'rows': new_table}


def tat_cell_mask_all(instance: dict, seed: int, mask_token="''") -> dict:
    """
    masking out all relevant and answer cells for TAT. 
    """
    relevant_cells = instance['rel_cell']
    new_rows = instance['table'][1:]
    for item in relevant_cells:
        new_rows[item[0]][item[1]] = mask_token
    new_header = instance['table'][0]
    return {'rows': new_rows, 'header': new_header}


def tat_cell_mask_random(instance: dict, seed: int, mask_token="''") -> dict:
    """
    masking out irrelevant cells randomly for TAT. 
    """
    import random
    random.seed(seed)
    relevant_cells = instance['rel_cell']
    targets = [[i, j] for i in range(len(instance['table'])-1)
               for j in range(len(instance['table'][0]))]
    targets = [item for item in targets if item not in relevant_cells]
    select_cell_idx = random.sample(targets, k=len(relevant_cells))
    new_rows = instance['table'][1:]
    for item in select_cell_idx:
        new_rows[item[0]][item[1]] = mask_token
    new_header = instance['table'][0]
    return {'rows': new_rows, 'header': new_header}


def shuffling_relevant_row_tat(instance: dict, seed: int) -> dict:
    """
    randomly shuffling rows containing relevant cells for TAT.
    """
    import random
    random.seed(seed)
    relevant_cells = instance['rel_cell']
    target_row_idx = list(set([item[0] for item in relevant_cells]))
    target_col_idx = list(set([item[1] for item in relevant_cells]))
    new_target_row_idx = target_row_idx
    if not len(target_row_idx) < 2:
        while new_target_row_idx == target_row_idx:
            new_target_row_idx = random.sample(
                target_row_idx, k=len(target_row_idx))
    else:
        # randomly move to someplace
        to_change = random.sample([i for i in range(
            len(instance['table'])-1) if i != target_row_idx[0]], k=1)[0]
        tv = target_row_idx[0]
        target_row_idx = [tv, to_change]
        new_target_row_idx = [to_change, tv]
    row_change = {ori: now for ori, now in zip(
        target_row_idx, new_target_row_idx)}
    new_table = []
    for i, row in enumerate(instance['table'][1:]):
        if i in target_row_idx:
            new_table.append(instance['table'][1:][row_change[i]])
        else:
            new_table.append(row)
    new_header = instance['table'][0]
    new_rows = new_table
    return {'rows': new_rows, 'header': new_header}
