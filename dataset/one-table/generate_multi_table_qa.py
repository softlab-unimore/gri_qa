import os
import numpy as np
import pandas as pd


def convert_unit_value(values, from_unit, to_unit, conversion_map):
    if from_unit == to_unit:
        return values  # No conversion needed
    if from_unit not in conversion_map or to_unit not in conversion_map[from_unit]:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported.")
    factor = conversion_map[from_unit][to_unit]
    return np.array(values) * factor


def compute_answer(keys, values, units, years, op, q_type, target_unit, conversion_map):
    # Get values related to specified years
    if years != -1:
        topk_values = []
        for val in values:
            items = [val[k] for k in years]
            topk_values.append(items)
    else:
        topk = np.min([len(arr) for arr in values])
        topk_values = [list(arr.values())[:topk] for arr in values]

    # Normalize values
    if conversion_map is not None:
        norm_topk_values = np.array(
            [convert_unit_value(topk_values[i], units[i], target_unit, conversion_map) for i in range(len(topk_values))]
        )
    else:
        norm_topk_values = np.array(topk_values)

    # Apply operation
    if op == 'mean':
        out_values = np.mean(norm_topk_values, axis=1)
    elif op == 'sum':
        out_values = np.sum(norm_topk_values, axis=1)
    elif op == 'diff_perc':
        # last year is in the first position
        assert norm_topk_values.shape[1] == 2
        out_values = np.abs(norm_topk_values[:, 1] - norm_topk_values[:, 0]) / norm_topk_values[:, 1] * 100
    elif op == 'diff':
        # last year is in the first position
        assert norm_topk_values.shape[1] == 2
        out_values = norm_topk_values[:, 0] - norm_topk_values[:, 1]
    elif op == '-':
        if norm_topk_values.shape[1] > 1:
            out_values = norm_topk_values
        else:
            out_values = np.reshape(norm_topk_values, (-1))
    else:
        raise ValueError(f"Operation {op} is not supported.")

    # Apply comparator
    if q_type == 'max':
        out_ix = np.argmax(out_values)
        out_company = '_'.join(keys[out_ix].split('_')[:-3])
        out_value = out_values[out_ix]
    elif q_type == 'min':
        out_ix = np.argmin(out_values)
        out_company = '_'.join(keys[out_ix].split('_')[:-3])
        out_value = out_values[out_ix]
    elif q_type == 'sum':
        out_company = None
        out_value = np.sum(out_values)
    elif q_type == 'mean':
        out_company = None
        out_value = np.mean(out_values)
    elif q_type == 'rank_asc':
        if len(out_values) == 2:
            topk = 2
        else:
            topk = np.random.randint(2, len(out_values))
        order = np.argsort(out_values)[:topk]
        out_value = out_values[order]
        out_company = np.array(['_'.join(keys[x].split('_')[:-3]) for x in order])
    elif q_type == 'rank_desc':
        if len(out_values) == 2:
            topk = 2
        else:
            topk = np.random.randint(2, len(out_values))
        order = np.argsort(out_values)[::-1][:topk]
        out_value = out_values[order]
        out_company = np.array(['_'.join(keys[x].split('_')[:-3]) for x in order])
    else:
        raise ValueError(f"Operation {q_type} is not supported.")

    if isinstance(out_value, np.ndarray):
        assert not pd.isnull(out_value).all(), "Found null answer!"
    else:
        assert not pd.isnull(out_value), "Found null answer!"

    out_value = np.round(out_value, 2)

    return out_company, out_value


def generate_multi_answers(keys, values, units, target_years, op, q_type, conversion_map, num_companies=5, num_qa=5):
    if units is not None:
        assert len({len(keys), len(values), len(units)}) == 1, "Keys, values and units must be of same length."
    answers = []
    it = 0
    cache = []
    loop_it = 0
    while it < num_qa:
        order = np.random.permutation(range(len(keys)))[:num_companies]
        if str(sorted(order)) in cache and loop_it < num_qa * 2:
            loop_it += 1
            continue
        loop_it = 0

        sel_keys = [keys[ix] for ix in order]
        sel_values = [values[ix] for ix in order]
        if units is not None:
            sel_units = [units[ix] for ix in order]
            target_unit = np.random.choice(sel_units)
        else:
            sel_units = None
            target_unit = None
        cache.append(str(sorted(order)))
        it += 1

        if target_years is None:
            years = set(sel_values[0].keys())
            for item in sel_values:
                years &= set(item.keys())
            years = sorted(list(years), reverse=True)
        else:
            years = target_years

        answer_company, answer_value = compute_answer(sel_keys, sel_values, sel_units, years, op, q_type, target_unit,
                                                      conversion_map)
        answers.append({
            'companies': sorted(sel_keys), 'answer_value': answer_value, 'answer_company': answer_company,
            'unit': target_unit, 'years': years
        })

    return answers


def check_value_consistency(keys, rows, cols, vals):
    dataset_dir = os.path.join('dataset', 'annotation')

    # Loop over companies
    all_rows = []
    all_cols = []
    for ix, key in enumerate(keys):
        # Retrieve table file name
        dir_name = '_'.join(key.split('_')[:-2])
        file_name = f"{'_'.join(key.split('_')[-2:])}.csv"

        # Get the number of rows in the header
        if cols[ix] is None:  # The pdf should be skipped
            continue
        num_header_rows = np.max([len(x) if isinstance(x, list) else 1 for x in cols[ix].values()])

        # Read the table
        df = pd.read_csv(
            os.path.join(dataset_dir, dir_name, file_name), header=list(range(int(num_header_rows))), sep=';'
        )

        # Get current values to check
        target_row = rows[ix]
        target_col_map = cols[ix]
        curr_val = vals[ix]
        target_cols = []

        # If target row is None, the value has not been to checked
        if target_row is None:
            continue

        # Loop over the values
        for col, val in curr_val.items():
            # Select the target column from the table
            # If the header is multi-index, loop over all the levels
            if isinstance(target_col_map[col], list):
                col_values = df.copy()
                target_cols.append([x[0] for x in df.columns].index(str(target_col_map[col][0])) + 1)
                for col_level in target_col_map[col]:
                    col_values = col_values[str(col_level)]
            else:
                col_values = df[str(target_col_map[col])]
                target_cols.append(list(df.columns).index(str(target_col_map[col])) + 1)

            # Select the cell value from the table
            # If the row reports an integer, it refers directly to row index
            # otherwise it is a tuple reporting the combination of operations used to obtain that value
            if isinstance(target_row, tuple):  # Value obtained as combination of operations
                args = target_row[0]
                ops = target_row[1]
                # Obtain the values associated to the indexes
                comb_row_vals = []
                for arg in args:  # Multiple args
                    # Shift index to align with dataframe indexing
                    comb_row_vals.append(float(col_values[arg - int(num_header_rows) - 1]))
                # Apply in order the sequence of operators on the specified values
                cell_value = comb_row_vals
                for op in ops:
                    if op == '+':
                        cell_value = np.sum(cell_value)
                    elif op == '*':
                        cell_value = np.prod(cell_value)
                    elif op == '/100':
                        cell_value = np.divide(cell_value, 100)
                    elif op == 'perc':
                        cell_value = (cell_value[0] / cell_value[1]) * 100
                    else:
                        raise ValueError("Wrong operator!")
            else:
                # Shift index to align with dataframe indexing
                cell_value = col_values[target_row - int(num_header_rows) - 1]

            # Check the equivalence between target and extracted values
            try:
                true_val = round(float(cell_value), 2)
            except ValueError:
                true_val = round(float(cell_value.replace('%', '')), 2)
            extracted_val = round(float(val), 2)
            assert true_val == extracted_val, f"Mismatch in values: true={true_val}, extracted={extracted_val}!"

            all_rows.append(target_row[0] if isinstance(target_row, tuple) else [target_row])
            all_cols.append(target_cols)

    return all_rows, all_cols


def total_energy_qa(keys, values, units, energy_conv_map, num_companies, mode):
    if mode == 'complex':
        q1 = 'What is the highest average energy consumed in the last <years> years among the following companies?'
        op1 = 'mean'
        q_type1 = 'max'
        cat1 = 'multistep_sup'
    elif mode == 'simple':
        q1 = 'What is the highest energy consumed in the last <years> years among the following companies?'
        op1 = '-'  # FIXME: Maybe max?
        q_type1 = 'max'
        cat1 = 'sup'
    else:
        raise ValueError("Wrong mode!")

    q1_answers = generate_multi_answers(keys, values, units, None, op1, q_type1, energy_conv_map,
                                        num_companies=num_companies)
    for x in q1_answers:
        new_q1 = q1.replace('<years>', str(len(x['years'])))
        x['question'] = new_q1
        x['operation'] = cat1

    if mode == 'complex':
        q2 = 'What is the lowest total energy consumption over the last <years> years among the following companies?'
        op2 = 'sum'
        q_type2 = 'min'
        cat2 = 'multistep_sup'
    elif mode == 'simple':
        q2 = 'What is the lowest energy consumption over the last <years> years among the following companies?'
        op2 = '-'  # FIXME: Maybe min?
        q_type2 = 'min'
        cat2 = 'sup'
    else:
        raise ValueError("Wrong mode!")

    q2_answers = generate_multi_answers(keys, values, units, None, op2, q_type2, energy_conv_map,
                                        num_companies=num_companies)
    for x in q2_answers:
        new_q2 = q2.replace('<years>', str(len(x['years'])))
        x['question'] = new_q2
        x['operation'] = cat2

    if mode == 'complex':
        q3 = 'What is the largest percentage variation in energy consumption from 2023 to 2022 among the following companies?'
        op3 = 'diff_perc'
        q_type3 = 'max'
        cat3 = 'multistep_sup'
    elif mode == 'simple':
        q3 = 'What is the largest energy consumption from 2022 to 2023 among the following companies?'
        op3 = '-'  # FIXME: Maybe min?
        q_type3 = 'max'
        cat3 = 'sup'
    else:
        raise ValueError("Wrong mode!")

    years3 = [2023, 2022]
    q3_answers = generate_multi_answers(keys, values, None, years3, op3, q_type3, None,
                                        num_companies=num_companies)
    for x in q3_answers:
        x['question'] = q3
        x['operation'] = cat3

    if mode == 'complex':
        q4 = 'What is the total energy consumption generated by the following companies in the last <years> years?'
        op4 = 'sum'
        q_type4 = 'sum'
        cat4 = 'multistep_sum'
    elif mode == 'simple':
        q4 = 'What is the total energy consumption generated by the following companies in the last <years> years?'
        op4 = '-'  # FIXME: Maybe sum?
        q_type4 = 'sum'
        cat4 = 'sum'
    else:
        raise ValueError("Wrong mode!")

    q4_answers = generate_multi_answers(keys, values, units, None, op4, q_type4, energy_conv_map,
                                        num_companies=num_companies)
    for x in q4_answers:
        x['question'] = q4.replace('<years>', str(len(x['years'])))
        x['operation'] = cat4

    if mode == 'complex':
        q5 = 'What is the average percentage variation in energy consumption from 2023 to 2022 among the following companies?'
        op5 = 'diff_perc'
        q_type5 = 'mean'
        cat5 = 'multistep_mean'
    elif mode == 'simple':
        q5 = 'What is the average energy consumption from 2023 to 2022 among the following companies?'
        op5 = '-'
        q_type5 = 'mean'
        cat5 = 'mean'
    else:
        raise ValueError("Wrong mode!")
    years5 = [2023, 2022]
    q5_answers = generate_multi_answers(keys, values, None, years5, op5, q_type5, None,
                                        num_companies=num_companies)
    for x in q5_answers:
        x['question'] = q5
        x['operation'] = cat5

    if mode == 'complex':
        q6 = 'What are the top <top> highest values of average energy consumed in the last <years> years (sorted in <order> order) among the following companies?'
        op6 = 'mean'
        cat6 = 'multistep_rank'
    elif mode == 'simple':
        q6 = 'What are the top <top> highest values of energy consumed in the last <years> years (sorted in <order> order) among the following companies?'
        op6 = '-'
        cat6 = 'rank'
    else:
        raise ValueError("Wrong mode!")
    q6a_answers = generate_multi_answers(keys, values, units, None, op6, 'rank_asc', energy_conv_map,
                                         num_companies=num_companies)
    for x in q6a_answers:
        x['question'] = q6.replace('<order>', 'ascending')
    q6b_answers = generate_multi_answers(keys, values, units, None, op6, 'rank_desc', energy_conv_map,
                                         num_companies=num_companies)
    for x in q6b_answers:
        x['question'] = q6.replace('<order>', 'descending')
    q6_answers = q6a_answers + q6b_answers
    for x in q6_answers:
        new_q6 = x['question'].replace('<years>', str(len(x['years'])))
        new_q6 = new_q6.replace('<top>', str(len(x['answer_value'])))
        x['question'] = new_q6
        x['operation'] = cat6

    return q1_answers + q2_answers + q3_answers + q4_answers + q5_answers + q6_answers


def ren_energy_qa(keys, values, num_companies):
    q1 = 'What is the largest percentage of energy consumption from renewable sources in <year> among the following companies?'
    op1 = '-'
    q_type1 = 'max'
    q1a_answers = generate_multi_answers(keys, values, None, [2023], op1, q_type1, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q1a_answers:
        new_q1 = q1.replace('<year>', '2023')
        x['question'] = new_q1
        x['operation'] = 'multistep_sup'
    q1b_answers = generate_multi_answers(keys, values, None, [2022], op1, q_type1, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q1b_answers:
        new_q1 = q1.replace('<year>', '2022')
        x['question'] = new_q1
        x['operation'] = 'multistep_sup'
    q1_answers = q1a_answers + q1b_answers

    q2 = 'What is the maximum average percentage consumption of energy from renewable sources in the years 2023 and 2022 among the following companies?'
    op2 = 'mean'
    q_type2 = 'max'
    q2_answers = generate_multi_answers(keys, values, None, [2023, 2022], op2, q_type2, None,
                                        num_qa=2, num_companies=num_companies)
    for x in q2_answers:
        x['question'] = q2
        x['operation'] = 'multistep_sup'

    q3 = 'What is the highest change in the percentage consumption of energy from renewable sources between the years 2023 and 2022 for the following companies?'
    op3 = 'diff'
    q_type3 = 'max'
    q3_answers = generate_multi_answers(keys, values, None, [2023, 2022], op3, q_type3,
                                        None, num_qa=2, num_companies=num_companies)
    for x in q3_answers:
        x['question'] = q3
        x['operation'] = 'multistep_sup'

    q4 = 'What is the average percentage of energy consumption from renewable sources in <year> among the following companies?'
    op4 = '-'
    q_type4 = 'mean'
    q4a_answers = generate_multi_answers(keys, values, None, [2023], op4, q_type4, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q4a_answers:
        new_q4 = q4.replace('<year>', '2023')
        x['question'] = new_q4
        x['operation'] = 'multistep_mean'
    q4b_answers = generate_multi_answers(keys, values, None, [2022], op4, q_type4, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q4b_answers:
        new_q4 = q4.replace('<year>', '2022')
        x['question'] = new_q4
        x['operation'] = 'multistep_mean'
    q4_answers = q4a_answers + q4b_answers

    return q1_answers + q2_answers + q3_answers + q4_answers


def energy_qa(num_companies):
    energy_conv_map = {
        'MWh': {
            'TWh': 1e-6,
            'million kWh': 0.001,
            'TJ': 0.0036,
            'GWh': 0.001,
            'GJ': 3.6
        },
        'TWh': {
            'MWh': 1e6,
            'million kWh': 1000,
            'TJ': 3600,
            'GWh': 1000,
            'GJ': 3.6e6
        },
        'million kWh': {
            'MWh': 1000,
            'TWh': 0.001,
            'TJ': 3.6,
            'GWh': 1,
            'GJ': 3600
        },
        'TJ': {
            'MWh': 277.778,
            'TWh': 0.000277778,
            'million kWh': 0.277778,
            'GWh': 0.277778,
            'GJ': 1000
        },
        'GWh': {
            'MWh': 1000,
            'TWh': 0.001,
            'million kWh': 1,
            'TJ': 3.6,
            'GJ': 3600
        },
        'GJ': {
            'MWh': 0.277778,
            'TWh': 2.77778e-7,
            'million kWh': 0.277778,
            'GWh': 0.000277778,
            'TJ': 0.001
        }
    }
    keys = [
        'OTC_SU_2023_310_1', 'NYSE_TTE_2023_381_0', 'OTC_DPSGY_2023_3_0', 'heidelberg-materials_2023_374_0',
        'prosiebensat1-media_2023_64_0', 'brenntag_2023_110_0', 'deutsche-telekom-ag_2023_46_0',
        'NASDAQ_DASTY_2023_117_0', 'terna_2023_287_1', 'munich-re_2023_47_1'
    ]
    cols = [
        {2023: 2023, 2022: 2022, 2021: 2021, 2020: 2020},
        {2023: 2023, 2022: 2022, 2021: 2021, 2020: 2020},
        {2023: 2023, 2022: '2022 adjusted', 2021: 2021, 2020: '2020 adjusted'},
        {2023: 2023, 2022: 2022, 2021: 2021},
        {2023: 2023, 2022: 2022},
        {2023: 2023, 2022: 'Previous year: 2022'},
        {2023: 2023, 2022: 2022, 2021: 2021, 2020: 2020},
        {2023: 2023, 2022: 2022},
        {2023: ['Terna', 2023], 2022: ['Terna', 2022], 2021: ['Terna', 2021]},
        {2023: ['Energy consumption', 2023], 2022: ['Energy consumption', 'Prev. year']},
    ]
    rows = {
        'total': [4, 25, 4, 16, 2, None, 4, 4, 5, 11],
        'renewable': [
            11,
            ([26, 25], ['perc']),
            ([5, 4], ['perc']),
            19,
            ([3, 2], ['perc']),
            None,
            None,
            6,
            None,
            None
        ]
    }
    total_keys = [
        'OTC_SU_2023_310_1', 'NYSE_TTE_2023_381_0', 'OTC_DPSGY_2023_3_0', 'heidelberg-materials_2023_374_0',
        'prosiebensat1-media_2023_64_0', 'deutsche-telekom-ag_2023_46_0',
        'NASDAQ_DASTY_2023_117_0', 'terna_2023_287_1', 'munich-re_2023_47_1'
    ]
    total_values = [
        {2023: 1124327, 2022: 1201276, 2021: 1325491, 2020: 1216845},
        {2023: 157, 2022: 166, 2021: 148, 2020: 147},
        {2023: 35056, 2022: 34493, 2021: 30486, 2020: 27427},
        {2023: 329775, 2022: 347068, 2021: 363226},
        {2023: 33.09, 2022: 33.26},
        {2023: 12241, 2022: 13253, 2021: 13323, 2020: 12843},
        {2023: 71218, 2022: 82766},
        {2023: 793077.6, 2022: 806399, 2021: 812033.5},
        {2023: 250916, 2022: 313698}
    ]
    total_units = ['MWh', 'TWh', 'million kWh', 'TJ', 'GWh', 'GWh', 'MWh', 'GJ', 'MWh']
    total_questions = total_energy_qa(total_keys, total_values, total_units, energy_conv_map, num_companies)

    ren_keys = [
        'OTC_SU_2023_310_1', 'NYSE_TTE_2023_381_0', 'OTC_DPSGY_2023_3_0', 'heidelberg-materials_2023_374_0',
        'prosiebensat1-media_2023_64_0', 'NASDAQ_DASTY_2023_117_0'
    ]
    ren_values = [
        {2023: 62.9, 2022: 57.3, 2021: 50.6},
        {2023: 1.2739, 2022: 0.60},
        {2023: 8.735, 2022: 6.57, 2021: 5.99},
        {2023: 12.3, 2022: 10.4, 2021: 9.1},
        {2023: 67.06, 2022: 66.1155},
        {2023: 84, 2022: 84}
    ]

    total_ixs = [keys.index(x) for x in total_keys]
    tot_row_ixs, tot_col_ixs = check_value_consistency(
        total_keys,
        [rows['total'][ix] for ix in total_ixs],
        [cols[ix] for ix in total_ixs],
        total_values)
    ren_ixs = [keys.index(x) for x in ren_keys]
    ren_row_ixs, ren_col_ixs = check_value_consistency(
        ren_keys,
        [rows['renewable'][ix] for ix in ren_ixs],
        [cols[ix] for ix in ren_ixs],
        ren_values
    )

    ren_questions = ren_energy_qa(ren_keys, ren_values, num_companies)

    questions = total_questions + ren_questions

    for q in questions:
        q['GRI'] = 302

    return questions


def direct_emission_qa(keys, values, units, conv_map, num_companies):
    q1 = 'What is the highest average direct GHG emissions (Scope 1) produced in the years 2023 and 2022 among the following companies?'
    years1 = [2023, 2022]
    op1 = 'mean'
    q_type1 = 'max'
    q1_answers = generate_multi_answers(keys, values, units, years1, op1, q_type1, conv_map,
                                        num_companies=num_companies)
    for x in q1_answers:
        x['question'] = q1
        x['operation'] = 'multistep_sup'

    q2 = 'What is the average percentage variation of direct GHG emissions from 2022 to 2023 among the following companies?'
    op2 = 'diff_perc'
    q_type2 = 'mean'
    q2_answers = generate_multi_answers(keys, values, None, [2023, 2022], op2, q_type2,
                                        None, num_qa=2, num_companies=num_companies)
    for x in q2_answers:
        x['question'] = q2
        x['operation'] = 'multistep_mean'

    q3 = 'What are the <k> <sup> values of average direct GHG emissions (Scope 1) produced in the years 2023 and 2022 (sorted in <order> order) among the following companies?'
    op3 = 'mean'
    q3a_answers = generate_multi_answers(keys, values, units, [2023, 2022], op3, 'rank_desc',
                                         conv_map, num_companies=num_companies)
    for x in q3a_answers:
        x['question'] = q3.replace('<order>', 'descending').replace('<sup>', 'highest')
    q3b_answers = generate_multi_answers(keys, values, units, [2023, 2022], op3, 'rank_asc',
                                         conv_map, num_companies=num_companies)
    for x in q3b_answers:
        x['question'] = q3.replace('<order>', 'ascending').replace('<sup>', 'lowest')
    q3_answers = q3a_answers + q3b_answers
    for x in q3_answers:
        new_q3 = x['question'].replace('<k>', str(len(x['answer_value'])))
        x['question'] = new_q3
        x['operation'] = 'multistep_rank'

    return q1_answers + q2_answers + q3_answers


def total_emission_qa(keys, values, units, conv_map, num_companies):
    q1 = 'What is the highest total emissions (both direct and indirect) of GHG generated in <year> among the following companies?'
    op1 = '-'
    q_type1 = 'max'
    q1a_answers = generate_multi_answers(keys, values, units, [2023], op1, q_type1, conv_map,
                                         num_companies=num_companies)
    q1b_answers = generate_multi_answers(keys, values, units, [2022], op1, q_type1, conv_map,
                                         num_companies=num_companies)
    for x1, x2 in zip(q1a_answers, q1b_answers):
        x1['question'] = q1.replace('<year>', '2023')
        x2['question'] = q1.replace('<year>', '2022')
        x1['operation'] = 'multistep_sup'
        x2['operation'] = 'multistep_sup'
    q1_answers = q1a_answers + q1b_answers

    q2 = 'What is the lowest total GHG emissions (both direct and indirect) in the last 2 years among the following companies?'
    op2 = 'sum'
    q_type2 = 'min'
    q2_answers = generate_multi_answers(keys, values, units, [2023, 2022], op2, q_type2, conv_map,
                                        num_companies=num_companies)
    for x in q2_answers:
        x['question'] = q2
        x['operation'] = 'multistep_sup'

    return q1_answers + q2_answers


def scope_ratio_emission_qa(keys, values, units, conv_map, num_companies):
    q1 = 'What is the highest percentage ratio of GHG emissions Scope 1 to Scope 3 in <year> among the following companies?'
    op1 = '-'
    q_type1 = 'max'
    q1a_answers = generate_multi_answers(keys, values, None, [2023], op1, q_type1, None,
                                         num_companies=num_companies)
    q1b_answers = generate_multi_answers(keys, values, None, [2022], op1, q_type1, None,
                                         num_companies=num_companies)
    for x1, x2 in zip(q1a_answers, q1b_answers):
        x1['question'] = q1.replace('<year>', '2023')
        x2['question'] = q1.replace('<year>', '2022')
        x1['operation'] = 'multistep_sup'
        x2['operation'] = 'multistep_sup'
    q1_answers = q1a_answers + q1b_answers

    q2 = 'What is the average of the percentage ratios of GHG emissions Scope 1 to Scope 3 in <year> among the following companies?'
    op2 = '-'
    q_type2 = 'mean'
    q2a_answers = generate_multi_answers(keys, values, None, [2023], op2, q_type2, None,
                                         num_companies=num_companies)
    q2b_answers = generate_multi_answers(keys, values, None, [2022], op2, q_type2, None,
                                         num_companies=num_companies)
    for x1, x2 in zip(q2a_answers, q2b_answers):
        x1['question'] = q2.replace('<year>', '2023')
        x2['question'] = q2.replace('<year>', '2022')
        x1['operation'] = 'multistep_mean'
        x2['operation'] = 'multistep_mean'
    q2_answers = q2a_answers + q2b_answers

    return q1_answers + q2_answers


def emission_qa(num_companies):
    keys = [
        'vivendi_2023_105_0',
        'terna_2023_286_0',
        'NASDAQ_DASTY_2023_141_0',
        'munich-re_2023_46_0',
        'axa_2023_179_0',
        'NYSE_AZ_2023_60_0',
        # 'commerzbank_2023_69_0',
        'OTC_RWNEF_2023_79_0',
        # 'OTC_DPSGY_2023_66_0',
        'OTC_ESOCF_2023_137_0',
        'OTC_BAYZF_2023_63_0',
        'OTC_ADDDF_2023_84_0'
    ]
    cols = [
        {2023: 2023, 2022: 2022},
        {2023: ['TERNA', 2023], 2022: ['TERNA', 2022], 2021: ['TERNA', 2021]},
        {2023: [2023, 'Value'], 2022: ['Unnamed: 3_level_0', 2022], 2021: ['Unnamed: 4_level_0', 2021]},
        {2023: [2023, 'tCO2e'], 2022: ['Prev. year', 'tCO2e']},
        {2023: 2023, 2022: 2022},
        {2023: 2023, 2022: 2022},
        # None,
        {2023: 2023, 2022: 2022},
        # None,
        {2023: 2023, 2022: 2022},
        {2023: ['Unnamed: 2_level_0', 2023], 2022: ['Unnamed: 1_level_0', 2022]},
        {2023: 2023, 2022: 2022},
    ]
    rows = {
        'scope_1': [2, 3, 5, 3, 3, 2, 2, 2, 3, 2],
        'scope_2_location': [4, None, None, None, 6, 4, 3, 3, None, None],
        'scope_2_market': [3, None, None, None, 5, 3, None, 4, 4, 7],
        'scope_2': [
            ([4, 3], ['+']),
            None,
            10,
            5,
            ([6, 5], ['+']),
            ([4, 3], ['+']),
            3,
            ([4, 3], ['+']),
            4,
            7
        ],
        'scope_3': [7, None, 14, 6, ([8, 14], ['+']), 5, 4, 5, 6, 12],
        'total': [
            ([2, 4, 3, 7], ['+']),
            5,
            23,
            15,
            ([16, 17], ['+']),
            ([2, 4, 3, 5], ['+']),
            ([2, 3, 4], ['+']),
            ([2, 4, 3, 5], ['+']),
            ([3, 4, 6], ['+']),
            18
        ]
    }
    scope_1 = [
        {2023: 10291, 2022: 11228},
        {2023: 71724.8, 2022: 72477.1, 2021: 68942},
        {2023: 4178, 2022: 4476, 2021: 3950},
        {2023: 33093, 2022: 43664},
        {2023: 21598, 2022: 21382},
        {2023: 31774, 2022: 30953},
        # {2023: 17418},
        {2023: 61.9, 2022: 85.4},
        # {2023: 8.25, 2022: 8.3},
        {2023: 34.51, 2022: 53.07},
        {2023: 1.89, 2022: 1.91},
        {2023: 21779, 2022: 21856}
    ]
    scope_2_location = [
        {2023: 22042, 2022: 22603},
        {},
        {},
        {},
        {2023: 42423, 2022: 45848},
        {2023: 112228, 2022: 138339},
        # {2023: 59367},
        {2023: 0.2, 2022: 0.1},
        # {},
        {2023: 3.28, 2022: 3.82},
        {},
        {}
    ]
    scope_2_market = [
        {2023: 14221, 2022: 19496},
        {},
        {},
        {},
        {2023: 30712, 2022: 37172},
        {2023: 7929, 2022: 30490},
        # {2023: 12867},
        {},
        # {2023: 0.05, 2022: 0.07},
        {2023: 4.51, 2022: 5.1},
        {2023: 1.11, 2022: 1.12},
        {2023: 142457, 2022: 142293}
    ]
    scope_2 = [
        {2023: 36263, 2022: 42099},
        {},
        {2023: 3193, 2022: 3324, 2021: 12500},
        {2023: 14249, 2022: 18310},
        {2023: 73135, 2022: 83020},
        {2023: 120157, 2022: 168829},
        # {2023: 72234},
        {2023: 0.2, 2022: 0.1},
        # {2023: 0.05, 2022: 0.07},
        {2023: 7.79, 2022: 8.92},
        {2023: 1.11, 2022: 1.12},
        {2023: 142457, 2022: 142293}
    ]
    scope_3 = [
        {2023: 735018, 2022: 770748},
        {},
        {2023: 179523, 2022: 168709, 2021: 123269},
        {2023: 3039435, 2022: 3130824},
        {2023: 343632, 2022: 373959},
        {2023: 96745, 2022: 92467},
        # {2023: 46306},
        {2023: 21.6, 2022: 23.8},
        # {2023: 24.97, 2022: 28.22},
        {2023: 56.53, 2022: 71.04},
        {2023: 9.18, 2022: 9.72},
        {2023: 5894811, 2022: 7635784}
    ]
    scope1_scope3_ratio = [{y: (s1[y] / s3[y]) * 100 for y in s1} if list(s1) == list(s3) else {} for s1, s3 in
                           zip(scope_1, scope_3)]
    total = [
        {2023: 781572, 2022: 824075},
        {2023: 1602382.5, 2022: 1807523.7, 2021: 1727284.6},
        {2023: 186894, 2022: 176510, 2021: 139719},
        {2023: 3086777, 2022: 3192798},
        {2023: 459963, 2022: 499745},
        {2023: 248676, 2022: 292249},
        # {2023: 135958},
        {2023: 83.7, 2022: 109.3},
        # {2023: 33.27, 2022: 36.59},
        {2023: 98.83, 2022: 133.03},
        {2023: 12.18, 2022: 12.75},
        {2023: 6059047, 2022: 7799933}
    ]

    check_value_consistency(keys, rows['scope_1'], cols, scope_1)
    check_value_consistency(keys, rows['scope_2_market'], cols, scope_2_market)
    check_value_consistency(keys, rows['scope_2_location'], cols, scope_2_location)
    check_value_consistency(keys, rows['scope_2'], cols, scope_2)
    check_value_consistency(keys, rows['scope_3'], cols, scope_3)
    check_value_consistency(keys, rows['total'], cols, total)

    units = [
        'TCO2eq', 'Tonnes of CO2 equivalent', 'tCO2-eq', 'tCO2e', 'tCO2eq', 'tCO2e',
        # 'Tonnes CO2 equivalents',
        'million mt CO2e',
        # 'Million metric tons of CO2e',
        'MtCO2eq', 'Million metric tons of CO2 equivalents', 'tons CO2e'
    ]

    conversion_to_tCO2eq = {
        'TCO2eq': 1,
        'Tonnes of CO2 equivalent': 1,
        'tCO2-eq': 1,
        'tCO2e': 1,
        'tCO2eq': 1,
        'Tonnes CO2 equivalents': 1,
        'million mt CO2e': 1e-6,
        'Million metric tons of CO2e': 1e-6,
        'MtCO2eq': 1e-6,
        'Million metric tons of CO2 equivalents': 1e-6,
        'tons CO2e': 1
    }
    emission_conv_map = {
        unit: {
            other_unit: (
                    conversion_to_tCO2eq[other_unit] / conversion_to_tCO2eq[unit]
            ) for other_unit in units
        } for unit in units
    }

    direct_questions = direct_emission_qa(keys, scope_1, units, emission_conv_map, num_companies)
    total_questions = total_emission_qa(keys, total, units, emission_conv_map, num_companies)
    scope_ratio_questions = scope_ratio_emission_qa(
        keys[:1] + keys[2:],
        scope1_scope3_ratio[:1] + scope1_scope3_ratio[2:],
        units[:1] + units[2:],
        emission_conv_map,
        num_companies
    )

    questions = direct_questions + total_questions + scope_ratio_questions

    for q in questions:
        q['GRI'] = 305

    return questions


def total_waste_qa(keys, values, units, conv_map, num_companies):
    q1 = 'What is the highest average amount of waste generated in the years 2023 and 2022 among the following companies?'
    years1 = [2023, 2022]
    op1 = 'mean'
    q_type1 = 'max'
    q1_answers = generate_multi_answers(keys, values, units, years1, op1, q_type1, conv_map,
                                        num_companies=num_companies)
    for x in q1_answers:
        x['question'] = q1
        x['operation'] = 'multistep_sup'

    q2 = 'What is the lowest percentage variation of waste generated from 2022 to 2023 among the following companies?'
    years2 = [2023, 2022]
    op2 = 'diff_perc'
    q_type2 = 'min'
    q2_answers = generate_multi_answers(keys, values, None, years2, op2, q_type2, None,
                                        num_companies=num_companies)
    for x in q2_answers:
        x['question'] = q2
        x['operation'] = 'multistep_sup'

    return q1_answers + q2_answers


def haz_perc_waste_qa(keys, values, units, conv_map, num_companies):
    q1 = 'What is the highest percentage of hazardous waste generated in <year> among the following companies?'
    op1 = '-'
    q_type1 = 'max'
    q1a_answers = generate_multi_answers(keys, values, None, [2023], op1, q_type1, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q1a_answers:
        new_q1 = q1.replace('<year>', '2023')
        x['question'] = new_q1
        x['operation'] = 'multistep_sup'
    q1b_answers = generate_multi_answers(keys, values, None, [2022], op1, q_type1, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q1b_answers:
        new_q1 = q1.replace('<year>', '2022')
        x['question'] = new_q1
        x['operation'] = 'multistep_sup'
    q1_answers = q1a_answers + q1b_answers

    q2 = 'What is the average percentage of hazardous waste generated in <year> among the following companies?'
    op2 = '-'
    q_type2 = 'mean'
    q2a_answers = generate_multi_answers(keys, values, None, [2023], op2, q_type2, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q2a_answers:
        new_q2 = q2.replace('<year>', '2023')
        x['question'] = new_q2
        x['operation'] = 'multistep_mean'
    q2b_answers = generate_multi_answers(keys, values, None, [2022], op2, q_type2, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q2b_answers:
        new_q2 = q2.replace('<year>', '2022')
        x['question'] = new_q2
        x['operation'] = 'multistep_mean'
    q2_answers = q2a_answers + q2b_answers

    q3 = 'What are the <k> <sup> percentages of hazardous waste generated in <year> (sorted in <order> order) among the following companies?'
    op3 = '-'
    q3a_answers = generate_multi_answers(keys, values, None, [2023], op3, 'rank_desc',
                                         None, num_qa=2, num_companies=num_companies)
    for x in q3a_answers:
        x['question'] = (q3.replace('<year>', '2023').replace('<order>', 'descending')
                         .replace('<sup>', 'highest'))
    q3b_answers = generate_multi_answers(keys, values, None, [2022], op3, 'rank_asc',
                                         None, num_qa=2, num_companies=num_companies)
    for x in q3b_answers:
        x['question'] = (q3.replace('<year>', '2022').replace('<order>', 'ascending')
                         .replace('<sup>', 'lowest'))
    q3_answers = q3a_answers + q3b_answers
    for x in q3_answers:
        new_q3 = x['question'].replace('<k>', str(len(x['answer_value'])))
        x['question'] = new_q3
        x['operation'] = 'multistep_rank'

    return q1_answers + q2_answers + q3_answers


def haz_waste_qa(keys, values, units, conv_map, num_companies):
    q1 = 'What is the average amount of hazardous waste obtained by summing the values related to the years 2023 and 2022 among the following companies?'
    years1 = [2023, 2022]
    op1 = 'sum'
    q_type1 = 'mean'
    q1_answers = generate_multi_answers(keys, values, units, years1, op1, q_type1, conv_map, num_qa=2,
                                        num_companies=num_companies)
    for x in q1_answers:
        x['question'] = q1
        x['operation'] = 'multistep_mean'

    q2 = 'What are the <k> <sup> amounts of hazardous waste obtained by summing the values related to the years 2023 and 2022 (sorted in <order> order) among the following companies?'
    op2 = 'sum'
    q2a_answers = generate_multi_answers(keys, values, units, [2023, 2022], op2, 'rank_desc',
                                         conv_map, num_qa=2, num_companies=num_companies)
    for x in q2a_answers:
        x['question'] = q2.replace('<order>', 'descending').replace('<sup>', 'highest')
    q2b_answers = generate_multi_answers(keys, values, units, [2023, 2022], op2, 'rank_asc', conv_map,
                                         num_qa=2, num_companies=num_companies)
    for x in q2b_answers:
        x['question'] = q2.replace('<order>', 'ascending').replace('<sup>', 'lowest')
    q2_answers = q2a_answers + q2b_answers
    for x in q2_answers:
        new_q2 = x['question'].replace('<k>', str(len(x['answer_value'])))
        x['question'] = new_q2
        x['operation'] = 'multistep_rank'

    return q1_answers + q2_answers


def non_vs_haz_waste_qa(keys, values, units, conv_map, num_companies):
    q1 = 'What is the highest percentage of hazardous compared to non-hazardous waste produced in <year> among the following companies?'
    op1 = '-'
    q_type1 = 'max'
    q1a_answers = generate_multi_answers(keys, values, None, [2023], op1, q_type1, None,
                                         num_qa=2, num_companies=num_companies)
    q1b_answers = generate_multi_answers(keys, values, None, [2022], op1, q_type1, None,
                                         num_qa=2, num_companies=num_companies)
    for x in q1a_answers:
        x['question'] = q1.replace('<year>', '2023')
        x['operation'] = 'multistep_sup'
    for x in q1b_answers:
        x['question'] = q1.replace('<year>', '2022')
        x['operation'] = 'multistep_sup'
    q1_answers = q1a_answers + q1b_answers

    return q1_answers


def waste_qa(num_companies):
    keys = [
        'OTC_SU_2023_309_0',
        'NYSE_TTE_2023_387_3',
        'heidelberg-materials_2023_358_0',
        'terna_2023_295_0',
        'vivendi_2023_140_0',
        'NASDAQ_DASTY_2023_141_0',
        'OTC_BAMGF_2023_311_0',
        'OTC_CRERF_2023_79_0'
    ]
    cols = [
        {2023: 2023, 2022: 2022, 2021: 2021, 2020: 2020},
        {2023: 2023, 2022: 2022, 2021: 2021, 2020: 2020, 2019: 2019},
        {2023: 2023, 2022: 2022, 2021: 2021},
        {2023: ['Terna', 2023], 2022: ['Terna', 2022], 2021: ['Terna', 2021]},
        {2023: 2023, 2022: 2022},
        {2023: [2023, 'Value'], 2022: ['Unnamed: 3_level_0', 2022]},
        {2023: 2023, 2022: 2022, 2021: 2021, 2020: 2020, 2019: 2019},
        {2023: 2023, 2022: 2022}
    ]
    rows = {
        'total': [3, 3, 21, 3, 9, 31, 2, 3],
        'haz': [11, 5, ([23, 21], ['*', '/100']), 38, ([4, 5], ['+']), 33,
                None, None],
        'non_haz': [5, 4, ([22, 21], ['*', '/100']), 20, 8, 32, None, None]
    }
    total = [
        {2023: 124139, 2022: 131402, 2021: 136816, 2020: 125292},
        {2023: 521, 2022: 498, 2021: 500, 2020: 501, 2019: 662},
        {2023: 476.5, 2022: 953.1, 2021: 1276.7},
        {2023: 7671.6, 2022: 9078.7, 2021: 8524.7},
        {2023: 10007, 2022: 11388},
        {2023: 931.3, 2022: 1321.5},
        {2023: 927880, 2022: 818387, 2021: 829498, 2020: 775459, 2019: 780911},
        {2023: 671, 2022: 587}
    ]
    haz = [
        {2023: 7573, 2022: 8091, 2021: 8549, 2020: 7685},
        {2023: 202, 2022: 176, 2021: 165, 2020: 198, 2019: 288},
        {2023: 28.59, 2022: 19.062, 2021: 89.369},
        {2023: 4700.6, 2022: 5886.2, 2021: 5451.4},
        {2023: 793, 2022: 2958},
        {2023: 57.3, 2022: 47.5},
        {},
        {}
    ]
    haz_perc = [{k: (h_item[k] / t_item[k]) * 100 for k in t_item} if h_item != {} else {} for t_item, h_item in
                zip(total, haz)]
    non_haz = [
        {2023: 116566, 2022: 123311, 2021: 128267, 2020: 117607},
        {2023: 319, 2022: 322, 2021: 335, 2020: 303, 2019: 375},
        {2023: 447.91, 2022: 934.038, 2021: 1187.331},
        {2023: 2971, 2022: 3192.5, 2021: 3073.3},
        {2023: 9214, 2022: 8430},
        {2023: 874, 2022: 1274},
        {},
        {}
    ]
    non_vs_haz_perc = [{k: (h_item[k] / nh_item[k]) * 100 for k in h_item} if h_item != {} else {} for h_item, nh_item
                       in
                       zip(haz, non_haz)]

    check_value_consistency(keys, rows['total'], cols, total)
    check_value_consistency(keys, rows['haz'], cols, haz)
    check_value_consistency(keys, rows['non_haz'], cols, non_haz)

    units = ['tons', 'kt', 'kt', 'tonnes', 'tons', 'tons', 't', 'tonnes']

    conversion_to_metric_tons = {
        'kt': 1e-3,
        'tonnes': 1,
        'tons': 1,
        't': 1
    }
    waste_conv_map = {
        unit: {
            other_unit: (
                    conversion_to_metric_tons[other_unit] / conversion_to_metric_tons[unit]
            ) for other_unit in units
        } for unit in units
    }

    total_questions = total_waste_qa(keys, total, units, waste_conv_map, num_companies)
    haz_perc_questions = haz_perc_waste_qa(keys[:-2], haz_perc[:-2], units[:-2], waste_conv_map, num_companies)
    haz_questions = haz_waste_qa(keys[:-2], haz[:-2], units[:-2], waste_conv_map, num_companies)
    non_vs_haz_questions = non_vs_haz_waste_qa(keys[:-2], non_vs_haz_perc[:-2], units[:-2], waste_conv_map,
                                               num_companies)

    questions = total_questions + haz_perc_questions + haz_questions + non_vs_haz_questions

    for q in questions:
        q['GRI'] = 306

    return questions


def total_water_qa(keys, values, units, conv_map, num_companies):
    q1 = 'What is the average total water consumption across the following companies, calculated as the sum of their 2023 and 2022 consumption?'
    years1 = [2023, 2022]
    op1 = 'sum'
    q_type1 = 'mean'
    q1_answers = generate_multi_answers(keys, values, units, years1, op1, q_type1, conv_map, num_qa=2,
                                        num_companies=num_companies)
    for x in q1_answers:
        x['question'] = q1
        x['operation'] = 'multistep_mean'

    q2 = 'What is the smallest percentage variation in water consumption from 2022 to 2023 among the following companies?'
    years2 = [2023, 2022]
    op2 = 'diff_perc'
    q_type2 = 'min'
    q2_answers = generate_multi_answers(keys, values, None, years2, op2, q_type2, None, num_qa=1,
                                        num_companies=num_companies)
    for x in q2_answers:
        x['question'] = q2
        x['operation'] = 'multistep_sup'

    q3 = 'What is the highest water consumption calculated as the average of the 2023 and 2022 consumption for the following companies?'
    years3 = [2023, 2022]
    op3 = 'mean'
    q_type3 = 'max'
    q3_answers = generate_multi_answers(keys, values, units, years3, op3, q_type3, conv_map, num_qa=5,
                                        num_companies=num_companies)
    for x in q3_answers:
        x['question'] = q3
        x['operation'] = 'multistep_sup'

    q4 = 'What are the top <k> <sup> water consumption values (in <order> order) obtained by summing the 2023 and 2022 consumption for the following companies?'
    op4 = 'sum'
    q4a_answers = generate_multi_answers(keys, values, units, [2023, 2022], op4, 'rank_desc',
                                         conv_map, num_qa=2, num_companies=num_companies)
    for x in q4a_answers:
        x['question'] = q4.replace('<order>', 'descending').replace('<sup>', 'highest')
    q4b_answers = generate_multi_answers(keys, values, units, [2023, 2022], op4, 'rank_asc', conv_map,
                                         num_qa=2, num_companies=num_companies)
    for x in q4b_answers:
        x['question'] = q4.replace('<order>', 'ascending').replace('<sup>', 'lowest')
    q4_answers = q4a_answers + q4b_answers
    for x in q4_answers:
        new_q4 = x['question'].replace('<k>', str(len(x['answer_value'])))
        x['question'] = new_q4
        x['operation'] = 'multistep_rank'

    return q1_answers + q2_answers + q3_answers + q4_answers


def water_qa(num_companies):
    keys = [
        'OTC_ESOCF_2023_139_0',
        'munich-re_2023_47_0',
        'OTC_BAMGF_2023_310_0',
        'OTC_CRERF_2023_72_0',
        'OTC_BAYZF_2023_132_0'
    ]
    rows = [
        5,
        3,
        2,
        4,
        23
    ]
    cols = [
        {2023: 2023, 2022: 2022},
        {2023: [2023, ' '], 2022: ['Prev. year', ' ']},
        {2023: 2023, 2022: 2022, 2021: 2021, 2020: 2020, 2019: 2019},
        {2023: '2023 Result', 2022: '2022 Result'},
        {2023: 2023, 2022: 2022},
    ]
    total = [
        {2023: 35.4, 2022: 45.2},
        {2023: 432730, 2022: 476997},
        {2023: 5049144, 2022: 4840161, 2021: 4924477, 2020: 4722310, 2019: 5417428},
        {2023: 8.2, 2022: 12.2},
        {2023: 6.78, 2022: 6.66}
    ]

    check_value_consistency(keys, rows, cols, total)

    units = ['millions of m3', 'm3', 'm3', 'millions of cu.m', 'million cubic meters']
    conversion_to_m3 = {
        'millions of m3': 1e-6,
        'm3': 1,
        'millions of cu.m': 1e-6,
        'million cubic meters': 1e-6
    }
    water_conv_map = {
        unit: {
            other_unit: (
                    conversion_to_m3[other_unit] / conversion_to_m3[unit]
            ) for other_unit in units
        } for unit in units
    }

    questions = total_water_qa(keys, total, units, water_conv_map, num_companies)

    for q in questions:
        q['GRI'] = 303

    return questions


def build_q_strings(questions):
    out_questions = []
    for q in questions:
        if q['unit'] is not None:
            q['question'] = q['question'].replace('?', f" in {q['unit']}?")
        q['out'] = q['answer_value']
        out_questions.append(q)

        new_q = q.copy()
        c = new_q['answer_company']
        company_existence_cond = not pd.isnull(c).all() if isinstance(c, np.ndarray) else not pd.isnull(c)
        if company_existence_cond:
            if new_q['question'].startswith("What is"):
                new_q['question'] = new_q['question'].replace('What is', "Which company has")
            elif q['question'].startswith("What are"):
                new_q['question'] = new_q['question'].replace('What are', "Which companies have")
            else:
                raise ValueError(f"Wrong question format!")
            new_q['out'] = new_q['answer_company']
            out_questions.append(new_q)

    assert all(
        [not pd.isnull(x['out']).all() if isinstance(x['out'], np.ndarray) else not pd.isnull(x['out'])
         for x in out_questions])
    for q in out_questions:
        if isinstance(q['out'], np.ndarray):
            q['out'] = ", ".join(map(str, q['out']))

    return out_questions


def remove_qa_duplicates(qa_tab):
    # Check duplicates based on the pairs (question, output)
    # where from the question is removed the measurement unit
    qa_tab['out_tmp'] = qa_tab['out'].astype(str)
    qa_tab['question_tmp'] = qa_tab.apply(
        lambda x: x['question'].replace(x['unit'], '') if not pd.isnull(x['unit']) else x['question'],
        axis=1
    )

    qa_tab = qa_tab.drop_duplicates(subset=['question_tmp', 'out_tmp'])
    qa_tab.drop(['out_tmp', 'question_tmp'], axis=1, inplace=True)

    return qa_tab


def generate_qa(num_tables):
    energy_questions = energy_qa(num_tables)
    emission_questions = emission_qa(num_tables)
    waste_questions = waste_qa(num_tables)
    water_questions = water_qa(num_tables)
    all_questions = energy_questions + emission_questions + waste_questions + water_questions
    out_questions = build_q_strings(all_questions)

    out = remove_qa_duplicates(pd.DataFrame(out_questions))

    return out


def generate_multi_qa_datasets(num_table_list, max_dataset_size=150, random_seed=42):
    data = {}
    for num_tables in num_table_list:
        qa_data = generate_qa(num_tables=num_tables)
        qa_data['num_tables'] = num_tables
        data[num_tables] = qa_data

    # # Remove duplicate QA across datasets
    # # To achieve it, concat the datasets, deduplicate and split back the datasets
    # concat_data = pd.concat([data[num_tables] for num_tables in num_table_list])
    # clean_concat_data = remove_qa_duplicates(concat_data)
    # data = {num_tables: df for num_tables, df in clean_concat_data.groupby(['num_tables'])}

    out_data = {num_tables: df.drop(['num_tables'], axis=1) for num_tables, df in data.items()}
    out_data = {num_tables: df.sample(n=max_dataset_size, random_state=random_seed).reset_index(drop=True) for
                num_tables, df in out_data.items()}
    return out_data


def main():
    random_seed = 42
    np.random.seed(random_seed)
    datasets = generate_multi_qa_datasets(
        num_table_list=[2, 3, 5],
        max_dataset_size=150,
        random_seed=random_seed
    )
    for num_tables, df in datasets.items():
        df.to_csv(f'gri-qa_multitable{num_tables}.csv')


if __name__ == '__main__':
    main()
