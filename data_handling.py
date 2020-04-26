import pandas as pd
import datetime
import numpy as np
from dateutil import parser
import re
from collections import Counter


def extract_phases_from_string(string):
    return re.split('\|', string)

def extract_drug_from_string(string):
    splitted = re.split('\|', string)

    drugs = []
    for s in splitted:
        m = re.search('(Drug:\s)([\s\d\w]*)', s)
        if m:
            drug = m.group(2)
            if 'Placebo' in drug:
                next
            elif 'Hydroxychloroquine' in drug:
                drugs.append('Hydroxychloroquine')
            elif 'Lopinavir' in drug:
                drugs.append('Lopinavir')
            else:
                drugs.append(drug)
    return drugs

def extract_intervention_type_from_string(string):
    # TODO: Extract other intentions types apart from 'Drug' also

    if pd.isna(string):
        return 'Other'
    elif 'Drug' in string:
        return 'Drug'
    else:
        return 'Other'

def extract_age_from_string(string):
    # TODO include minutely data
    if 'Years and older' in string:
        m = re.search('(\d+)(\sYears\sand\solder)', string)
        if m:
            return [int(m.group(1)), 99]
        else:
            return np.nan
    elif 'Months and older' in string:
        m = re.search('(\d+)(\sMonths\sand\solder)', string)
        if m:
            return [round(int(m.group(1))/12, ndigits=3), 99]
        else:
            return np.nan
    elif 'up to ' in string:
        m = re.search('(up\sto\s)(\d+)(\sYears)', string)
        if m:
            return [0, m.group(2)]
    elif 'Years to' in string:
        m = re.search('(\d+)(\sYears\sto\s)(\d+)(\sYears)', string)
        if m:
            return [int(m.group(1)), int(m.group(3))]
        else:
            return np.nan
    else:
        return np.nan

def process_raw_table(raw_table):
    # Transform date columns to timedate
    raw_table.loc[:, 'Completion Date'] = raw_table.apply(lambda row: np.nan if pd.isna(row['Completion Date']) else parser.parse(row['Completion Date']), axis=1)
    raw_table.loc[:, 'First Posted'] = raw_table.apply(lambda row: np.nan if pd.isna(row['First Posted']) else parser.parse(row['First Posted']), axis=1)
    raw_table.loc[:, 'Results First Posted'] = raw_table.apply(lambda row: np.nan if pd.isna(row['Results First Posted']) else parser.parse(row['Results First Posted']), axis=1)
    raw_table.loc[:, 'Last Update Posted'] = raw_table.apply(lambda row: np.nan if pd.isna(row['Last Update Posted']) else parser.parse(row['Last Update Posted']), axis=1)

    # Extract min and max age
    raw_table.loc[:, 'Age'] = raw_table.apply(lambda row: np.nan if pd.isna(row['Age']) else extract_age_from_string(row['Age']), axis=1)

    # Extract phases
    raw_table['Phases'] = raw_table.apply(lambda row: [] if pd.isna(row['Phases']) else extract_phases_from_string(row['Phases']), axis=1)

    # Extract drugs
    raw_table['Intervention Type'] = raw_table.apply(lambda row: extract_intervention_type_from_string(row['Interventions']), axis=1)
    raw_table['Drugs'] = raw_table.apply(lambda row: [] if pd.isna(row['Interventions']) else extract_drug_from_string(row['Interventions']), axis=1)

    return raw_table

def get_distinct_drugs(table):
    drugs = []
    for i in table['Drugs']:
        for j in i:
            drugs.append(j)
    return list(set(drugs))

def get_most_common_drugs(table):
    drugs = []
    for i in table['Drugs']:
        for j in i:
            drugs.append(j)

    drugs_df = pd.DataFrame(list(zip(Counter(drugs).keys(), Counter(drugs).values())), columns=['Drug', 'Number Of Trials'])
    return drugs_df.sort_values(by=['Number Of Trials'], ascending=False, ignore_index=True)

def restrict_table_to_drugs(table, list_of_drugs):
    return table[table.apply(lambda x: any(i in x['Drugs'] for i in list_of_drugs), axis=1)]

def get_active_agent_scores(table):
    distinct_drugs = get_distinct_drugs(table)
    drug_table = []
    for drug in distinct_drugs:
        num_active_studies = table.apply(lambda x: drug in x['Drugs'], axis=1).sum()

        last_week = np.datetime64(datetime.date.today() - datetime.timedelta(days=7))
        table_changes_last_week = table.loc[table['Last Update Posted'] > last_week]
        updates_last_week = table_changes_last_week.apply(lambda x: drug in x['Drugs'], axis=1).sum()

        num_phase_2 = table.apply(lambda x: drug in x['Drugs'] and 'Phase 2' in x['Phases'], axis=1).sum()
        num_phase_3 = table.apply(lambda x: drug in x['Drugs'] and 'Phase 3' in x['Phases'], axis=1).sum()
        num_phase_4 = table.apply(lambda x: drug in x['Drugs'] and 'Phase 4' in x['Phases'], axis=1).sum()

        table_drug = table[table.apply(lambda x: drug in x['Drugs'], axis=1)]
        table_drug = table_drug[table_drug['Age'].notna()]
        min_age = table_drug.loc[:, 'Age'].apply(lambda x: x[0]).min()
        max_age = table_drug.loc[:, 'Age'].apply(lambda x: x[1]).min()

        drug_table.append([drug,
                           num_active_studies,
                           updates_last_week,
                           num_phase_2,
                           num_phase_3,
                           num_phase_4,
                           min_age,
                           max_age])

    columns = ['Active Agent',
               'Number Trials',
               'Number Updates Last Week',
               'Trials in Phase 2',
               'Trials in Phase 3',
               'Trials in Phase 4',
               'Min Age',
               'Max Age']

    score_dataframe = pd.DataFrame(drug_table, columns = columns)
    score_dataframe['Score'] = round(np.log(score_dataframe['Number Updates Last Week']/score_dataframe['Number Trials'] *(score_dataframe['Number Trials']
                                                                                                                     + 2 * score_dataframe['Trials in Phase 2']
                                                                                                                     + 3*score_dataframe['Trials in Phase 3']
                                                                                                                     + 4*score_dataframe['Trials in Phase 4']))/6, ndigits=2)
    score_dataframe = score_dataframe.sort_values('Score', ignore_index=True, ascending=False)
    return score_dataframe
