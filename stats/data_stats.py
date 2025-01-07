import pandas as pd

def main():
    path = "../dataset/one-table/gri-qa_extra.csv"

    df = pd.read_csv(path)

    full_dataset_size = len(df)
    error_size = len(df[df['Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)'] == 2])
    hier_size = len(df[df['Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)'] == 3])

    df_extr = df[
        df['Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)'].isin([0, 1]) |
        pd.isna(df['Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)'])
    ]
    df_hier = df[df['Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)'] == 3]

    extr_size = len(df_extr)

    df_correct = pd.concat([df_extr, df_hier])

    count_pdfs = df_correct.groupby("pdf name")["gri"].count()
    count_pdfs_extr = df_extr.groupby("pdf name")["gri"].count()
    count_pdfs_hier = df_hier.groupby("pdf name")["gri"].count()
    count_gri = df_correct.groupby("gri")["checked"].count()
    count_gri_extr = df_extr.groupby("gri")["checked"].count()
    count_gri_hier = df_hier.groupby("gri")["checked"].count()
        
    count_nr_tables_extr = len(df_extr.groupby(["pdf name", "page nbr", "table nbr"]))
    count_nr_tables_hier = len(df_hier.groupby(["pdf name", "page nbr", "table nbr"]))
    count_nr_tables_total = len(df_correct.groupby(["pdf name", "page nbr", "table nbr"]))
    
    df_correct['table id'] = df_correct['pdf name'] + '_' + df_correct['page nbr'].astype(str) + '_' + df_correct['table nbr'].astype(str)
    df_extr['table id'] = df_extr['pdf name'] + '_' + df_extr['page nbr'].astype(str) + '_' + df_extr['table nbr'].astype(str)
    df_hier['table id'] = df_hier['pdf name'] + '_' + df_hier['page nbr'].astype(str) + '_' + df_hier['table nbr'].astype(str)

    nr_tables_per_gri = df_correct.groupby('gri')['table id'].nunique()
    nr_tables_per_gri_extr = df_extr.groupby('gri')['table id'].nunique()
    nr_tables_per_gri_hier = df_hier.groupby('gri')['table id'].nunique()

    nr_gri_per_table = df_correct.groupby('table id')['gri'].nunique()
    nr_gri_per_table_extr = df_extr.groupby('table id')['gri'].nunique()
    nr_gri_per_table_hier = df_hier.groupby('table id')['gri'].nunique()

    # Calculate the frequency distribution: count how many tables have a specific number of unique GRIs
    frequency_distribution = nr_gri_per_table.value_counts()
    frequency_distribution_extr = nr_gri_per_table_extr.value_counts()
    frequency_distribution_hier = nr_gri_per_table_hier.value_counts()
    
    #how many tables for each company
    nr_tables_per_company = df_correct.groupby('pdf name')['table id'].nunique()
    nr_tables_per_company_extr = df_extr.groupby('pdf name')['table id'].nunique()
    nr_tables_per_company_hier = df_hier.groupby('pdf name')['table id'].nunique()
    
    #how many questions on average per table
    avg_questions_per_table = df_correct.groupby('table id')['question'].count().mean()
    avg_questions_per_table_extr = df_extr.groupby('table id')['question'].count().mean()
    avg_questions_per_table_hier = df_hier.groupby('table id')['question'].count().mean()

    print_summary(
        full_dataset_size, error_size, hier_size, extr_size,
        count_pdfs, count_pdfs_extr, count_pdfs_hier,
        count_gri, count_gri_extr, count_gri_hier,
        count_nr_tables_extr, count_nr_tables_hier, count_nr_tables_total,
        nr_tables_per_gri, nr_tables_per_gri_extr, nr_tables_per_gri_hier,
        frequency_distribution, frequency_distribution_extr, frequency_distribution_hier,
        nr_tables_per_company, nr_tables_per_company_extr, nr_tables_per_company_hier,
        avg_questions_per_table, avg_questions_per_table_extr, avg_questions_per_table_hier
    )

def print_summary(
    full_dataset_size, error_size, hier_size, extr_size,
    count_pdfs, count_pdfs_extr, count_pdfs_hier,
    count_gri, count_gri_extr, count_gri_hier,
    count_nr_tables_extr, count_nr_tables_hier, count_nr_tables_total,
    nr_tables_per_gri, nr_tables_per_gri_extr, nr_tables_per_gri_hier,
    frequency_distribution, frequency_distribution_extr, frequency_distribution_hier,
    nr_tables_per_company, nr_tables_per_company_extr, nr_tables_per_company_hier,
    avg_questions_per_table, avg_questions_per_table_extr, avg_questions_per_table_hier
):
    print(f"Total dataset size: {full_dataset_size}")
    print(f"Number of unrelated samples (2): {error_size}")
    print(f"Number of hierarchical samples (3): {hier_size}")
    print(f"Dataset size for simple extraction (0-1): {extr_size}")
    print()

    print("--- PDF DISTRIBUTION TOTAL ---")
    print()
    for pdf, count in count_pdfs.items():
        print(f"\t- {pdf}: {count}")
    print()

    print("--- PDF DISTRIBUTION EXTR ---")
    print()
    for pdf, count in count_pdfs_extr.items():
        print(f"\t- {pdf}: {count}")
    print()

    print("--- PDF DISTRIBUTION HIER ---")
    print()
    for pdf, count in count_pdfs_hier.items():
        print(f"\t- {pdf}: {count}")
    print()

    print("--- GRI DISTRIBUTION TOTAL ---")
    print()
    for gri, count in count_gri.items():
        print(f"\t- {gri}: {count}")
    print()

    print("--- GRI DISTRIBUTION EXTR ---")
    print()
    for gri, count in count_gri_extr.items():
        print(f"\t- {gri}: {count}")
    print()

    print("--- GRI DISTRIBUTION HIER ---")
    print()
    for gri, count in count_gri_hier.items():
        print(f"\t- {gri}: {count}")
    print()
    
    print(f"Number of tables in extr dataset: {count_nr_tables_extr}")
    print(f"Number of tables in hier dataset: {count_nr_tables_hier}")
    print(f"Number of tables in total: {count_nr_tables_total}")
    
    print(f"Number of tables per GRI in total: {nr_tables_per_gri}")
    print(f"Number of tables per GRI in extr: {nr_tables_per_gri_extr}")
    print(f"Number of tables per GRI in hier: {nr_tables_per_gri_hier}")
    
    print(f"Number of GRIs per table in total: {frequency_distribution}")
    print(f"Number of GRIs per table in extr: {frequency_distribution_extr}")
    print(f"Number of GRIs per table in hier: {frequency_distribution_hier}")
    
    print(f"Number of tables per company in total: {nr_tables_per_company}")
    print(f"Number of tables per company in extr: {nr_tables_per_company_extr}")
    print(f"Number of tables per company in hier: {nr_tables_per_company_hier}")

    print(f"Average number of questions per table in total: {avg_questions_per_table}")
    print(f"Average number of questions per table in extr: {avg_questions_per_table_extr}")
    print(f"Average number of questions per table in hier: {avg_questions_per_table_hier}")
    

if __name__ == "__main__":
    main()
