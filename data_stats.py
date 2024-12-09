import pandas as pd

def main():
    path = "./dataset/gri-qa_extra.csv"

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

    count_pdfs_extr = df_extr.groupby("pdf name")["gri"].count().to_dict()
    count_pdfs_hier = df_hier.groupby("pdf name")["gri"].count().to_dict()
    count_gri_extr = df_extr.groupby("gri")["checked"].count().to_dict()
    count_gri_hier = df_hier.groupby("gri")["checked"].count().to_dict()

    print_summary(
        full_dataset_size, error_size, hier_size, extr_size,
        count_pdfs_extr, count_pdfs_hier, count_gri_extr, count_gri_hier
    )

def print_summary(
    full_dataset_size, error_size, hier_size, extr_size,
    count_pdfs_extr, count_pdfs_hier, count_gri_extr, count_gri_hier
):
    print(f"Total dataset size: {full_dataset_size}")
    print(f"Number of unrelated samples (2): {error_size}")
    print(f"Number of hierarchical samples (3): {hier_size}")
    print(f"Dataset size for simple extraction (0-1): {extr_size}")
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

if __name__ == "__main__":
    main()
