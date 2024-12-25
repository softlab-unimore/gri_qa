import pandas as pd

gri_dict = {
  "301-1.1": ["301-1.1"],
  "301-2.1": ["301-2.1"],
  "301-2.2": ["301-3.1"],
  "301-2.3": ["301-2.2"],
  "301-2.5": ["301-3.2"],
  "302-1.1": ["302-1.1"],
  "302-1.11": ["302-1.7"],
  "302-1.12": ["302-1.8"],
  "302-1.13": ["302-1.9"],
  "302-1.16": ["302-1.10"],
  "302-1.17": ["302-1.11"],
  "302-1.20": ["302-1.12"],
  "302-1.21": ["302-5.1"],
  "302-1.23": ["302-3.3"],
  "302-1.24": ["302-5.2"],
  "302-1.25": ["302-1.13"],
  "302-1.26": ["302-4.1"],
  "302-1.27": ["302-1.14"],
  "302-1.29": ["302-1.15"],
  "302-1.3": ["302-1.2"],
  "302-1.30": ["302-4.2"],
  "302-1.31": ["302-2.1"],
  "302-1.32": ["302-1.16"],
  "302-1.4": ["302-1.3"],
  "302-1.5": ["302-3.4"],
  "302-1.7": ["302-1.4"],
  "302-1.8": ["302-1.5"],
  "302-1.9": ["302-1.6"],
  "302-3.1": ["302-3.1"],
  "302-3.2": ["302-3.2"],
  "303-3.1": ["303-3.1"],
  "303-3.10": ["303-3.8"],
  "303-3.11": ["303-3.9"],
  "303-3.12": ["303-3.10"],
  "303-3.13": ["303-3.11"],
  "303-3.2": ["303-3.2"],
  "303-3.3": ["303-3.3"],
  "303-3.4": ["303-3.4"],
  "303-3.5": ["303-3.5"],
  "303-3.6": ["303-3.6"],
  "303-3.7": ["303-3.7"],
  "303-4.1": ["303-4.1"],
  "303-4.10": ["303-4.10"],
  "303-4.11": ["303-4.11"],
  "303-4.2": ["303-4.2"],
  "303-4.3": ["303-4.3"],
  "303-4.4": ["303-4.4"],
  "303-4.5": ["303-4.5"],
  "303-4.6": ["303-4.6"],
  "303-4.7": ["303-4.7"],
  "303-4.8": ["303-4.8"],
  "303-4.9": ["303-4.9"],
  "303-5.1": ["303-5.1"],
  "303-5.10": ["303-5.10"],
  "303-5.2": ["303-5.2"],
  "303-5.3": ["303-5.3"],
  "303-5.4": ["303-5.4"],
  "303-5.5": ["303-5.5"],
  "303-5.6": ["303-5.6"],
  "303-5.7": ["303-5.7"],
  "303-5.8": ["303-5.8"],
  "303-5.9": ["303-5.9"],
  "304-3.1": ["304-3.1"],
  "304-4.1": ["304-4.1"],
  "304-4.2": ["304-3.1"],
  "304-4.3": ["304-1.1"],
  "304-4.4": ["304-2.1"],
  "304-4.5": ["304-2.2"],
  "305-1.1": ["305-1.1"],
  "305-1.10": ["305-*.3"],
  "305-1.11": ["305-*.4"],
  "305-1.12": ["305-*.5"],
  "305-1.13": ["305-5.2"],
  "305-1.14": ["305-5.3"],
  "305-1.17": ["305-7.7"],
  "305-1.18": ["305-*.6"],
  "305-1.19": ["305-5.4"],
  "305-1.4": ["305-*.1"],
  "305-1.5": ["305-1.5"],
  "305-1.6": ["305-1.6"],
  "305-1.7": ["305-1.7"],
  "305-1.9": ["305-*.2"],
  "305-2.1": ["305-2.1"],
  "305-2.2": ["305-2.2"],
  "305-2.4": ["305-2.3"],
  "305-2.5": ["305-2.4"],
  "305-3.1": ["305-3.1"],
  "305-3.2": ["305-3.2"],
  "305-3.3": ["305-3.3"],
  "305-3.4": ["305-3.4"],
  "305-3.5": ["305-*.7"],
  "305-4.1": ["305-4.1"],
  "305-4.2": ["305-4.2"],
  "305-5.1": ["305-5.1"],
  "305-7.1": ["305-7.1"],
  "305-7.2": ["305-7.2"],
  "305-7.3": ["305-7.3"],
  "305-7.4": ["305-7.4"],
  "305-7.5": ["305-7.5"],
  "305-7.6": ["305-7.6"],
  "306-3.1": ["306-3.1"],
  "306-3.10": ["306-3.8"],
  "306-3.2": ["306-3.2"],
  "306-3.3": ["306-3.3"],
  "306-3.4": ["306-4.2"],
  "306-3.6": ["306-3.4"],
  "306-3.7": ["306-3.5"],
  "306-3.8": ["306-3.6"],
  "306-3.9": ["306-3.7"],
  "306-4.2": ["306-4.1"],
  "306-5.1": ["306-5.1"],
  "306-5.2": ["306-3.9"],
  "306-5.3": ["306-5.2"],
  "308-2.1": ["308-1.1"],
}

df = pd.read_csv("gri-qa_extra2.csv")
df = df[df["Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)"] != 2]
for i, row in df.iterrows():
    new_el = []
    gris = [el.strip() for el in row["gri_finegrained"].split(", ")]
    for gri in gris:
        found = False
        for k,v in gri_dict.items():
            if gri == k:
                new_el.append(v[0])
                found = True
                break
        if not found:
            print(gri)

    new_el = ', '.join(new_el)
    df.at[i,"gri_finegrained"] = new_el
df.to_csv("gri-qa_extra3.csv")