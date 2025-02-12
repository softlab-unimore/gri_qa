system_prompt = """You will be provided with an HTML table that is structurally correct but contains some incorrect numerical values.
Additionally, you will be given a second table with the correct numerical values.
Correct the first table by inserting the correct values and return it in CSV format.
Write only the CSV file. Do not write anything else.
"""

human_prompt = """# TABLE 1

{}

# TABLE 2

{}
"""
