system_prompt = """Ti verrà fornita una tabella HTML strutturalmente corretta ma con alcuni valori numeri sbagliati.
Inoltre, te ne verrà fornita una seconda con i valori numerici corretti.
Correggi la prima tabella inserendo i valori corretti e ritornala in formato csv.
Scrivi esclusivamente il file csv. Non scrivere nient'altro.
"""

human_prompt = """# TABELLA 1

{}

# TABELLA 2

{}
"""