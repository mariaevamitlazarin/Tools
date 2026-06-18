"""make_sample.py — builds a messy CSV exercising every cleaning case."""
import pandas as pd

rows = [
    # name,            city,           state, signup_date,  age,   salary,        active
    ["  Ana Souza ",   "são paulo",    "SP",  "2021-03-15", "34",  "R$ 4.500,00", "sim"],
    ["BRUNO LIMA",     "SAO PAULO",    "sp",  "15/03/2021", "29",  "3200.50",     "TRUE"],
    ["Ana  Souza",     "Sao Paulo",    "SP",  "2021-03-15", "34",  "4500",        "1"],   # near-dup of row 0
    ["Carlos Mendes",  "rio de janeiro","RJ", "2020-13-01", "-5",  "R$ 7.800,00", "nao"], # bad date, bad age
    ["DÉBORA SÃ£O",    "Belo Horizonte","mg", "2099-01-01", "41",  "5,000.00",    "yes"], # mojibake, future date
    ["Ana Souza",      "sp",           "SP",  "2021-03-15", "34",  "4500",        "sim"], # dup
    ["Eduardo Rocha",  "curtiba",      "PR",  "01-07-2019", "200", "n/a",         "false"],# typo city, impossible age
    ["",               "",             "",    "",           "",    "",            ""],     # empty row
    ["Fernanda Alves", "Salvador",     "BA",  "2022-08-20", "27",  "R$ 6.100,00", "Sim"],
    ["GUSTAVO -",      "recife",       "pe",  "10/12/2023", "38",  "  8200 ",     "0"],
]
df = pd.DataFrame(rows, columns=[
    "Nome ", "Cidade", "Estado", "Data de Inscrição", "Idade", "Salário (R$)", "Ativo"
])
df.to_csv("sample_messy.csv", index=False, encoding="utf-8")
print("wrote sample_messy.csv")
print(df)
