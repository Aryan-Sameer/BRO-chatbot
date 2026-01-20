import pandas as pd

def split_into_tables(df):
    tables = []
    current_table = []

    for _, row in df.iterrows():
        # Check if the row is fully empty
        if all(str(x).strip() == "" for x in row):
            if current_table:
                tables.append(pd.DataFrame(current_table).reset_index(drop=True))
                current_table = []
        else:
            current_table.append(row)

    if current_table:
        tables.append(pd.DataFrame(current_table).reset_index(drop=True))

    return tables


def parse_excel_or_csv(file_path: str) -> list[str]:
    chunks = []

    # Read CSV
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, dtype=str)
        df = df.fillna("")
        sheets = {"Sheet1": df}

    # Read Excel
    else:
        sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)
        sheets = {name: df.fillna("") for name, df in sheets.items()}

    for sheet_name, df in sheets.items():
        tables = split_into_tables(df)

        for table_index, table in enumerate(tables):
            table = table.fillna("").astype(str)

            # Process each row
            for _, row in table.iterrows():
                parsed = ", ".join(
                    [f"{col}: {row[col]}" for col in table.columns if row[col].strip()]
                )
                chunks.append(f"{sheet_name}_table{table_index+1} - {parsed}")

    return chunks
