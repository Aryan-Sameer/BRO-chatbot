import pandas as pd

def parse_excel_or_csv(file_path: str) -> list[str]:
    """
    Reads any Excel (.xls/.xlsx) or CSV file and converts it
    into plain text chunks formatted like:
    'for row 1 - col1: value1, col2: value2, ...'

    Returns a list of text blocks.
    """
    chunks = []

    if file_path.endswith(".csv"):
        # Read CSV safely
        df = pd.read_csv(file_path, dtype=str).fillna("")
        sheets = {"Sheet1": df}
    else:
        # Read all sheets in Excel
        sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)
        sheets = {name: df.fillna("") for name, df in sheets.items()}

    for sheet_name, df in sheets.items():
        for index, row in df.iterrows():
            # Format each row like the reference
            parsed_text = ", ".join([f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()])
            if parsed_text.strip():
                chunks.append(f"Row {index + 1} in sheet {sheet_name} - {parsed_text},")

    return chunks
