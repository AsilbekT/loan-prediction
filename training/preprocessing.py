import cudf
import pandas as pd

def load_clean_data(filepath):
    # Step 1: Read real column headers using pandas (row 0)
    true_headers = pd.read_csv(filepath, nrows=0).columns
    headers = [str(col).strip() for col in true_headers]

    # Step 2: Load cuDF DataFrame using those headers (skip metadata row)
    df = cudf.read_csv(filepath, skiprows=1, names=headers)
    df.columns = df.columns.str.strip()

    # Step 3: Filter target values
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    df['loan_status'] = (df['loan_status'] == 'Charged Off').astype('int32')

    # Step 4: Drop target-leaking or unnecessary columns
    leaky_cols = [
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
        "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
        "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
        "funded_amnt_inv"
    ]
    drop_cols = ['id', 'member_id', 'emp_title', 'url', 'title', 'zip_code', 'desc'] + leaky_cols
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Step 5: Drop columns with excessive nulls
    df = df.dropna(thresh=0.7 * len(df), axis=1)

    # Step 6: Encode categoricals and fill nulls
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
        if df[col].null_count > 0:
            if df[col].dtype.kind in ['i', 'f']:
                df[col] = df[col].fillna(-1)
            else:
                df[col] = df[col].fillna(0)

    return df, df.columns.tolist()