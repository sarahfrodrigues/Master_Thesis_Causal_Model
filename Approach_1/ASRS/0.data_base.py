import os
import pandas as pd

# =====================================================
# PATHS
# =====================================================

RAW_DATA_FOLDER = (
    r'C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\ASRS_RAW_DATA'
)
OUTPUT_FILE = (
    r'C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\ASRS_ALL.csv'
)

# =====================================================
# COLUMNS TO KEEP
# =====================================================

COLUMNS_TO_KEEP = [
    '_ACN',
    'Time_Date',
    'Events_Anomaly',
    'Events_Detector',
    'Events_Result',
    'Assessments_Contributing Factors / Situations',
    'Assessments_Primary Problem',
    'Report 1_Narrative'
]

# =====================================================
# PARAMETERS
# =====================================================
MIN_NARR_CHAR = 500
MIN_NARR_WORDS = 100
MAX_NARR_WORDS = 600

# =====================================================
# BUILD COMBINED HEADER
# =====================================================

first_file = os.path.join(RAW_DATA_FOLDER, 'ASRS_2006.csv')

header_rows = pd.read_csv(first_file, nrows=2, header=None)

combined_headers = []

for col_idx in range(header_rows.shape[1]):
    h1 = str(header_rows.iloc[0, col_idx]).strip()
    h2 = str(header_rows.iloc[1, col_idx]).strip()

    if h1 != 'nan' and h2 != 'nan':
        column_name = f"{h1}_{h2}"
    elif h1 != 'nan':
        column_name = h1
    else:
        column_name = h2

    combined_headers.append(column_name)

# =====================================================
# LOAD ALL ASRS FILES
# =====================================================

dataframes = []

for filename in os.listdir(RAW_DATA_FOLDER):

    if not filename.endswith('.csv'):
        continue

    file_path = os.path.join(RAW_DATA_FOLDER, filename)
    print(f"Processing {filename}")

    df = pd.read_csv(
        file_path,
        skiprows=2,
        header=None,
        names=combined_headers
    )

    df = df[COLUMNS_TO_KEEP]
    dataframes.append(df)

# =====================================================
# MERGE DATA
# =====================================================

ASRS_df = pd.concat(dataframes, ignore_index=True)

ASRS_df = ASRS_df.dropna(how='all')
ASRS_df = ASRS_df.drop_duplicates(subset='_ACN')

# =====================================================
# FILTER NARRATIVES BY WORDS AND CHARACTERS
# =====================================================

ASRS_df['Report 1_Narrative'] = ASRS_df['Report 1_Narrative'].astype(str)

# Word count filter
ASRS_df['narr_word_count'] = ASRS_df['Report 1_Narrative'].str.split().str.len()
ASRS_df = ASRS_df[
    (ASRS_df['narr_word_count'] >= MIN_NARR_WORDS) &
    (ASRS_df['narr_word_count'] <= MAX_NARR_WORDS)
]

# Character count filter
ASRS_df['narr_char_count'] = ASRS_df['Report 1_Narrative'].str.len()
ASRS_df = ASRS_df[ASRS_df['narr_char_count'] >= MIN_NARR_CHAR]

# Drop helper columns
ASRS_df = ASRS_df.drop(columns=['narr_word_count', 'narr_char_count'])


# =====================================================
# SAVE
# =====================================================

ASRS_df.to_csv(OUTPUT_FILE, index=False)

print("Data base saved, code completed")
