import torchaudio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def is_valid_row(row):
    try:
        arr, sr = torchaudio.load(row["path"])
        if arr.shape[0] == 1 and arr.shape[-1] * sr > 3 * sr:
            return True
    except Exception as e:
        print(f"Error loading file {row['path']}: {e}")
    return False

# 데이터 읽기
df = pd.read_csv("sampled_train_150k_1.csv")

# 병렬 처리로 유효한 행 필터링
with ThreadPoolExecutor() as executor:
    valid_mask = list(executor.map(is_valid_row, [row for _, row in df.iterrows()]))

# 유효한 데이터만 필터링
df_cleaned = df[valid_mask]

# 결과 저장
df_cleaned.to_csv("sampled_train_150k_fast2.csv", index=False)
