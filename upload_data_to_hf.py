import os
import pandas as pd
import librosa
import numpy as np

# 폴더 경로와 클래스 이름 설정
base_path = "/data/donggukang/data/korean_dialect_sentence_audio"
folders = {
    "split_Gangwon-do": "Gangwon-do",
    "split_Jeolla-do": "Jeolla-do",
    "split_Chungcheong-do": "Chungcheong-do",
    "split_Jeju-do": "Jeju-do",
    "split_Seoul_Gyeonggi-do": "Seoul_Gyeonggi-do",
    "split_Gyeongsang-do": "Gyeongsang-do",
}
from datasets import Dataset, Features, Sequence, Value, ClassLabel


df = pd.read_csv("sampled_full/csv")

data_list = []
for idx,row in df.iterrows():
    audio_path = row["path"]
    dialect = row["dialect"]
    name = row["name"]

    # librosa로 WAV 로드 (16kHz로 통일)
    y, sr = librosa.load(audio_path, sr=16000)
    y = y.astype(np.float32)
    data_list.append({
        "name": name,
        "array": y,            # 실제 오디오 샘플 배열
        "sampling_rate": sr,   # 16000
        "dialect": dialect,    # 문자열 라벨
    })


data_list = []
for path, dialect in zip(audio_paths, dialect_labels):
    # 원하는 샘플링레이트로 로드 (여기서는 16kHz)
    y, sr = librosa.load(path, sr=16000)
    # float32로 캐스팅(부동소수점 정밀도 줄여 용량 감소)
    y = y.astype(np.float32)

    data_list.append({
        "array": y,            # 실제 파형 배열
        "sampling_rate": sr,   # 샘플링레이트
        "dialect": dialect     # 라벨/메타정보
    })

# 방언 라벨 목록(필요시):
dialect_names = ["Gyeongsang-do", "Jeolla-do", "Chungcheong-do", "Jeju-do", "Gangwon-do", "Seoul_Gyeonggi-do"]

features = Features({
    "array": Sequence(Value("float32")),
    "sampling_rate": Value("int32"),
    "dialect": ClassLabel(names=dialect_names),
})


# 데이터 저장용 리스트
data = []

# 폴더별 파일 탐색 및 데이터 추가
data = [
    {"name": file_name, "path": os.path.join(folder_path, file_name), "dialect": label}
    for folder_name, label in folders.items()  # 폴더와 라벨 순회
    for folder_path in [os.path.join(base_path, folder_name)]  # 폴더 경로 생성
    if os.path.exists(folder_path)  # 폴더 존재 확인
    for file_name in os.listdir(folder_path)  # 폴더 내 파일 탐색
    if os.path.isfile(os.path.join(folder_path, file_name))  # 파일만 처리
]

# DataFrame 생성
df = pd.DataFrame(data)
df_group_counts = df.groupby('dialect').size()
df.to_csv("out_preprocessed2.csv", index=False)
print(df_group_counts)