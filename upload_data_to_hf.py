import os
import pandas as pd

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