import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import torchaudio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import time
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("process2.log", encoding="utf-8"),
        logging.StreamHandler()  # 콘솔에도 출력하고 싶다면
    ]
)
logger = logging.getLogger(__name__)

def get_sorted_wav_files(root_directory, label):
    cache_file = f"{label}_cache.pickle"
    if Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
            wav_files = cache_data
            return wav_files
    root_directory = Path(root_directory) if isinstance(root_directory, str) else root_directory
    wav_files = []
    subdirs = [p for p in root_directory.iterdir() if p.is_dir()]
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda subdir: list(subdir.glob("**/*.wav")), subdirs)    
        for sublist in results:
            wav_files.extend(sublist)
    
    wav_files = sorted(wav_files, key=lambda p: p.name)
    with open(cache_file, "wb") as f:
        pickle.dump(wav_files, f)    
    return wav_files

def get_sorted_metadata_files(root_directory, label):
    cache_file = f"{label}_label_cache.pickle"
    if Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
            files = cache_data
            return files
    root_directory = Path(root_directory) if isinstance(root_directory, str) else root_directory
    files = []
    subdirs = [p for p in root_directory.iterdir() if p.is_dir()]
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda subdir: list(subdir.glob("**/*_metadata.txt")), subdirs)    
        for sublist in results:
            files.extend(sublist)
    files = sorted(files, key=lambda p: p.name)
    with open(cache_file, "wb") as f:
        pickle.dump(files, f)
    return files

root_dir = "/data/donggukang/data"

paths = {
    "Gyeongsang-do": {
        "raw_path": f"{root_dir}/014.한국어_방언_발화_데이터(경상도)",
    },
    "Jeolla-do": {
        "raw_path": f"{root_dir}/015.한국어_방언_발화_데이터(전라도)",
    },
    "Chungcheong-do": {
        "raw_path": f"{root_dir}/017.한국어_방언_발화_데이터(충청도)",
    },
    "Jeju-do": {
        "raw_path": f"{root_dir}/016.한국어_방언_발화_데이터(제주도)",
    },
    "Gangwon-do": {
        "raw_path": f"{root_dir}/013.한국어_방언_발화_데이터(강원도)",
    },
    "Seoul_Gyeonggi-do": {
        "raw_path": f"{root_dir}/009.한국인대화음성",
    }
}
label_list = list(paths.keys())


def load_origin():
    data = []
    for label,path_info in paths.items():
        wav_files = get_sorted_wav_files(path_info["raw_path"], label)
        # raw_path=path_info["raw_path"]
        # label_path=path_info["label_path"]
        if label=="Seoul_Gyeonggi-do": # 경기도의 경우 이미 문장단위로 되어 있기때문에 문장단위로 쪼개지 않고, 라벨파일을 활용하여, 서울/경기 클래스만 데이터프레임에 추가.
            label_files = get_sorted_metadata_files(path_info["raw_path"], label)
            filtered_set = set()
            for label_file_path in label_files:
                with open(label_file_path, 'r', encoding='utf-8') as lf:
                    lines = lf.readlines()
                    for line in lines:
                        items = line.split("|")
                        # 원하는 인덱스(예: items[-3])가 "1"(서울/경기)인지 확인. 
                        if items[-3].strip() == "1":
                            filtered_set.add(items[0].split('/')[-1].split('.')[0])
            
            for p in tqdm(wav_files):
                name = str(p).split('/')[-1].split('.')[0]
                if name in filtered_set:
                    try:
                        #s = torchaudio.load(p)
                        data.append({
                            "name": name,
                            "path": p,
                            "dialect": label
                        })
                    except Exception as e:
                        print(str(p), e)
                        pass
        else:
            for p in tqdm(wav_files):
                name = str(p).split('/')[-1].split('.')[0]
                try:
                    #y, sr = torchaudio.load(p)
                    data.append({
                        "name": name,
                        "path": p,
                        "dialect": label
                    })
                except Exception as e:
                    print(str(p), e)
                    pass

    df = pd.DataFrame(data)
    print(len(df))
    return df


def split_save_sentence(df, min_sec=3, out_path="/data/donggukang/data/korean_dialect_sentence_audio"):
    # split by sentence, remove first sentence, remove short(<3s) sentence, remove non dialect sentence(dialect form, standard form)
    label_dict = {}
    for label,path_info in paths.items():
        if label == "Seoul_Gyeonggi-do":
            continue
        label_dict[label] = list(Path(path_info["raw_path"]).glob("**/*.json"))
    
    sentence_data = []
    for row in tqdm(df.itertuples(), desc="Processing audio files", total=len(df), position=0):
        if row.dialect == "Seoul_Gyeonggi-do": 
            out_dir = os.path.join(out_path, f"split_{row.dialect}")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, os.path.basename(row.path))
            try:
                if os.path.exists(output_path):
                    sentence_data.append({
                        "path": output_path,
                        "dialect": row.dialect,
                    })
                    continue
                speech, sr = torchaudio.load(row.path)
            except Exception as e:
                logger.error(f"Error loading audio file {row.path}: {e}")
                continue
            if sr * min_sec > speech.shape[-1]: # 3초 이내면 포함시키지 않음.
                logger.info(f"audio file {row.path} is too short. < {min_sec} ")
                continue
            else:
                sentence_data.append({
                    "path": output_path,
                    "dialect": row.dialect,
                })
                shutil.copy2(str(row.path), out_dir)
        
        else:
            out_dir = os.path.join(out_path, f"split_{row.dialect}")
            os.makedirs(out_dir, exist_ok=True)

            label_full_path = None
            for label_path in label_dict[row.dialect]:
                if row.name == label_path.stem:
                    label_full_path = label_path
                    break
            if label_full_path is None:
                logger.warning(f"Label file not found for {row.name} in dialect {row.dialect}")
                continue

            label_name = os.path.basename(label_full_path)

            speech, sr = None, None

            try:
                with open(label_full_path, "r", encoding="utf-8-sig") as f:
                    label_json = json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON from {label_full_path}: {e}")
                continue
                
            u_list = label_json["utterance"]
            for i,u_info in enumerate(u_list): #tqdm(enumerate(u_list), total=len(u_list), desc=f"Preprocessing And Splitting sentence {audio_base_name}", position=1):
                if i==0: # 첫 어절은 무조건 0초부터 시작되므로 첫 어절은 패스.
                    continue
                name = row.name + f"_{i}" + ".wav"
                output_path = os.path.join(out_dir, name)
                if os.path.exists(output_path):
                    sentence_data.append({
                        "path": output_path,
                        "dialect": row.dialect,
                    })
                    continue
                start = u_info["start"]
                end = u_info["end"]

                if start is None or end is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    logger.warning(f"Invalid start/end in {label_name}: start={start}, end={end}")
                    continue

                if u_info["standard_form"] == u_info["dialect_form"]: # 만약 표준어표현과 방언표현이 같으면 건너뜀.
                    continue
                
                # 3초 미만의 문장은 제외.
                if end-start < 3:
                    continue
                
                if speech is None:
                    try:
                        speech, sr = torchaudio.load(row.path)
                    except Exception as e:
                        logger.error(f"Error loading audio file {row.path}: {e}")
                        break

                trimmed_speech = speech[:, int(start*sr):int(end*sr)]

                if trimmed_speech.numel() == 0:
                    logger.warning(f"Empty audio segment for file {output_path}")
                    continue

                if torch.isnan(trimmed_speech).any() or torch.isinf(trimmed_speech).any():
                    logger.warning(f"Invalid audio data (NaN or Inf) for file {output_path}")
                    continue

                try:
                    torchaudio.save(output_path, trimmed_speech, sr)
                    sentence_data.append({
                        "path": output_path,
                        "dialect": row.dialect,
                        # "sr": sr,
                        # "duration": round(trimmed_speech.shape[-1]/sr, 2)
                    })
                except Exception as e:
                    logger.error(f"Error saving audio segment to {output_path}: {e}")

    df2 = pd.DataFrame(sentence_data)
    df2.to_csv('out_preprocessed.csv', index=False)
    return df2

if __name__ == "__main__":
    out_path="/data/donggukang/data/korean_dialect_sentence_audio"
    if not os.path.exists("out.csv"):
        df = load_origin()
        df.to_csv('out.csv', index=False)
    else:
        df = pd.read_csv("out.csv")

    df2 = split_save_sentence(df, out_path=out_path)