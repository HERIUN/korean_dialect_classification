import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import torchaudio

from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor
import pickle
import time
import logging
import shutil
from preprocess_data import get_sorted_wav_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/elder_preprocess.log", encoding="utf-8"),
        #logging.StreamHandler()  # 콘솔에도 출력하고 싶다면
    ],
    force=True
)
logger = logging.getLogger(__name__)

root_dir = "/data/donggukang/data"

paths = {
    "Gyeongsang-do": {
        "old_raw_path" : f"{root_dir}/139-1.중·노년층_한국어_방언_데이터_(강원도,_경상도)",
        "old_label_path" : f"{root_dir}/139-1.중·노년층_한국어_방언_데이터_(강원도,_경상도)"
    },
    "Jeolla-do": {
        "old_raw_path" : f"{root_dir}/139-2.중·노년층_한국어_방언_데이터_(충청도,_전라도,_제주도)",
        "old_label_path" : f"{root_dir}/139-2.중·노년층_한국어_방언_데이터_(충청도,_전라도,_제주도)"
    },
    "Chungcheong-do": {
        "old_raw_path" : f"{root_dir}/139-2.중·노년층_한국어_방언_데이터_(충청도,_전라도,_제주도)",
        "old_label_path" : f"{root_dir}/139-2.중·노년층_한국어_방언_데이터_(충청도,_전라도,_제주도)"
    },
    "Jeju-do": {
        "old_raw_path" : f"{root_dir}/139-2.중·노년층_한국어_방언_데이터_(충청도,_전라도,_제주도)",
        "old_label_path" : f"{root_dir}/139-2.중·노년층_한국어_방언_데이터_(충청도,_전라도,_제주도)"
    },
    "Gangwon-do": {
        "old_raw_path" : f"{root_dir}/139-1.중·노년층_한국어_방언_데이터_(강원도,_경상도)",
        "old_label_path" : f"{root_dir}/139-1.중·노년층_한국어_방언_데이터_(강원도,_경상도)"
    },
}
label_list = list(paths.keys())

code_to_label = { # 파일이름에 해당지역코드
    "cc" : "Chungcheong-do",
    "jj" : "Jeju-do",
    "gw" : "Gangwon-do",
    "jl" : "Jeolla-do",
    "gs" : "Gyeongsang-do"
}

def hhmmss_msec_to_seconds(time_str: str) -> float:
    # 문자열을 ':'로 분리하면 ["HH", "MM", "SS.MSEC"] 형태가 됨
    hh, mm, ss_msec = time_str.split(":")

    # 초와 밀리초를 '.'로 분리
    ss, msec = ss_msec.split(".")

    hours = int(hh)
    minutes = int(mm)
    seconds = int(ss)
    milliseconds = int(msec)

    # 총 초 계산
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds

def get_sorted_label_files(root_directory, label):
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
        results = executor.map(lambda subdir: list(subdir.glob("**/*.json")), subdirs)    
        for sublist in results:
            files.extend(sublist)
    files = sorted(files, key=lambda p: p.name)
    with open(cache_file, "wb") as f:
        pickle.dump(files, f)
    return files

def audio_label_mapping(audio_path_list, label_path_list):
    dict_a = {Path(p).stem: p for p in audio_path_list}
    dict_b = {Path(p).stem: p for p in label_path_list}

    common_stems = set(dict_a.keys()) & set(dict_b.keys())

    paired_dict = {stem: (dict_a[stem], dict_b[stem]) for stem in common_stems}

    return paired_dict


def preprocess_elder_data(speaker_speech_n=30, min_duration=3, out_path="/data/donggukang/data/korean_dialect_sentence_audio"):
    data = []
    gw_gs_path = paths["Gyeongsang-do"]["old_raw_path"]
    gw_gs_audio_files = get_sorted_wav_files(gw_gs_path, "elder_gw_gs")
    gw_gs_label_files = get_sorted_label_files(gw_gs_path, "elder_gw_gs")
    gw_gs_audo_label_dict = audio_label_mapping(gw_gs_audio_files, gw_gs_label_files)

    cc_jl_jj_path = paths["Jeolla-do"]["old_raw_path"]
    cc_jl_jj_audio_files = get_sorted_wav_files(cc_jl_jj_path, "elder_cc_jl_jj")
    cc_jl_jj_label_files = get_sorted_label_files(cc_jl_jj_path, "elder_cc_jl_jj")
    cc_jl_jj_audio_label_dict = audio_label_mapping(cc_jl_jj_audio_files, cc_jl_jj_label_files)

    waves_labels = [gw_gs_audo_label_dict, cc_jl_jj_audio_label_dict]
    
    data = []
    speaker_id_count_dict = defaultdict(int)
    for al_dict in waves_labels:
        for stem, (audio_path, label_path) in tqdm(al_dict.items(), total=len(al_dict.keys())):
            file_info = stem.split('_')
            speaker_region_code = file_info[3][len("speaker"):len("speaker")+2]
            assert speaker_region_code in code_to_label.keys()
            
            speaker_id = file_info[3][len("speaker"):]
            dialect_label = code_to_label[speaker_region_code]

            out_dir = os.path.join(out_path, f"split_elder_{dialect_label}")
            os.makedirs(out_dir, exist_ok=True)

            # split by senetence using label
            try:
                with open(label_path, "r") as f:
                    label_json = json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON from {label_path}: {e}")
                continue

            try:
                y, sr = torchaudio.load(audio_path)
            except Exception as e:
                logger.error(f"Error reading audio from {label_path}: {e}")
                continue

            sentences = label_json["transcription"]["sentences"]

            for si, sentence in enumerate(sentences):
                start = hhmmss_msec_to_seconds(sentence["startTime"]) #"00:00:00.560"
                end = hhmmss_msec_to_seconds(sentence["endTime"])

                if (start is None or end is None
                     or not isinstance(start, (int, float))
                     or not isinstance(end, (int, float))
                     or end-start < min_duration):
                    logger.warning(f"Invalid or too short start/end in {stem}: start={start}, end={end}")
                    continue

                trimmed_speech = y[:, int(start*sr):int(end*sr)]

                if trimmed_speech.numel() == 0:
                    logger.warning(f"Empty audio segment for file {output_path}")
                    continue

                if torch.isnan(trimmed_speech).any() or torch.isinf(trimmed_speech).any():
                    logger.warning(f"Invalid audio data (NaN or Inf) for file {output_path}")
                    continue

                if sentence["dialect"] == sentence["standard"]:
                    continue

                try:
                    output_path = os.path.join(out_dir, stem+f'_{si}'+'.wav')
                    torchaudio.save(output_path, trimmed_speech, sr)
                    data.append({
                        "path": output_path,
                        "dialect": code_to_label[speaker_region_code]
                    })
                    speaker_id_count_dict[speaker_id] += 1
                except Exception as e:
                    print(e)
                
                # 지정된 speaker의 문장개수가 speaker_speech_n을 초과하면 다음 오디오 파일로 넘어감.
                if speaker_id_count_dict[speaker_id] > speaker_speech_n:
                    break

    df = pd.DataFrame(data)
    logger.info(f"Total saved rows: {len(df)}")
    return df


if __name__ == "__main__":
    speaker_speech_n=30
    min_duration=3
    out_path="/data/donggukang/data/korean_dialect_sentence_audio"
    df = preprocess_elder_data(speaker_speech_n, min_duration, out_path)
    df.to_csv(f'old_out_processed.csv', index=False)
