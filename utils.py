import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

def get_sorted_wav_files_with_cache(raw_path, cache_file="wav_cache.pkl", cache_validity=3600):
    """
    지정된 경로(raw_path) 내의 모든 wav 파일을 재귀적으로 검색하여 파일 이름 순으로 정렬한 리스트를 반환합니다.
    캐시 파일(cache_file)을 사용해 결과를 저장하며, 캐시 유효시간(cache_validity)이 지난 경우에는 재검색합니다.
    
    Args:
        raw_path (str or Path): 검색할 루트 경로.
        cache_file (str): 캐시 결과를 저장할 파일명.
        cache_validity (int): 캐시 유효시간(초). 기본값은 3600초(1시간)입니다.
        
    Returns:
        List[Path]: 정렬된 wav 파일들의 Path 객체 리스트.
    """
    raw_path = Path(raw_path)
    
    # 캐시 파일이 존재하는지 확인
    if Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        # 캐시 파일의 생성시간이 유효하면 바로 반환
        cache_time, cached_path, wav_files = cache_data
        if time.time() - cache_time < cache_validity and cached_path == str(raw_path.resolve()):
            print("캐시를 사용합니다.")
            return wav_files
    
    # 캐시가 없거나 유효하지 않은 경우, 새로 검색
    print("파일 검색을 수행합니다.")
    # 우선 raw_path 자체의 wav 파일들을 수집
    wav_files = list(raw_path.glob("*.wav"))
    
    # 하위 디렉터리 검색
    subdirs = [p for p in raw_path.iterdir() if p.is_dir()]
    with ThreadPoolExecutor() as executor:
        # 각 하위 디렉터리에서 wav 파일들을 검색
        results = executor.map(lambda subdir: list(subdir.glob("**/*.wav")), subdirs)
    
    for file_list in results:
        wav_files.extend(file_list)
    
    # 파일 이름 기준으로 정렬
    wav_files = sorted(wav_files, key=lambda p: p.name)
    
    # 캐시 저장: (저장 시간, 검색한 디렉터리 경로, 결과)
    with open(cache_file, "wb") as f:
        pickle.dump((time.time(), str(raw_path.resolve()), wav_files), f)
    
    return wav_files

# 사용 예시
# path_info = {"raw_path": "/your/path/here"}
# sorted_wav_files = get_sorted_wav_files_with_cache(path_info["raw_path"])
# for wav in sorted_wav_files:
#     print(wav)
