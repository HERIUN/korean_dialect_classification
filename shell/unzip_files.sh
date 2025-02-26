#!/bin/bash

# 1. GNU parallel이 설치되어 있는지 확인
if ! command -v parallel &>/dev/null; then
  echo "GNU parallel이 필요합니다. 설치 후 다시 시도하세요."
  exit 1
fi


# 2. 압축 파일 탐색 후 병렬 처리
#    - find로 검색한 파일 목록을 parallel로 넘겨줌
#    - parallel에서 -j 옵션을 통해 몇 개의 프로세스를 동시에 실행할지 지정
#    - --bar 옵션은 진행률 바를 표시합니다(필요 없으면 제거).
#    - --halt soon,fail=1 은 하나라도 실패하면 전체를 중단합니다(옵션은 상황에 따라 조절하세요).
PARALLEL_CMD=$(which parallel)
find . -type f \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) \
| parallel -j "$(nproc)" --progress --halt soon,fail=1 '
    archive_file={}
    archive_dir="$(dirname "$archive_file")"
    archive_name="$(basename "$archive_file")"

    echo "---------------------------------------------"
    echo "Extracting: $archive_file"

    case "$archive_name" in
        *.zip)
            unzip -o -q "$archive_file" -d "$archive_dir"
            ;;
        *.tar)
            tar -xf "$archive_file" -C "$archive_dir"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "$archive_file" -C "$archive_dir"
            ;;
        *)
            echo "Unknown archive format: $archive_name"
            ;;
    esac

    echo "Deleting: $archive_file"
    rm -f "$archive_file"
'

echo -e "\n모든 압축 파일을 병렬로 해제 후 삭제했습니다."