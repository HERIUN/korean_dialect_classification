import yt_dlp
from pydub import AudioSegment

def download_and_trim_youtube_audio(
    youtube_url: str,
    start_time: float,
    end_time: float,
    output_wav: str = "output.wav"
):
    """
    youtube_url: 유튜브 영상 링크
    start_time: 잘라낼 구간 시작(초 단위)
    end_time:   잘라낼 구간 끝(초 단위)
    output_wav: 최종 저장할 WAV 파일명
    """

    # 1) yt-dlp 옵션 설정
    #    - bestaudio/best로 다운로드
    #    - postprocessors로 ffmpeg를 사용해 WAV 파일로 추출
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "downloaded_audio.%(ext)s",  # 임시 파일명 패턴
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    # 2) 다운로드 수행
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])  # 다운로드 후 "downloaded_audio.wav" 생성

    # 3) pydub로 다운로드된 파일 로드 (파일명이 "downloaded_audio.wav"일 것)
    sound = AudioSegment.from_file("downloaded_audio.wav", format="wav")

    # 4) pydub는 millisecond 단위 사용하므로 1000 곱
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)

    # 5) 구간 추출
    trimmed_sound = sound[start_ms:end_ms]

    # 6) 추출한 구간을 WAV로 내보내기
    trimmed_sound.export(output_wav, format="wav")
    print(f"'{output_wav}' 파일로 저장 완료! (구간: {start_time}~{end_time}초)")

# ========== 사용 예시 ==========
if __name__ == "__main__":
    # 예: 유튜브 영상 URL과 구간 설정
    youtube_url = "https://www.youtube.com/watch?v=XsOPmnPaxAA" #gs
    youtube_url = "https://www.youtube.com/watch?v=117w9OF5mwM" #jl
    youtube_url = "https://www.youtube.com/watch?v=7NUl8tx8uUc" #jj
    youtube_url = "https://www.youtube.com/watch?v=4ahQpgLyg3g" #Cc
    youtube_url = "https://www.youtube.com/watch?v=Q79YdQ3yRJM" #gw

    start_sec = 51  # 시작 시점(초)
    end_sec = 74    # 끝지점 
    output_filename = "/confirmed_data/gw/result_gw2.wav"
    
    download_and_trim_youtube_audio(
        youtube_url,
        start_time=start_sec,
        end_time=end_sec,
        output_wav=output_filename
    )
