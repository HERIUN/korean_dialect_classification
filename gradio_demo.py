import os
os.environ["GRADIO_TEMP_DIR"] = "./tmp"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import gradio as gr
import torch
import torchaudio
import librosa
from transformers import AutoProcessor, AutoConfig, pipeline
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch.nn.functional as F
import random

# 모델 및 프로세서 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "HERIUN/wav2vec-bert-korean-dialect-recognition"
example_audio_root = "./confirmed_data"
audio_classifier = pipeline(
    "audio-classification", 
    model=model_name,
    device=device,  # GPU 환경이면 device=0 설정 가능 (예: device=0)
)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
config = AutoConfig.from_pretrained(model_name)

def speech_file_to_array_fn(path, sampling_rate=feature_extractor.sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(orig_freq=_sampling_rate, new_freq=sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def classify_audio(filepath, max_sec=20):
    if filepath is None:
        return "오디오를 업로드해주세요."

    y, sr = librosa.load(filepath, sr=16000)
    max_samples = 16000 * max_sec
    if len(y) > max_samples:
        max_start = len(y) - max_samples
        start = random.randint(0,max_start)
        y = y[start : start + max_samples]
        gr.Warning(f"[WARNING] 입력 오디오가 {max_sec}초를 초과하여 초반부 {max_sec}초 구간만 사용했습니다.", duration=5)
    results = audio_classifier(y, top_k=6)
    #results = audio_classifier(filepath, top_k=6)

    # [(label, score), (label, score), ...]
    formatted_result = ""
    label_score_dict = {}
    for res in results:
        label = res["label"]
        score = res["score"]
        label_score_dict[res["label"]] = float(res["score"])
        formatted_result += f"{label}: {score:.4f}\n"
    print(formatted_result)

    return os.path.basename(filepath), label_score_dict

demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(sources=["microphone","upload"], type="filepath", label="음성 입력(또는 업로드)"),
    outputs=[
        gr.Textbox(label="업로드된 파일 이름"),
        gr.Label(num_top_classes=6, label="class"),
    ],
    title="Korean Dialect Classification(한국어 방언 분류기)",
    description="음성 데이터를 업로드하거나 녹음하여 분류합니다.",
    examples=[
        f"{example_audio_root}/gs/result_gs1.wav",
        f"{example_audio_root}/gs/result_gs2.wav",
        f"{example_audio_root}/gs/result_gs4.wav",
        # f"{example_audio_root}/gs/result_gs3.wav",
        # f"{example_audio_root}/jl/result_jl2.wav", 
        # f"{example_audio_root}/jl/result_jl3.wav",
        # f"{example_audio_root}/jl/result_jl4.wav",
        f"{example_audio_root}/jl/result_jl5.wav",
        # f"{example_audio_root}/jl/result_jl6.wav",
        # f"{example_audio_root}/jl/result_jl7.wav",
        # f"{example_audio_root}/cc/result_cc1.wav",
        # f"{example_audio_root}/cc/result_cc2.wav",
        f"{example_audio_root}/cc/result_cc3.wav",
        f"{example_audio_root}/cc/result_cc4.wav",
        f"{example_audio_root}/jj/result_jj_0.wav",
        f"{example_audio_root}/jj/result_jj2.wav",
        f"{example_audio_root}/jj/result_jj3.wav",
        f"{example_audio_root}/gw/result_gw.wav",
        # f"{example_audio_root}/gw/result_gw1.wav",
        # f"{example_audio_root}/gw/result_gw2.wav",
    ]
)


if __name__ == "__main__":
    demo.launch(share=True)
    # gr.Interface.from_pipeline(audio_classifier).launch(share=True)