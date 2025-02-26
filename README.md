wav2vec_bert 2.0을 speech encoder로 사용하고 뒷단에 classifier를 붙여서 한국어 방언 음성 분류(audio classification)하는 코드


1. 데이터셋 준비
- aihub에 가입하여 아래 데이터에 대한 요청승인을 받는다.
  - 한국어 대화 음성(경기도)
  - 한국어 방언 발화(경상도)
  - 한국어 방언 발화(충청도)
  - 한국어 방언 발화(강원도)
  - 한국어 방언 발화(제주도)
  - 한국어 방언 발화(전라도)
  - 중·노년층 한국어 방언 데이터(강원도, 경상도)
  - 중·노년층 한국어 방언 데이터(충청도, 전라도, 제주도)
  - 한국인 대화음성
- 데이터 다운로드를 환경변수 AIHUB_ID와 AI_HUBPW를 등록한 후 shell내부의 스크립트를 이용해 데이터를 다운로드 한다.[aihubshell 사용법](https://www.aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100) (기존 aihubshell은 병렬 다운로드 기능이 안되니, 여기있는 aihubshell을 사용할것.)
  - download_raw_data.sh : "한국어 방언 발화", "한국인 대화음성" 데이터의 ***일부*** 원천데이터(.wav)파일과 필요한 label(문장단위 정보(timestamp)및 발화자 정보)파일을 다운로드한다.
     - 만약 전체 데이터를 다운로드 받고 싶을경우 filekey 옵션을 제거하여 다운로드 진행.
     - 충청도의 경우 train데이터의 라벨파일이 대부분 없다는 점 참고.
  - download_elder_data.sh : 중장년층의 방언 데이터 다운로드
      - 1인이 발화 따라말하기
      - 2인이 한가지 주제에 대한 1,2안에 대해 각각 의견을 말하는 패턴
      - 1인이 질문에 대한 1,2안에 대한 답변 말하기(V) : 가장 자연스러움. 양도 제일 많음.
- unzip_download_files.sh : 각 압축파일을 해제하고(unzip 설치 필요) 압축해제가 전부 끝날시 용량 확보를 위해 압축파일을 삭제하는 스크립트.
- preprocess_data.py : 음성파일의 경로와 라벨파일의 경로를 코드에서 수정한 후 실행한다. 방언데이터의 경우 label을 활용해 문장단위로 자르고(첫번째 문장은 초반부에 침묵이 있으므로 제외), 표준텍스트(standard_form)와 방언텍스트(dialect_form)가 다를경우 사용한다(hard condition). 서울/경기 음성은 "한국인 대화음성"데이터셋에서 필터링되어 선택된다.
- preprocess_old_data.py : 중장년층의 음성을 각 파일별 3초이상의 문장들중 화자별 최대 n개 문장만 샘플링.
- filter2.py : 안열리는 오디오 파일등을 필터링.
- 최종적으로 name(파일이름)	path(audio파일경로) dialect(방언종류) 컬럼을 가지는 csv파일들(train,eval)을 생성하면 됨.

2. Train & Evaluation
- 학습과 평가 모두 train_wav2vec_bert.py파일로 진행하며, 인자로 json파일을 받는다. json파일에는 학습,평가,기타설정값들을 설정할 수 있다.
- .env_sample을 참고하여 huggingface token 준비하여 .env로 이름 변경.
- train : ```python train_wav2vec_bert.py train_wav2vec_bert_resume.json```
- eval : ```python train_wav2vec_bert.py eval_wav2vec_bert.json```
- 학습이 끝나면 자동으로 huggingface에 업로드하는 기능과 wandb도 있으니 참고.
- 현재 speech encoder가 기본적으로 freeze되어있으나 freeze 안해보는것도 좋을듯
- DDP로 하고싶으면 train.sh 참고
- single gpu는 train_wav2vec_bert_single_gpu.json 참고
- pretrained model : https://huggingface.co/HERIUN/wav2vec-bert-korean-dialect-recognition

3. Demo
- ```python gradio_demo.py```
- 유튜브링크로부터 음성 클립하는 코드 : ```python youtube_clip.py```

[ ] LoRA finetuning
[ ] upload_data_to_hf.py : 로컬에 저장된 문장 음성파일들을 읽어, train,test로 쪼갠후 전처리작업을 거친후 huggingface에 업로드한다.

