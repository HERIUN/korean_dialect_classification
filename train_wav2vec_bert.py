#ref : https://github.com/huggingface/transformers/blob/main/examples/pytorch/audio-classification/run_audio_classification.py
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from random import randint
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from huggingface_hub import login

import torch.distributed as dist
from collections import Counter

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2BertForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
import librosa
import wandb
from utils import get_label_weight
from torch.nn.modules.loss import CrossEntropyLoss
import torch
from dotenv import load_dotenv

load_dotenv()
token = os.environ["HF_TOKEN"]

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("korean_dialect_classification", mode="a", encoding="utf-8")
        ],
    )
logger = logging.getLogger(__name__)

def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]

def load_checkpoint_safely(checkpoint_path, device="cuda"):
    """Load checkpoint safely, even if GPU count has changed."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Always load to CPU first
    
    # Check if the checkpoint has CUDA RNG states
    if "cuda" in checkpoint.get("rng_state", {}):
        checkpoint_rng_state = checkpoint["rng_state"]["cuda"]
        num_gpus_checkpoint = len(checkpoint_rng_state)
        num_gpus_current = torch.cuda.device_count()
        
        if num_gpus_checkpoint != num_gpus_current:
            print(f"[WARNING] GPU count mismatch! Checkpoint has {num_gpus_checkpoint} GPUs, but current system has {num_gpus_current} GPUs.")
            
            # Adjust the RNG state size
            adjusted_rng_state = [checkpoint_rng_state[i % num_gpus_checkpoint] for i in range(num_gpus_current)]
            checkpoint["rng_state"]["cuda"] = adjusted_rng_state
    
    return checkpoint

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
# Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
# `predictions` and `label_ids` fields) and has to return a dictionary string to float.
def compute_metrics(eval_pred):
    """Computes accuracy, precision, recall, f1 for multiclass predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    
    # 1) Accuracy
    accuracy_result = accuracy_metric.compute(predictions=predictions, references=references)
    
    # 2) Precision
    precision_result = precision_metric.compute(
        predictions=predictions, 
        references=references, 
        average="macro",  # or "weighted",
        zero_division=0
    )
    
    # 3) Recall
    recall_result = recall_metric.compute(
        predictions=predictions, 
        references=references, 
        average="macro",
        zero_division=0
    )
    
    # 4) F1
    f1_result = f1_metric.compute(
        predictions=predictions, 
        references=references, 
        average="macro",
    )
    
    return {
        "accuracy": accuracy_result["accuracy"],
        "precision": precision_result["precision"],
        "recall": recall_result["recall"],
        "f1": f1_result["f1"]
    }


def speech_file_to_array_fn(path):
    y, sr = librosa.load(path, sr=16000)
    return y


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of a dataset from the datasets package"})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the training audio paths and labels."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the validation audio paths and labels."}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    label_column_name: str = field(
        default="label", metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: float = field(
        default=20,
        metadata={"help": "Audio clips will be randomly cut to this length during training if the value is set."},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_mask: bool = field(
        default=True, metadata={"help": "Whether to generate an attention mask in the feature extractor."}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    label_list = [
        "Gyeongsang-do",
        "Jeolla-do",
        "Chungcheong-do",
        "Jeju-do",
        "Gangwon-do",
        "Seoul_Gyeonggi-do"
    ]
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.token = token
    
    if training_args.local_rank in [-1, 0]:
        wandb.init(
            project="wav2vec2_bert_cls",
            entity="ehdrndd",
            resume="auto" if training_args.do_train else "allow",  # 이전 실행이 있으면 이어서 실행. auto로 하면, run id도 기존에 쓰던걸 씀.
        )
    else:
        # 나머지 프로세스는 disabled 모드로 돌려두면 실제 W&B 로그가 생기지 않음
        wandb.init(mode="disabled")

    
    login(token=model_args.token)
    training_args.token = model_args.token
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_audio_classification", model_args, data_args)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,  # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model_input_name = feature_extractor.model_input_names[0]

    data_files = {
        "train": data_args.train_file, 
        "eval": data_args.eval_file,
    }
    logger.info("loading dataset")
    if training_args.local_rank <= 0:
        raw_datasets = load_dataset("csv", data_files=data_files, delimiter=",")
    if training_args.parallel_mode.value == 'distributed':
        dist.barrier()
        if training_args.local_rank > 0:
            # 다시 한 번 load_dataset을 호출하거나, 이미 캐싱된 상태라면 캐시에서 불러옵니다.
            raw_datasets = load_dataset("csv", data_files=data_files, delimiter=",")
    logger.info("loading dataset done")
    
    def train_transforms(batch):
        """Apply train_transforms across a batch. sample random part of audio"""
        subsampled_wavs = []
        input_column = "path"
        audio_arrays=[]
        # audio_arrays = [speech_file_to_array_fn(path) for path in batch[input_column]]
        for path in batch[input_column]:
            try:
                # Read the audio file and append the result to audio_arrays
                audio = speech_file_to_array_fn(path)
                if not isinstance(audio, (list, np.ndarray)) or len(audio) == 0:
                    raise ValueError(f"Invalid audio array for file: {path}")
                audio_arrays.append(audio)
            except Exception as e:
                # Log the file name and error if something goes wrong
                logger.error(f"Error processing file {path}: {e}")

        for audio in audio_arrays:
            wav = random_subsample(
                audio, max_length=data_args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
            )
            subsampled_wavs.append(wav)
        try:
            inputs = feature_extractor(subsampled_wavs, sampling_rate=feature_extractor.sampling_rate)
        except Exception as e:
            logger.error(f"error batch info : {batch}")

        output_batch = {model_input_name: inputs.get(model_input_name)}
        # output_batch["labels"] = list(batch[data_args.label_column_name])
        output_batch["labels"] = [int(label2id[label_name]) for label_name in batch[data_args.label_column_name]]

        return output_batch

    def val_transforms(batch):
        """Apply val_transforms across a batch. trim linear part of audio"""
        # wavs = [audio["array"] for audio in batch[data_args.audio_column_name]]
        input_column = "path"
        audio_arrays = [speech_file_to_array_fn(path) for path in batch[input_column]]
        wavs = []
        for audio in audio_arrays:
            wav_length = len(audio)
            chunk_length = int(feature_extractor.sampling_rate * data_args.max_length_seconds)
            if wav_length > chunk_length:
                wavs.append(audio[:chunk_length])
            else:
                wavs.append(audio)
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        # output_batch["labels"] = list(batch[data_args.label_column_name])
        output_batch["labels"] = [int(label2id[label_name]) for label_name in batch[data_args.label_column_name]]

        return output_batch


    label2id, id2label = {}, {}
    for i, label in enumerate(label_list):
        label2id[label] = str(i)
        id2label[str(i)] = label

    train_labels = raw_datasets["train"][data_args.label_column_name]  # e.g. ["Seoul_Gyeonggi-do", "Jeolla-do", ...]
    counts = Counter(train_labels)
    class_weights = get_label_weight(counts, label_list)
    def myloss(outputs, labels, num_items_in_batch=None):
        logits = outputs.get('logits')
        w = class_weights.to(logits.device)
        closs = CrossEntropyLoss(w)
        loss = closs(logits, labels)
        return loss

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="audio-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    # freeze the convolutional waveform encoder
    if model_args.freeze_feature_encoder:
        if isinstance(model, Wav2Vec2BertForSequenceClassification):
            model.freeze_base_model()
        else:
            model.freeze_feature_encoder()
            
    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        raw_datasets["train"].set_transform(train_transforms, output_all_columns=False)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        raw_datasets["eval"].set_transform(val_transforms, output_all_columns=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        compute_loss_func=myloss,
        compute_metrics=compute_metrics,
        processing_class=feature_extractor,
    )

    # Training
    logger.info("### Start train")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # Write model card and (optionally) push to hub
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "audio-classification",
            "dataset": data_args.dataset_name,
            "tags": ["audio-classification"],
        }
        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
            trainer.create_model_card(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()