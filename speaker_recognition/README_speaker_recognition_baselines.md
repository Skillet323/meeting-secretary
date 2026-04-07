# Speaker Recognition Baselines

This package provides a compact benchmark for speaker recognition / speaker identification experiments.

## What is included

### 1. ECAPA-TDNN baseline

A frozen pretrained SpeechBrain ECAPA model (`speechbrain/spkrec-ecapa-voxceleb`) with a trainable linear classifier head.

### 2. WavLM baseline

A frozen pretrained Microsoft WavLM speaker-verification backbone (`microsoft/wavlm-base-sv`) with a trainable linear classifier head.

### 3. X-vector TDNN baseline

A compact end-to-end x-vector style model trained from scratch on log-mel features.

## Manifest format

CSV with at least:

* `path` — audio file path
* `label` — speaker id / class label

Optional:

* `split` — `train`, `valid`, `test`
* `utt_id` — utterance id

### Example

```csv
path,label,split,utt_id
Audio1.wav,speaker_01,train,0001a
Audio2.wav,speaker_02,train,0002a
Audio4_with_speaker_as_Audio1.wav,speaker_01,valid,0001b
Audio3.wav,speaker_03,test,0003a
```


## Parameters of `speaker_recognition_baselines.py`

### Required parameters

#### `--manifest`

Path to the CSV manifest file. This is the main input that tells the script where audio files are and which speaker label belongs to each file.

#### `--model`

Selects the model family to train. Available values:

* `ecapa` — pretrained ECAPA-TDNN embeddings + classifier head
* `wavlm` — pretrained WavLM speaker-verification backbone + classifier head
* `xvector` — compact x-vector style baseline trained from scratch

## Data split parameters

#### `--split-col`

Name of the column in the manifest that stores the split name. Default: `split`.

#### `--train-split`

Name of the split used for training. Default: `train`.

#### `--valid-split`

Name of the split used for validation. Default: `valid`.

#### `--no-valid`

Disables validation completely. Useful when you want to train on all available data.

## Optimization parameters

#### `--batch-size`

Number of audio examples processed in one training step. Larger values are faster but use more GPU memory.

#### `--epochs`

Number of training epochs.

#### `--lr`

Learning rate for the AdamW optimizer.

#### `--seed`

Random seed for reproducibility.

## Audio preprocessing parameters

#### `--target-sr`

Target sampling rate used for loading and resampling audio. Default: `16000`.

#### `--max-seconds`

Maximum duration of each audio segment in seconds. Longer files are truncated to the first `max-seconds` seconds.

## Pretrained backbone parameters

#### `--freeze-backbone`

Freezes the pretrained encoder for `ecapa` and `wavlm`. Only the classification head is trained. This is the default.

#### `--no-freeze-backbone`

Allows fine-tuning of the pretrained backbone.

## Augmentation parameters

#### `--augment`

Enables simple waveform augmentation during training.

This currently adds a lightweight mix of:

* random gain,
* random circular shift,
* speed perturbation,
* additive noise.

## Output parameters

#### `--out`

File path where the trained checkpoint will be saved. Default: `checkpoint.pt`.

## Example runs

```bash
python speaker_recognition_baselines.py \
  --manifest data/manifest.csv \
  --model ecapa \
  --train-split train \
  --valid-split valid \
  --epochs 5 \
  --batch-size 8 \
  --augment \
  --out ecapa.pt
```

```bash
python speaker_recognition_baselines.py \
  --manifest data/manifest.csv \
  --model wavlm \
  --train-split train \
  --valid-split valid \
  --epochs 5 \
  --batch-size 4 \
  --freeze-backbone \
  --out wavlm.pt
```

```bash
python speaker_recognition_baselines.py \
  --manifest data/manifest.csv \
  --model xvector \
  --train-split train \
  --valid-split valid \
  --epochs 20 \
  --batch-size 16 \
  --augment \
  --out xvector.pt
```