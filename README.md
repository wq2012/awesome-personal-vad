# Awesome Personal VAD [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

## Table of contents

* [Overview](#overview)
* [Software](#software)
  * [Personal VAD implementations](#personal-vad-implementations)
  * [Target-Speaker VAD (TS-VAD) implementations](#target-speaker-vad-ts-vad-implementations)
  * [Speech and diarization toolkits](#speech-and-diarization-toolkits)
  * [Speaker embedding extractors](#speaker-embedding-extractors)
  * [Data simulation and augmentation](#data-simulation-and-augmentation)
* [Pretrained models](#pretrained-models)
  * [Personal VAD 1.0 & 2.0 collection](#personal-vad-10--20-collection)
  * [Target-Speaker VAD (TS-VAD) models](#target-speaker-vad-ts-vad-models)
* [Demos](#demos)
* [Datasets](#datasets)
  * [Conversational and meeting datasets](#conversational-and-meeting-datasets)
  * [Speech and speaker corpora for simulation](#speech-and-speaker-corpora-for-simulation)
  * [Noise and room impulse response corpora](#noise-and-room-impulse-response-corpora)
* [Publications](#publications)
  * [Special topics](#special-topics)
    * [Personal VAD (Speaker-Conditioned VAD)](#personal-vad-speaker-conditioned-vad)
    * [Joint Personal/Target-Speaker VAD, extraction, and ASR](#joint-personaltarget-speaker-vad-extraction-and-asr)
    * [Online and streaming Target-Speaker VAD](#online-and-streaming-target-speaker-vad)
    * [Multi-channel and spatial Target-Speaker VAD](#multi-channel-and-spatial-target-speaker-vad)
    * [Audio-Visual Target-Speaker VAD](#audio-visual-target-speaker-vad)
  * [Other](#other) ([2026](#2026) · [2025](#2025) · [2024](#2024) · [2023](#2023) · [2022](#2022) · [2021](#2021) · [2020](#2020))
* [Other learning materials](#other-learning-materials)
  * [Project pages and slides](#project-pages-and-slides)
  * [Theses](#theses)
  * [Books](#books)
* [Star History](#star-history)

## Overview

This is a curated list of awesome **Personal Voice Activity Detection (Personal VAD)** — also known as **Speaker-Conditioned Voice Activity Detection** or **Target-Speaker Voice Activity Detection (TS-VAD)** — papers, libraries, pretrained models, demos, datasets, and other resources.

Unlike standard Voice Activity Detection (VAD), which classifies each audio frame into speech vs. non-speech regardless of who is speaking, **Personal VAD** and **Target-Speaker VAD** condition the detector on one or more target speakers' voice profiles (such as d-vectors, x-vectors, i-vectors, or reference speech segments) to detect frame-level voice activity exclusively for the enrolled target speaker(s) while ignoring competing speakers and background noise. It serves both as an ultra-lightweight on-device gatekeeper for personalized streaming speech recognition and full-duplex voice assistants, and as a core neural building block for overlap-aware multi-speaker diarization.

The purpose of this repo is to organize the world's resources for Personal VAD and Target-Speaker VAD, and make them universally accessible and useful.

To add items to this page, simply send a pull request. ([contributing guide](CONTRIBUTING.md))

## Software

### Personal VAD implementations

| Link | Language | Description |
| ---- | -------- | ----------- |
| [personal_vad](https://github.com/wq2012/personal_vad) ![GitHub stars](https://img.shields.io/github/stars/wq2012/personal_vad?style=social) [![Build Status](https://github.com/wq2012/personal_vad/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/wq2012/personal_vad/actions/workflows/pythonapp.yml) [![PyPI Version](https://img.shields.io/pypi/v/personal-vad.svg)](https://pypi.org/project/personal-vad/) | Python & TensorFlow & TFLite | Standalone open-source reproduction of **Personal VAD 1.0** ([Odyssey 2020](https://arxiv.org/abs/1908.04284)) and **Personal VAD 2.0** ([Interspeech 2022](https://arxiv.org/abs/2204.03793)). Implements all conditioning architectures (`SC`, `ST`, `ET`, `SET`, `FiLM`, Speaker Pre-Net), Weighted Pairwise Loss (`WPL`), Unified Enrolled & Enrollment-Less training, and stateful FP32/INT8 TFLite streaming inference. |
| [personalVAD](https://github.com/pirxus/personalVAD) ![GitHub stars](https://img.shields.io/github/stars/pirxus/personalVAD?style=social) | Python & PyTorch & Kaldi | Unofficial PyTorch and Kaldi implementation of [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/1908.04284) (`ET`, `ST`, `SET`, Weighted Pairwise Loss, and joint frame-level x-vector training) from Šimon Sedláček's bachelor's thesis at BUT Speech@FIT. |
| [Personal-vad-2.0](https://github.com/fclearner/Personal-vad-2.0) ![GitHub stars](https://img.shields.io/github/stars/fclearner/Personal-vad-2.0?style=social) | Python & PyTorch | PyTorch implementation of [Personal VAD 2.0: Optimizing Personal Voice Activity Detection for On-Device Speech Recognition](https://arxiv.org/abs/2204.03793) with a 4-layer causal Conformer encoder, 2-layer Speaker Pre-Net, FiLM cosine conditioning, and CAM++ speaker embeddings. |
| [Personal_VAD](https://github.com/chenCFD/Personal_VAD) ![GitHub stars](https://img.shields.io/github/stars/chenCFD/Personal_VAD?style=social) | Python & PyTorch | Lightweight PyTorch implementation of Personal VAD 1.0 (`ET` Embedding-Conditioned Training) using `Resemblyzer` d-vectors and synthetic concatenated LibriSpeech mixtures. |

### Target-Speaker VAD (TS-VAD) implementations

| Link | Language | Description |
| ---- | -------- | ----------- |
| [kaldi-asr (CHiME-6 Track 2 TS-VAD)](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5c_track2) ![GitHub stars](https://img.shields.io/github/stars/kaldi-asr/kaldi?style=social) | Bash & Python & C++ | Official Kaldi recipe (`local/ts-vad/`) introducing [Target-Speaker Voice Activity Detection (TS-VAD)](https://arxiv.org/abs/2005.07272) for multi-speaker diarization in dinner-party scenarios. |
| [NSD-MS2S](https://github.com/liyunlongaaa/NSD-MS2S) ![GitHub stars](https://img.shields.io/github/stars/liyunlongaaa/NSD-MS2S?style=social) | Python & PyTorch & Bash | Official PyTorch implementation of [Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding with Sequence-to-Sequence Architecture](https://arxiv.org/abs/2309.09180) (`NSD-MS2S` & `NSD-MS2S-SSMoE`), the CHiME-7/8 DASR winning diarization system. |
| [NSD-MA-MSE](https://github.com/Maokui-He/NSD-MA-MSE) ![GitHub stars](https://img.shields.io/github/stars/Maokui-He/NSD-MA-MSE?style=social) | Python & PyTorch & Bash | Official PyTorch implementation of [ANSD-MA-MSE: Adaptive Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding](https://doi.org/10.1109/TASLP.2023.3265199), extending TS-VAD with memory-aware multi-speaker embeddings. |
| [TS-VAD](https://github.com/dodohow1011/TS-VAD) ![GitHub stars](https://img.shields.io/github/stars/dodohow1011/TS-VAD?style=social) | Python & PyTorch | PyTorch implementation of Target-Speaker Voice Activity Detection (TS-VAD) for CHiME-6 multi-speaker diarization. |
| [AED-TSVAD](https://github.com/Clovermax/AED-TSVAD) ![GitHub stars](https://img.shields.io/github/stars/Clovermax/AED-TSVAD?style=social) | Python & PyTorch | Official PyTorch implementation and pretrained weights for Attention-Based Encoder-Decoder Target-Speaker Voice Activity Detection (`AED-TSVAD`) with WavLM representations across AMI, AISHELL-4, AliMeeting, CHiME-6, DIHARD, VoxConverse, and NOTSOFAR-1. |
| [AliMeeting](https://github.com/yufan-aslp/AliMeeting) ![GitHub stars](https://img.shields.io/github/stars/yufan-aslp/AliMeeting?style=social) | Python & PyTorch & Bash | Official ICASSP 2022 M2MeT Challenge baseline repository providing multi-channel meeting diarization recipes and data preparation for TS-VAD evaluation on the AliMeeting corpus. |
| [misp2022_baseline](https://github.com/mispchallenge/misp2022_baseline) ![GitHub stars](https://img.shields.io/github/stars/mispchallenge/misp2022_baseline?style=social) | Python & PyTorch & Bash | Official ICASSP 2023 MISP 2022 Challenge Track 1 baseline implementing **Audio-Visual Target-Speaker Voice Activity Detection (AV-TS-VAD)** combining lip ROI visual embeddings, CNN audio features, and speaker i-vectors. |
| [MISP-2023-Challenge-Baseline](https://github.com/mispchallenge/MISP-2023-Challenge-Baseline) ![GitHub stars](https://img.shields.io/github/stars/mispchallenge/MISP-2023-Challenge-Baseline?style=social) | Python & PyTorch & Bash | Official ICASSP 2024 MISP 2023 Challenge baseline for Audio-Visual Target Speaker Extraction (AVTSE) and multimodal speaker activity conditioning. |
| [MISP-2025-AVSD-Baseline](https://github.com/mispchallenge/MISP-2025-AVSD-Baseline) ![GitHub stars](https://img.shields.io/github/stars/mispchallenge/MISP-2025-AVSD-Baseline?style=social) | Python & PyTorch & Bash | Official Audio-Visual Speaker Diarization (AVSD) baseline for the MISP 2025 Challenge featuring a Conformer+BLSTM visual front-end and BLSTM AV-TS-VAD decoder. |
| [TS-VAD-plus](https://github.com/AnshKapadia/TS-VAD-plus) ![GitHub stars](https://img.shields.io/github/stars/AnshKapadia/TS-VAD-plus?style=social) | Python & PyTorch | Transformer-based TS-VAD pipeline (`TS-VAD+`) upgrading TS-VAD with WavLM frame features, ECAPA-TDNN speaker profiles, VBx initialization, and Memory-Aware Attention. |
| [TSVAD_pytorch](https://github.com/adnan-azmat/TSVAD_pytorch) ![GitHub stars](https://img.shields.io/github/stars/adnan-azmat/TSVAD_pytorch?style=social) | Python & PyTorch & C | End-to-end PyTorch TS-VAD training and evaluation pipeline integrating WeSpeaker/ECAPA-TDNN speaker embeddings, WavLM-Base+, VBx initial clustering, and TS-VAD refinement. |
| [chunk-level_TS-VAD](https://github.com/nttcslab-sp/chunk-level_TS-VAD) ![GitHub stars](https://img.shields.io/github/stars/nttcslab-sp/chunk-level_TS-VAD?style=social) | Python & PyTorch | Official NTT implementation of Target-Speaker Voice Activity Detection with Chunk-Level Speaker Queries (ICASSP 2026). |
| [ME-TSVAD](https://github.com/ME-TSVAD/ME-TSVAD) ![GitHub stars](https://img.shields.io/github/stars/ME-TSVAD/ME-TSVAD?style=social) | Python & PyTorch | PyTorch implementation of Multi-Enrollment / Memory-Enhanced Target-Speaker Voice Activity Detection (`ME-TSVAD`) with WavLM and ResNet speaker encoders. |

### Speech and diarization toolkits

| Link | Language | Description |
| ---- | -------- | ----------- |
| [FunASR](https://github.com/modelscope/FunASR) ![GitHub stars](https://img.shields.io/github/stars/modelscope/FunASR?style=social) | Python & PyTorch | Industrial-grade speech recognition toolkit from Alibaba DAMO Academy featuring FSMN-VAD, CAM++ speaker verification, and speaker-attributed ASR / diarization pipelines. |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/NeMo?style=social) | Python & PyTorch | Scalable generative AI and speech toolkit featuring MarbleNet VAD, TitaNet speaker embeddings, Sortformer, and Multi-scale Diarization Decoder (MSDD) — a target-speaker-conditioned neural diarizer. |
| [pyannote-audio](https://github.com/pyannote/pyannote-audio) ![GitHub stars](https://img.shields.io/github/stars/pyannote/pyannote-audio?style=social) | Python & PyTorch | Neural building blocks for speaker diarization: voice activity detection, overlapped speech detection, speaker change detection, and speaker embedding. |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) ![GitHub stars](https://img.shields.io/github/stars/speechbrain/speechbrain?style=social) | Python & PyTorch | Open-source all-in-one PyTorch speech toolkit providing pretrained ECAPA-TDNN speaker encoders, VAD models, and LibriParty recipes. |
| [ESPnet](https://github.com/espnet/espnet) ![GitHub stars](https://img.shields.io/github/stars/espnet/espnet?style=social) | Python & PyTorch | End-to-end speech processing toolkit including neural speaker diarization (EEND, EEND-EDA, EEND-SS), target-speaker ASR, and CHiME-6 / AliMeeting / MISP recipes. |
| [3D-Speaker](https://github.com/modelscope/3D-Speaker) ![GitHub stars](https://img.shields.io/github/stars/modelscope/3D-Speaker?style=social) | Python & PyTorch & C++ | Open-source toolkit for single- and multi-modal (audio-visual) speaker verification, recognition, and speaker diarization (CAM++, ERes2Net, overlap detection, and ONNX runtime). |
| [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) ![GitHub stars](https://img.shields.io/github/stars/BUTSpeechFIT/DiariZen?style=social) | Python & PyTorch | Speaker diarization toolkit from BUT Speech@FIT powered by AudioZen, pyannote, and WavLM. |
| [VBx](https://github.com/BUTSpeechFIT/VBx) ![GitHub stars](https://img.shields.io/github/stars/BUTSpeechFIT/VBx?style=social) | Python | Variational Bayes HMM over x-vectors (`VBx`), widely used as the first-pass clustering initializer to estimate target-speaker profiles for TS-VAD. |

### Speaker embedding extractors

| Link | Language | Description |
| ---- | -------- | ----------- |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) ![GitHub stars](https://img.shields.io/github/stars/wenet-e2e/wespeaker?style=social) | Python & PyTorch & C++ | Research and production toolkit for speaker verification and embedding extraction (ResNet, CAM++, ERes2Net, ECAPA-TDNN, ReDimNet), widely used to extract target speaker profiles for Personal VAD and TS-VAD. |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) ![GitHub stars](https://img.shields.io/github/stars/resemble-ai/Resemblyzer?style=social) | Python & PyTorch | Python package for extracting GE2E d-vector voice embeddings (`256-D`), used for target speaker conditioning in several open-source Personal VAD implementations. |
| [conformer-speaker-encoder](https://huggingface.co/tflite-hub/conformer-speaker-encoder) | TFLite | On-device Conformer speaker encoder model in TensorFlow Lite format for extracting target speaker embeddings. |
| [ReDimNet](https://github.com/IDRnD/redimnet) ![GitHub stars](https://img.shields.io/github/stars/IDRnD/redimnet?style=social) | Python & PyTorch | Official PyTorch implementation and pretrained weights for [Reshape Dimensions Network for Speaker Recognition](https://arxiv.org/abs/2407.18223). |

### Data simulation and augmentation

| Link | Language | Description |
| ---- | -------- | ----------- |
| [EEND_dataprep](https://github.com/BUTSpeechFIT/EEND_dataprep) ![GitHub stars](https://img.shields.io/github/stars/BUTSpeechFIT/EEND_dataprep?style=social) | Bash & Python | Realistic multi-speaker conversation simulation toolkit from [From Simulated Mixtures to Simulated Conversations as Training Data for End-to-End Neural Diarization](https://arxiv.org/abs/2204.00890), used for pretraining TS-VAD and EEND models. |
| [jsalt2020_simulate](https://github.com/jsalt2020-asrdiar/jsalt2020_simulate) ![GitHub stars](https://img.shields.io/github/stars/jsalt2020-asrdiar/jsalt2020_simulate?style=social) | Bash & Python | Multi-speaker conversation simulation recipes developed at JSALT 2020 for training Target-Speaker VAD and EEND models. |
| [SCATSVAD/DataSimu](https://github.com/SCATSVAD/DataSimu) ![GitHub stars](https://img.shields.io/github/stars/SCATSVAD/DataSimu?style=social) | Python | Multi-channel static and moving-speaker Room Impulse Response (`SonicSim`) data simulation toolkit for spatial/channel-aware TS-VAD ([demo page](https://scatsvad.github.io/DataSimu/)). |
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) ![GitHub stars](https://img.shields.io/github/stars/LCAV/pyroomacoustics?style=social) | Python & C++ | Room acoustics simulation and audio array processing package used to synthesize reverberant multi-speaker mixtures for Personal VAD and TS-VAD training. |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) ![GitHub stars](https://img.shields.io/github/stars/DavidDiazGuerra/gpuRIR?style=social) | Python & CUDA | GPU-accelerated Room Impulse Response (RIR) generator using the image source method for multi-style training augmentation. |

## Pretrained models

### Personal VAD 1.0 & 2.0 collection

The [**Personal VAD Hugging Face Collection (`wq2012/personal-vad`)**](https://huggingface.co/collections/wq2012/personal-vad) provides 13 open-source pretrained checkpoints covering all architectures from **Personal VAD 1.0** ([Odyssey 2020](https://arxiv.org/abs/1908.04284)) and **Personal VAD 2.0** ([Interspeech 2022](https://arxiv.org/abs/2204.03793)), trained and evaluated on the 8-language Multilingual LibriSpeech (MLS) benchmark. Each model repository includes Safetensors (`model.safetensors`), Keras weights (`model.weights.h5`), stateful FP32 TFLite (`model_fp32.tflite`), stateful 8-bit quantized TFLite (`model_quantized.tflite`), and the 256-D open-set speaker subspace (`speaker_subspace.npz`):

| Model | Architecture | Parameters (INT8 TFLite) | Description |
| ----- | ------------ | ------------------------ | ----------- |
| [wq2012/personal-vad-v1-sc](https://huggingface.co/wq2012/personal-vad-v1-sc) | PVAD 1.0 (`SC`) | 64.2K (`136.7 KB`) | **Score Combination Baseline**: 2-layer 64-unit unidirectional LSTM Standard VAD combined post-hoc with online 5th/95th percentile rescaled frame-level speaker verification cosine similarity. |
| [wq2012/personal-vad-v1-st-ce](https://huggingface.co/wq2012/personal-vad-v1-st-ce) | PVAD 1.0 (`ST` + `CE`) | 64.5K (`137.9 KB`) | **Score-Conditioned Training**: 2-layer 64-unit LSTM conditioned on 40-D log-Mel features and 1-D frame-level speaker verification cosine score (`[x_t, s_t]`), trained with Cross-Entropy loss. |
| [wq2012/personal-vad-v1-et-ce](https://huggingface.co/wq2012/personal-vad-v1-et-ce) | PVAD 1.0 (`ET` + `CE`) | 129.8K (`265.4 KB`) | **Embedding-Conditioned Training**: 2-layer 64-unit LSTM conditioned on 40-D log-Mel features and the 256-D target speaker d-vector (`[x_t, e_target]`), trained with Cross-Entropy loss. |
| [wq2012/personal-vad-v1-et-wpl](https://huggingface.co/wq2012/personal-vad-v1-et-wpl) | PVAD 1.0 (`ET` + `WPL`) | 129.8K (`265.4 KB`) | **Embedding-Conditioned Training + Weighted Pairwise Loss**: Canonical Odyssey 2020 Personal VAD 1.0 model trained with Weighted Pairwise Loss (`WPL`) to prioritize `<tss, ns>` and `<tss, ntss>` boundaries. |
| [wq2012/personal-vad-v1-set-ce](https://huggingface.co/wq2012/personal-vad-v1-set-ce) | PVAD 1.0 (`SET` + `CE`) | 130.1K (`266.0 KB`) | **Score & Embedding Conditioned Training**: 2-layer 64-unit LSTM conditioned on acoustic features, target d-vector, and frame-level cosine score (`[x_t, e_target, s_t]`). |
| [wq2012/personal-vad-v2-lstm-concat](https://huggingface.co/wq2012/personal-vad-v2-lstm-concat) | PVAD 2.0 (`B0`) | 1.78M (`2.01 MB`) | **3-Layer LayerNorm LSTM + Input Concat**: PVAD 2.0 Architecture `B0` baseline with 512-D stacked log-Mel frontend (30 ms frame step) and input d-vector concatenation. |
| [wq2012/personal-vad-v2-conformer-concat](https://huggingface.co/wq2012/personal-vad-v2-conformer-concat) | PVAD 2.0 (`E0`) | 700K (`881.0 KB`) | **Causal Conformer + Input Concat**: PVAD 2.0 Architecture `E0` using a 4-layer 64-D causal streaming Conformer backbone with input d-vector concatenation. |
| [wq2012/personal-vad-v2-conformer-film-dvector](https://huggingface.co/wq2012/personal-vad-v2-conformer-film-dvector) | PVAD 2.0 (`E1`) | 716K (`906.8 KB`) | **Causal Conformer + `FILM_DVECTOR`**: PVAD 2.0 Architecture `E1` using Feature-wise Linear Modulation (`FiLM`) conditioned on the target speaker d-vector `e_target`. |
| [wq2012/personal-vad-v2-conformer-film-cos](https://huggingface.co/wq2012/personal-vad-v2-conformer-film-cos) | PVAD 2.0 (`E2`) | 1.06M (`1.31 MB`) | **Causal Conformer + Speaker Pre-Net + `FILM_COS`**: PVAD 2.0 Architecture `E2` using a 2-layer Conformer Speaker Pre-Net and FiLM modulation on the cosine similarity score `s_t`. |
| [wq2012/personal-vad-v2-conformer-film-dvector-cos](https://huggingface.co/wq2012/personal-vad-v2-conformer-film-dvector-cos) | PVAD 2.0 (`E3` / `E4`) | 1.09M (`1.37 MB`) | **Causal Conformer + Speaker Pre-Net + `FILM_DVECTOR_COS`**: Highest-accuracy enrolled PVAD 2.0 architecture (`E3` FP32 / `E4` INT8) combining both `[e_target, s_t]` in the FiLM generator. |
| [wq2012/personal-vad-v2-conformer-unified](https://huggingface.co/wq2012/personal-vad-v2-conformer-unified) | PVAD 2.0 (`E5`) | 1.09M (`1.37 MB`) | **Unified Enrolled & Enrollment-Less Causal Conformer**: PVAD 2.0 Architecture `E5` trained with Algorithm 1 (`p_0 = 0.2` zero d-vector replacement) to operate as both a Personal VAD (`e_target != 0`) and a Standard VAD (`e_target = 0`). |
| [wq2012/personal-vad-v2-lstm-std-vad](https://huggingface.co/wq2012/personal-vad-v2-lstm-std-vad) | PVAD 2.0 (`B1`) | 1.71M (`1.91 MB`) | **3-Layer LayerNorm LSTM Standard VAD**: PVAD 2.0 Architecture `B1` 2-class speaker-agnostic Standard VAD baseline. |
| [wq2012/personal-vad-v2-conformer-std-vad](https://huggingface.co/wq2012/personal-vad-v2-conformer-std-vad) | PVAD 2.0 (`B2`) | 683K (`856.2 KB`) | **4-Layer Causal Conformer Standard VAD**: PVAD 2.0 Architecture `B2` 2-class speaker-agnostic Standard VAD baseline. |

### Target-Speaker VAD (TS-VAD) models

| Model | Framework | Description |
| ----- | --------- | ----------- |
| [Kaldi CHiME-6 Track 2 TS-VAD Models (`m12`)](https://kaldi-asr.org/models/m12) | Kaldi | Official pretrained CHiME-6 Track 2 TS-VAD diarization and ASR models released by STC-innovations and JHU ([mirror archive](https://github.com/yuri-hohlov/ts-vad-data)). |
| [AED-TSVAD Checkpoints](https://github.com/Clovermax/AED-TSVAD) | PyTorch | Pretrained Attention-Based Encoder-Decoder TS-VAD checkpoints evaluated across AMI, AISHELL-4, AliMeeting, CHiME-6, DIHARD, VoxConverse, and NOTSOFAR-1. |
| [NSD-MA-MSE AMI Checkpoint](https://github.com/Maokui-He/NSD-MA-MSE) | PyTorch | Pretrained Memory-Aware Multi-Speaker Embedding TS-VAD checkpoint trained on the AMI Meeting Corpus. |

## Demos

| Link | Type | Description |
| ---- | ---- | ----------- |
| [Personal VAD (1.0 & 2.0) Interactive Demo](https://huggingface.co/spaces/wq2012/personal_vad) | Hugging Face Space | Interactive browser-based demo for Personal Voice Activity Detection (1.0 & 2.0). Upload or record a short enrollment clip of a target speaker and a multi-speaker conversation audio file to visualize frame-level target-speaker voice activity (`tss` / `ns` / `ntss`) and isolate the target speaker's speech in real time. |
| [Google Speaker-ID Personal VAD Audio Demo](https://google.github.io/speaker-id/publications/PersonalVAD/) | Project Page | Official companion page for [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/1908.04284) with architecture diagrams, slide deck, and demonstration examples. |
| [SCA-TSVAD Multi-Channel Simulation Demo](https://scatsvad.github.io/DataSimu/) | Project Page | Interactive audio and spatial trajectory demonstrations of static and moving multi-speaker room simulations for multi-channel TS-VAD. |

## Datasets

### Conversational and meeting datasets

* [CHiME-6](https://www.chimechallenge.org/) ([OpenSLR 150](https://www.openslr.org/150/)): Multi-microphone dinner-party conversational speech corpus with heavy overlap and reverberation; the foundational benchmark where TS-VAD was introduced.
* [The AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) ([License CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)): 100 hours of multi-modal English meeting recordings, widely used for evaluating TS-VAD and memory-aware neural diarization.
* [The AliMeeting Corpus](https://www.openslr.org/119/) ([Baseline repo](https://github.com/yufan-aslp/AliMeeting)): 118.75 hours of real multi-channel Mandarin meetings from the ICASSP 2022 M2MeT Challenge.
* [AISHELL-4](https://www.openslr.org/111/): 120-hour Mandarin multi-channel meeting corpus with 4–8 speakers per session for speech enhancement, separation, recognition, and target-speaker diarization.
* [VoxConverse](https://github.com/joonson/voxconverse): Over 50 hours of multi-speaker "in-the-wild" YouTube conversations with high speech overlap.
* [DIHARD III](https://catalog.ldc.upenn.edu/LDC2022S12) ([Eval](https://catalog.ldc.upenn.edu/LDC2022S14)): Multi-domain hard diarization benchmark spanning clinical, restaurant, web video, court, and sociolinguistic recordings.
* [MISP 2022 / 2023 / 2025 Corpus](https://mispchallenge.github.io/): Large-scale Chinese home-TV multi-microphone and multi-camera conversational corpus used as the primary benchmark for Audio-Visual TS-VAD (AV-TS-VAD).
* [NOTSOFAR-1](https://www.chimechallenge.org/current/task2/index): CHiME-8 Distant Meeting Transcription and Diarization dataset (300+ meetings across 30 rooms).
* [LibriParty](https://www.openslr.org/101/): Simulated multi-speaker cocktail-party dataset created from LibriSpeech, MUSAN background noise, and room impulse responses for VAD and target-speaker speech detection.
* [LibriCSS](https://github.com/chenzhuo1011/libri_css): Continuous speech separation and meeting diarization benchmark derived from LibriSpeech re-recordings.
* [The ICSI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/icsi/) ([License CC BY 4.0](https://groups.inf.ed.ac.uk/ami/icsi/license.shtml)): 72 hours of natural research meetings at ICSI Berkeley.

### Speech and speaker corpora for simulation

* [LibriSpeech](https://www.openslr.org/12/): 1,000 hours of 16 kHz read English speech; the standard corpus for generating concatenated and partially overlapped multi-speaker utterances for Personal VAD 1.0/2.0 and TS-VAD training.
* [Multilingual LibriSpeech (MLS)](https://www.openslr.org/94/) ([Hugging Face](https://huggingface.co/datasets/facebook/multilingual_librispeech)): Large-scale multilingual read speech corpus across 8 European languages (`de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt`), used to train and evaluate the open-source [Personal VAD 1.0 & 2.0 models](https://huggingface.co/collections/wq2012/personal-vad).
* [VoxCeleb 1 & 2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/): Over 7,000 speakers extracted from YouTube interviews, widely used to train target speaker embedding extractors and simulated TS-VAD mixtures.
* [CN-Celeb 1 & 2](https://www.openslr.org/82/): Large-scale multi-genre Chinese speaker recognition corpus.
* [VoxBlink2](https://voxblink2.github.io/): 100K+ speaker recognition corpus for training large-scale speaker embedding extractors.

### Noise and room impulse response corpora

* [MUSAN](https://www.openslr.org/17/): A corpus of music, speech, and noise used for multi-style training (`MTR`) augmentation in Personal VAD and TS-VAD.
* [RIRS_NOISES](https://www.openslr.org/28/): Simulated and real room impulse responses and isotropic noises for reverberant data augmentation.
* [AudioSet](https://research.google.com/audioset/): Large-scale ontology of manually annotated audio events used as background noise sources.

## Publications

### Special topics

#### Personal VAD (Speaker-Conditioned VAD)

* [Adaptive Speaker Embedding Self-Augmentation for Personal Voice Activity Detection with Short Enrollment Speech](https://arxiv.org/abs/2601.12769), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2026
* [HyWA: Architecture-Preserving Personalized Voice Activity Detection for Full-Duplex Voice Assistants](https://arxiv.org/abs/2510.12947), *arXiv preprint arXiv:2510.12947*, 2025
* [FireRedChat: A Pluggable, Full-Duplex Voice Interaction System with Cascaded and Semi-Cascaded Implementations](https://arxiv.org/abs/2509.06502), *arXiv preprint arXiv:2509.06502*, 2025
* [Noise-Robust Target-Speaker Voice Activity Detection Through Self-Supervised Pretraining](https://arxiv.org/abs/2501.03184), *arXiv preprint arXiv:2501.03184*, 2025
* [Comparative Analysis of Personalized Voice Activity Detection Systems: Assessing Real-World Effectiveness](https://arxiv.org/abs/2406.09443), *Proc. Interspeech*, 2024
* [Investigation of Speaker Representation for Target-Speaker Speech Processing](https://arxiv.org/abs/2410.11243), *IEEE Spoken Language Technology Workshop (SLT)*, 2024
* [Personal Voice Activity Detection with Ultra-Short Reference Speech](https://doi.org/10.1109/APSIPAASC63619.2024.10849125), *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2024
* [Self-supervised Pretraining for Robust Personalized Voice Activity Detection in Adverse Conditions](https://arxiv.org/abs/2312.16613), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [AS-PVAD: A Frame-Wise Personalized Voice Activity Detection Network with Attentive Score Loss](https://doi.org/10.1109/ICASSP48485.2024.10446941), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [Efficient Personal Voice Activity Detection with Wake Word Reference Speech](https://doi.org/10.1109/ICASSP48485.2024.10447922), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [SVVAD: Personal Voice Activity Detection for Speaker Verification](https://arxiv.org/abs/2305.19581), *Proc. Interspeech*, 2023
* [Array Configuration-Agnostic Personal Voice Activity Detection Based on Spatial Coherence](https://arxiv.org/abs/2304.08887), *Proc. INTER-NOISE*, 2023
* [In-Ear-Voice: Towards Milli-Watt Audio Enhancement With Bone-Conduction Microphones for In-Ear Sensing Platforms](https://arxiv.org/abs/2309.02393), *ACM/IEEE Conference on Internet of Things Design and Implementation (IoTDI)*, 2023
* [Personal VAD 2.0: Optimizing Personal Voice Activity Detection for On-Device Speech Recognition](https://arxiv.org/abs/2204.03793) ([Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=cB62SPcAAAAJ&citation_for_view=cB62SPcAAAAJ:sSrBHYA8nusC)), *Proc. Interspeech*, 2022
* [Enrollment-less training for personalized voice activity detection](https://arxiv.org/abs/2106.12132), *Proc. Interspeech*, 2021
* [Personalizing Speech Start Point and End Point Detection in ASR Systems from Speaker Embeddings](https://doi.org/10.1109/SLT48900.2021.9383603), *IEEE Spoken Language Technology Workshop (SLT)*, 2021
* [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/1908.04284) ([Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=cB62SPcAAAAJ&citation_for_view=cB62SPcAAAAJ:KxtntwgDAa4C)), *Proc. Odyssey The Speaker and Language Recognition Workshop*, 2020
* [End-to-End Speaker-Dependent Voice Activity Detection](https://arxiv.org/abs/2009.09906), *National Conference on Man-Machine Speech Communication (NCMMSC)*, 2019

#### Joint Personal/Target-Speaker VAD, extraction, and ASR

* [Grounding Spoken LLMs in Multi-Speaker Audio via Diarization Conditioning](https://arxiv.org/abs/2606.18134), *Proc. Interspeech*, 2026
* [SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper](https://arxiv.org/abs/2601.19194), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2026
* [End-to-End Target Speaker Speech Recognition with Voice Activity Detection Fusion](https://doi.org/10.1016/j.dsp.2026.105966), *Digital Signal Processing*, 2026
* [Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection](https://arxiv.org/abs/2501.03612), *Computer Speech & Language*, 2025
* [DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition](https://arxiv.org/abs/2501.00114), *Computer Speech & Language*, 2025
* [Target Speaker ASR with Whisper](https://arxiv.org/abs/2409.09543), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025
* [Robust Target Speaker Diarization and Separation via Augmented Speaker Embedding Sampling](https://arxiv.org/abs/2508.06393), *Proc. Interspeech*, 2025
* [TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings](https://arxiv.org/abs/2303.03849), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2024
* [USED: Universal Speaker Extraction and Diarization](https://arxiv.org/abs/2309.10674), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2024
* [Joint Inference of Speaker Diarization and ASR with Multi-Stage Information Sharing](https://doi.org/10.1109/ICASSP48485.2024.10446724), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [Continuous Target Speech Extraction: Enhancing Personalized Diarization and Extraction on Complex Recordings](https://arxiv.org/abs/2401.15993), *arXiv preprint arXiv:2401.15993*, 2024
* [DCF-DS: Deep Cascade Fusion of Diarization and Separation for Speech Recognition under Realistic Single-Channel Conditions](https://arxiv.org/abs/2411.06667), *arXiv preprint arXiv:2411.06667*, 2024
* [Breaking the trade-off in personalized speech enhancement with cross-task knowledge distillation](https://arxiv.org/abs/2211.02944), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023
* [Simultaneous Speech Extraction for Multiple Target Speakers under the Meeting Scenarios](https://arxiv.org/abs/2206.08525), *National Conference on Man-Machine Speech Communication (NCMMSC)*, 2023
* [Transcribe-to-Diarize: Neural Speaker Diarization for Unlimited Number of Speakers using End-to-End Speaker-Attributed ASR](https://arxiv.org/abs/2110.03151), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022
* [Sparsely Overlapped Speech Training in the Time Domain: Joint Learning of Target Speech Separation and Personal VAD Benefits](https://arxiv.org/abs/2106.14371), *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2021
* [WASE: Learning When to Attend for Speaker Extraction in Cocktail Party Environments](https://arxiv.org/abs/2106.07016), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2021
* [Multi-user VoiceFilter-Lite via Attentive Speaker Embedding](https://arxiv.org/abs/2107.01201), *IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)*, 2021

#### Online and streaming Target-Speaker VAD

* [EEND-SAA: Enrollment-Less Main Speaker Voice Activity Detection Using Self-Attention Attractors](https://arxiv.org/abs/2509.11957), *arXiv preprint arXiv:2509.11957*, 2025
* [Sequence-to-Sequence Neural Diarization with Automatic Speaker Detection and Representation](https://arxiv.org/abs/2411.13849), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2025
* [Online Neural Speaker Diarization With Target Speaker Tracking](https://arxiv.org/abs/2310.08696), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2024
* [Online Neural Speaker Diarization with Spectral Clustering for Meeting Scenarios](https://doi.org/10.1109/ISCSLP63861.2024.10800669), *International Symposium on Chinese Spoken Language Processing (ISCSLP)*, 2024
* [Online Target Speaker Voice Activity Detection for Speaker Diarization](https://arxiv.org/abs/2207.05920), *Proc. Interspeech*, 2022
* [Online Speaker Diarization with Core Samples Selection](https://doi.org/10.21437/Interspeech.2022-10363), *Proc. Interspeech*, 2022

#### Multi-channel and spatial Target-Speaker VAD

* [Selective Channel Attention based Target Speaker Voice Activity Detection for Speaker Diarization under AD-HOC Microphone Array Settings](https://doi.org/10.21437/Interspeech.2025-1749), *Proc. Interspeech*, 2025
* [Multi-Channel Sequence-to-Sequence Neural Diarization: Experimental Results for The MISP 2025 Challenge](https://doi.org/10.21437/Interspeech.2025-1262), *Proc. Interspeech*, 2025
* [Incorporating Spatial Cues in Modular Speaker Diarization for Multi-Channel Multi-Party Meetings](https://arxiv.org/abs/2409.16803), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025
* [A Spatial Long-Term Iterative Mask Estimation Approach for Multi-Channel Speaker Diarization and Speech Recognition](https://doi.org/10.1109/ICASSP48485.2024.10446168), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [Semi-Supervised Multi-Channel Speaker Diarization With Cross-Channel Attention](https://doi.org/10.1109/ASRU57964.2023.10389615), *IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)*, 2023
* [Cross-Channel Attention-Based Target Speaker Voice Activity Detection: Experimental Results for M2MeT Challenge](https://arxiv.org/abs/2202.02687), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022
* [The CUHK-TENCENT Speaker Diarization System for the ICASSP 2022 Multi-Channel Multi-Party Meeting Transcription Challenge](https://arxiv.org/abs/2202.01986), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022
* [Polynomial Eigenvalue Decomposition-Based Target Speaker Voice Activity Detection in the Presence of Competing Talkers](https://doi.org/10.1109/IWAENC53105.2022.9914796), *IEEE International Workshop on Acoustic Signal Enhancement (IWAENC)*, 2022

#### Audio-Visual Target-Speaker VAD

* [Multi-Input Multi-Output Target-Speaker Voice Activity Detection For Unified, Flexible, and Robust Audio-Visual Speaker Diarization](https://arxiv.org/abs/2401.08052), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2025
* [The Multimodal Information Based Speech Processing (MISP) 2025 Challenge: Audio-Visual Diarization and Recognition](https://arxiv.org/abs/2505.13971), *Proc. Interspeech*, 2025
* [Cross-attention and Self-attention for Audio-visual Speaker Diarization in MISP-Meeting Challenge](https://arxiv.org/abs/2506.02621), *Proc. Interspeech*, 2025
* [Quality-Aware End-to-End Audio-Visual Neural Speaker Diarization](https://arxiv.org/abs/2410.22350), *arXiv preprint arXiv:2410.22350*, 2024
* [The Multimodal Information Based Speech Processing (MISP) 2022 Challenge: Audio-Visual Diarization And Recognition](https://arxiv.org/abs/2303.06326), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023
* [The WHU-Alibaba Audio-Visual Speaker Diarization System for the MISP 2022 Challenge](https://doi.org/10.1109/ICASSP49357.2023.10095802), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023
* [CASA-Net: Cross-attention and Self-attention for End-to-End Audio-visual Speaker Diarization](https://doi.org/10.1109/APSIPAASC58517.2023.10317320), *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2023
* [Joint Training or Not: An Exploration of Pre-trained Speech Models in Audio-Visual Speaker Diarization](https://arxiv.org/abs/2312.04131), *arXiv preprint arXiv:2312.04131*, 2023
* [End-to-End Audio-Visual Neural Speaker Diarization](https://www.isca-archive.org/interspeech_2022/he22c_interspeech.html), *Proc. Interspeech*, 2022

### Other

#### 2026
* [Learning Speaker-Activity-Conditioned WavLM Speaker Embeddings with Real Conversational Data](https://arxiv.org/abs/2610.25183), *IEEE Spoken Language Technology Workshop (SLT)*, 2026
* [Target-Speaker Voice Activity Detection with Chunk-Level Speaker Queries](https://github.com/nttcslab-sp/chunk-level_TS-VAD), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2026
* [Attention-Based Encoder-Decoder Target-Speaker Voice Activity Detection for Robust Speaker Diarization](https://github.com/Clovermax/AED-TSVAD), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2026

#### 2025
* [Flow-TSVAD: Target-Speaker Voice Activity Detection via Latent Flow Matching for Speaker Diarization](https://arxiv.org/abs/2409.04859), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025
* [Guided Speaker Embedding](https://arxiv.org/abs/2410.12182), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025
* [Mitigating Non-Target Speaker Bias in Guided Speaker Embedding](https://arxiv.org/abs/2506.12500), *Proc. Interspeech*, 2025
* [TS-VAD+: Modularized Target-Speaker Voice Activity Detection for Robust Speaker Diarization](https://doi.org/10.1109/APSIPAASC65261.2025.11249081), *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2025
* [Exploring Speaker Diarization with Mixture of Experts](https://arxiv.org/abs/2506.14750), *arXiv preprint arXiv:2506.14750*, 2025

#### 2024
* [Profile-Error-Tolerant Target-Speaker Voice Activity Detection](https://arxiv.org/abs/2309.12521), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding with Sequence-to-Sequence Architecture](https://arxiv.org/abs/2309.09180), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [Prompt-Driven Target Speech Diarization](https://arxiv.org/abs/2310.14823), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [Target Speech Diarization with Multimodal Prompts](https://arxiv.org/abs/2406.07198), *Proc. Interspeech*, 2024
* [Attention-Based Encoder-Decoder End-to-End Neural Diarization With Embedding Enhancer](https://arxiv.org/abs/2309.06672), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2024

#### 2023
* [Target-Speaker Voice Activity Detection via Sequence-to-Sequence Prediction](https://arxiv.org/abs/2210.16127), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023
* [Target Speaker Voice Activity Detection with Transformers and Its Integration with End-to-End Neural Diarization](https://arxiv.org/abs/2208.13085), *IEEE Spoken Language Technology Workshop (SLT)*, 2023
* [ANSD-MA-MSE: Adaptive Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding](https://doi.org/10.1109/TASLP.2023.3265199), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2023
* [Attention-based Encoder-Decoder Network for End-to-End Neural Speaker Diarization with Target Speaker Attractor](https://arxiv.org/abs/2305.10704), *Proc. Interspeech*, 2023
* [Unsupervised Adaptation with Quality-Aware Masking to Improve Target-Speaker Voice Activity Detection for Speaker Diarization](https://doi.org/10.21437/Interspeech.2023-1502), *Proc. Interspeech*, 2023
* [Multi-Target Extractor and Detector for Unknown-Number Speaker Diarization](https://arxiv.org/abs/2203.16007), *IEEE Signal Processing Letters*, 2023
* [Weakly Supervised Target-Speaker Voice Activity Detection](https://doi.org/10.1109/BIGCOM61073.2023.00041), *IEEE International Conference on Big Data Computing and Communications (BIGCOM)*, 2023
* [The USTC-NERCSLIP Systems for the CHiME-7 DASR Challenge](https://arxiv.org/abs/2308.14638), *Proc. CHiME-7 Workshop*, 2023
* [PP-MeT: A Real-World Personalized Prompt Based Meeting Transcription System](https://arxiv.org/abs/2309.16247), *IEEE Automatic Speech Recognition and Understanding Workshop (ASRU M2MeT 2.0)*, 2023

#### 2022
* [Incorporating End-to-End Framework Into Target-Speaker Voice Activity Detection](https://doi.org/10.1109/ICASSP43922.2022.9747772), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022
* [Speaker Embedding-aware Neural Diarization for Flexible Number of Speakers with Textual Information](https://arxiv.org/abs/2111.13694), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022
* [Speaker Embedding-aware Neural Diarization: an Efficient Framework for Overlapping Speech Diarization in Meeting Scenarios](https://arxiv.org/abs/2203.09767), *Proc. Interspeech*, 2022
* [Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis](https://arxiv.org/abs/2211.10243), *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2022
* [The USTC-Ximalaya System for the ICASSP 2022 Multi-Channel Multi-Party Meeting Transcription (M2MeT) Challenge](https://arxiv.org/abs/2202.04855), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022
* [Closing the Gap between Single-User and Multi-User VoiceFilter-Lite](https://arxiv.org/abs/2202.12169), *Proc. Odyssey The Speaker and Language Recognition Workshop*, 2022

#### 2021
* [Target-Speaker Voice Activity Detection with Improved i-Vector Estimation for Unknown Number of Speaker](https://arxiv.org/abs/2108.03342), *Proc. Interspeech*, 2021
* [Scenario-Dependent Speaker Diarization for DIHARD-III Challenge](https://doi.org/10.21437/Interspeech.2021-516), *Proc. Interspeech*, 2021
* [Integration of Speech Separation, Diarization, and Recognition for Multi-Speaker Meetings: System Description, Comparison, and Analysis](https://arxiv.org/abs/2011.02014), *IEEE Spoken Language Technology Workshop (SLT)*, 2021
* [DOVER-Lap: A Method for Combining Overlap-aware Diarization Outputs](https://arxiv.org/abs/2011.01997), *IEEE Spoken Language Technology Workshop (SLT)*, 2021

#### 2020
* [Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario](https://arxiv.org/abs/2005.07272), *Proc. Interspeech*, 2020
* [The STC System for the CHiME-6 Challenge](https://doi.org/10.21437/CHiME.2020-3), *Proc. 6th International Workshop on Speech Processing in Everyday Environments (CHiME)*, 2020

## Other learning materials

### Project pages and slides

* [Personal VAD: Speaker-Conditioned Voice Activity Detection — Official Project Page](https://google.github.io/speaker-id/publications/PersonalVAD/) and [Odyssey 2020 Presentation Slides (PDF)](https://google.github.io/speaker-id/publications/PersonalVAD/resources/odyssey2020_personalvad_slides.pdf) by Shaojin Ding, Quan Wang, Shuo-yiin Chang, Li Wan, and Ignacio Lopez Moreno
* [Awesome Speaker Diarization](https://github.com/wq2012/awesome-diarization) by Quan Wang

### Theses

* [Personal Voice Activity Detection (PDF)](https://github.com/pirxus/personalVAD/blob/master/xsedla1h_thesis.pdf), Bachelor's thesis by Šimon Sedláček, Brno University of Technology (BUT Speech@FIT), 2021
* [TS-VAD+: Transformer-Based Target-Speaker Voice Activity Detection (PDF)](https://github.com/AnshKapadia/TS-VAD-plus/blob/main/Docs/Master's_Thesis.pdf), Master's thesis by Ansh Kapadia, Nanyang Technological University (NTU), 2023

### Books

* [Voice Identity Techniques: From core algorithms to engineering practice (Chinese)](https://github.com/wq2012/VoiceIdentityBook) by Quan Wang, 2020

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wq2012/awesome-personal-vad&type=Date)](https://star-history.com/#wq2012/awesome-personal-vad&Date)
