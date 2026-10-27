# Awesome Personal VAD [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

## Table of contents

* [Overview](#overview)
* [Software](#software)
  * [Personal VAD implementations](#personal-vad-implementations)
  * [Target-Speaker VAD (TS-VAD) implementations](#target-speaker-vad-ts-vad-implementations)
* [Pretrained models](#pretrained-models)
  * [Personal VAD 1.0 & 2.0 collection](#personal-vad-10--20-collection)
  * [Target-Speaker VAD (TS-VAD) models](#target-speaker-vad-ts-vad-models)
* [Demos](#demos)
* [Publications](#publications)
  * [Personal VAD (Speaker-Conditioned VAD)](#personal-vad-speaker-conditioned-vad)
  * [Target-Speaker VAD (TS-VAD)](#target-speaker-vad-ts-vad)
* [Other learning materials](#other-learning-materials)
  * [Tutorials](#tutorials)
  * [Project pages and slides](#project-pages-and-slides)
  * [Theses](#theses)
  * [Books](#books)
* [Star History](#star-history)

## Overview

This is a curated list of awesome **Personal Voice Activity Detection (Personal VAD)** — also known as **Speaker-Conditioned Voice Activity Detection** or **Target-Speaker Voice Activity Detection (TS-VAD)** — papers, libraries, pretrained models, demos, and other resources.

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
| [TS-VAD](https://github.com/dodohow1011/TS-VAD) ![GitHub stars](https://img.shields.io/github/stars/dodohow1011/TS-VAD?style=social) | Python & PyTorch | PyTorch implementation of Target-Speaker Voice Activity Detection (TS-VAD) for CHiME-6 multi-speaker diarization. |
| [AED-TSVAD](https://github.com/Clovermax/AED-TSVAD) ![GitHub stars](https://img.shields.io/github/stars/Clovermax/AED-TSVAD?style=social) | Python & PyTorch | Official PyTorch implementation and pretrained weights for Attention-Based Encoder-Decoder Target-Speaker Voice Activity Detection (`AED-TSVAD`) with WavLM representations across AMI, AISHELL-4, AliMeeting, CHiME-6, DIHARD, VoxConverse, and NOTSOFAR-1. |
| [misp2022_baseline](https://github.com/mispchallenge/misp2022_baseline) ![GitHub stars](https://img.shields.io/github/stars/mispchallenge/misp2022_baseline?style=social) | Python & PyTorch & Bash | Official ICASSP 2023 MISP 2022 Challenge Track 1 baseline implementing **Audio-Visual Target-Speaker Voice Activity Detection (AV-TS-VAD)** combining lip ROI visual embeddings, CNN audio features, and speaker i-vectors. |
| [MISP-2025-AVSD-Baseline](https://github.com/mispchallenge/MISP-2025-AVSD-Baseline) ![GitHub stars](https://img.shields.io/github/stars/mispchallenge/MISP-2025-AVSD-Baseline?style=social) | Python & PyTorch & Bash | Official Audio-Visual Speaker Diarization (AVSD) baseline for the MISP 2025 Challenge featuring a Conformer+BLSTM visual front-end and BLSTM AV-TS-VAD decoder. |
| [TS-VAD-plus](https://github.com/AnshKapadia/TS-VAD-plus) ![GitHub stars](https://img.shields.io/github/stars/AnshKapadia/TS-VAD-plus?style=social) | Python & PyTorch | Transformer-based TS-VAD pipeline (`TS-VAD+`) upgrading TS-VAD with WavLM frame features, ECAPA-TDNN speaker profiles, VBx initialization, and Memory-Aware Attention. |
| [TSVAD_pytorch](https://github.com/adnan-azmat/TSVAD_pytorch) ![GitHub stars](https://img.shields.io/github/stars/adnan-azmat/TSVAD_pytorch?style=social) | Python & PyTorch & C | End-to-end PyTorch TS-VAD training and evaluation pipeline integrating WeSpeaker/ECAPA-TDNN speaker embeddings, WavLM-Base+, VBx initial clustering, and TS-VAD refinement. |
| [chunk-level_TS-VAD](https://github.com/nttcslab-sp/chunk-level_TS-VAD) ![GitHub stars](https://img.shields.io/github/stars/nttcslab-sp/chunk-level_TS-VAD?style=social) | Python & PyTorch | Official NTT implementation of Target-Speaker Voice Activity Detection with Chunk-Level Speaker Queries (ICASSP 2026). |
| [ME-TSVAD](https://github.com/ME-TSVAD/ME-TSVAD) ![GitHub stars](https://img.shields.io/github/stars/ME-TSVAD/ME-TSVAD?style=social) | Python & PyTorch | PyTorch implementation of Multi-Enrollment / Memory-Enhanced Target-Speaker Voice Activity Detection (`ME-TSVAD`) with WavLM and ResNet speaker encoders. |
| [SCATSVAD/DataSimu](https://github.com/SCATSVAD/DataSimu) ![GitHub stars](https://img.shields.io/github/stars/SCATSVAD/DataSimu?style=social) | Python | Multi-channel static and moving-speaker Room Impulse Response (`SonicSim`) data simulation toolkit for Selective Channel Attention Target-Speaker VAD ([demo page](https://scatsvad.github.io/DataSimu/)). |

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

## Demos

| Link | Type | Description |
| ---- | ---- | ----------- |
| [Personal VAD (1.0 & 2.0) Interactive Demo](https://huggingface.co/spaces/wq2012/personal_vad) | Hugging Face Space | Interactive browser-based demo for Personal Voice Activity Detection (1.0 & 2.0). Upload or record a short enrollment clip of a target speaker and a multi-speaker conversation audio file to visualize frame-level target-speaker voice activity (`tss` / `ns` / `ntss`) and isolate the target speaker's speech in real time. |
| [Google Speaker-ID Personal VAD Audio Demo](https://google.github.io/speaker-id/publications/PersonalVAD/) | Project Page | Official companion page for [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/1908.04284) with architecture diagrams, slide deck, and demonstration examples. |
| [SCA-TSVAD Multi-Channel Simulation Demo](https://scatsvad.github.io/DataSimu/) | Project Page | Interactive audio and spatial trajectory demonstrations of static and moving multi-speaker room simulations for multi-channel TS-VAD. |

## Publications

### Personal VAD (Speaker-Conditioned VAD)

* [Adaptive Speaker Embedding Self-Augmentation for Personal Voice Activity Detection with Short Enrollment Speech](https://arxiv.org/abs/2601.12769), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2026
* [HyWA: Architecture-Preserving Personalized Voice Activity Detection for Full-Duplex Voice Assistants](https://arxiv.org/abs/2510.12947), *arXiv preprint arXiv:2510.12947*, 2025
* [Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection](https://arxiv.org/abs/2501.03612), *Computer Speech & Language*, 2025
* [Comparative Analysis of Personalized Voice Activity Detection Systems: Assessing Real-World Effectiveness](https://arxiv.org/abs/2406.09443), *Proc. Interspeech*, 2024
* [Investigation of Speaker Representation for Target-Speaker Speech Processing](https://arxiv.org/abs/2410.11243), *IEEE Spoken Language Technology Workshop (SLT)*, 2024
* [Personal Voice Activity Detection with Ultra-Short Reference Speech](https://doi.org/10.1109/APSIPAASC63619.2024.10849125), *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2024
* [Self-supervised Pretraining for Robust Personalized Voice Activity Detection in Adverse Conditions](https://arxiv.org/abs/2312.16613), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [AS-PVAD: A Frame-Wise Personalized Voice Activity Detection Network with Attentive Score Loss](https://doi.org/10.1109/ICASSP48485.2024.10446941), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [Efficient Personal Voice Activity Detection with Wake Word Reference Speech](https://doi.org/10.1109/ICASSP48485.2024.10447922), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [SVVAD: Personal Voice Activity Detection for Speaker Verification](https://arxiv.org/abs/2305.19581), *Proc. Interspeech*, 2023
* [Array Configuration-Agnostic Personal Voice Activity Detection Based on Spatial Coherence](https://arxiv.org/abs/2304.08887), *Proc. INTER-NOISE*, 2023
* [Personal VAD 2.0: Optimizing Personal Voice Activity Detection for On-Device Speech Recognition](https://arxiv.org/abs/2204.03793) ([Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=cB62SPcAAAAJ&citation_for_view=cB62SPcAAAAJ:sSrBHYA8nusC)), *Proc. Interspeech*, 2022
* [Enrollment-less training for personalized voice activity detection](https://arxiv.org/abs/2106.12132), *Proc. Interspeech*, 2021
* [Personalizing Speech Start Point and End Point Detection in ASR Systems from Speaker Embeddings](https://doi.org/10.1109/SLT48900.2021.9383603), *IEEE Spoken Language Technology Workshop (SLT)*, 2021
* [Sparsely Overlapped Speech Training in the Time Domain: Joint Learning of Target Speech Separation and Personal VAD Benefits](https://arxiv.org/abs/2106.14371), *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2021
* [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/1908.04284) ([Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=cB62SPcAAAAJ&citation_for_view=cB62SPcAAAAJ:KxtntwgDAa4C)), *Proc. Odyssey The Speaker and Language Recognition Workshop*, 2020
* [End-to-End Speaker-Dependent Voice Activity Detection](https://arxiv.org/abs/2009.09906), *National Conference on Man-Machine Speech Communication (NCMMSC)*, 2019

### Target-Speaker VAD (TS-VAD)

* [Target-Speaker Voice Activity Detection with Chunk-Level Speaker Queries](https://github.com/nttcslab-sp/chunk-level_TS-VAD), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2026
* [End-to-End Target Speaker Speech Recognition with Voice Activity Detection Fusion](https://doi.org/10.1016/j.dsp.2026.105966), *Digital Signal Processing*, 2026
* [Flow-TSVAD: Target-Speaker Voice Activity Detection via Latent Flow Matching for Speaker Diarization](https://arxiv.org/abs/2409.04859), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025
* [Multi-Input Multi-Output Target-Speaker Voice Activity Detection For Unified, Flexible, and Robust Audio-Visual Speaker Diarization](https://arxiv.org/abs/2401.08052), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2025
* [Selective Channel Attention based Target Speaker Voice Activity Detection for Speaker Diarization under AD-HOC Microphone Array Settings](https://doi.org/10.21437/Interspeech.2025-1749), *Proc. Interspeech*, 2025
* [TS-VAD+: Modularized Target-Speaker Voice Activity Detection for Robust Speaker Diarization](https://doi.org/10.1109/APSIPAASC65261.2025.11249081), *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2025
* [Noise-Robust Target-Speaker Voice Activity Detection Through Self-Supervised Pretraining](https://arxiv.org/abs/2501.03184), *arXiv preprint arXiv:2501.03184*, 2025
* [EEND-SAA: Enrollment-Less Main Speaker Voice Activity Detection Using Self-Attention Attractors](https://arxiv.org/abs/2509.11957), *arXiv preprint arXiv:2509.11957*, 2025
* [Profile-Error-Tolerant Target-Speaker Voice Activity Detection](https://arxiv.org/abs/2309.12521), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2024
* [TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings](https://arxiv.org/abs/2303.03849), *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*, 2024
* [Target-Speaker Voice Activity Detection via Sequence-to-Sequence Prediction](https://arxiv.org/abs/2210.16127), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023
* [Target Speaker Voice Activity Detection with Transformers and Its Integration with End-to-End Neural Diarization](https://arxiv.org/abs/2208.13085), *2022 IEEE Spoken Language Technology Workshop (SLT)*, 2023
* [Unsupervised Adaptation with Quality-Aware Masking to Improve Target-Speaker Voice Activity Detection for Speaker Diarization](https://doi.org/10.21437/Interspeech.2023-1502), *Proc. Interspeech*, 2023
* [Weakly Supervised Target-Speaker Voice Activity Detection](https://doi.org/10.1109/BIGCOM61073.2023.00041), *IEEE International Conference on Big Data Computing and Communications (BIGCOM)*, 2023
* [Online Target Speaker Voice Activity Detection for Speaker Diarization](https://arxiv.org/abs/2207.05920), *Proc. Interspeech*, 2022
* [Incorporating End-to-End Framework Into Target-Speaker Voice Activity Detection](https://doi.org/10.1109/ICASSP43922.2022.9747772), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022
* [Cross-Channel Attention-Based Target Speaker Voice Activity Detection: Experimental Results for M2MeT Challenge](https://arxiv.org/abs/2202.02687), *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022
* [Polynomial Eigenvalue Decomposition-Based Target Speaker Voice Activity Detection in the Presence of Competing Talkers](https://doi.org/10.1109/IWAENC53105.2022.9914796), *IEEE International Workshop on Acoustic Signal Enhancement (IWAENC)*, 2022
* [Target-Speaker Voice Activity Detection with Improved i-Vector Estimation for Unknown Number of Speaker](https://arxiv.org/abs/2108.03342), *Proc. Interspeech*, 2021
* [Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario](https://arxiv.org/abs/2005.07272), *Proc. Interspeech*, 2020

## Other learning materials

### Tutorials

* [Towards inclusive, personalised and empathetic spoken language systems (PDF)](https://drive.google.com/file/d/15Ct9ARY1teT3za_QBd12CSK64PFn-mAq/view?usp=drive_link), *Interspeech 2026 Tutorial*, 2026

### Project pages and slides

* [Personal VAD: Speaker-Conditioned Voice Activity Detection — Official Project Page](https://google.github.io/speaker-id/publications/PersonalVAD/) and [Odyssey 2020 Presentation Slides (PDF)](https://google.github.io/speaker-id/publications/PersonalVAD/resources/odyssey2020_personalvad_slides.pdf) by Shaojin Ding, Quan Wang, Shuo-yiin Chang, Li Wan, and Ignacio Lopez Moreno

### Theses

* [Personal Voice Activity Detection (PDF)](https://github.com/pirxus/personalVAD/blob/master/xsedla1h_thesis.pdf), Bachelor's thesis by Šimon Sedláček, Brno University of Technology (BUT Speech@FIT), 2021
* [TS-VAD+: Transformer-Based Target-Speaker Voice Activity Detection (PDF)](https://github.com/AnshKapadia/TS-VAD-plus/blob/main/Docs/Master's_Thesis.pdf), Master's thesis by Ansh Kapadia, Nanyang Technological University (NTU), 2023

### Books

* [Voice Identity Techniques: From core algorithms to engineering practice (Chinese)](https://github.com/wq2012/VoiceIdentityBook) by Quan Wang, 2020 (Section 6.4.2 covers Personal VAD)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wq2012/awesome-personal-vad&type=Date)](https://star-history.com/#wq2012/awesome-personal-vad&Date)
