# SigLIP2 드레스 Pattern·Length 파인튜닝

GLAMI-1M 드레스 데이터의 패턴(pattern)·기장(length) 속성을 기준으로 SigLIP2를 LoRA
파인튜닝한 프로젝트입니다. 아래 순서(데이터셋 → 환경 → 학습)대로 재현할 수 있습니다.

## 이 저장소에 포함된 것 / 안 된 것

이 저장소에는 **코드만** 포함되어 있습니다. 데이터셋과 모델 가중치는 포함하지 않았으며, 코드 실행으로 재생성하거나(공개 체크포인트는 자동 다운로드) 별도로 구해야 합니다.

**미포함 (재현 시 필요)**
- GLAMI-1M 원본 — `build_glami_dataset/`을 돌리려면 `~/Downloads/GLAMI-1M-dataset/`에 직접 받아둬야 합니다 (본 프로젝트 데이터의 출처이므로 별도 배포하지 않았습니다). 이게 있어야 `datasets/GLAMI-1M-dresses-pattern-length/`가 만들어지고, 그래야 학습(`siglip2_finetune/`)을 돌릴 수 있습니다.
- `datasets/pexels-dresses/` (크롭 안 된 원본 Pexels 드레스 사진 100장 + `manifest.json`) — `build_pexels_dataset/`로 크롭 데이터셋(`datasets/pexels-dresses-crops/`)을 만들 때 필요합니다.
- `weights/siglip2/` — 공개 HuggingFace 체크포인트라 코드 최초 실행 시 자동 다운로드됩니다.
- `build_pexels_dataset/weights/`(YOLO-World `yolov8s-worldv2.pt` + 내부 CLIP 텍스트 인코더) — `crop_dress.py` 첫 실행 시 `ultralytics`가 둘 다 자동으로 받아옵니다(직접 검증함).
- 각 폴더의 `results/` — 코드 재실행으로 다시 만들어지는 산출물입니다.

## 저장소 구조

| 폴더 | 내용 |
|---|---|
| `build_glami_dataset/` | GLAMI-1M → 패턴/기장 라벨 데이터셋 전처리 |
| `build_pexels_dataset/` | Pexels 드레스 사진 → 드레스 크롭 + pattern/length 캡션 생성 전처리 |
| `siglip2_finetune/` | LoRA 파인튜닝 스크립트 + 설정(config) |
| `datasets/` | 전처리 결과물이 생성되는 위치 (이미지 + manifest.json, 저장소에는 미포함) |
| `weights/` | 베이스 모델 / 파인튜닝 가중치가 저장되는 위치 (저장소에는 미포함) |

---

## 1. 데이터셋을 어떻게 만들었는지

### GLAMI-1M-dresses-pattern-length (학습 + 정답 검증셋)

`build_glami_dataset/`에서 한 번에 실행합니다 (이미 만들어진 결과물이 있으면 해당 단계는 자동으로 건너뜁니다).

```bash
cd build_glami_dataset
python3 run_pipeline.py          # 4단계 파이프라인을 순서대로 실행
python3 run_pipeline.py --force  # 이미 있는 결과물도 전부 다시 생성
```

내부적으로 아래 4단계가 순서대로 실행됩니다:

| 단계 | 스크립트 | 결과물 |
|---|---|---|
| 1 | `build_glami_dresses.py` | GLAMI-1M 원본(`~/Downloads/GLAMI-1M-dataset/`, 저장소에 미포함)에서 dresses 카테고리만 필터링 → `datasets/GLAMI-1M-dresses/` |
| 2 | `build_dataset.py` | 로케일별 구조화된 description 필드에서 pattern/length 원문 추출 + `normalize.py`로 공통 영어 라벨로 정규화·병합 → `results/pattern_length_dataset.jsonl` |
| 3 | `export_dataset.py` | 라벨이 있는 이미지만 복사 + 최종 manifest 작성 → `datasets/GLAMI-1M-dresses-pattern-length/` |
| 4 | `add_english_descriptions.py` | 상품명(5개 언어)을 영어로 번역해 `description` 캡션 문장 생성 (manifest에 필드 추가) |

`extract.py`는 이 체인에 포함되지 않습니다 — `normalize.py`의 번역 사전을 손으로 만들 때 참고한 **원문 raw 값 진단용 덤프**(`results/extracted.jsonl` + 로케일별 값 분포 출력) 스크립트라, 다른 단계가 그 출력 파일을 읽지 않습니다. 원문 값 분포를 다시 보고 싶을 때만 `python3 extract.py`로 따로 실행하면 됩니다.

- pattern_label / length_label은 **쇼핑몰이 실제로 기재한 구조화 상품 스펙**에서 파싱한 것 — 사람이 만든 라벨이 아니라 실데이터입니다. 하나라도 파싱이 안 된 항목은 채워 넣지 않고 그냥 제외합니다(`normalize.py` 참고).
- **표본 수가 너무 적은 클래스는 제외했습니다** (`build_dataset.py`의 `MIN_CLASS_COUNT = 50` 기준 — 학습·평가하기엔 너무 적은 클래스라 판단):
  - pattern에서 제외: `two_tone`(38장), `paisley`(37), `ethnic`(29), `graphic`(26), `gradient`(18), `moto_print`(18), `oriental`(11), `graphic_text`(4), `camo`(1)
  - length에서 제외: `half_length`(7장)
  - 두 라벨 중 하나라도 제외 대상이면 그 아이템 전체가 최종 데이터셋에서 빠집니다.
- 최종 4,125장, 8개 pattern 클래스(`solid`, `floral`, `striped`, `animal`, `polka_dot`, `mixed`, `checkered`, `logo`) · 5개 length 클래스(`mini`, `knee_midi`, `three_quarter`, `seven_eighth`, `maxi`).

### pexels-dresses-crops (일반화 검증셋)

`build_pexels_dataset/`에서 한 번에 실행합니다 (GLAMI 쪽과 동일하게, 이미 있는 결과물은 자동으로 건너뜁니다).

```bash
cd build_pexels_dataset
python3 run_pipeline.py          # 2단계 파이프라인을 순서대로 실행
python3 run_pipeline.py --force  # 이미 있는 결과물도 전부 다시 생성
```

| 단계 | 스크립트 | 결과물 |
|---|---|---|
| 1 | `crop_dress.py` | YOLO-World(open-vocabulary, class="dress")로 사진마다 신뢰도 최고 박스 1개 검출 후 그 영역만 크롭 → `datasets/pexels-dresses-crops/` |
| 2 | `build_pexels_crops_manifest.py` | Claude가 크롭된 사진을 직접 보고 판단한 pattern/length 라벨(`pexels_labels.py`) + 영어 `description` 캡션 추가 → `manifest.json` |

`detect_dress_face.py`는 이 체인에 포함되지 않습니다 — `crop_dress.py`를 작성하기 전에 YOLO-World 드레스 검출 + MediaPipe 얼굴 검출 조합이 잘 동작하는지 **눈으로 확인해본 QA용 스크립트**라, 최종 크롭 파이프라인은 이 얼굴 제외 로직을 쓰지 않습니다(`crop_dress.py`는 얼굴 제외 없이 드레스 박스만 그대로 크롭). 검출 품질을 다시 눈으로 보고 싶을 때만 `python3 detect_dress_face.py`로 따로 실행하면 됩니다.

- 이 데이터셋은 판매자 메타데이터가 없어서, `pexels_labels.py`에 있는 **pattern_label/length_label은 Claude가 크롭된 사진 100장을 직접 보고 판단한 값**입니다. 원본 alt 텍스트는 hem length를 거의 언급하지 않아 텍스트 파싱이 불가능했기 때문입니다.
- 크롭 때문에 밑단이 잘려서 안 보이는 사진은 **몸 비율로 length를 추정**했습니다.
- GLAMI와 달리 정답이 아니라 **AI 추정 pseudo-label**로 취급해야 합니다 (`build_pexels_crops_manifest.py` 상단 주석 및 생성된 `manifest.json`의 `source` 필드에 명시).

### 데이터 분할 방식

| 데이터셋 | 분할 | 비고 |
|---|---|---|
| GLAMI-1M-dresses-pattern-length | train 3,572 / test 553 | GLAMI-1M 원본이 제공하는 train/test 분할을 그대로 승계 — 직접 임의로 나누지 않음. `manifest.json`의 `split` 필드로 구분 |
| pexels-dresses-crops | 분할 없음 (100장 전량) | 학습에는 전혀 사용하지 않는, **본 적 없는 도메인**(실제 스트릿 사진) 일반화 확인용 held-out 데이터 |

파인튜닝은 GLAMI의 train split(3,572장)만 사용하며, GLAMI test 553장과 Pexels 100장은 학습 중 한 번도 보지 않은 이미지입니다.

---

## 2. 가상 환경 생성 및 Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows는 .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 전체 내용:

```
torch==2.8.0
transformers==4.57.6
peft==0.17.1
pillow==11.3.0
numpy==2.0.2
huggingface_hub==0.36.2
```

- Apple Silicon Mac이면 PyTorch가 자동으로 MPS 백엔드를 사용합니다. MPS가 없으면 CPU로 동작합니다(느림).
- `google/siglip2-base-patch16-224` 체크포인트는 최초 실행 시 자동 다운로드되어 `weights/`에 캐시됩니다.
- `build_pexels_dataset/`을 돌리려면 `ultralytics`(YOLO-World 크롭)가 추가로 필요합니다. `detect_dress_face.py`(QA용)는 `mediapipe`, `opencv-python`도 필요합니다.

---

## 3. Fine-tuning

```bash
cd siglip2_finetune
python3 finetune_siglip2_lora.py
```

- **방식**: LoRA — 양쪽 타워(vision/text)의 attention Q/K/V/O 프로젝션 4개 전부 + vision/text projection head 풀 파인튜닝. 나머지(MLP, 임베딩, sigmoid loss의 `logit_scale`/`logit_bias`)는 사전학습 값으로 동결.
- **손실 함수**: SigLIP2 자체의 sigmoid contrastive loss (`model(..., return_loss=True)`).
- 학습이 끝나면 `weights/fine-tuning/siglip2_lora_adapter_<run_name>/`에 LoRA 어댑터 + 학습된 head가 저장됩니다.

### 주요 설정

`siglip2_finetune/config.py`에 전부 정의되어 있습니다 (현재 값 = run2, 최종 채택 설정).

```python
run_name = "run2_bs32_ep15_cosine"
ckpt = "google/siglip2-base-patch16-224"

batch_size = 32
epochs = 15
seed = 0
max_text_length = 64

lora_lr = 1e-4          # LoRA 파라미터 learning rate
head_lr = 1e-4          # projection head learning rate (동일한 단일 lr)
warmup_ratio = 0.1      # 전체 step의 10% linear warmup 후 cosine decay

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = ("q_proj", "k_proj", "v_proj", "out_proj")
modules_to_save = ("vision_model.head", "text_model.head")
```

다른 조합으로 재학습하고 싶으면 이 파일 값을 바꾸고 `run_name`도 같이 바꿔서(덮어쓰기 방지) 실행하면 됩니다.

### 모델 Weight

베이스 모델은 공개 HuggingFace 체크포인트라 코드 최초 실행 시 자동 다운로드됩니다. LoRA 어댑터는 위 학습 스크립트를 실행하면 생성됩니다.
