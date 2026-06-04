# Skin AI Analyst — 개발 위키

> README가 "무엇을 만들었나"라면, 이 위키는 **"왜 그렇게 만들었나 · 어떻게 디버깅했나 · 어디까지가 한계인가"**를 다룹니다.
> 발표·인수인계·재현을 위한 내부 기록입니다.

## 목차
1. [설계 철학과 의사결정](#1-설계-철학과-의사결정)
2. [얼굴 분할: 랜드마크 앵커 방식의 탄생](#2-얼굴-분할-랜드마크-앵커-방식의-탄생)
3. [도메인 갭 — 이 프로젝트 최대의 적](#3-도메인-갭--이-프로젝트-최대의-적)
4. [분류 학습의 함정과 처방](#4-분류-학습의-함정과-처방)
5. [회귀 학습과 점수 캘리브레이션](#5-회귀-학습과-점수-캘리브레이션)
6. [추론 정책 결정 기록](#6-추론-정책-결정-기록)
7. [백엔드 통합 시 빠진 함정](#7-백엔드-통합-시-빠진-함정)
8. [배포 파이프라인 상세](#8-배포-파이프라인-상세)
9. [알려진 한계와 향후 과제](#9-알려진-한계와-향후-과제)
10. [용어집 / 부위·항목 매핑](#10-용어집--부위항목-매핑)

---

## 1. 설계 철학과 의사결정

### 왜 "부위별 단일 모델" 멀티헤드인가
한 장의 얼굴에 8개 부위 × 여러 항목이 있는데, 처음 후보는 (a) 단일 거대 멀티태스크 네트워크,
(b) 부위마다 독립 ResNet50이었습니다. (b)를 택한 이유:

- AI Hub 라벨이 **부위별로 결측이 제각각** — 어떤 사진은 볼만, 어떤 사진은 이마만 라벨이 있음.
  부위별 독립 모델이면 각자 가진 데이터로만 학습/검증할 수 있어 결측 처리가 단순해짐.
- 부위마다 텍스처 스케일(이마 주름 vs 볼 모공)이 달라, 헤드만 공유하는 것보다 백본까지 분리하는 편이
  수렴이 안정적이었음.
- 추론 시 일부 부위가 안 잡혀도(예: 머리카락이 이마를 가림) 나머지 부위 점수는 그대로 산출 가능.

### 왜 분류와 회귀를 둘 다 쓰나
AI Hub는 **사람이 매긴 등급(0~N)** 과 **장비 실측값(수분량/모공 수 등)** 을 모두 제공합니다.
- 등급(분류)은 사용자에게 직관적이지만 거칠다(예: 모공 6단계).
- 실측값(회귀)은 정밀하지만 절대 스케일이라 사용자가 해석하기 어렵다.

→ **회귀 결과를 0~100 점수로 환산해 메인 화면에 쓰고, 분류 등급은 보조 지표로** 둠.
단, 회귀는 도메인 갭에 취약(§3)해서 항목별 **신뢰도 티어(정밀/참고/한계)** 를 함께 노출.

---

## 2. 얼굴 분할: 랜드마크 앵커 방식의 탄생

### 1차 시도 (실패): 얼굴 bbox 비율 곱
"얼굴 박스의 좌상단에서 가로 30%, 세로 20% 지점이 이마" 식의 비율 곱은
**거리·각도·얼굴형이 바뀌면 영역이 어긋남**. 셀카는 학습 데이터(정면 고정 촬영)와 구도가 달라
패치가 엉뚱한 곳을 잘랐음.

### 2차 시도 (채택): 해부학 랜드마크 그룹 min/max + pad
`ai/inference.py`의 `FACEPART_LANDMARKS`는 각 부위의 학습 bbox 안에 실제로 들어오던
MediaPipe 468 랜드마크 인덱스를 모아둔 것입니다. 추론 시 그 랜드마크들의 min/max로 박스를 만들고
`FACEPART_PAD`로 미세 확장.

- 장점: 얼굴 크기·거리·각도와 무관하게 **동일 해부학 영역**을 자름 → 학습/추론 구도 일치
- 튜닝 흔적: forehead는 헤어라인 쪽 `pad_top` 크게, glabellus는 눈썹까지만 위로, perocular는
  측두부 과확장 방지 등 부위별 pad를 손으로 맞춤
- chin 패치 정의는 2026-05-21에 17/18/200 랜드마크 제거 실험을 했으나 회귀 r 개선이 미미해
  (평균 Δ+0.005, elasticity는 오히려 −0.02) **원복** — 학습 시점 패치 정의와 추론 crop을
  일치시키는 것이 더 중요했기 때문 (§3).

### area 4/6의 미스터리
오른쪽 눈가(4)·오른쪽 볼(6)은 전용 체크포인트가 없습니다. 학습 코드(`model.py`)에서 area를
`str`로 비교하는 자리와 `int`로 비교하는 자리가 섞여 있어, **"4/6은 3/5 모델 + 좌우 flip"
의도가 실제로는 flip 없이 동작**했습니다. 추론도 의도가 아니라 **실제 학습 동작에 맞춰** flip 없이
3/5 모델을 그대로 적용합니다. (의도대로 flip을 켜면 학습/추론 불일치로 정확도가 떨어짐)

---

## 3. 도메인 갭 — 이 프로젝트 최대의 적

### 증상
회귀 모델이 valid에서는 r 0.7~0.9인데, 실제 셀카 추론에서는 색소(count)를 빼면 r이 0~0.1까지
**붕괴**. 분류는 상대적으로 견뎠지만 회귀가 무너짐.

### 원인
- 학습 패치: 데이터셋 제작 시 **사람이 라벨링한 bbox로 crop**
- 추론 패치: **MediaPipe 랜드마크 기반 crop**
- 둘은 같은 부위라도 **여백·중심·종횡비가 미묘하게 다름**. 회귀는 절대 측정값을 예측하므로 이
  분포 이동(domain shift)에 분류보다 훨씬 민감했음.

### 처방
1. **학습 패치를 추론과 동일한 MediaPipe crop으로 전량 재추출** (`extract_patches_mediapipe.py`)
   → 색소 외 항목 r 0~0.1 → **0.5~0.7 회복**.
2. 그래도 남는 절대 스케일 차이는 **percentile 캘리브레이션**(§5)으로 흡수.
3. 패치 정의를 함부로 바꾸지 않음 — chin 실험 원복(§2)도 같은 맥락.

> 교훈: **"학습 전처리 = 추론 전처리"는 회귀에서 타협 불가**. 한 픽셀의 crop 차이가 r을 반토막낸다.

---

## 4. 분류 학습의 함정과 처방

### 클래스 불균형
`label_distribution.txt` 기준 일부 등급이 극단적으로 희소(예: pore 등급5, dryness 양극단, sagging
등급6은 단 13장). 그대로 balanced weight를 쓰면 가중치가 14배까지 튀어 **학습이 진동**.

- 처방 A: balanced 가중치에 **sqrt** 적용 → pore 등급5 14.3→3.78로 완화 (`model.py CLASS_WEIGHTS`)
- 처방 B: sagging 등급6(13장)은 등급5로 **클램프**해 6-class로 운영
- 처방 C: forehead_wrinkle/wrinkle은 고등급 과소·저등급 과다예측 경향 → 다수 등급 가중치 ↓,
  고등급 가중치 ↑로 수동 재조정

### 결측 라벨 처리
collate 단계에서 누락/invalid를 `-1` sentinel로 마킹하고, 손실 계산에서 `mask = target >= 0`으로
제외. 부위별 데이터 가용량이 달라도 같은 배치에서 공존 가능.

### 모델 선정 기준
분류는 valid **±1 accuracy** 최대, 회귀는 valid **L1 최소** 기준으로 체크포인트를 저장.
(`--areas` 필터로 이번 epoch에 안 돈 부위는 `val_acc.count==0`이면 skip — None 모델 저장 크래시 방지)

---

## 5. 회귀 학습과 점수 캘리브레이션

### 정규화 / 역정규화
회귀 타깃은 0~1로 정규화해 학습하고, 추론에서 `REG_DENORM`으로 실측 스케일 복원:
moisture ×100, Ra ×100(μm), pore ×3000(개), count ×350(개), R2 ×1.

### percentile rank 캘리브레이션 (`calibrate_inference_domain.py` → `reg_calibration_mp.json`)
모델 raw 출력을 그대로 점수화하면 도메인 shift로 전부 한쪽에 쏠립니다. 대신:

1. 학습 valid 분포의 percentile 배열을 저장
2. 추론 raw가 그 분포의 **몇 percentile에 위치하는지** linear interpolation으로 역산
3. 방향에 따라 severity 환산:
   - 높을수록 양호(moisture/R²): `severity = 100 - rank`
   - 낮을수록 양호(Ra/pore/count): `severity = rank`
4. 건강점수 `= 100 - severity` (0~100 클램프)

캘리브레이션 파일이 없으면 raw 단순 환산으로 fallback(도메인 보정 없음).

### 신뢰도 티어
test split Pearson r로 항목을 분류해 UI/발표에 함께 표기:

| 티어 | 항목 | 의미 |
|---|---|---|
| 정밀 | count(색소) r0.92, pore(모공) r0.82 | 신뢰 가능 |
| 참고 | Ra, moisture, R2 (r 0.5~0.7) | 경향 참고 수준 |
| 한계 | chin_elasticity r0.39 | 평균회귀로 보수적 — 단독 판단 금물 |

> 종합 점수를 단순 균등평균으로 내면 약한 항목의 평균회귀가 전체를 보수적으로 끌어내립니다.
> 그래서 CLI는 종합점수를 빼고 항목별+티어로 제시하고, 백엔드는 UI 단일값이 필요해 균등평균을
> 쓰되 항목별 점수/티어를 함께 내려보냅니다.

---

## 6. 추론 정책 결정 기록

### TTA (좌우 flip 평균)
augmentation 재학습(`100%_augw`) 이후 모델은 flip aug로 학습됐으므로 추론에서도 TTA를 켭니다
(`SkinEngine.tta = True`). aug 없이 학습한 구버전 체크포인트에선 TTA가 오히려 정확도를 떨어뜨려
CLI 기본값은 off로 두고 `--tta`로 명시.

### Hybrid argmax / Expected-value
§3.2(README) 정책. 핵심 근거 한 줄: **"±1이 낮은 항목만" E-value로 멀리 튀는 오차를 줄이고,
이미 잘 맞는 항목(pigmentation/dryness)은 argmax를 유지**해 소수등급 recall 붕괴를 피함.
- pore는 ±1만 통과(Macro-F1 19, 등급0/4/5 recall 0%) — **구조적 한계로 수용**.

### shape 분기 forward
회귀에서 패치가 128보다 길면 학습 코드와 동일하게 **좌/우(또는 상/하)로 쪼개 forward 후 합산**.
정확히 128×128인 케이스만 TTA flip 평균 추가(긴 케이스는 학습 시점에 이미 split+flip 합산이라 중복 회피).

---

## 7. 백엔드 통합 시 빠진 함정

- **venv 의존성 핀 고정**: `mediapipe==0.10.9`(0.10.10+에서 legacy `mp.solutions` 네임스페이스
  제거됨), `natsort`, `opencv-python-headless`(GUI 의존 없는 서버용). 이 조합이 깨지면 추론이 import
  단계에서 죽음.
- **PyTorch는 CPU 휠 강제**: `--extra-index-url .../whl/cpu`. CUDA 휠은 2GB+ 라 HF Spaces 빌드 실패.
- **모델 싱글톤 + prewarm**: `skin_inference.SkinEngine`은 프로세스당 1회 로드. FastAPI lifespan에서
  `prewarm()` 호출해 첫 요청 30초 지연 제거. prewarm이 실패해도 서버는 뜨고, 첫 요청 시 재시도.
- **얼굴 미감지 = 사용자 입력 오류**: `extract_faceparts`가 빈 dict면 `ValueError` → `diagnosis.py`가
  업로드 파일 삭제 후 **HTTP 400**. (500이 아니라 400 — 서버 버그가 아니라 사진 문제)
- **결과 JSON 풀 저장**: DB `regression` 컬럼에 raw 0~1 실수까지 저장 → 추후 점수 정책이 바뀌어도
  재집계 가능.

---

## 8. 배포 파이프라인 상세

### 프론트엔드 (Vercel)
- `wep/skinai` CRA 빌드. `public/`는 그대로 서빙 — `manifest.json`, `service-worker.js`,
  `.well-known/assetlinks.json`.
- `vercel.json` 헤더: 서비스워커는 `no-cache, no-store, must-revalidate`, manifest는
  `max-age=0, must-revalidate` (오래된 SW가 박히는 문제 방지).
- 서비스워커: shell precache + navigation은 network-first, 정적 자산은 cache-first, API/비-GET/
  cross-origin은 건너뜀.

### 백엔드 (HF Spaces · Docker)
- `Dockerfile` python:3.9-slim, libgl1/libglib2.0-0(mediapipe·opencv 런타임 의존).
- requirements 먼저 COPY해 의존성 레이어 캐시 → 코드 변경 시 재설치 안 함.
- **체크포인트는 이미지에 안 넣고** 빌드 타임에 `snapshot_download(MODEL_REPO)`로 받음
  (Space repo 1GB LFS 한계 회피). public repo면 토큰 불필요.
- `WORKDIR /app/backend`에서 `uvicorn main:app --port ${PORT:-7860}`.

### 안드로이드 앱 (Bubblewrap TWA)
- TWA = **얇은 Chrome 컨테이너**. `https://ai-skin-analyst.vercel.app`를 매 실행마다 새로 로드.
  → **CSS/JS/콘텐츠 변경은 APK 재빌드 불필요**, Vercel 재배포만 하면 다음 실행 때 반영.
  (단, 서비스워커 캐시로 첫 실행이 구버전일 수 있어 앱 백그라운드 종료 후 재실행하면 해결)
- **APK 재빌드가 필요한 경우만**: 앱 이름/아이콘/테마색(manifest), package ID, SHA-256/keystore 변경.
- Digital Asset Links: `public/.well-known/assetlinks.json`이 도메인↔APK SHA-256 지문을 연결해
  주소창 없는 전체화면을 보장.
- 🔒 `apk/android.keystore`는 **앱 서명 키 — 절대 커밋 금지**(`.gitignore`에 `apk/` 등록). 분실하면
  같은 앱으로 업데이트 불가.

### 빌드 환경 메모
- JDK **17** 필요(21 아님). Gradle 데몬 OOM 시 `apk/gradle.properties`의 `org.gradle.jvmargs`를
  `-Xmx1024m`로 낮춤.
- bubblewrap 스크립트 실행은 PowerShell ExecutionPolicy(CurrentUser RemoteSigned) 필요.
  stdin 파이프 시 inquirer가 죽으므로 **대화형 셸에서 직접 실행**.

---

## 9. 알려진 한계와 향후 과제

### 모델 한계
- chin 회귀(특히 elasticity r0.39)는 여전히 약함 — `100%/_mp_chinfix` 전량 재추출 후 재학습이 다음 단계.
- pore 분류는 ±1만 통과하고 strict/Macro-F1이 낮음 — 등급 경계가 텍스처만으로는 모호한 구조적 한계.
- 모든 회귀는 **단일 정면 셀카** 기준 — 조명/화장/카메라 화질에 민감. 의료 진단 아님.

### 운영 한계
- 추론 서버 CPU 단일 — 동시 요청이 늘면 큐잉 지연. (현재 prewarm 싱글톤 1개)
- Gemini API spend cap — 챗봇 호출량 관리 필요.
- MySQL 운영 시 커넥션/마이그레이션 전략 미정(현재 `create_all` idempotent 자동 생성).

### 백로그 (우선순위 순)
1. chin 전량 재추출(`skin_mp_patch_chinfix`) + 재학습 → `inference.py` area 8을 `100%/_mp_chinfix`로 전환
2. 캘리브레이션 재생성
3. 분류 A-실험 (가중치/aug 추가 튜닝)
4. 발표용 한계 정리 문서화

---

## 10. 용어집 / 부위·항목 매핑

### 부위(area) ↔ 모델
| area | 부위 | 분류 출력차원 | 회귀 항목 | 비고 |
|---|---|---|---|---|
| 0 | 전체 얼굴 | — | count(색소) | 회귀 전용, face crop |
| 1 | 이마 | 13 (fw+pigmentation) | moisture, R2 | |
| 2 | 미간 | 7 (wrinkle) | — | 회귀 데이터 없음 |
| 3 | 왼쪽 눈가 | 7 (wrinkle) | Ra | |
| 4 | 오른쪽 눈가 | (3 모델 공유) | (3 공유) | flip 미적용 |
| 5 | 왼쪽 볼 | 12 (pigmentation+pore) | moisture, R2, pore | |
| 6 | 오른쪽 볼 | (5 모델 공유) | (5 공유) | flip 미적용 |
| 7 | 입술 | 5 (dryness) | — | 회귀 데이터 없음 |
| 8 | 턱 | 6 (sagging) | moisture, R2 | r 약함 |

### 항목 키 ↔ 의미
| 키 | 의미 | 종류 | 점수 방향 |
|---|---|---|---|
| pigmentation / count | 색소침착 | 분류/회귀 | 적을수록 양호 |
| wrinkle / forehead_wrinkle / Ra | 주름·거칠기 | 분류/회귀 | 적을수록 양호 |
| pore | 모공 | 분류/회귀 | 적을수록 양호 |
| dryness / moisture | 건조·수분 | 분류/회귀 | 수분 많을수록 양호 |
| sagging / R2 | 처짐·탄력 | 분류/회귀 | 탄력 높을수록 양호 |

### 건강점수 5단계 (`_grade_label`)
80~100 매우 양호 · 60~80 양호 · 40~60 보통 · 20~40 주의 · 0~20 심각

---

*문서 관리: Team Skinmate · 최신 갱신 2026-06.*
</content>
