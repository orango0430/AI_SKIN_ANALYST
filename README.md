# Skin AI Analyst

AI 기반 피부 진단 웹·앱 서비스. 단국대학교 졸업작품 (Team Skinmate).
얼굴 사진 1장으로 8개 부위 × 11개 항목의 피부 상태를 진단하고,
0~100 점수로 환산해 사용자에게 표시합니다.

## 구성

| 폴더/파일 | 설명 |
|---|---|
| `ai/` | AI 모델 코드 (학습·추론·전처리·테스트) |
| `ai/inference.py` | 분류·회귀 추론 통합 진입점 |
| `ai/model.py`, `ai/data_loader.py`, `ai/regression_data_loader.py` | 모델·데이터 모듈 |
| `ai/regression_train.py` | 회귀 재학습 (CCC 손실, 옵티마이저 유지) |
| `ai/extract_patches_mediapipe.py` | MediaPipe crop 패치 추출 (도메인 정합) |
| `ai/main.py` | 분류 학습 진입점 |
| `backend/` | FastAPI 서버 (AI 통합) |
| `wep/skinai/` | React 프론트엔드 (Vercel 배포 대상) |

## 빠른 실행

### 백엔드
```bash
cd backend
./venvs/myapi/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```
서버 기동 시 모델 로드 30~60초 → `[SkinEngine] ready` 출력 후 요청 받음.

### 프론트엔드
```bash
cd wep/skinai
npm install
npm start          # http://localhost:3000
```
`.env.local`에 `REACT_APP_API_BASE_URL=http://localhost:8000` 지정.

### 추론 CLI 단독
```bash
cd ai
python inference.py \
  --image ../test_img.jpg \
  --checkpoint "../checkpoint2/class/100%_augw/1,2,3" \
  --reg_checkpoint "../checkpoint2/regression/100%/_mp" \
  --calibration ../reg_calibration_mp.json --tta
```

## 기술 스택

- **AI**: PyTorch · ResNet50 × (분류 6 + 회귀 5) · MediaPipe Face Mesh
- **백엔드**: FastAPI · SQLAlchemy · MySQL · JWT
- **프론트엔드**: React 18 · React Router · Vanilla CSS
- **배포**: Vercel (frontend) · HuggingFace Spaces (backend, 예정)

## 모델 성능 (test split)

| 회귀 항목 | Pearson r | 신뢰도 |
|---|---|---|
| pigmentation_count | 0.92 | 정밀 |
| l_cheek_pore | 0.82 | 정밀 |
| l_perocular_wrinkle_Ra | 0.72 | 참고 |
| l_cheek_elasticity_R2 | 0.70 | 참고 |
| l_cheek_moisture | 0.63 | 참고 |
| chin_moisture | 0.49 | 참고 |
| chin_elasticity_R2 | 0.39 | 한계 |

**핵심 발견**: 학습 패치(bbox crop)와 추론 패치(MediaPipe crop) 간 도메인 갭이
회귀 추론 저하의 주범. 전량 MediaPipe 재추출 후 재학습으로 색소 외 항목 r 0~0.1 → 0.5~0.7 회복.

분류는 ±1 정확도 ≥ 85% 확보 (가중치 조정 + augmentation + Expected-value 하이브리드).

## 데이터셋

**AI Hub - 한국인 피부상태 측정 데이터** (NIA-019-028) 기반.
10~50대 1,100명 × 3장비 × 13각도 = 약 13만장 안면 이미지.

데이터셋 문의: 단국대학교 컴퓨터학과 박사과정 이정호 (72210297@dankook.ac.kr)

## 팀

Team Skinmate · 단국대학교 졸업작품 (2025)
