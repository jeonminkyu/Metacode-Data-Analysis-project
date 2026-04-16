# Metacode-Data-Analysis-project
메타코드 데이터분석 프로젝트 저장소입니다.
# FC Online 4 데이터 분석 프로젝트

**패키지 출시가 유저 이탈과 매출에 미치는 영향 분석**

## 프로젝트 개요

FC Online 4의 패키지 상품이 게임 경제에 미치는 영향을 데이터 기반으로 분석합니다. 패키지를 통해 특정 OVR 선수가 대량 공급되면 거래소 가격이 하락하고, 이는 유저 구단가치 감소와 이탈로 이어집니다. 이 연관성,상관성,인과성순으로 확대되어 각 관계를  검증하고 최적 패키지 OVR 정책(액션아이템)을 도출합니다.

### 인과 체인

```
패키지 출시 → OVR 가격 하락 → 구단가치 감소 → 유저 이탈 → 매출 손실 > 패키지 매출
```

### 분석 규모

| 항목 | 규모 |
|------|------|
| 분석 기간 | 2025.01 ~ 2025.04 (4개월) |
| 유저 수 | 50,000명 |
| 총 데이터 | 약 407만 건 (5개 테이블) |
| 패키지 출시 | 3회 (1/18, 2/20, 3/20) |

## 프로젝트 구조

```
fc_online4_analysis/
├── data/                          # 생성된 CSV 데이터
│   ├── user_profile.csv           # 유저 프로필 (50,000건)
│   ├── login_logs.csv             # 로그인 로그 (3,470,000건)
│   ├── package_purchase.csv       # 패키지 구매 (84,800건)
│   ├── trade_market.csv           # 거래소 (32,070건)
│   └── daily_club_value.csv       # 일별 구단가치 (480,000건)
├── src/
│   ├── font_setup.py              # 한글 폰트 및 공통 스타일 설정
│   ├── 01_generate_data.py        # 데이터 생성 (~970줄)
│   ├── 02_eda_visualization.py    # EDA 시각화 18개 차트 (~1,100줄)
│   ├── 03_ml_churn_model.py       # ML 이탈 예측 모델 (~860줄)
│   └── 04_scenario_simulation.py  # OVR 시나리오 시뮬레이션 (~680줄)
├── outputs/
│   ├── figures/                   # 생성된 시각화 이미지 (42개)
│   ├── FC_Online4_v2_프레젠테이션.pptx
│   └── FC_Online4_분석_레포트.docx
└── README.md
```

## 실행 방법

### 환경 설정

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### 순차 실행

```bash
cd fc_online4_analysis

# 1단계: 데이터 생성 (5개 CSV)
python src/01_generate_data.py

# 2단계: EDA 시각화 (18개 차트)
python src/02_eda_visualization.py

# 3단계: ML 이탈 예측 모델 (6개 차트)
python src/03_ml_churn_model.py

# 4단계: 시나리오 시뮬레이션 (4개 차트)
python src/04_scenario_simulation.py
```

모든 시각화 결과는 `outputs/figures/`에 저장됩니다.

## 분석 파이프라인

### Stage 1: 데이터 생성 (`01_generate_data.py`)

5개 테이블을 생성합니다. 실제 게임 데이터처럼 자연스러운 분포와 패턴을 반영합니다.

- **K-Means(k=4)** 구단가치 기반 유저 그룹 분류: 0~10조(7%), 10~100조(54%), 100~1000조(35%), 1000조이상(4%)
- 패키지 OVR 확률: 133(40%), 130(19%), 135(5%), 136(1%)
- 이탈 정의: 30일 미접속

### Stage 2: EDA 시각화 (`02_eda_visualization.py`)

15개 EDA 차트 + 3개 가설 검증 차트를 생성합니다.

| 차트 | 내용 |
|------|------|
| fig_01 | K-Means Elbow Method |
| fig_02 | 그룹별 유저 분포 + OVR 바이올린 플롯 |
| fig_03 | 패키지 OVR 등장 확률 버블차트 |
| fig_05 | 패키지 구매 패턴 시계열 |
| fig_07 | 그룹별 구단가치 지수 추이 |
| fig_08 | OVR별 선수 가격 변동 추이 |
| fig_09 | DID(이중차분법) 분석 |
| fig_10 | 그룹별 이탈률 |
| fig_11 | 구단가치 변동 민감도 분석 |
| fig_12 | 패키지 매출 vs 이탈 손실 |
| fig_a1 | [가설1] OVR 중첩도 vs 구단가치 하락 |
| fig_a2 | [가설2] 패키지 전후 DID 분석 |
| fig_a3 | [가설3] 인과 체인: 상관분석(r=0.872) + Granger Causality |

### Stage 3: ML 모델링 (`03_ml_churn_model.py`)

EDA에서 검증된 피처로 이탈 예측 분류 모델을 구축합니다.

- **시간 분할**: 관측(1/1~3/1) → 피처 / 결과(3/1~4/30) → 라벨
- **모델**: Random Forest (AUC 0.727, Accuracy 74.1%) + Logistic Regression (AUC 0.713)
- **핵심 피처**: pkg_cumulative_shock, ovr_pkg_overlap (EDA 가설과 일치)

### Stage 4: 시나리오 시뮬레이션 (`04_scenario_simulation.py`)

패키지 OVR을 조정했을 때의 매출/손실 변화를 시뮬레이션합니다.

- 총 기대수익 = 패키지 매출(OVR 133 중심 Gaussian) + 잔존 유저 기대수익(비이탈 유저 × 월 비용 × 8개월)
- RF predict_proba() 기반 유저별 이탈 확률 직접 예측
- 최적 OVR: **131** (현행 132 → 1단계 하향, 순영향 +0.8억 개선)

## 핵심 결과

| 지표 | 현행 (OVR 132) | 최적 (OVR 131) | 변화 |
|------|---------------|---------------|------|
| 전체 이탈률 | 39.8% | 39.6% | -0.2%p |
| 10~100조 이탈률 | 51.6% | 51.2% | -0.4%p |
| 총 기대수익 | 502.8억 | 502.0억 | -0.8억 |
| 이탈 손실 | 135.5억 | 134.0억 | -1.5억 |
| **순영향** | **+367.2억** | **+368.0억** | **+0.8억 개선** |

## 기술 스택

- **언어**: Python 3.10+
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **통계**: scipy (Pearson 상관, DID)
- **머신러닝**: scikit-learn (Random Forest, Logistic Regression, K-Means)

## 3가지 가설과 검증

1. **OVR 중첩도 → 구단가치 하락** (fig_a1): 유저 보유 OVR이 패키지 높은확률 OVR(130~134)과 겹칠수록 구단가치 하락이 크다 → **검증 완료**
2. **10~100조 그룹 최대 피해** (fig_a2): 패키지 전후 DID 분석에서 10~100조 그룹의 하락률이 가장 큼 → **검증 완료**
3. **인과적 방향성 시사** (fig_a3): Pearson r=0.872 (강한 상관) + Granger Causality Test (비유의, p>0.05) → **DID·상관분석으로 방향성 확인, 완전한 인과 증명에는 A/B Test 필요**
