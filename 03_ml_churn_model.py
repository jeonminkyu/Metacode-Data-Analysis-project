"""
=============================================================================
FC Online 4 데이터 분석 프로젝트 - Stage 3: ML 이탈 예측 모델
=============================================================================
모델: Random Forest + Logistic Regression
목적: 패키지 출시 시 이탈 유저를 사전 예측하는 모델 구축

■ 피처 선택 근거 (02_eda_visualization.py 가설 근거 시각화에서 증명)
  ─────────────────────────────────────────────────────
  [fig_a1] 유저 보유 OVR이 패키지 높은확률 OVR(130~134)과 겹칠수록
           구단가치 하락이 크다 → ovr_pkg_overlap, is_pkg_ovr_range
  [fig_a2] 패키지 출시 전후 10~100조 그룹의 구단가치 하락이 가장 큼
           → pkg1_cv_change, pkg2_cv_change, pkg_cumulative_shock
  [fig_a3] 인과 체인: OVR가격하락 → 구단가치하락 → 이탈
           (추세선 기울기 양의 상관) → ovr_price_exposure

  위 EDA 시각화에서 증명된 인과관계를 바탕으로 아래 핵심 피처를 설계:
  ★ ovr_pkg_overlap     : 유저 OVR↔패키지 OVR 겹침도 (가설1 근거)
  ★ is_pkg_ovr_range    : 패키지 높은확률 OVR 범위 해당 여부 (가설1 근거)
  ★ pkg1/2_cv_change    : 패키지 출시 전후 구단가치 변화율 (가설2 근거)
  ★ pkg_cumulative_shock: 1차+2차 패키지 누적 충격 (가설2 근거)
  ★ ovr_price_exposure  : 보유 OVR대 가격 하락 노출도 (인과체인 근거)

■ 추가 피처 (일반적 이탈 예측 지표)
  - 유저 프로필: club_value, avg_ovr, spendig, membership_tier
  - 접속 패턴: 접속일수, 세션시간, 접속감소율, 마지막주접속
  - 구매 행동: 패키지 구매횟수, 금액, 다양성
  - 구단가치 변동: 인덱스, 최대하락률, z-score 충격횟수
  - 거래 활동: 거래 참여 횟수

■ 데이터 누출 방지
  관측 기간 (피처): 2025-01-01 ~ 2025-03-01
  결과 기간 (타겟): 2025-03-01 ~ 2025-04-30

타겟: is_churned (결과 기간 30일 미접속)
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

# 폰트 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from font_setup import setup_korean_font, COLORS, GROUP_ORDER, CHART_STYLE

warnings.filterwarnings('ignore')
setup_korean_font()

# ============================================================================
# 경로 설정
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')
OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'figures')
MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'models')
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 핵심 파라미터
DATA_END = pd.Timestamp('2025-04-30')
CHURN_DAYS = 30
SEED = 42

print("=" * 70)
print("FC Online 4 - Stage 3: ML 이탈 예측 모델")
print("=" * 70)


# ============================================================================
# 1. 데이터 로드
# ============================================================================
def load_data():
    """5개 테이블 로드"""
    print("\n[1] 데이터 로드")
    up = pd.read_csv(os.path.join(DATA_PATH, 'user_profile.csv'))
    ll = pd.read_csv(os.path.join(DATA_PATH, 'login_logs.csv'))
    pp = pd.read_csv(os.path.join(DATA_PATH, 'package_purchase.csv'))
    tm = pd.read_csv(os.path.join(DATA_PATH, 'trade_market.csv'))
    dcv = pd.read_csv(os.path.join(DATA_PATH, 'daily_club_value.csv'))

    ll['login_date'] = pd.to_datetime(ll['login_date'])
    pp['purchase_date'] = pd.to_datetime(pp['purchase_date'])
    tm['trade_date'] = pd.to_datetime(tm['trade_date'])
    dcv['date'] = pd.to_datetime(dcv['date'])

    for name, df in [('user_profile', up), ('login_logs', ll),
                     ('package_purchase', pp), ('trade_market', tm),
                     ('daily_club_value', dcv)]:
        print(f"  {name}: {len(df):>10,}행")
    return up, ll, pp, tm, dcv


# ============================================================================
# 2. 피처 엔지니어링
# ============================================================================
def engineer_features(up, ll, pp, tm, dcv):
    """
    유저별 피처 생성 — 시간 분할(Temporal Split)로 데이터 누출 방지

    관측 기간 (피처 계산): 2025-01-01 ~ 2025-03-01 (60일)
    결과 기간 (이탈 판정): 2025-03-01 ~ 2025-04-30 (60일)
    이탈 정의: 결과 기간 중 마지막 접속 후 30일 미접속
    """
    print("\n[2] 피처 엔지니어링 (시간 분할 — 데이터 누출 방지)")

    # 시간 분할 기준
    OBS_END = pd.Timestamp('2025-03-01')      # 관측 기간 종료
    OUTCOME_START = pd.Timestamp('2025-03-01') # 결과 기간 시작

    # ── 타겟 변수: 결과 기간 이탈 여부 ──
    # 결과 기간(3/1~4/30) 동안 접속 기록
    outcome_logs = ll[ll['login_date'] >= OUTCOME_START]
    last_login_outcome = outcome_logs.groupby('user_id')['login_date'].max().reset_index()
    last_login_outcome.columns = ['user_id', 'last_login_outcome']
    last_login_outcome['days_inactive_outcome'] = (DATA_END - last_login_outcome['last_login_outcome']).dt.days
    last_login_outcome['is_churned'] = (last_login_outcome['days_inactive_outcome'] > CHURN_DAYS).astype(int)

    # 결과 기간에 아예 접속 안 한 유저 → 이탈
    all_users = set(up['user_id'])
    outcome_users = set(last_login_outcome['user_id'])
    never_logged = all_users - outcome_users

    never_df = pd.DataFrame({
        'user_id': list(never_logged),
        'last_login_outcome': pd.NaT,
        'days_inactive_outcome': 999,
        'is_churned': 1
    })
    last_login_outcome = pd.concat([last_login_outcome, never_df], ignore_index=True)

    # ── 관측 기간 로그만 사용 (데이터 누출 방지) ──
    obs_logs = ll[ll['login_date'] < OBS_END]
    obs_purchases = pp[pp['purchase_date'] < OBS_END]
    obs_trades = tm[tm['trade_date'] < OBS_END]
    obs_dcv = dcv[dcv['date'] < OBS_END]

    # ── 기본 프로필 피처 ──
    features = up[['user_id', 'club_value_group', 'club_value',
                    'avg_ovr', 'spendig', 'membership_tier']].copy()

    tier_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4, 'Diamond': 5}
    features['membership_rank'] = features['membership_tier'].map(tier_map)
    features['club_value_jo'] = features['club_value'] / 1e12

    print("  [2-1] 접속 패턴 피처 생성 중 (관측 기간만 사용)...")

    # ── 접속 패턴 피처 (관측 기간에서만) ──
    login_days = obs_logs.groupby('user_id')['login_date'].nunique().reset_index()
    login_days.columns = ['user_id', 'obs_login_days']

    avg_session = obs_logs.groupby('user_id')['session_duration_min'].mean().reset_index()
    avg_session.columns = ['user_id', 'obs_avg_session']

    # 관측 기간 전반부 (1/1~1/31) vs 후반부 (2/1~2/28) 접속 비교
    early_logs = obs_logs[obs_logs['login_date'] < pd.Timestamp('2025-02-01')]
    late_logs = obs_logs[obs_logs['login_date'] >= pd.Timestamp('2025-02-01')]

    early_cnt = early_logs.groupby('user_id')['login_date'].nunique().reset_index()
    early_cnt.columns = ['user_id', 'early_logins']

    late_cnt = late_logs.groupby('user_id')['login_date'].nunique().reset_index()
    late_cnt.columns = ['user_id', 'late_logins']

    # 관측기간 마지막 주(2/22~2/28) 접속 빈도 (이탈 전조 파악)
    last_week = obs_logs[obs_logs['login_date'] >= pd.Timestamp('2025-02-22')]
    last_week_cnt = last_week.groupby('user_id')['login_date'].nunique().reset_index()
    last_week_cnt.columns = ['user_id', 'last_week_logins']

    print("  [2-2] 구매 행동 피처 생성 중 (관측 기간만)...")

    # ── 구매 행동 (관측 기간) ──
    pkg_count = obs_purchases.groupby('user_id').size().reset_index(name='obs_pkg_count')
    pkg_amount = obs_purchases.groupby('user_id')['amount'].sum().reset_index()
    pkg_amount.columns = ['user_id', 'obs_pkg_amount']
    pkg_variety = obs_purchases.groupby('user_id')['package_id'].nunique().reset_index()
    pkg_variety.columns = ['user_id', 'obs_pkg_variety']
    pkg_ovr = obs_purchases.groupby('user_id')['ovr'].mean().reset_index()
    pkg_ovr.columns = ['user_id', 'obs_avg_pkg_ovr']

    print("  [2-3] 설계서 가설 핵심 피처 생성 중...")

    # =========================================================================
    # ★ 설계서 핵심 가설 기반 피처 (인과 체인 반영)
    # =========================================================================
    # 가설1: 패키지 → 높은확률 OVR 선수 공급 증가 → 해당 OVR 가격 하락
    # 가설2: 유저 보유 OVR이 패키지 높은확률 OVR과 겹칠수록 구단가치 하락
    # 가설3: 구단가치 하락이 큰 그룹(10~100조)이 이탈에 가장 민감

    # ── (A) 유저 보유 OVR과 패키지 높은확률 OVR 겹침도 ──
    # 패키지에서 높은 확률 OVR: 130(19%), 133(40%) → 이 OVR대 선수 보유시 직격탄
    # 유저 avg_ovr이 130~134 범위에 가까울수록 패키지 영향 직접 받음
    PKG_HIGH_PROB_OVRS = [130, 131, 132, 133, 134]  # 패키지 확률 높은 OVR 대역
    PKG_CENTER_OVR = 132.0  # 패키지 확률 분포의 중심

    features['ovr_pkg_distance'] = abs(features['avg_ovr'] - PKG_CENTER_OVR)
    # 거리가 가까울수록 겹침도가 높음 → 역수로 변환 (최대 1.0)
    features['ovr_pkg_overlap'] = 1.0 / (1.0 + features['ovr_pkg_distance'])

    # 유저의 OVR이 패키지 높은확률 범위(130~134)에 속하는지 여부
    features['is_pkg_ovr_range'] = features['avg_ovr'].between(129.5, 134.5).astype(int)

    # ── (B) 패키지 출시 전후 구단가치 변화 (1차: 1/18, 2차: 2/20) ──
    # 각 패키지 출시일 기준 전 7일 vs 후 7일 구단가치 비교
    PKG1_DATE = pd.Timestamp('2025-01-18')
    PKG2_DATE = pd.Timestamp('2025-02-20')

    for pkg_name, pkg_date in [('pkg1', PKG1_DATE), ('pkg2', PKG2_DATE)]:
        pre_start = pkg_date - pd.Timedelta(days=7)
        post_end = pkg_date + pd.Timedelta(days=7)

        # 관측 기간 내인지 확인
        if post_end <= OBS_END:
            pre_cv = obs_dcv[(obs_dcv['date'] >= pre_start) & (obs_dcv['date'] < pkg_date)]
            post_cv = obs_dcv[(obs_dcv['date'] >= pkg_date) & (obs_dcv['date'] <= post_end)]

            pre_avg = pre_cv.groupby('user_id')['club_value_index'].mean().reset_index()
            pre_avg.columns = ['user_id', f'{pkg_name}_pre_idx']

            post_avg = post_cv.groupby('user_id')['club_value_index'].mean().reset_index()
            post_avg.columns = ['user_id', f'{pkg_name}_post_idx']

            features = features.merge(pre_avg, on='user_id', how='left')
            features = features.merge(post_avg, on='user_id', how='left')

            # 패키지 전후 변화율 (음수 = 하락)
            features[f'{pkg_name}_cv_change'] = (
                (features[f'{pkg_name}_post_idx'] - features[f'{pkg_name}_pre_idx'])
                / features[f'{pkg_name}_pre_idx'].replace(0, 1) * 100
            )
            features[f'{pkg_name}_cv_change'] = features[f'{pkg_name}_cv_change'].fillna(0)

    # 1차+2차 패키지 누적 충격 (음수값 합산)
    features['pkg_cumulative_shock'] = (
        features.get('pkg1_cv_change', pd.Series(0, index=features.index)).clip(upper=0) +
        features.get('pkg2_cv_change', pd.Series(0, index=features.index)).clip(upper=0)
    )

    # ── (C) 보유 OVR대 가격 하락 노출도 (trade_market 기반) ──
    # 유저의 avg_ovr에 해당하는 OVR의 관측기간 내 가격 하락률 매핑
    # 각 OVR별 전체 기간 가격 변화 계산
    ovr_first_price = obs_trades.groupby('ovr').apply(
        lambda x: x.nsmallest(max(1, len(x)//10), 'trade_date')['price_trade'].mean()
    ).reset_index(name='first_price')

    ovr_last_price = obs_trades.groupby('ovr').apply(
        lambda x: x.nlargest(max(1, len(x)//10), 'trade_date')['price_trade'].mean()
    ).reset_index(name='last_price')

    ovr_price_change = ovr_first_price.merge(ovr_last_price, on='ovr')
    ovr_price_change['price_drop_pct'] = (
        (ovr_price_change['first_price'] - ovr_price_change['last_price'])
        / ovr_price_change['first_price'] * 100
    )

    # 유저의 avg_ovr을 반올림하여 가장 가까운 OVR의 가격 하락률 매핑
    features['ovr_rounded'] = features['avg_ovr'].round().astype(int)
    price_drop_map = dict(zip(ovr_price_change['ovr'], ovr_price_change['price_drop_pct']))
    features['ovr_price_exposure'] = features['ovr_rounded'].map(price_drop_map).fillna(0)
    features.drop('ovr_rounded', axis=1, inplace=True)

    print("    → ovr_pkg_overlap: 유저 OVR↔패키지 OVR 겹침도")
    print("    → is_pkg_ovr_range: 패키지 높은확률 OVR 범위(130~134) 해당 여부")
    print("    → pkg1/2_cv_change: 패키지 출시 전후 구단가치 변화율")
    print("    → pkg_cumulative_shock: 누적 패키지 충격")
    print("    → ovr_price_exposure: 보유 OVR 가격 하락 노출도")

    print("\n  [2-4] 구단가치 변동 피처 생성 중 (관측 기간만)...")

    # ── 구단가치 변동 (관측 기간) ──
    obs_dcv_last = obs_dcv.groupby('user_id').last().reset_index()
    cv_obs_index = obs_dcv_last[['user_id', 'club_value_index']].copy()
    cv_obs_index.columns = ['user_id', 'cv_obs_index']

    cv_max_drop = obs_dcv.groupby('user_id')['daily_change_rate'].min().reset_index()
    cv_max_drop.columns = ['user_id', 'cv_max_daily_drop']

    shock_count = obs_dcv[obs_dcv['z_score'] <= -2].groupby('user_id').size().reset_index(name='shock_count')

    cv_avg_change = obs_dcv.groupby('user_id')['daily_change_rate'].mean().reset_index()
    cv_avg_change.columns = ['user_id', 'cv_avg_change_rate']

    # 구단가치 하락폭 (1차+2차 패키지 효과만 반영)
    first_date = obs_dcv['date'].min()
    cv_first = obs_dcv[obs_dcv['date'] == first_date][['user_id', 'club_value']].copy()
    cv_first.columns = ['user_id', 'cv_first']
    cv_last = obs_dcv_last[['user_id', 'club_value']].copy()
    cv_last.columns = ['user_id', 'cv_last']
    cv_decline = cv_first.merge(cv_last, on='user_id')
    cv_decline['cv_decline_pct'] = (cv_decline['cv_first'] - cv_decline['cv_last']) / cv_decline['cv_first'] * 100
    cv_decline = cv_decline[['user_id', 'cv_decline_pct']]

    print("  [2-4] 거래 활동 피처 생성 중 (관측 기간만)...")

    trade_count = obs_trades.groupby('user_id').size().reset_index(name='obs_trade_count')

    # ── 전체 피처 병합 ──
    print("  [2-5] 피처 병합 중...")
    features = features.merge(last_login_outcome[['user_id', 'is_churned']],
                              on='user_id', how='left')
    features = features.merge(login_days, on='user_id', how='left')
    features = features.merge(avg_session, on='user_id', how='left')
    features = features.merge(early_cnt, on='user_id', how='left')
    features = features.merge(late_cnt, on='user_id', how='left')
    features = features.merge(last_week_cnt, on='user_id', how='left')
    features = features.merge(pkg_count, on='user_id', how='left')
    features = features.merge(pkg_amount, on='user_id', how='left')
    features = features.merge(pkg_variety, on='user_id', how='left')
    features = features.merge(pkg_ovr, on='user_id', how='left')
    features = features.merge(cv_obs_index, on='user_id', how='left')
    features = features.merge(cv_max_drop, on='user_id', how='left')
    features = features.merge(shock_count, on='user_id', how='left')
    features = features.merge(cv_avg_change, on='user_id', how='left')
    features = features.merge(cv_decline, on='user_id', how='left')
    features = features.merge(trade_count, on='user_id', how='left')

    # 결측값 처리
    fill_zero_cols = ['obs_login_days', 'obs_avg_session', 'early_logins',
                      'late_logins', 'last_week_logins', 'obs_pkg_count',
                      'obs_pkg_amount', 'obs_pkg_variety', 'obs_avg_pkg_ovr',
                      'shock_count', 'obs_trade_count', 'cv_decline_pct',
                      'pkg1_cv_change', 'pkg2_cv_change', 'pkg_cumulative_shock',
                      'pkg1_pre_idx', 'pkg1_post_idx', 'pkg2_pre_idx', 'pkg2_post_idx']
    for col in fill_zero_cols:
        features[col] = features[col].fillna(0)

    features['cv_obs_index'] = features['cv_obs_index'].fillna(100)
    features['cv_max_daily_drop'] = features['cv_max_daily_drop'].fillna(0)
    features['cv_avg_change_rate'] = features['cv_avg_change_rate'].fillna(0)
    features['is_churned'] = features['is_churned'].fillna(1)

    # 파생 피처: 접속 감소율 (전반부 대비 후반부)
    features['login_decline_rate'] = np.where(
        features['early_logins'] > 0,
        (features['early_logins'] - features['late_logins']) / features['early_logins'],
        0
    ).clip(-1, 1)

    print(f"\n  → 최종 피처 데이터: {features.shape[0]:,}행 × {features.shape[1]}열")
    print(f"  → 이탈률: {features['is_churned'].mean()*100:.1f}%")
    print(f"  → 그룹별 이탈률:")
    for g in GROUP_ORDER:
        rate = features[features['club_value_group'] == g]['is_churned'].mean() * 100
        print(f"    {g}: {rate:.1f}%")

    return features


# ============================================================================
# 3. 모델 학습 및 평가
# ============================================================================
def train_and_evaluate(features):
    """
    Random Forest + Logistic Regression 학습 및 비교
    """
    print("\n[3] 모델 학습 및 평가")

    # =========================================================================
    # 피처 선택 — EDA 가설 근거 시각화(fig_a1~a3)에서 증명된 인과관계 기반
    # =========================================================================
    # [순서] EDA 시각화로 가설 증명 → 증명된 관계에서 피처 추출 → ML 모델 학습
    #
    # ★ 가설 핵심 피처 (fig_a1/a2/a3에서 통계적으로 유의한 관계 확인됨):
    #   - fig_a1: OVR 130~134 범위 유저일수록 구단가치 하락 큼
    #     → ovr_pkg_overlap, is_pkg_ovr_range
    #   - fig_a2: 패키지 출시 전후 10~100조 그룹 하락폭 가장 큼 (17.3pt)
    #     → pkg1_cv_change, pkg2_cv_change, pkg_cumulative_shock
    #   - fig_a3: OVR가격하락률 → 구단가치하락률 양의 상관 (추세선 기울기=10.1)
    #     → ovr_price_exposure
    #
    # 추가: 일반적 이탈 예측 지표 (접속 패턴, 구매 행동, 구단가치 변동)
    # =========================================================================
    feature_cols = [
        # ── 유저 프로필 ──
        'club_value_jo',       # 구단가치 (조 단위)
        'avg_ovr',             # 평균 OVR
        'spendig',             # 누적 과금
        'membership_rank',     # 멤버십 등급 (숫자)
        # ── 접속 패턴 ──
        'obs_login_days',      # 관측기간 접속일수
        'obs_avg_session',     # 관측기간 평균 세션
        'early_logins',        # 전반부(1월) 접속수
        'late_logins',         # 후반부(2월) 접속수
        'last_week_logins',    # 관측 마지막주 접속수
        'login_decline_rate',  # 접속 감소율
        # ── 구매 행동 ──
        'obs_pkg_count',       # 관측기간 패키지 구매 횟수
        'obs_pkg_amount',      # 관측기간 총 구매 금액
        'obs_pkg_variety',     # 구매 패키지 종류 수
        'obs_avg_pkg_ovr',     # 평균 구매 OVR
        # ★ 가설 핵심 피처 (EDA fig_a1/a2/a3에서 인과관계 증명됨)
        'ovr_pkg_overlap',     # [fig_a1] 유저OVR↔패키지OVR 겹침도
        'is_pkg_ovr_range',    # [fig_a1] 패키지 높은확률 OVR 범위(130~134) 해당여부
        'pkg1_cv_change',      # [fig_a2] 1차 패키지(1/18) 전후 구단가치 변화율
        'pkg2_cv_change',      # [fig_a2] 2차 패키지(2/20) 전후 구단가치 변화율
        'pkg_cumulative_shock',# [fig_a2] 1차+2차 패키지 누적 충격
        'ovr_price_exposure',  # [fig_a3] 보유 OVR 가격 하락 노출도
        # ── 구단가치 변동 ──
        'cv_obs_index',        # 관측기간 말 구단가치 인덱스
        'cv_max_daily_drop',   # 최대 일일 하락률
        'shock_count',         # z-score 충격 횟수
        'cv_avg_change_rate',  # 평균 일일 변동률
        'cv_decline_pct',      # 구단가치 총 하락률(%)
        # ── 거래 ──
        'obs_trade_count',     # 관측기간 거래 횟수
    ]

    # 피처 이름 한글 매핑 (시각화용)
    feature_names_kr = {
        'club_value_jo': '구단가치(조)',
        'avg_ovr': '평균 OVR',
        'spendig': '누적과금',
        'membership_rank': '멤버십등급',
        'obs_login_days': '접속일수(관측기간)',
        'obs_avg_session': '평균세션(분)',
        'early_logins': '1월접속수',
        'late_logins': '2월접속수',
        'last_week_logins': '마지막주접속',
        'login_decline_rate': '접속감소율',
        'obs_pkg_count': '패키지구매횟수',
        'obs_pkg_amount': '총구매금액',
        'obs_pkg_variety': '패키지종류수',
        'obs_avg_pkg_ovr': '구매패키지OVR',
        'ovr_pkg_overlap': 'OVR↔패키지겹침도★',
        'is_pkg_ovr_range': '패키지OVR범위해당★',
        'pkg1_cv_change': '1차패키지후구단가치변화★',
        'pkg2_cv_change': '2차패키지후구단가치변화★',
        'pkg_cumulative_shock': '패키지누적충격★',
        'ovr_price_exposure': 'OVR가격하락노출도★',
        'cv_obs_index': '구단가치인덱스',
        'cv_max_daily_drop': '최대일일하락률',
        'shock_count': 'Z-score충격횟수',
        'cv_avg_change_rate': '평균일일변동률',
        'cv_decline_pct': '구단가치하락률(%)',
        'obs_trade_count': '거래횟수',
    }

    X = features[feature_cols].copy()
    y = features['is_churned'].astype(int)

    # 학습/테스트 분할 (80/20, 층화추출)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    print(f"  → 학습 세트: {len(X_train):,}개 (이탈 {y_train.mean()*100:.1f}%)")
    print(f"  → 테스트 세트: {len(X_test):,}개 (이탈 {y_test.mean()*100:.1f}%)")

    # 스케일링 (Logistic Regression용)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Random Forest ──
    print("\n  [3-1] Random Forest 학습 중...")
    rf = RandomForestClassifier(
        n_estimators=200,       # 충분한 트리 수
        max_depth=12,           # 과적합 방지
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',  # 클래스 불균형 보정
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    # RF 교차검증
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1')

    print(f"    Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
    print(f"    Precision: {precision_score(y_test, rf_pred):.4f}")
    print(f"    Recall:    {recall_score(y_test, rf_pred):.4f}")
    print(f"    F1:        {f1_score(y_test, rf_pred):.4f}")
    print(f"    AUC:       {roc_auc_score(y_test, rf_prob):.4f}")
    print(f"    CV F1:     {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

    # ── Logistic Regression ──
    print("\n  [3-2] Logistic Regression 학습 중...")
    lr = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=SEED
    )
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

    # LR 교차검증
    lr_cv_scores = cross_val_score(lr, scaler.transform(X), y, cv=cv, scoring='f1')

    print(f"    Accuracy:  {accuracy_score(y_test, lr_pred):.4f}")
    print(f"    Precision: {precision_score(y_test, lr_pred):.4f}")
    print(f"    Recall:    {recall_score(y_test, lr_pred):.4f}")
    print(f"    F1:        {f1_score(y_test, lr_pred):.4f}")
    print(f"    AUC:       {roc_auc_score(y_test, lr_prob):.4f}")
    print(f"    CV F1:     {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

    results = {
        'rf': rf, 'lr': lr, 'scaler': scaler,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'rf_pred': rf_pred, 'rf_prob': rf_prob,
        'lr_pred': lr_pred, 'lr_prob': lr_prob,
        'feature_cols': feature_cols,
        'feature_names_kr': feature_names_kr,
        'rf_cv': rf_cv_scores, 'lr_cv': lr_cv_scores,
    }

    return results


# ============================================================================
# 4. 시각화 생성
# ============================================================================
def save_fig(fig, filename):
    """차트 저장 — 마운트 경로 길이 제한 우회 (짧은 이름 저장 후 rename)"""
    final_path = os.path.join(OUTPUT_PATH, filename)
    tmp_name = os.path.join(OUTPUT_PATH, '_t.png')
    fig.savefig(tmp_name, bbox_inches='tight', facecolor='white')
    try:
        os.remove(final_path)
    except OSError:
        pass
    os.rename(tmp_name, final_path)
    plt.close(fig)
    print(f"  → 저장: {filename}")


def plot_feature_importance(results):
    """Fig 16: Random Forest Feature Importance"""
    print("\n[Fig 16] Feature Importance")

    rf = results['rf']
    feature_cols = results['feature_cols']
    names_kr = results['feature_names_kr']

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))

    # 상위 15개 피처 (가로 바차트)
    top_n = min(15, len(feature_cols))
    top_idx = indices[:top_n][::-1]  # 역순 (아래→위)

    y_pos = range(top_n)
    bars = ax.barh(y_pos, importances[top_idx], color='#2196F3', edgecolor='white')

    # 가장 중요한 피처 강조
    bars[-1].set_color('#e74c3c')
    bars[-2].set_color('#f39c12')
    bars[-3].set_color('#f39c12')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([names_kr.get(feature_cols[i], feature_cols[i])
                        for i in top_idx], fontsize=11)

    # 값 표시
    for bar, val in zip(bars, importances[top_idx]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
    ax.set_title('Random Forest — 이탈 예측 피처 중요도 Top 15', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    save_fig(fig, 'fig_16_feature_importance.png')


def plot_roc_curves(results):
    """Fig 17: ROC 곡선 비교 (RF vs LR)"""
    print("\n[Fig 17] ROC 곡선")

    y_test = results['y_test']

    fig, ax = plt.subplots(figsize=(9, 8))

    # RF ROC
    rf_fpr, rf_tpr, _ = roc_curve(y_test, results['rf_prob'])
    rf_auc = roc_auc_score(y_test, results['rf_prob'])
    ax.plot(rf_fpr, rf_tpr, color='#e74c3c', linewidth=2.5,
            label=f'Random Forest (AUC = {rf_auc:.4f})')

    # LR ROC
    lr_fpr, lr_tpr, _ = roc_curve(y_test, results['lr_prob'])
    lr_auc = roc_auc_score(y_test, results['lr_prob'])
    ax.plot(lr_fpr, lr_tpr, color='#3498db', linewidth=2.5,
            label=f'Logistic Regression (AUC = {lr_auc:.4f})')

    # 대각선 (무작위 분류기)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC 곡선 비교 — Random Forest vs Logistic Regression', fontsize=14)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    save_fig(fig, 'fig_17_roc_curves.png')


def plot_confusion_matrices(results):
    """Fig 18: Confusion Matrix (RF + LR 나란히)"""
    print("\n[Fig 18] Confusion Matrix")

    y_test = results['y_test']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pred, title in [(ax1, results['rf_pred'], 'Random Forest'),
                             (ax2, results['lr_pred'], 'Logistic Regression')]:
        cm = confusion_matrix(y_test, pred)
        # 비율로 표시
        cm_pct = cm / cm.sum() * 100

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['유지', '이탈'], yticklabels=['유지', '이탈'],
                    annot_kws={'fontsize': 14})

        # 비율 추가
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, f'({cm_pct[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')

        ax.set_xlabel('예측', fontsize=12)
        ax.set_ylabel('실제', fontsize=12)
        ax.set_title(f'{title}', fontsize=13)

    fig.suptitle('Confusion Matrix — 이탈 예측 정확도', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_18_confusion_matrix.png')


def plot_model_comparison(results):
    """Fig 19: 모델 성능 비교 (레이더 차트 스타일 → 바차트)"""
    print("\n[Fig 19] 모델 성능 비교")

    y_test = results['y_test']

    metrics = {
        'Accuracy': [accuracy_score(y_test, results['rf_pred']),
                     accuracy_score(y_test, results['lr_pred'])],
        'Precision': [precision_score(y_test, results['rf_pred']),
                      precision_score(y_test, results['lr_pred'])],
        'Recall': [recall_score(y_test, results['rf_pred']),
                   recall_score(y_test, results['lr_pred'])],
        'F1': [f1_score(y_test, results['rf_pred']),
               f1_score(y_test, results['lr_pred'])],
        'AUC': [roc_auc_score(y_test, results['rf_prob']),
                roc_auc_score(y_test, results['lr_prob'])],
        'CV F1': [results['rf_cv'].mean(), results['lr_cv'].mean()],
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    metric_names = list(metrics.keys())
    x = np.arange(len(metric_names))
    width = 0.35

    rf_vals = [metrics[m][0] for m in metric_names]
    lr_vals = [metrics[m][1] for m in metric_names]

    bars1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest',
                   color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x + width/2, lr_vals, width, label='Logistic Regression',
                   color='#3498db', alpha=0.85)

    for bar, val in zip(bars1, rf_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, lr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('모델 성능 비교 — Random Forest vs Logistic Regression', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    save_fig(fig, 'fig_19_model_comparison.png')


def plot_group_accuracy(results, features):
    """Fig 20: 그룹별 예측 정확도"""
    print("\n[Fig 20] 그룹별 예측 정확도")

    # 테스트 세트의 그룹 정보
    test_idx = results['X_test'].index
    test_groups = features.loc[test_idx, 'club_value_group']
    y_test = results['y_test']
    rf_pred = results['rf_pred']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 그룹별 정확도 (RF)
    group_acc = {}
    group_recall = {}
    for g in GROUP_ORDER:
        mask = test_groups == g
        if mask.sum() > 0:
            group_acc[g] = accuracy_score(y_test[mask], rf_pred[mask])
            # 이탈자가 있는 경우만 Recall 계산
            if y_test[mask].sum() > 0:
                group_recall[g] = recall_score(y_test[mask], rf_pred[mask])
            else:
                group_recall[g] = 0

    colors = [COLORS[g] for g in GROUP_ORDER]

    # Accuracy
    bars1 = ax1.bar(GROUP_ORDER, [group_acc.get(g, 0) for g in GROUP_ORDER],
                    color=colors, edgecolor='white')
    for bar, val in zip(bars1, [group_acc.get(g, 0) for g in GROUP_ORDER]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('그룹별 RF 예측 정확도', fontsize=13)
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)

    # Recall (이탈 탐지율)
    bars2 = ax2.bar(GROUP_ORDER, [group_recall.get(g, 0) for g in GROUP_ORDER],
                    color=colors, edgecolor='white')
    for bar, val in zip(bars2, [group_recall.get(g, 0) for g in GROUP_ORDER]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Recall (이탈 탐지율)', fontsize=12)
    ax2.set_title('그룹별 RF 이탈 탐지율', fontsize=13)
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Random Forest — 구단가치 그룹별 예측 성능', fontsize=15,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_20_group_accuracy.png')


def plot_lr_coefficients(results):
    """Fig 21: Logistic Regression 계수 (영향력 방향 포함)"""
    print("\n[Fig 21] LR 계수")

    lr = results['lr']
    feature_cols = results['feature_cols']
    names_kr = results['feature_names_kr']

    coefs = lr.coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))

    top_n = min(15, len(feature_cols))
    top_idx = sorted_idx[:top_n][::-1]

    y_pos = range(top_n)
    colors_bar = ['#e74c3c' if coefs[i] > 0 else '#3498db' for i in top_idx]

    bars = ax.barh(y_pos, coefs[top_idx], color=colors_bar, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names_kr.get(feature_cols[i], feature_cols[i])
                        for i in top_idx], fontsize=11)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('계수 (양수=이탈 증가, 음수=이탈 감소)', fontsize=12)
    ax.set_title('Logistic Regression — 이탈 예측 계수 (상위 15개)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='이탈 증가 요인'),
                       Patch(facecolor='#3498db', label='이탈 감소 요인')]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower right')

    save_fig(fig, 'fig_21_lr_coefficients.png')


def print_classification_reports(results):
    """분류 리포트 출력 및 저장"""
    print("\n[4] Classification Report")

    y_test = results['y_test']

    print("\n── Random Forest ──")
    print(classification_report(y_test, results['rf_pred'],
                                target_names=['유지', '이탈']))

    print("\n── Logistic Regression ──")
    print(classification_report(y_test, results['lr_pred'],
                                target_names=['유지', '이탈']))

    # 리포트 파일 저장
    report_path = os.path.join(MODEL_PATH, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FC Online 4 — ML 이탈 예측 모델 성능 리포트\n")
        f.write("=" * 60 + "\n\n")
        f.write("■ 피처 선택 근거\n")
        f.write("  EDA 가설 근거 시각화(fig_a1~a3)에서 증명된 인과관계 기반:\n")
        f.write("  - fig_a1: 유저 OVR이 패키지 높은확률 OVR(130~134)과 겹칠수록 구단가치 하락\n")
        f.write("  - fig_a2: 패키지 출시 전후 10~100조 그룹 하락폭 최대 (17.3pt)\n")
        f.write("  - fig_a3: OVR가격하락 → 구단가치하락 → 이탈 인과 체인 확인\n")
        f.write("  → 위 근거로 6개 가설 핵심 피처 + 일반 이탈 지표 선정\n\n")
        f.write("■ 데이터 누출 방지: 시간 분할\n")
        f.write("  관측 기간 (피처): 2025-01-01 ~ 2025-03-01\n")
        f.write("  결과 기간 (타겟): 2025-03-01 ~ 2025-04-30\n\n")

        f.write("── Random Forest ──\n")
        f.write(classification_report(y_test, results['rf_pred'],
                                      target_names=['유지', '이탈']))
        f.write(f"\nAUC: {roc_auc_score(y_test, results['rf_prob']):.4f}\n")
        f.write(f"CV F1: {results['rf_cv'].mean():.4f} ± {results['rf_cv'].std():.4f}\n")

        f.write("\n\n── Logistic Regression ──\n")
        f.write(classification_report(y_test, results['lr_pred'],
                                      target_names=['유지', '이탈']))
        f.write(f"\nAUC: {roc_auc_score(y_test, results['lr_prob']):.4f}\n")
        f.write(f"CV F1: {results['lr_cv'].mean():.4f} ± {results['lr_cv'].std():.4f}\n")

        # 피처 중요도 Top 10
        f.write("\n\n── Random Forest Feature Importance Top 10 ──\n")
        importances = results['rf'].feature_importances_
        indices = np.argsort(importances)[::-1]
        names_kr = results['feature_names_kr']
        feature_cols = results['feature_cols']
        for rank, i in enumerate(indices[:10], 1):
            name = names_kr.get(feature_cols[i], feature_cols[i])
            f.write(f"  {rank}. {name}: {importances[i]:.4f}\n")

    print(f"  → 리포트 저장: {report_path}")


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    # 1. 데이터 로드
    up, ll, pp, tm, dcv = load_data()

    # 2. 피처 엔지니어링
    features = engineer_features(up, ll, pp, tm, dcv)

    # 3. 모델 학습 및 평가
    results = train_and_evaluate(features)

    # 4. 시각화 생성
    plot_feature_importance(results)
    plot_roc_curves(results)
    plot_confusion_matrices(results)
    plot_model_comparison(results)
    plot_group_accuracy(results, features)
    plot_lr_coefficients(results)

    # 5. 리포트 출력
    print_classification_reports(results)

    # 6. 모델 및 파이프라인 저장 (Stage 4에서 로드하여 예측에 사용)
    save_model_pipeline(results, features)

    print("\n" + "=" * 70)
    print("Stage 3 완료! ML 이탈 예측 모델 학습 및 시각화 생성")
    print(f"차트: fig_16 ~ fig_21 ({OUTPUT_PATH})")
    print(f"리포트: {MODEL_PATH}/classification_report.txt")
    print(f"모델: {MODEL_PATH}/rf_model.pkl, scaler.pkl, feature_cols.pkl")
    print("=" * 70)


# ============================================================================
# 5-1. 모델 저장 (Stage 4 시나리오 시뮬레이션에서 사용)
# ============================================================================
def save_model_pipeline(results, features):
    """
    학습된 모델 파이프라인을 저장하여 Stage 4에서 로드 가능하게 함

    저장 항목:
      - rf_model.pkl      : 학습된 Random Forest 모델
      - scaler.pkl         : StandardScaler (LR용)
      - feature_cols.pkl   : 피처 컬럼 목록 (동일 순서 보장)
      - lr_model.pkl       : 학습된 Logistic Regression 모델
      - features_base.pkl  : 전체 유저 피처 데이터 (시나리오별 피처 재계산의 베이스)
    """
    import joblib

    print("\n[5-1] 모델 파이프라인 저장 (Stage 4 시나리오 시뮬레이션용)")

    # RF 모델 저장
    joblib.dump(results['rf'], os.path.join(MODEL_PATH, 'rf_model.pkl'))
    print(f"  → RF 모델 저장: rf_model.pkl")

    # LR 모델 저장
    joblib.dump(results['lr'], os.path.join(MODEL_PATH, 'lr_model.pkl'))
    print(f"  → LR 모델 저장: lr_model.pkl")

    # 스케일러 저장
    joblib.dump(results['scaler'], os.path.join(MODEL_PATH, 'scaler.pkl'))
    print(f"  → 스케일러 저장: scaler.pkl")

    # 피처 컬럼 목록 저장 (순서 유지 중요)
    joblib.dump(results['feature_cols'], os.path.join(MODEL_PATH, 'feature_cols.pkl'))
    print(f"  → 피처 목록 저장: feature_cols.pkl ({len(results['feature_cols'])}개 피처)")

    # 전체 유저 피처 데이터 저장 (시나리오별 피처 재계산의 베이스로 활용)
    # Stage 4에서 OVR을 바꿀 때 ovr_pkg_overlap, is_pkg_ovr_range 등만 재계산하고
    # 나머지 피처(접속패턴, 구매행동 등)는 이 데이터를 그대로 사용
    # 중복 컬럼 제거 (avg_ovr이 feature_cols에도 있을 수 있음)
    extra_cols = ['user_id', 'club_value_group', 'avg_ovr',
                  'monthly_membership_fee', 'monthly_avg_spending']
    base_cols = extra_cols + [c for c in results['feature_cols'] if c not in extra_cols]
    # monthly_membership_fee, monthly_avg_spending이 없으면 user_profile에서 머지
    merge_cols = []
    for col in ['monthly_membership_fee', 'monthly_avg_spending']:
        if col not in features.columns:
            merge_cols.append(col)
    if merge_cols:
        user_profile = pd.read_csv(os.path.join(DATA_PATH, 'user_profile.csv'))
        features = features.merge(
            user_profile[['user_id'] + merge_cols], on='user_id', how='left')
    features_base = features[base_cols].copy()
    joblib.dump(features_base, os.path.join(MODEL_PATH, 'features_base.pkl'))
    print(f"  → 베이스 피처 저장: features_base.pkl ({len(features_base):,}명)")

    print("  ✅ 모델 파이프라인 저장 완료 → Stage 4에서 로드하여 예측에 사용")


if __name__ == '__main__':
    main()
