"""
=============================================================================
FC Online 4 데이터 분석 프로젝트 - Stage 4: ML 기반 패키지 OVR 시나리오 시뮬레이션
=============================================================================
목적:
  Stage 3에서 학습·저장한 ML 이탈 예측 모델(Random Forest)을 로드하여,
  패키지의 평균 OVR을 조절했을 때 유저별 이탈 확률이 어떻게 변하는지
  직접 predict_proba()로 예측 → 최적 패키지 OVR 제안

핵심 로직:
  1. Stage 3 저장 파이프라인 로드 (RF 모델 + 스케일러 + 피처목록 + 유저 피처)
  2. 각 OVR 시나리오마다 "제어 가능한 피처"만 재계산:
     - ovr_pkg_overlap: 유저OVR ↔ 새 패키지OVR 중심 거리 기반 겹침도
     - is_pkg_ovr_range: 새 패키지 높은확률 OVR 범위 해당 여부
  3. 나머지 피처(접속패턴, 구매행동, 구단가치변동 등)는 관측 데이터 그대로 유지
  4. model.predict_proba() → 유저별 이탈 확률 → 그룹별/전체 이탈률 집계
  5. 매출/손실 계산 → 최적 OVR 제안

배경 (EDA에서 증명됨):
  - fig_a1: 유저 OVR이 패키지 높은확률 OVR(130~134)과 겹칠수록 구단가치 하락 큼
  - fig_a2: 패키지 출시 전후 10~100조 그룹 하락폭 가장 큼 (17.3pt)
  - fig_a3: OVR가격하락 → 구단가치하락 → 이탈 인과체인 (r=0.872)
  - fig_12/12b: 이탈 손실(135.5억) vs 총 기대수익(502.8억, 패키지+잔존유저)

출력:
  - fig_22: 시나리오별 그룹 이탈률 비교 (Grouped Bar)
  - fig_23: 시나리오별 총 이탈률 + 매출/손실 비교
  - fig_24: 최적 OVR 중심점 탐색 곡선
  - fig_25: 최종 액션 아이템 요약 대시보드
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import joblib

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

# 이탈 손실 계산 기간
LOSS_MONTHS = 8  # 이탈 유저의 예상 미래 손실 기간 (개월)
# 월 기대수익은 user_profile의 monthly_membership_fee + monthly_avg_spending에서 직접 참조
# → load_model_pipeline()에서 features_base에 포함

# 패키지 매출 가우시안 파라미터
# 패키지 매력도는 OVR 133 부근에서 최대 (대다수 유저가 원하는 OVR대)
# 너무 낮으면 → 안 사고, 너무 높으면 → 비싸서 못 삼
REVENUE_CENTER = 133  # 매출 최대 OVR
REVENUE_SIGMA = 4.0   # 매출 감소 속도
BASE_PKG_REVENUE = 27.4   # 패키지 판매 매출 최대치 (억)
RETENTION_MONTHS = 8  # 잔존 유저 기대수익 산정 기간 (손실과 동일)

print("=" * 70)
print("FC Online 4 - Stage 4: ML 기반 패키지 OVR 시나리오 시뮬레이션")
print("=" * 70)


# ============================================================================
# 1. Stage 3 모델 파이프라인 로드
# ============================================================================
def load_model_pipeline():
    """
    Stage 3에서 저장한 모델 파이프라인을 로드
    - RF 모델, 스케일러, 피처 목록, 유저 베이스 피처
    """
    print("\n[1] ML 모델 파이프라인 로드 (Stage 3 산출물)")

    rf_model = joblib.load(os.path.join(MODEL_PATH, 'rf_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_PATH, 'feature_cols.pkl'))
    features_base = joblib.load(os.path.join(MODEL_PATH, 'features_base.pkl'))

    print(f"  → RF 모델 로드 완료 (n_estimators={rf_model.n_estimators})")
    print(f"  → 피처 목록: {len(feature_cols)}개")
    print(f"  → 유저 베이스 피처: {len(features_base):,}명")

    return rf_model, scaler, feature_cols, features_base


# ============================================================================
# 2. 시나리오 정의
# ============================================================================
def define_scenarios():
    """
    패키지 OVR 중심점을 변경한 시나리오 정의
    각 시나리오는 패키지에서 높은 확률로 등장하는 OVR 범위를 변경
    """
    scenarios = {
        '현행 (130~134)': {
            'center': 132, 'range': (130, 134),
            'description': '현재 패키지 설정 (OVR 130~134 중심)',
            'color': '#e74c3c',
        },
        '하향A (126~130)': {
            'center': 128, 'range': (126, 130),
            'description': '저가 선수 중심으로 대폭 하향',
            'color': '#3498db',
        },
        '소폭하향B (128~132)': {
            'center': 130, 'range': (128, 132),
            'description': '소폭 하향 (현행 대비 OVR -2)',
            'color': '#2ecc71',
        },
        '상향C (134~138)': {
            'center': 136, 'range': (134, 138),
            'description': '고가 선수 중심으로 상향',
            'color': '#f39c12',
        },
        '대폭상향D (136~140)': {
            'center': 138, 'range': (136, 140),
            'description': '최고가 선수 중심으로 대폭 상향',
            'color': '#9b59b6',
        },
    }
    return scenarios


# ============================================================================
# 3. ML 기반 시뮬레이션 엔진
# ============================================================================
def recalculate_scenario_features(features_base, feature_cols, pkg_center, pkg_range):
    """
    새로운 패키지 OVR 시나리오에 맞춰 "제어 가능한 피처"만 재계산

    변경되는 피처 (패키지 OVR에 의존):
      - ovr_pkg_overlap: 유저OVR ↔ 새 패키지OVR 거리 기반 겹침도
      - is_pkg_ovr_range: 새 패키지 높은확률 OVR 범위 해당 여부

    유지되는 피처 (관측 데이터 기반 — 시나리오와 무관):
      - 접속 패턴, 구매 행동, 구단가치 변동, 거래 활동 등
      - pkg1/2_cv_change, pkg_cumulative_shock (이미 발생한 과거 데이터)
      - ovr_price_exposure (관측 기간 내 실제 가격 변동)

    Args:
        features_base: Stage 3에서 저장한 전체 유저 피처 DataFrame
        feature_cols: 모델이 사용하는 피처 컬럼 순서
        pkg_center: 새 패키지 OVR 중심점
        pkg_range: (min_ovr, max_ovr) 새 패키지 높은확률 OVR 범위
    Returns:
        X_scenario: 시나리오용 피처 DataFrame (feature_cols 순서)
    """
    X = features_base[feature_cols].copy()

    # ── (A) ovr_pkg_overlap 재계산 ──
    # 유저 avg_ovr과 새 패키지 OVR 중심 간 거리 → 겹침도
    # Stage 3 원본: 1.0 / (1.0 + abs(avg_ovr - PKG_CENTER_OVR))
    ovr_distance = abs(features_base['avg_ovr'] - pkg_center)
    X['ovr_pkg_overlap'] = 1.0 / (1.0 + ovr_distance)

    # ── (B) is_pkg_ovr_range 재계산 ──
    # 유저 avg_ovr이 새 패키지 높은확률 OVR 범위에 해당하는지
    # Stage 3 원본: avg_ovr.between(129.5, 134.5)
    X['is_pkg_ovr_range'] = features_base['avg_ovr'].between(
        pkg_range[0] - 0.5, pkg_range[1] + 0.5
    ).astype(int)

    return X


def simulate_scenario_ml(rf_model, features_base, feature_cols, scenario_config):
    """
    ML 모델(RF)로 시나리오별 유저 이탈 확률을 직접 예측

    프로세스:
      1. 시나리오에 맞게 피처 재계산
      2. rf_model.predict_proba() → 유저별 이탈 확률
      3. 그룹별 평균 이탈률 + 이탈 유저 수 집계
    """
    pkg_center = scenario_config['center']
    pkg_range = scenario_config['range']

    # 시나리오 피처 재계산
    X_scenario = recalculate_scenario_features(
        features_base, feature_cols, pkg_center, pkg_range)

    # ML 모델로 유저별 이탈 확률 예측
    churn_proba = rf_model.predict_proba(X_scenario)[:, 1]

    # 그룹별 결과 집계
    results = {}
    for group in GROUP_ORDER:
        mask = features_base['club_value_group'] == group
        group_proba = churn_proba[mask]
        n_users = mask.sum()

        if n_users == 0:
            results[group] = {'churn_rate': 0, 'n_churned': 0, 'n_users': 0}
            continue

        # 그룹 평균 이탈 확률 = 예상 이탈률
        avg_churn_rate = group_proba.mean() * 100
        n_churned = int(n_users * group_proba.mean())

        results[group] = {
            'churn_rate': avg_churn_rate,
            'n_churned': n_churned,
            'n_users': n_users,
            'avg_proba': group_proba.mean(),
            'median_proba': np.median(group_proba),
        }

    # 전체 이탈률
    total_churned = sum(r['n_churned'] for r in results.values())
    total_users = sum(r['n_users'] for r in results.values())
    results['전체'] = {
        'churn_rate': churn_proba.mean() * 100,
        'n_churned': total_churned,
        'n_users': total_users,
    }

    return results


def calculate_revenue_loss(scenario_results, pkg_center, features_base):
    """
    시나리오별 총 기대수익 vs 이탈 손실 계산

    총 기대수익 = 패키지 판매 매출 + 잔존(비이탈) 유저의 향후 기대수익
      - 패키지 매출: OVR 133 중심 Gaussian bell curve
      - 잔존 수익: 이탈하지 않은 유저의 (멤버십+과금) × 8개월
    이탈 손실 = 이탈 유저 × (월 멤버십 + 월 과금) × 8개월

    → 순영향 = 총 기대수익 - 이탈 손실
    → OVR 최적화는 "이탈을 줄여 잔존 수익을 높이면서 패키지 매출도 유지"하는 균형점
    """
    # (1) 패키지 판매 매출: 종 모양 곡선 (OVR 133에서 최대)
    rev_factor = np.exp(-0.5 * ((pkg_center - REVENUE_CENTER) / REVENUE_SIGMA) ** 2)
    pkg_revenue = BASE_PKG_REVENUE * rev_factor

    # (2) 잔존 유저 기대수익: 이탈하지 않은 유저의 향후 멤버십+과금 수익
    retention_revenue = 0
    total_loss = 0
    for group in GROUP_ORDER:
        group_mask = features_base['club_value_group'] == group
        n_total = group_mask.sum()
        n_churned = scenario_results[group]['n_churned']
        n_retained = n_total - n_churned

        # 그룹별 평균 월 기대수익 (데이터 기반)
        avg_fee = features_base.loc[group_mask, 'monthly_membership_fee'].mean()
        avg_spending = features_base.loc[group_mask, 'monthly_avg_spending'].mean() \
            if 'monthly_avg_spending' in features_base.columns else 0
        monthly_rev = avg_fee + avg_spending

        # 잔존 유저 수익 (억 단위)
        retention_revenue += n_retained * monthly_rev * RETENTION_MONTHS / 1e8
        # 이탈 유저 손실 (억 단위)
        total_loss += n_churned * monthly_rev * LOSS_MONTHS / 1e8

    # 총 기대수익 = 패키지 매출 + 잔존 유저 수익
    total_revenue = pkg_revenue + retention_revenue

    return total_revenue, total_loss


# ============================================================================
# 4. 전체 시나리오 시뮬레이션 실행
# ============================================================================
def run_all_scenarios(rf_model, features_base, feature_cols):
    """모든 시나리오에 대해 ML 모델 기반 이탈률 예측 및 결과 집계"""
    print("\n[2] ML 모델 기반 시나리오 시뮬레이션 실행")

    scenarios = define_scenarios()
    all_results = {}

    for name, config in scenarios.items():
        print(f"\n  ▶ {name}: {config['description']}")

        # ML 모델로 이탈 확률 예측
        sim_results = simulate_scenario_ml(
            rf_model, features_base, feature_cols, config)

        # 매출/손실 계산
        revenue, loss = calculate_revenue_loss(sim_results, config['center'], features_base)

        all_results[name] = {
            'config': config,
            'sim_results': sim_results,
            'revenue': revenue,
            'loss': loss,
            'net_impact': revenue - loss,
        }

        print(f"    전체 이탈률: {sim_results['전체']['churn_rate']:.1f}% "
              f"(ML predict_proba 기반)")
        print(f"    매출: {revenue:.1f}억 / 손실: {loss:.1f}억 / 순영향: {revenue-loss:+.1f}억")

        for g in GROUP_ORDER:
            r = sim_results[g]
            print(f"    {g}: 이탈률 {r['churn_rate']:.1f}%, "
                  f"이탈 {r['n_churned']:,}명")

    return all_results


# ============================================================================
# 5. 최적 OVR 탐색 (연속 ML 예측)
# ============================================================================
def find_optimal_ovr(rf_model, features_base, feature_cols):
    """
    패키지 OVR 중심점을 125~142까지 1단위로 변경하며
    ML 모델로 이탈률 예측 → 매출/손실/순영향 계산 → 최적점 탐색
    """
    print("\n[3] ML 기반 최적 OVR 중심점 탐색")

    ovr_range = np.arange(125, 143, 1)
    results_list = []

    for center in ovr_range:
        pkg_range = (center - 2, center + 2)
        config = {'center': center, 'range': pkg_range}

        # ML 모델로 예측
        sim = simulate_scenario_ml(rf_model, features_base, feature_cols, config)

        # 매출/손실 계산
        revenue, loss = calculate_revenue_loss(sim, center, features_base)

        results_list.append({
            'ovr_center': center,
            'total_churn_rate': sim['전체']['churn_rate'],
            'revenue': revenue,
            'loss': loss,
            'net_impact': revenue - loss,
            'churn_0_10': sim['0~10조']['churn_rate'],
            'churn_10_100': sim['10~100조']['churn_rate'],
            'churn_100_1000': sim['100~1000조']['churn_rate'],
            'churn_1000_plus': sim['1000조이상']['churn_rate'],
        })

    df = pd.DataFrame(results_list)

    # 최적점: 순영향(매출-손실)이 최대인 OVR
    optimal_idx = df['net_impact'].idxmax()
    optimal_ovr = df.loc[optimal_idx, 'ovr_center']

    print(f"\n  ★ 최적 패키지 OVR 중심점: {optimal_ovr}")
    print(f"    이탈률: {df.loc[optimal_idx, 'total_churn_rate']:.1f}% (ML 예측)")
    print(f"    매출: {df.loc[optimal_idx, 'revenue']:.1f}억")
    print(f"    손실: {df.loc[optimal_idx, 'loss']:.1f}억")
    print(f"    순영향: {df.loc[optimal_idx, 'net_impact']:+.1f}억")

    return df, optimal_ovr


# ============================================================================
# 6. 시각화
# ============================================================================
def save_fig(fig, filename):
    """차트 저장 — 마운트 경로 길이 제한 우회 (짧은 이름 저장 후 rename)"""
    final_path = os.path.join(OUTPUT_PATH, filename)
    tmp_name = os.path.join(OUTPUT_PATH, '_t.png')
    fig.savefig(tmp_name, bbox_inches='tight', facecolor='white', dpi=150)
    try:
        os.remove(final_path)
    except OSError:
        pass
    os.rename(tmp_name, final_path)
    plt.close(fig)
    print(f"  → 저장: {filename}")


def fig_22_scenario_churn_comparison(all_results):
    """Fig 22: 시나리오별 그룹 이탈률 비교 (Grouped Bar) — ML 예측 기반"""
    print("\n[Fig 22] 시나리오별 그룹 이탈률 비교 (ML 예측)")

    scenario_names = list(all_results.keys())
    n_scenarios = len(scenario_names)
    n_groups = len(GROUP_ORDER)

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(n_groups)
    width = 0.15

    for i, name in enumerate(scenario_names):
        churn_rates = [all_results[name]['sim_results'][g]['churn_rate']
                       for g in GROUP_ORDER]
        offset = (i - n_scenarios / 2 + 0.5) * width
        bars = ax.bar(x + offset, churn_rates, width,
                      label=name,
                      color=all_results[name]['config']['color'],
                      alpha=0.85, edgecolor='white')

        # 현행 시나리오만 값 표시
        if '현행' in name:
            for bar, val in zip(bars, churn_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold',
                        color='red')

    ax.set_xlabel('구단가치 그룹', fontsize=12)
    ax.set_ylabel('ML 예측 이탈률 (%)', fontsize=12)
    ax.set_title('패키지 OVR 시나리오별 그룹 이탈률 비교 (RF model.predict_proba 기반)\n'
                 '→ OVR 중심을 하향 조절하면 10~100조 그룹 이탈률 감소',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=11)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)

    # 핵심 메시지
    ax.text(0.02, 0.95,
            '▼ 패키지 OVR을 하향하면 10~100조 그룹 이탈 감소\n'
            '▲ 패키지 OVR을 상향하면 100~1000조/1000조이상 이탈 증가\n'
            '※ Random Forest predict_proba() 기반 예측',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            va='top')

    save_fig(fig, 'fig_22_scenario_churn.png')


def fig_23_revenue_loss_comparison(all_results):
    """Fig 23: 시나리오별 매출/손실/순영향 비교"""
    print("\n[Fig 23] 시나리오별 매출 vs 손실 비교")

    scenario_names = list(all_results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # (좌) 매출 vs 손실 비교
    x = np.arange(len(scenario_names))
    width = 0.35

    revenues = [all_results[n]['revenue'] for n in scenario_names]
    losses = [all_results[n]['loss'] for n in scenario_names]

    bars1 = ax1.bar(x - width/2, revenues, width, label='패키지 매출',
                    color='#2ecc71', alpha=0.85)
    bars2 = ax1.bar(x + width/2, losses, width, label='이탈 손실',
                    color='#e74c3c', alpha=0.85)

    for bar, val in zip(bars1, revenues):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}', ha='center', fontsize=9)

    ax1.set_ylabel('금액 (억 원)', fontsize=12)
    ax1.set_title('시나리오별 매출 vs 이탈 손실', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.split(' ')[0] for n in scenario_names],
                        fontsize=9, rotation=15)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # (우) 순영향 (매출 - 손실)
    net_impacts = [all_results[n]['net_impact'] for n in scenario_names]
    bar_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in net_impacts]

    bars3 = ax2.bar(x, net_impacts, 0.6, color=bar_colors, alpha=0.85,
                    edgecolor='white')

    for bar, val in zip(bars3, net_impacts):
        y_pos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.5
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:+.1f}억', ha='center', fontsize=10, fontweight='bold')

    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('순영향 (매출 - 손실, 억 원)', fontsize=12)
    ax2.set_title('시나리오별 순영향 (ML 예측 기반)\n(양수 = 이득, 음수 = 손해)',
                  fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.split(' ')[0] for n in scenario_names],
                        fontsize=9, rotation=15)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('패키지 OVR 조절에 따른 수익성 분석 (ML 이탈 예측 기반)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_23_revenue_loss_scenario.png')


def fig_24_optimal_ovr_curve(optimal_df, optimal_ovr):
    """Fig 24: 최적 OVR 탐색 곡선 — ML 예측 기반"""
    print("\n[Fig 24] 최적 OVR 탐색 곡선 (ML 예측)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    df = optimal_df

    # (상) 이탈률 곡선 — 그룹별 (ML 예측)
    ax1.plot(df['ovr_center'], df['churn_10_100'], '-o', color=COLORS['10~100조'],
             linewidth=2.5, markersize=6, label='10~100조 (핵심 그룹)')
    ax1.plot(df['ovr_center'], df['churn_0_10'], '-s', color=COLORS['0~10조'],
             linewidth=1.5, markersize=4, label='0~10조')
    ax1.plot(df['ovr_center'], df['churn_100_1000'], '-^', color=COLORS['100~1000조'],
             linewidth=1.5, markersize=4, label='100~1000조')
    ax1.plot(df['ovr_center'], df['churn_1000_plus'], '-d', color=COLORS['1000조이상'],
             linewidth=1.5, markersize=4, label='1000조이상')

    # 현행 위치 표시
    ax1.axvline(132, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.text(132.2, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 25,
             '현행\n(132)', color='red', fontsize=10, fontweight='bold')

    # 최적 위치 표시
    ax1.axvline(optimal_ovr, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.text(optimal_ovr + 0.2, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 20,
             f'최적\n({int(optimal_ovr)})', color='green', fontsize=10, fontweight='bold')

    ax1.set_ylabel('ML 예측 이탈률 (%)', fontsize=12)
    ax1.set_title('패키지 OVR 중심점에 따른 그룹별 이탈률 변화 (RF predict_proba 기반)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # (하) 순영향 곡선
    ax2.fill_between(df['ovr_center'], df['net_impact'], 0,
                     where=(df['net_impact'] >= 0),
                     color='#2ecc71', alpha=0.3, label='순이익 구간')
    ax2.fill_between(df['ovr_center'], df['net_impact'], 0,
                     where=(df['net_impact'] < 0),
                     color='#e74c3c', alpha=0.3, label='순손실 구간')
    ax2.plot(df['ovr_center'], df['net_impact'], '-o', color='#2c3e50',
             linewidth=2.5, markersize=6)

    # 매출선과 손실선
    ax2.plot(df['ovr_center'], df['revenue'], '--', color='#2ecc71',
             linewidth=1.5, alpha=0.7, label='매출')
    ax2.plot(df['ovr_center'], df['loss'], '--', color='#e74c3c',
             linewidth=1.5, alpha=0.7, label='손실')

    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axvline(132, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.axvline(optimal_ovr, color='green', linestyle='--', alpha=0.5, linewidth=1.5)

    # 최적점 강조
    opt_row = df[df['ovr_center'] == optimal_ovr].iloc[0]
    ax2.scatter(optimal_ovr, opt_row['net_impact'], s=200, color='green',
                zorder=5, edgecolors='black', linewidths=2)
    ax2.annotate(f'최적: OVR {int(optimal_ovr)}\n순영향: {opt_row["net_impact"]:+.1f}억',
                 xy=(optimal_ovr, opt_row['net_impact']),
                 xytext=(20, 20), textcoords='offset points',
                 fontsize=11, fontweight='bold', color='green',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax2.set_xlabel('패키지 OVR 중심점', fontsize=12)
    ax2.set_ylabel('금액 (억 원)', fontsize=12)
    ax2.set_title('패키지 OVR 중심점에 따른 매출·손실·순영향 (ML 기반)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig_24_optimal_ovr_curve.png')


def fig_25_action_item_dashboard(all_results, optimal_df, optimal_ovr):
    """Fig 25: 최종 액션 아이템 요약 대시보드"""
    print("\n[Fig 25] 최종 액션 아이템 대시보드")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── (1) 현행 vs 최적 이탈률 비교 ──
    ax1 = fig.add_subplot(gs[0, 0])
    current = all_results['현행 (130~134)']

    current_rates = [current['sim_results'][g]['churn_rate'] for g in GROUP_ORDER]

    # 최적 시나리오 결과 (optimal_df에서)
    opt_row = optimal_df[optimal_df['ovr_center'] == optimal_ovr].iloc[0]
    optimal_rates = [opt_row['churn_0_10'], opt_row['churn_10_100'],
                     opt_row['churn_100_1000'], opt_row['churn_1000_plus']]

    x = np.arange(len(GROUP_ORDER))
    width = 0.35
    ax1.bar(x - width/2, current_rates, width, label='현행 (132)', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, optimal_rates, width, label=f'제안 ({int(optimal_ovr)})',
            color='#2ecc71', alpha=0.8)

    for i, (c, o) in enumerate(zip(current_rates, optimal_rates)):
        diff = o - c
        color = '#2ecc71' if diff < 0 else '#e74c3c'
        ax1.text(i, max(c, o) + 0.5, f'{diff:+.1f}%p', ha='center',
                 fontsize=9, fontweight='bold', color=color)

    ax1.set_ylabel('ML 예측 이탈률 (%)')
    ax1.set_title('그룹별 이탈률: 현행 vs 제안\n(RF predict_proba 기반)',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['0~10조', '10~100조', '100~1000조', '1000조+'], fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ── (2) 매출/손실 비교 ──
    ax2 = fig.add_subplot(gs[0, 1])

    current_rev = current['revenue']
    current_loss = current['loss']
    opt_rev = opt_row['revenue']
    opt_loss = opt_row['loss']

    categories = ['현행\n매출', '현행\n손실', '제안\n매출', '제안\n손실']
    values = [current_rev, current_loss, opt_rev, opt_loss]
    colors_bar = ['#27ae60', '#c0392b', '#2ecc71', '#e74c3c']

    bars = ax2.bar(categories, values, color=colors_bar, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}억', ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('금액 (억 원)')
    ax2.set_title('매출/손실 비교', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # ── (3) 핵심 KPI 변화 ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    current_total_churn = current['sim_results']['전체']['churn_rate']
    optimal_total_churn = opt_row['total_churn_rate']
    churn_diff = optimal_total_churn - current_total_churn

    net_current = current_rev - current_loss
    net_optimal = opt_rev - opt_loss
    net_diff = net_optimal - net_current

    kpi_text = (
        f"━━━ 핵심 KPI 변화 ━━━\n\n"
        f"패키지 OVR 중심점\n"
        f"  현행: 132  →  제안: {int(optimal_ovr)}\n\n"
        f"전체 이탈률 (ML 예측)\n"
        f"  {current_total_churn:.1f}%  →  {optimal_total_churn:.1f}%  ({churn_diff:+.1f}%p)\n\n"
        f"10~100조 이탈률 (핵심)\n"
        f"  {current['sim_results']['10~100조']['churn_rate']:.1f}%  →  "
        f"{opt_row['churn_10_100']:.1f}%\n\n"
        f"순영향 (매출-손실)\n"
        f"  {net_current:+.1f}억  →  {net_optimal:+.1f}억  ({net_diff:+.1f}억)"
    )

    ax3.text(0.1, 0.95, kpi_text, transform=ax3.transAxes,
             fontsize=11, va='top',
             bbox=dict(boxstyle='round', facecolor='#f0f8ff', edgecolor='#3498db',
                       alpha=0.9))
    ax3.set_title('핵심 지표 변화', fontsize=12, fontweight='bold')

    # ── (4) 인과 체인 요약 (텍스트) ──
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    summary_text = (
        "■ 분석 결론 및 액션 아이템\n\n"
        "  [인과 체인] 패키지 OVR 확률분포 → 해당 OVR대 선수 공급과잉 → 선수 가격 하락 → 구단가치 하락 → 유저 이탈\n\n"
        "  [핵심 발견]\n"
        "  • EDA(fig_a1~a3): 유저 보유 OVR이 패키지 높은확률 OVR(130~134)과 겹칠수록 구단가치 하락이 크고, 10~100조 그룹이 가장 민감 (r=0.872)\n"
        "  • ML 모델: OVR 겹침도·패키지 충격·구단가치 하락률이 이탈 예측의 유의한 피처로 확인 (RF AUC=0.73)\n"
        "  • 시나리오 시뮬레이션: 학습된 RF 모델의 predict_proba()로 유저별 이탈 확률 직접 예측 → 시나리오별 비교\n"
        "  • 현행 패키지: 총 기대수익(패키지+잔존유저) 502.8억 vs 이탈 손실 135.5억 → 순영향 +367.2억\n\n"
        f"  [제안] 패키지 높은확률 OVR 중심을 132 → {int(optimal_ovr)}로 조절\n"
        f"  • 10~100조 그룹(전체 54%)의 OVR 겹침도 감소 → 이탈 확률 저하 (ML 예측)\n"
        f"  • 순영향: {net_current:+.1f}억 → {net_optimal:+.1f}억 (개선 {net_diff:+.1f}억)\n"
        f"  • 패키지 매출 소폭 감소를 감수하되, 이탈 방지를 통한 장기 수익 확보\n\n"
        "  [향후 과제]\n"
        "  • 실제 운영 데이터로 ML 예측 결과 검증 (A/B 테스트)\n"
        "  • 패키지 OVR 분포를 균등화하는 방안 검토 (특정 OVR 쏠림 방지)\n"
        "  • 이탈 고위험 유저 대상 맞춤 리텐션 프로그램 설계"
    )

    ax4.text(0.02, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, va='top',
             bbox=dict(boxstyle='round', facecolor='#fffde7', edgecolor='#f57f17',
                       alpha=0.9, linewidth=2))

    fig.suptitle('FC Online 4 — ML 기반 패키지 OVR 최적화 제안 대시보드',
                 fontsize=16, fontweight='bold', y=1.01)

    save_fig(fig, 'fig_25_action_dashboard.png')


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    # 1. ML 모델 파이프라인 로드
    rf_model, scaler, feature_cols, features_base = load_model_pipeline()

    # 2. 시나리오 시뮬레이션 (ML 예측)
    all_results = run_all_scenarios(rf_model, features_base, feature_cols)

    # 3. 최적 OVR 탐색 (ML 예측)
    optimal_df, optimal_ovr = find_optimal_ovr(rf_model, features_base, feature_cols)

    # 4. 시각화 생성
    fig_22_scenario_churn_comparison(all_results)
    fig_23_revenue_loss_comparison(all_results)
    fig_24_optimal_ovr_curve(optimal_df, optimal_ovr)
    fig_25_action_item_dashboard(all_results, optimal_df, optimal_ovr)

    print("\n" + "=" * 70)
    print("Stage 4 완료! ML 기반 패키지 OVR 시나리오 시뮬레이션")
    print(f"  예측 방식: Random Forest predict_proba()")
    print(f"  최적 패키지 OVR 중심점: {int(optimal_ovr)}")
    print(f"  차트: fig_22 ~ fig_25 ({OUTPUT_PATH})")
    print("=" * 70)


if __name__ == '__main__':
    main()
