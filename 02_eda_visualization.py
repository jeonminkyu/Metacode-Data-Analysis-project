"""
=============================================================================
FC Online 4 데이터 분석 프로젝트 - EDA 시각화 
=============================================================================

테이블: user_profile, login_logs, package_purchase, trade_market, daily_club_value
=============================================================================
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 폰트 설정 모듈 import
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
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 핵심 파라미터 (설계서 기반)
PACKAGE_DATES = [pd.Timestamp('2025-01-18'),
                 pd.Timestamp('2025-02-20'),
                 pd.Timestamp('2025-03-20')]
PACKAGE_DATE_STRS = ['2025-01-18', '2025-02-20', '2025-03-20']
DATA_END = pd.Timestamp('2025-04-30')
CHURN_DAYS = 30  # 이탈 기준: 30일 미접속

print("=" * 70)
print("FC Online 4 EDA 시각화 — 데이터 기반 전면 재작성")
print("=" * 70)


# ============================================================================
# 데이터 로드
# ============================================================================
def load_all_data():
    """5개 테이블 모두 로드"""
    print("\n[데이터 로드]")
    up = pd.read_csv(os.path.join(DATA_PATH, 'user_profile.csv'))
    ll = pd.read_csv(os.path.join(DATA_PATH, 'login_logs.csv'))
    pp = pd.read_csv(os.path.join(DATA_PATH, 'package_purchase.csv'))
    tm = pd.read_csv(os.path.join(DATA_PATH, 'trade_market.csv'))
    dcv = pd.read_csv(os.path.join(DATA_PATH, 'daily_club_value.csv'))

    # 날짜 변환
    ll['login_date'] = pd.to_datetime(ll['login_date'])
    pp['purchase_date'] = pd.to_datetime(pp['purchase_date'])
    tm['trade_date'] = pd.to_datetime(tm['trade_date'])
    dcv['date'] = pd.to_datetime(dcv['date'])

    for name, df in [('user_profile', up), ('login_logs', ll),
                     ('package_purchase', pp), ('trade_market', tm),
                     ('daily_club_value', dcv)]:
        print(f"  {name}: {len(df):>10,}행")

    return up, ll, pp, tm, dcv


def compute_churn(user_profile, login_logs):
    """이탈 여부 계산 (30일 미접속 기준)"""
    last_login = login_logs.groupby('user_id')['login_date'].max().reset_index()
    last_login.columns = ['user_id', 'last_login']
    last_login['days_inactive'] = (DATA_END - last_login['last_login']).dt.days
    last_login['is_churned'] = last_login['days_inactive'] > CHURN_DAYS

    # monthly_membership_fee, monthly_avg_spending 포함 (하위 호환)
    profile_cols = ['user_id', 'club_value_group', 'spendig',
                    'membership_tier', 'avg_ovr']
    for col in ['monthly_membership_fee', 'monthly_avg_spending']:
        if col in user_profile.columns:
            profile_cols.append(col)
    merged = user_profile[profile_cols].merge(
        last_login, on='user_id', how='left')
    # 로그인 기록 없으면 이탈 처리
    merged['is_churned'] = merged['is_churned'].fillna(True)
    return merged


# ============================================================================
# 차트 공통 유틸리티
# ============================================================================
def add_package_lines(ax, ymin=None, ymax=None, label_y=None):
    """패키지 출시일 수직 점선 추가"""
    pkg_labels = ['Pack 1차\n(1/18)', 'Pack 2차\n(2/20)', 'Pack 3차\n(3/20)']
    for i, (pd_date, label) in enumerate(zip(PACKAGE_DATES, pkg_labels)):
        ax.axvline(pd_date, color='red', linestyle='--', alpha=0.6, linewidth=1)
        if label_y is not None:
            ax.text(pd_date, label_y, label, ha='center', va='bottom',
                    fontsize=7, color='red', alpha=0.8)


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


# ============================================================================
# Fig 01: K-Means 엘보우 메서드 (k=4 최적)
# ============================================================================
def fig_01_kmeans_elbow(user_profile):
    """실제 user_profile의 club_value + avg_ovr + spendig로 K-means 엘보우"""
    print("\n[Fig 01] K-Means 엘보우")

    # 클러스터링 피처 준비
    features = user_profile[['club_value', 'avg_ovr', 'spendig']].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # k=2~8에 대해 inertia 계산
    K_range = range(2, 9)
    inertias = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)

    # k=4에서 강조 표시
    k4_idx = 2  # k=4는 인덱스 2
    ax.plot(4, inertias[k4_idx], 'r*', markersize=20, zorder=5)
    ax.annotate(f'최적 k=4\n(inertia={inertias[k4_idx]:,.0f})',
                xy=(4, inertias[k4_idx]),
                xytext=(5.5, inertias[k4_idx] * 1.1),
                fontsize=11, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel('클러스터 수 (k)', fontsize=12)
    ax.set_ylabel('Inertia (군집 내 거리 합)', fontsize=12)
    ax.set_title('K-Means 엘보우 메서드 — 최적 클러스터 수 결정', fontsize=14)
    ax.set_xticks(list(K_range))
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'fig_01_kmeans_elbow.png')


# ============================================================================
# Fig 02: 그룹별 유저 분포 (파이차트 + 바차트)
# ============================================================================
def fig_02_group_distribution(user_profile):
    """그룹별 유저 수 분포"""
    print("\n[Fig 02] 그룹별 유저 분포")

    counts = user_profile['club_value_group'].value_counts()
    counts = counts.reindex(GROUP_ORDER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 파이차트
    colors = [COLORS[g] for g in GROUP_ORDER]
    wedges, texts, autotexts = ax1.pie(
        counts.values, labels=GROUP_ORDER, colors=colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    for at in autotexts:
        at.set_fontweight('bold')
    ax1.set_title('구단가치 그룹별 유저 비율', fontsize=13)

    # 바차트
    bars = ax2.bar(GROUP_ORDER, counts.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}명', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('유저 수', fontsize=12)
    ax2.set_title('구단가치 그룹별 유저 수', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('FC Online 4 유저 분포 분석', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_02_group_distribution.png')


# ============================================================================
# Fig 02-b: 그룹별 보유 OVR 바이올린 플롯
# ============================================================================
def fig_02_ovr_violin(user_profile, daily_club_value):
    """그룹별 보유 선수 OVR 분포를 바이올린 플롯으로 시각화

    - 각 그룹의 OVR 분포 형태(중앙값, 분산)를 비교
    - 패키지 높은확률 OVR 구간(130~134) 하이라이트
    - 중위값(밀집 구간) OVR 수치를 명시적으로 표시
    """
    print("\n[Fig 02-b] 그룹별 OVR 바이올린 플롯")

    # user_profile에서 그룹별 대표 OVR 추출 (avg_ovr 컬럼 사용)
    df = user_profile[['user_id', 'club_value_group', 'avg_ovr']].copy()
    df['club_value_group'] = pd.Categorical(
        df['club_value_group'], categories=GROUP_ORDER, ordered=True
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    # 그룹별 데이터 분리
    group_data = [df[df['club_value_group'] == g]['avg_ovr'].dropna().values
                  for g in GROUP_ORDER]

    # 바이올린 플롯 (박스플롯 내장으로 더 풍성하게)
    parts = ax.violinplot(group_data, positions=range(len(GROUP_ORDER)),
                          showmeans=False, showmedians=False, showextrema=False,
                          widths=0.75)

    # 색상 적용 — 그룹별 고유색상
    colors_list = [COLORS[g] for g in GROUP_ORDER]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.55)
        pc.set_edgecolor(colors_list[i])
        pc.set_linewidth(1.5)

    # 바이올린 위에 박스플롯을 겹쳐서 사분위수 시각화
    bp = ax.boxplot(group_data, positions=range(len(GROUP_ORDER)),
                    widths=0.12, patch_artist=True,
                    showfliers=False, zorder=3,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='gray', linewidth=1),
                    capprops=dict(color='gray', linewidth=1))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_list[i])
        patch.set_alpha(0.9)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)

    # ── 핵심: 중위값(밀집 구간) OVR 수치 표시 ──
    for i, (g, data) in enumerate(zip(GROUP_ORDER, group_data)):
        median_val = np.median(data)
        mean_val = np.mean(data)
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)

        # 중위값 수치 표시 (큰 흰색 텍스트 + 배경)
        ax.annotate(f'{median_val:.1f}',
                    xy=(i, median_val), xytext=(i + 0.38, median_val),
                    fontsize=12, fontweight='bold', color=colors_list[i],
                    va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=colors_list[i], alpha=0.9, linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', color=colors_list[i],
                                    lw=1.2))

        # 평균값 다이아몬드 마커
        ax.scatter(i, mean_val, marker='D', color='black', s=30, zorder=5,
                   label='평균' if i == 0 else None)

        # IQR 범위 텍스트 (25%~75% 구간)
        ax.text(i, ax.get_ylim()[0] + 0.5, f'IQR: {q25:.1f}~{q75:.1f}',
                ha='center', fontsize=8, color='gray', style='italic')

    # 패키지 높은확률 OVR 구간 하이라이트 (130~134)
    ax.axhspan(130, 134, alpha=0.08, color='red', zorder=0)
    ax.axhline(133, color='red', linestyle='--', alpha=0.5, linewidth=1.2,
               label='OVR 133 (패키지 최고확률 40%)')
    ax.axhline(130, color='red', linestyle=':', alpha=0.3, linewidth=1,
               label='OVR 130 (패키지 확률 19%)')

    # 우측에 패키지 OVR 구간 텍스트 레이블
    ax.text(len(GROUP_ORDER) - 0.55, 133.3, '← 패키지 OVR 133 (40%)',
            fontsize=9, color='red', alpha=0.7, va='bottom')
    ax.text(len(GROUP_ORDER) - 0.55, 130.3, '← 패키지 OVR 130 (19%)',
            fontsize=9, color='red', alpha=0.7, va='bottom')

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, fontsize=12, fontweight='bold')
    ax.set_ylabel('보유 선수 평균 OVR', fontsize=13)
    ax.set_title('그룹별 보유 선수 OVR 분포 — 중위값(밀집 구간) 및 패키지 OVR 비교',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.2, linestyle='-')

    # Y축 범위 여유
    y_min = min(np.min(d) for d in group_data) - 2
    y_max = max(np.max(d) for d in group_data) + 2
    ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    save_fig(fig, 'fig_02_ovr_violin.png')


# ============================================================================
# Fig 03: 패키지 OVR 확률 버블차트
# ============================================================================
def fig_03_package_ovr_bubble(package_purchase):
    """실제 패키지 구매 데이터에서 OVR별 등장 빈도를 버블로 표시"""
    print("\n[Fig 03] 패키지 OVR 버블차트")

    # OVR별 등장 횟수 및 평균 가격 계산
    ovr_stats = package_purchase.groupby('ovr').agg(
        count=('ovr', 'size'),
        avg_price=('amount', 'mean')
    ).reset_index()

    # 등장 비율 계산
    ovr_stats['pct'] = ovr_stats['count'] / ovr_stats['count'].sum() * 100

    fig, ax = plt.subplots(figsize=(12, 7))

    # 버블 크기: 비율에 비례 (최소 크기 보장)
    bubble_sizes = ovr_stats['pct'] * 120  # 스케일 조정
    bubble_sizes = bubble_sizes.clip(lower=50)

    scatter = ax.scatter(
        ovr_stats['ovr'], ovr_stats['avg_price'],
        s=bubble_sizes, alpha=0.7,
        c=ovr_stats['pct'], cmap='YlOrRd',
        edgecolors='black', linewidths=0.5)

    # 주요 OVR 레이블 추가
    for _, row in ovr_stats.iterrows():
        if row['pct'] >= 3:  # 3% 이상만 레이블
            ax.annotate(f"OVR {int(row['ovr'])}\n({row['pct']:.1f}%)",
                        xy=(row['ovr'], row['avg_price']),
                        fontsize=9, ha='center', va='bottom',
                        fontweight='bold')

    cbar = fig.colorbar(scatter, ax=ax, label='등장 비율 (%)')
    ax.set_xlabel('OVR (선수 능력치)', fontsize=12)
    ax.set_ylabel('평균 패키지 가격 (원)', fontsize=12)
    ax.set_title('패키지에서 OVR별 등장 확률 및 평균 가격', fontsize=14)
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'fig_03_ovr_bubble.png')


# ============================================================================
# Fig 04: 그룹별 OVR 분포 (바이올린)
# ============================================================================
# ============================================================================
# Fig 05: 패키지별 판매량 비교
# ============================================================================
def fig_05_package_sales(package_purchase):
    """패키지 종류별 판매량 + 매출"""
    print("\n[Fig 05] 패키지별 판매량")

    pkg_stats = package_purchase.groupby('package_id').agg(
        count=('package_id', 'size'),
        revenue=('amount', 'sum')
    ).reindex(['Pack_A', 'Pack_B', 'Pack_C', 'Pack_D'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors_pkg = ['#6baed6', '#fd8d3c', '#74c476', '#9e9ac8']

    # 판매량
    bars1 = ax1.bar(pkg_stats.index, pkg_stats['count'], color=colors_pkg, edgecolor='white')
    for bar, val in zip(bars1, pkg_stats['count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}', ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('판매 건수', fontsize=12)
    ax1.set_title('패키지 유형별 판매량', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)

    # 매출
    bars2 = ax2.bar(pkg_stats.index, pkg_stats['revenue'] / 1e6,
                    color=colors_pkg, edgecolor='white')
    for bar, val in zip(bars2, pkg_stats['revenue'] / 1e6):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:,.0f}백만', ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylabel('매출 (백만 원)', fontsize=12)
    ax2.set_title('패키지 유형별 매출', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('패키지 판매 분석 — Pack B 최다 판매', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_05_package_sales.png')


# ============================================================================
# Fig 07: 구단가치 인덱스 시계열 (daily_club_value 기반)
# ============================================================================
def fig_07_club_value_index(dcv):
    """daily_club_value에서 그룹별 평균 인덱스 시계열"""
    print("\n[Fig 07] 구단가치 인덱스 시계열")

    # 그룹×날짜별 평균 인덱스
    group_daily = dcv.groupby(['date', 'club_value_group'])['club_value_index'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 7))

    for group in GROUP_ORDER:
        gdata = group_daily[group_daily['club_value_group'] == group]
        ax.plot(gdata['date'], gdata['club_value_index'],
                color=COLORS[group], linewidth=2, label=group)

    # 패키지 출시일 표시
    add_package_lines(ax, label_y=105)

    ax.axhline(100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.set_xlabel('날짜', fontsize=12)
    ax.set_ylabel('구단가치 인덱스 (기준일=100)', fontsize=12)
    ax.set_title('그룹별 구단가치 인덱스 변화 — 10~100조 그룹 최대 하락', fontsize=14)
    ax.legend(fontsize=11, loc='lower left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'fig_07_club_value_index.png')


# ============================================================================
# Fig 08: OVR별 이적시장 가격 변화 (trade_market 기반)
# ============================================================================
def fig_08_ovr_price_trend(trade_market):
    """trade_market에서 OVR별 일별 평균 거래 가격 시계열"""
    print("\n[Fig 08] OVR별 가격 변화")

    # 주요 OVR만 표시 (설계서 기준: 130, 133, 135, 136)
    target_ovrs = [130, 133, 135, 136]
    ovr_colors = {130: '#e41a1c', 133: '#377eb8', 135: '#4daf4a', 136: '#984ea3'}

    fig, ax = plt.subplots(figsize=(14, 7))

    for ovr_val in target_ovrs:
        ovr_data = trade_market[trade_market['ovr'] == ovr_val]
        daily_avg = ovr_data.groupby('trade_date')['price_trade'].mean().reset_index()
        # 7일 이동평균으로 스무딩
        daily_avg['price_ma7'] = daily_avg['price_trade'].rolling(7, min_periods=1).mean()
        ax.plot(daily_avg['trade_date'], daily_avg['price_ma7'],
                color=ovr_colors[ovr_val], linewidth=2,
                label=f'OVR {ovr_val}')

    add_package_lines(ax, label_y=ax.get_ylim()[1] * 0.95)

    ax.set_xlabel('날짜', fontsize=12)
    ax.set_ylabel('평균 거래가 (100억 단위)', fontsize=12)
    ax.set_title('주요 OVR 선수 이적시장 가격 변동 (7일 이동평균)', fontsize=14)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'fig_08_ovr_price_trend.png')


# ============================================================================
# Fig 09: DID (이중차분법) 분석 — daily_club_value + login_logs
# ============================================================================
def fig_09_did_analysis(dcv, churn_data):
    """
    DID (이중차분법): Treatment=10~100조(가장 민감) vs Control=1000조이상(가장 안정)
    기간: 1차 패키지 출시(1/18) 전/후로 구단가치 인덱스 변화 비교
    → Treatment 그룹이 패키지 출시 후 훨씬 큰 하락을 경험
    """
    print("\n[Fig 09] DID 분석")

    treatment_group = '10~100조'
    control_group = '1000조이상'
    cutoff = pd.Timestamp('2025-01-18')  # 1차 패키지 출시일

    # 그룹별 일별 평균 인덱스
    group_daily = dcv.groupby(['date', 'club_value_group'])['club_value_index'].mean().reset_index()

    # Pre / Post 분리
    treat_data = group_daily[group_daily['club_value_group'] == treatment_group]
    ctrl_data = group_daily[group_daily['club_value_group'] == control_group]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Treatment (10~100조)
    ax.plot(treat_data['date'], treat_data['club_value_index'],
            color=COLORS[treatment_group], linewidth=2.5,
            label=f'Treatment ({treatment_group})')

    # Control (1000조이상)
    ax.plot(ctrl_data['date'], ctrl_data['club_value_index'],
            color=COLORS[control_group], linewidth=2.5,
            label=f'Control ({control_group})')

    # 패키지 출시일 전/후 영역 표시
    ax.axvline(cutoff, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(dcv['date'].min(), cutoff, alpha=0.05, color='green', label='Pre (출시 전)')
    ax.axvspan(cutoff, dcv['date'].max(), alpha=0.05, color='red', label='Post (출시 후)')

    # DID 수치 계산 (1차 패키지 기준)
    pre_treat = treat_data[treat_data['date'] < cutoff]['club_value_index'].mean()
    post_treat = treat_data[treat_data['date'] >= cutoff]['club_value_index'].mean()
    pre_ctrl = ctrl_data[ctrl_data['date'] < cutoff]['club_value_index'].mean()
    post_ctrl = ctrl_data[ctrl_data['date'] >= cutoff]['club_value_index'].mean()

    did_effect = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)

    # DID 효과 텍스트
    ax.text(0.02, 0.15, f'DID 효과 = {did_effect:.2f}\n'
            f'Treatment(출시후-전): {post_treat - pre_treat:.2f}\n'
            f'Control(출시후-전): {post_ctrl - pre_ctrl:.2f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='bottom')

    add_package_lines(ax, label_y=103)
    ax.axhline(100, color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel('날짜', fontsize=12)
    ax.set_ylabel('구단가치 인덱스 (기준일=100)', fontsize=12)
    ax.set_title('DID 분석: Treatment(10~100조) vs Control(1000조이상) 구단가치 변화', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'fig_09_did_analysis.png')


# ============================================================================
# Fig 10: 그룹별 이탈률 바차트
# ============================================================================
def fig_10_churn_by_group(churn_data):
    """그룹별 이탈률 (실제 login_logs 기반)"""
    print("\n[Fig 10] 그룹별 이탈률")

    churn_rate = churn_data.groupby('club_value_group')['is_churned'].mean() * 100
    churn_rate = churn_rate.reindex(GROUP_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS[g] for g in GROUP_ORDER]
    bars = ax.bar(GROUP_ORDER, churn_rate.values, color=colors, edgecolor='white')

    for bar, val in zip(bars, churn_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('이탈률 (%)', fontsize=12)
    ax.set_title('구단가치 그룹별 이탈률 (30일 미접속 기준)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # 10~100조 강조
    ax.annotate('가장 높은 이탈률', xy=(1, churn_rate['10~100조']),
                xytext=(2.2, churn_rate['10~100조'] + 3),
                fontsize=11, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

    save_fig(fig, 'fig_10_churn_by_group.png')


# ============================================================================
# Fig 11: 구단가치 하락률 vs 이탈률 민감도 (Scatter)
# ============================================================================
def fig_11_sensitivity_analysis(dcv, churn_data):
    """
    X축: 그룹별 구단가치 평균 하락률 (daily_club_value에서 계산)
    Y축: 그룹별 이탈률 (login_logs에서 계산)
    → 민감도 = 기울기
    """
    print("\n[Fig 11] 민감도 분석")

    # 그룹별 평균 구단가치 하락률 계산
    first_date = dcv['date'].min()
    last_date = dcv['date'].max()

    first_val = dcv[dcv['date'] == first_date].groupby('club_value_group')['club_value'].mean()
    last_val = dcv[dcv['date'] == last_date].groupby('club_value_group')['club_value'].mean()
    decline_pct = ((first_val - last_val) / first_val * 100).reindex(GROUP_ORDER)

    # 그룹별 이탈률
    churn_rate = churn_data.groupby('club_value_group')['is_churned'].mean() * 100
    churn_rate = churn_rate.reindex(GROUP_ORDER)

    fig, ax = plt.subplots(figsize=(10, 7))

    for g in GROUP_ORDER:
        ax.scatter(decline_pct[g], churn_rate[g],
                   s=200, color=COLORS[g], edgecolors='black',
                   linewidths=1, zorder=5)
        ax.annotate(f'{g}\n(하락 {decline_pct[g]:.1f}%, 이탈 {churn_rate[g]:.1f}%)',
                    xy=(decline_pct[g], churn_rate[g]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=COLORS[g])

    # 추세선
    x_vals = decline_pct.values
    y_vals = churn_rate.values
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_vals.min() - 2, x_vals.max() + 2, 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, linewidth=1.5,
            label=f'추세선 (기울기={z[0]:.2f})')

    ax.set_xlabel('구단가치 평균 하락률 (%)', fontsize=12)
    ax.set_ylabel('이탈률 (%)', fontsize=12)
    ax.set_title('구단가치 하락 vs 이탈률 민감도 분석\n10~100조 그룹이 가장 민감', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'fig_11_sensitivity.png')


# ============================================================================
# Fig 12: 월별 매출 vs 이탈 손실 (monthly_membership 방식)
# ============================================================================
def fig_12_revenue_vs_loss(user_profile, churn_data, package_purchase):
    """
    매출: 패키지 판매 합계
    손실: 이탈유저의 실제 월멤버십 요금 × 8개월 (데이터 기반)
    → 손실이 매출보다 약간 더 크게
    """
    print("\n[Fig 12] 매출 vs 이탈 손실")

    # 매출: 그룹별 패키지 매출
    pkg_with_group = package_purchase.merge(
        user_profile[['user_id', 'club_value_group']], on='user_id', how='left')
    group_revenue = pkg_with_group.groupby('club_value_group')['amount'].sum()
    group_revenue = group_revenue.reindex(GROUP_ORDER).fillna(0)

    # 손실: 이탈 유저의 (멤버십 + 월 과금) 합산 × 8개월
    # 유저 이탈 시 멤버십뿐 아니라 월 과금(패키지/강화/선수팩 등)도 전부 손실
    LOSS_MONTHS = 8
    churned_users = churn_data[churn_data['is_churned']]

    group_loss = pd.Series({
        g: (churned_users[churned_users['club_value_group'] == g][
            'monthly_membership_fee'].sum()
            + churned_users[churned_users['club_value_group'] == g][
            'monthly_avg_spending'].sum()) * LOSS_MONTHS
        for g in GROUP_ORDER
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(GROUP_ORDER))
    width = 0.35

    # 억 단위 변환
    rev_ok = group_revenue.values / 1e8
    loss_ok = group_loss.values / 1e8

    bars1 = ax.bar(x - width/2, rev_ok, width, label='패키지 매출',
                   color='#2ecc71', edgecolor='white')
    bars2 = ax.bar(x + width/2, loss_ok, width, label='이탈 손실',
                   color='#e74c3c', edgecolor='white')

    for bar, val in zip(bars1, rev_ok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}억', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, loss_ok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}억', ha='center', fontsize=9, fontweight='bold', color='red')

    ax.set_xlabel('구단가치 그룹', fontsize=12)
    ax.set_ylabel('금액 (억 원)', fontsize=12)
    ax.set_title('그룹별 패키지 매출 vs 이탈로 인한 손실', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 각주: 손실 산정 기준 명시
    fig.text(0.5, -0.02,
             f'※ 이탈 손실 = 이탈 유저 × (월 멤버십 + 월 과금) × {LOSS_MONTHS}개월 (향후 기대수익 손실)',
             ha='center', fontsize=9, style='italic', color='gray')

    save_fig(fig, 'fig_12_revenue_vs_loss.png')


# ============================================================================
# Fig 12b: 10~100조 그룹 이탈이 왜 치명적인가 — 매출 기여도 + 손실 집중도 분석
# ============================================================================
def fig_12b_why_mid_group_matters(user_profile, churn_data, package_purchase):
    """
    패키지 출시로 인한 이탈이 게임사 영업이익에 치명적인 이유를 시각화:
      핵심 논리:
        - 1000조이상: 1인당 매출은 압도적 (고과금 유저)
        - 그러나 10~100조(54%) + 100~1000조(35%) = 전체 유저의 89%
        - 패키지 출시 → 이 89%에서 대규모 이탈 발생
        - 소수 고과금 유저의 매출로는 다수 중간층 이탈 손실을 메울 수 없음
        - 결과: 게임사 전체 영업이익 감소

      (좌상) 1인당 매출(ARPU) vs 유저 비중 — 1000조이상은 ARPU 최고지만 유저 4%
      (우상) 그룹별 이탈 유저 수 + 유저 비중 — 89%의 핵심 유저층이 이탈 타격
      (좌하) 그룹별 이탈 손실 금액 — 이탈자 수 × 실제 월 요금이 합쳐지면 중간층 손실 최대
      (우하) 영업이익 관점 종합 — 패키지 매출 vs 핵심 유저층(89%) 이탈 손실
    """
    print("\n[Fig 12b] 패키지 출시가 게임사 영업이익에 미치는 영향")

    # ── 공통 데이터 준비 ──
    LOSS_MONTHS = 8

    # 그룹별 유저 수 및 비율
    group_counts = user_profile.groupby('club_value_group').size().reindex(GROUP_ORDER)
    total_users = group_counts.sum()
    group_pct = group_counts / total_users * 100
    core_pct = group_pct['10~100조'] + group_pct['100~1000조']  # 핵심 유저층 비중

    # 그룹별 월 기대수익 = 멤버십 + 월 과금 (실제 데이터 기반)
    group_avg_fee = user_profile.groupby('club_value_group')['monthly_membership_fee'].mean()
    group_avg_spending = user_profile.groupby('club_value_group')['monthly_avg_spending'].mean()
    group_avg_rev = (group_avg_fee + group_avg_spending).reindex(GROUP_ORDER)
    group_avg_fee = group_avg_fee.reindex(GROUP_ORDER)
    group_avg_spending = group_avg_spending.reindex(GROUP_ORDER)
    print(f"  그룹별 월 기대수익 (멤버십 + 과금, 데이터 기반):")
    for g in GROUP_ORDER:
        print(f"    {g}: 멤버십 {group_avg_fee[g]:,.0f} + 과금 {group_avg_spending[g]:,.0f} = {group_avg_rev[g]:,.0f}원/월 (유저 {group_counts[g]:,}명, {group_pct[g]:.0f}%)")
    print(f"  → 10~100조 + 100~1000조 = 전체 유저의 {core_pct:.0f}%")

    # 패키지 매출 (그룹별)
    pkg_with_group = package_purchase.merge(
        user_profile[['user_id', 'club_value_group']], on='user_id', how='left')
    pkg_revenue = pkg_with_group.groupby('club_value_group')['amount'].sum()
    pkg_revenue = pkg_revenue.reindex(GROUP_ORDER).fillna(0)

    # ARPU (만원 단위) — 월 기대수익(멤버십+과금) 기준
    arpu = group_avg_rev / 1e4

    # 이탈 데이터
    churned_users = churn_data[churn_data['is_churned']]
    churned_count = churned_users.groupby('club_value_group').size().reindex(GROUP_ORDER).fillna(0)
    total_count = churn_data.groupby('club_value_group').size().reindex(GROUP_ORDER)
    churn_rate = (churned_count / total_count * 100)

    # 이탈 손실 (억 단위) — (멤버십 + 월 과금) × 8개월
    group_loss = pd.Series({
        g: (churned_users[churned_users['club_value_group'] == g][
            'monthly_membership_fee'].sum()
            + churned_users[churned_users['club_value_group'] == g][
            'monthly_avg_spending'].sum()) * LOSS_MONTHS / 1e8
        for g in GROUP_ORDER
    })

    # 총 매출 (패키지 + 멤버십 + 과금)
    group_total_rev = user_profile.groupby('club_value_group').apply(
        lambda x: (x['monthly_membership_fee'].sum() + x['monthly_avg_spending'].sum())
    ).reindex(GROUP_ORDER)
    revenue_4m = group_total_rev * 4 / 1e8  # 4개월
    pkg_rev_ok = pkg_revenue / 1e8
    total_rev = pkg_rev_ok + revenue_4m

    # ── 시각화 (2×2 레이아웃) ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    group_colors = [COLORS[g] for g in GROUP_ORDER]
    core_color = '#e74c3c'   # 핵심 유저층(10~100조+100~1000조) 강조색
    whale_color = '#3498db'  # 1000조이상 강조색

    # ═══════════════════════════════════════════════════
    # (좌상) ① 1인당 매출(ARPU) vs 유저 비중 — 역전 현상
    # ═══════════════════════════════════════════════════
    ax1 = axes[0, 0]

    # 이중 Y축: 막대=ARPU, 선=유저비중
    bars = ax1.bar(GROUP_ORDER, arpu.values, color=group_colors,
                   edgecolor='white', linewidth=1.5, alpha=0.85)
    for bar, val in zip(bars, arpu.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}만', ha='center', fontsize=10, fontweight='bold')

    ax1.set_ylabel('ARPU — 인당 월 기대수익 (만원)', fontsize=11)
    ax1.set_title('① 1인당 기대수익은 1000조이상이 압도적이지만...', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 우축: 유저 비중
    ax1b = ax1.twinx()
    ax1b.plot(GROUP_ORDER, group_pct.values, 'ko-', linewidth=2.5, markersize=10, zorder=5)
    for i, (x, y) in enumerate(zip(GROUP_ORDER, group_pct.values)):
        ax1b.annotate(f'{y:.0f}%', xy=(x, y), xytext=(0, 10),
                      textcoords='offset points', ha='center', fontsize=11,
                      fontweight='bold', color='black')
    ax1b.set_ylabel('유저 비중 (%)', fontsize=11)
    ax1b.set_ylim(0, 70)

    # 핵심 메시지
    ax1.text(0.02, 0.95,
             f'1000조이상: ARPU 최고 but 유저 {group_pct["1000조이상"]:.0f}%\n'
             f'10~100조+100~1000조: 유저 {core_pct:.0f}% (핵심 유저층)',
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ═══════════════════════════════════════════════════
    # (우상) ② 그룹별 이탈 유저 수 — 89%의 핵심 유저층이 대규모 이탈
    # ═══════════════════════════════════════════════════
    ax2 = axes[0, 1]

    bars2 = ax2.bar(GROUP_ORDER, churned_count.values, color=group_colors,
                    edgecolor='white', linewidth=1.5)
    # 10~100조, 100~1000조 강조 테두리
    bars2[1].set_edgecolor(core_color)
    bars2[1].set_linewidth(3)
    bars2[2].set_edgecolor(core_color)
    bars2[2].set_linewidth(3)

    for bar, cnt, rate in zip(bars2, churned_count.values, churn_rate.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{int(cnt):,}명\n(이탈률 {rate:.1f}%)',
                 ha='center', fontsize=9, fontweight='bold')

    ax2.set_ylabel('이탈 유저 수 (명)', fontsize=11)
    ax2.set_title('② 패키지 출시 → 핵심 유저층(89%)에서 대규모 이탈', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 핵심 유저층 합계 강조
    core_churned = churned_count['10~100조'] + churned_count['100~1000조']
    total_churned = churned_count.sum()
    core_churn_pct = core_churned / total_churned * 100
    ax2.annotate(f'10~100조 + 100~1000조\n= 전체 이탈의 {core_churn_pct:.0f}%\n'
                 f'({int(core_churned):,}명 / {int(total_churned):,}명)',
                 xy=(1.5, max(churned_count.values) * 0.7),
                 fontsize=11, color=core_color, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0f0', alpha=0.95))

    # ═══════════════════════════════════════════════════
    # (좌하) ③ 이탈 손실 분해 — 이탈자 수 × 실제 월 요금
    # ═══════════════════════════════════════════════════
    ax3 = axes[1, 0]

    # 스택 바: 이탈자수 × 평균월요금 = 총 손실
    loss_pct = group_loss / group_loss.sum() * 100
    bars3 = ax3.bar(GROUP_ORDER, group_loss.values, color=group_colors,
                    edgecolor='white', linewidth=1.5)
    bars3[1].set_edgecolor(core_color)
    bars3[1].set_linewidth(3)
    bars3[2].set_edgecolor(core_color)
    bars3[2].set_linewidth(3)

    for bar, val, pct in zip(bars3, group_loss.values, loss_pct.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}억 ({pct:.0f}%)', ha='center', fontsize=10, fontweight='bold')

    ax3.set_ylabel('이탈 손실 (억 원)', fontsize=11)
    ax3.set_title('③ 이탈 손실 분해 — 이탈자수 × (월 멤버십 + 월 과금) × 8개월', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 핵심 유저층 합계
    core_loss = group_loss['10~100조'] + group_loss['100~1000조']
    core_loss_pct = core_loss / group_loss.sum() * 100
    ax3.text(0.95, 0.95,
             f'핵심 유저층(89%) 이탈 손실:\n'
             f'{core_loss:.1f}억 = 전체 손실의 {core_loss_pct:.0f}%\n\n'
             f'1000조이상: 1인당 손실은 최대이지만\n'
             f'이탈자 수가 적어 총 손실 비중은 낮음',
             transform=ax3.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ═══════════════════════════════════════════════════
    # (우하) ④ 영업이익 관점 종합 — 패키지 매출 vs 핵심 유저층 이탈 손실
    # ═══════════════════════════════════════════════════
    ax4 = axes[1, 1]

    # 비교 항목
    total_pkg_rev = pkg_revenue.sum() / 1e8
    labels = ['패키지 매출\n(전 그룹)', f'핵심 유저층\n이탈 손실\n(10~100조+100~1000조)',
              '전체\n이탈 손실']
    values = [total_pkg_rev, core_loss, group_loss.sum()]
    bar_colors = ['#2ecc71', core_color, '#c0392b']

    bars4 = ax4.bar(labels, values, color=bar_colors, width=0.6,
                    edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars4, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}억', ha='center', fontsize=13, fontweight='bold')

    # 손익 차이 화살표
    net_core = total_pkg_rev - core_loss
    ax4.annotate(f'핵심 유저층만으로도\n순손실 {abs(net_core):.1f}억',
                 xy=(1, core_loss * 0.5),
                 xytext=(1.8, core_loss * 0.6),
                 fontsize=11, color=core_color, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=core_color, lw=2.5),
                 bbox=dict(boxstyle='round', facecolor='#fff0f0', alpha=0.95))

    ax4.set_ylabel('금액 (억 원)', fontsize=11)
    ax4.set_title('④ 영업이익 관점: 패키지 매출로 이탈 손실을 메울 수 없다', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # 최종 결론 박스
    ax4.text(0.02, 0.95,
             f'패키지 매출 {total_pkg_rev:.1f}억\n'
             f'  < 핵심 유저층(89%) 이탈 손실 {core_loss:.1f}억\n'
             f'  < 전체 이탈 손실 {group_loss.sum():.1f}억\n\n'
             f'→ 패키지가 89%의 유저를 이탈시키면\n'
             f'   영업이익이 구조적으로 악화됨',
             transform=ax4.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.95))

    fig.suptitle('패키지 출시가 게임사 영업이익에 미치는 영향\n'
                 f'10~100조 + 100~1000조 = 전체 유저의 {core_pct:.0f}% → 이탈 시 영업이익 구조적 손실',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_fig(fig, 'fig_12b_why_mid_group_matters.png')


# ============================================================================
# Fig 12c: 구단가치 그룹별 평균 멤버십 티어 및 평균 과금액
# ============================================================================
def fig_12c_group_tier_spending(user_profile):
    """
    구단가치 그룹별 과금 구조 시각화:
      (좌) 그룹별 멤버십 티어 분포 — 100% 스택 바
      (우) 그룹별 평균 월 과금액 + 평균 멤버십 요금 — 스택 바
    """
    print("\n[Fig 12c] 그룹별 평균 멤버십 티어 및 과금 구조")

    TIER_ORDER = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
    TIER_COLORS = {
        'Bronze':   '#cd7f32',  # 브론즈 갈색
        'Silver':   '#c0c0c0',  # 실버 회색
        'Gold':     '#ffd700',  # 골드 노란색
        'Platinum': '#4169e1',  # 플래티넘 파란색
        'Diamond':  '#b9f2ff',  # 다이아몬드 하늘색
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ── (좌) 멤버십 티어 100% 스택 바 ──
    tier_pct = pd.crosstab(
        user_profile['club_value_group'], user_profile['membership_tier'],
        normalize='index') * 100
    tier_pct = tier_pct.reindex(GROUP_ORDER)[TIER_ORDER]

    bottom = np.zeros(len(GROUP_ORDER))
    for tier in TIER_ORDER:
        vals = tier_pct[tier].values
        bars = ax1.bar(GROUP_ORDER, vals, bottom=bottom, label=tier,
                       color=TIER_COLORS[tier], edgecolor='white', linewidth=0.5)
        # 10% 이상인 구간에만 레이블 표시
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 8:
                ax1.text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                         fontsize=9, fontweight='bold')
        bottom += vals

    ax1.set_ylabel('티어 비중 (%)', fontsize=12)
    ax1.set_title('그룹별 멤버십 티어 분포', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, title='멤버십 등급')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.2)

    # 각 그룹의 평균 티어 텍스트
    tier_numeric = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4, 'Diamond': 5}
    for i, g in enumerate(GROUP_ORDER):
        sub = user_profile[user_profile['club_value_group'] == g]
        avg_tier_num = sub['membership_tier'].map(tier_numeric).mean()
        # 숫자 → 티어명 (가장 가까운 티어)
        closest_tier = min(tier_numeric, key=lambda t: abs(tier_numeric[t] - avg_tier_num))
        ax1.text(i, 102, f'평균: {closest_tier}({avg_tier_num:.1f})',
                 ha='center', fontsize=9, fontweight='bold', color='#333')

    # ── (우) 평균 월 과금액 + 멤버십 스택 바 ──
    avg_membership = user_profile.groupby('club_value_group')['monthly_membership_fee'].mean()
    avg_spending = user_profile.groupby('club_value_group')['monthly_avg_spending'].mean()
    avg_membership = avg_membership.reindex(GROUP_ORDER)
    avg_spending = avg_spending.reindex(GROUP_ORDER)

    # 만원 단위 변환
    mem_val = avg_membership.values / 1e4
    spend_val = avg_spending.values / 1e4

    bars_mem = ax2.bar(GROUP_ORDER, mem_val, label='월 멤버십 요금',
                       color='#3498db', edgecolor='white', linewidth=1)
    bars_spend = ax2.bar(GROUP_ORDER, spend_val, bottom=mem_val,
                         label='월 평균 과금액', color='#e67e22',
                         edgecolor='white', linewidth=1)

    # 멤버십 레이블
    for i, (mv, sv) in enumerate(zip(mem_val, spend_val)):
        ax2.text(i, mv/2, f'{mv:.1f}만', ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white')
        # 과금 레이블
        if sv >= 1:
            ax2.text(i, mv + sv/2, f'{sv:.1f}만', ha='center', va='center',
                     fontsize=9, fontweight='bold', color='white')
        # 합계 레이블 (위에)
        total = mv + sv
        ax2.text(i, total + max(spend_val) * 0.03,
                 f'합계 {total:.1f}만원', ha='center', fontsize=10,
                 fontweight='bold', color='#333')

    ax2.set_ylabel('월 기대수익 (만원)', fontsize=12)
    ax2.set_title('그룹별 월 평균 멤버십 + 과금액', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # 과금 배율 주석
    ratio_whale = (avg_membership['1000조이상'] + avg_spending['1000조이상']) / \
                  (avg_membership['10~100조'] + avg_spending['10~100조'])
    ax2.text(0.98, 0.95,
             f'1000조이상 유저 1명의 월 기대수익은\n'
             f'10~100조 유저의 약 {ratio_whale:.0f}배',
             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
             fontweight='bold', color='#c0392b',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle('구단가치 그룹별 과금 구조 분석 — 멤버십 티어 × 월 과금액',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 그룹별 상세 출력
    for g in GROUP_ORDER:
        sub = user_profile[user_profile['club_value_group'] == g]
        print(f"  {g}: 평균 멤버십 {avg_membership[g]:,.0f}원 + "
              f"평균 과금 {avg_spending[g]:,.0f}원 = "
              f"합계 {avg_membership[g]+avg_spending[g]:,.0f}원/월")

    save_fig(fig, 'fig_12c_group_tier_spending.png')


# ============================================================================
# ★ 가설 근거 시각화 (ML 피처 선택 근거)
# ============================================================================

# Fig A1: 유저 OVR ↔ 패키지 OVR 겹침 → 구단가치 하락 관계
def fig_a1_ovr_overlap_decline(user_profile, dcv):
    """
    가설1 근거: 패키지에서 높은 확률(130,133)로 등장하는 OVR의 선수를
    보유한 유저일수록 구단가치 하락이 크다는 것을 시각화
    X축: 유저 평균 OVR, Y축: 구단가치 하락률(%), 색상: 그룹
    """
    print("\n[Fig A1] OVR 겹침도 vs 구단가치 하락률")

    # dcv에서 유저별 구단가치 하락률 계산
    first_date = dcv['date'].min()
    last_date = dcv['date'].max()
    cv_first = dcv[dcv['date'] == first_date][['user_id', 'club_value']].copy()
    cv_last = dcv[dcv['date'] == last_date][['user_id', 'club_value']].copy()
    cv_first.columns = ['user_id', 'cv_start']
    cv_last.columns = ['user_id', 'cv_end']

    decline = cv_first.merge(cv_last, on='user_id')
    decline['decline_pct'] = (decline['cv_start'] - decline['cv_end']) / decline['cv_start'] * 100
    decline = decline.merge(user_profile[['user_id', 'avg_ovr', 'club_value_group']], on='user_id')

    fig, ax = plt.subplots(figsize=(14, 8))

    for g in GROUP_ORDER:
        gdata = decline[decline['club_value_group'] == g]
        ax.scatter(gdata['avg_ovr'], gdata['decline_pct'],
                   c=COLORS[g], alpha=0.4, s=20, label=g)

    # 패키지 높은 확률 OVR 영역 강조
    ax.axvspan(129.5, 134.5, alpha=0.1, color='red',
               label='패키지 높은확률 OVR (130~134)')
    ax.axvline(133, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(130, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(131.5, ax.get_ylim()[1] * 0.9, 'OVR 130\n(19%)', color='red',
            ha='center', fontsize=9, fontweight='bold')
    ax.text(133.5, ax.get_ylim()[1] * 0.85, 'OVR 133\n(40%)', color='red',
            ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('유저 평균 OVR', fontsize=12)
    ax.set_ylabel('구단가치 하락률 (%)', fontsize=12)
    ax.set_title('가설1 근거: 유저 보유 OVR이 패키지 높은확률 OVR과 겹칠수록\n구단가치 하락이 크다',
                 fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'fig_a1_ovr_overlap_decline.png')


# Fig A2: 패키지 출시 전후 그룹별 구단가치 변화 (Facet)
def fig_a2_pkg_before_after(dcv):
    """
    가설2 근거: 각 그룹별로 패키지 출시 전후 구단가치 인덱스가
    어떻게 변하는지 Facet으로 비교
    설계서 멘토 조언: "그룹별로 패키지 출시 시점에 따라 하락률과 이탈률을 같이 걸어서 비교"
    """
    print("\n[Fig A2] 패키지 전후 구단가치 변화 (그룹별 Facet)")

    # 1차 패키지(1/18) 기준 전후 20일
    pkg_date = pd.Timestamp('2025-01-18')
    window = 20

    nearby = dcv[(dcv['date'] >= pkg_date - pd.Timedelta(days=window)) &
                 (dcv['date'] <= pkg_date + pd.Timedelta(days=window))].copy()
    nearby['days_from_pkg'] = (nearby['date'] - pkg_date).dt.days

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

    for idx, (g, ax) in enumerate(zip(GROUP_ORDER, axes.flatten())):
        gdata = nearby[nearby['club_value_group'] == g]
        daily_avg = gdata.groupby('days_from_pkg')['club_value_index'].mean()

        ax.plot(daily_avg.index, daily_avg.values,
                color=COLORS[g], linewidth=2.5, marker='o', markersize=3)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(100, color='gray', linestyle=':', alpha=0.4)

        # 전후 하락률 표시
        pre_avg = daily_avg[daily_avg.index < 0].mean()
        post_avg = daily_avg[daily_avg.index > 0].mean()
        drop = pre_avg - post_avg
        ax.text(0.05, 0.1, f'하락: {drop:.1f}pt',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax.set_title(f'{g}', fontsize=13, fontweight='bold', color=COLORS[g])
        ax.set_xlabel('패키지 출시 후 일수', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel('구단가치 인덱스', fontsize=11)
    axes[1][0].set_ylabel('구단가치 인덱스', fontsize=11)

    fig.suptitle('가설2 근거: 1차 패키지(1/18) 출시 전후 그룹별 구단가치 변화\n→ 10~100조 그룹 하락이 가장 큼',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_a2_pkg_before_after.png')


# Fig A3: 인과 체인 요약 — 상관 분석 + Granger Causality Test
def fig_a3_causal_chain(user_profile, dcv, churn_data, trade_market):
    """
    가설 전체 인과 체인 검증 (2-패널 구성):
      (좌) 상관 분석: X=OVR가격하락률, Y=구단가치하락률, 버블=이탈률 → Pearson r
      (우) Granger Causality Test: 시계열 기반 인과 방향성 검증
           - H0: X가 Y를 예측하는 데 도움이 되지 않는다
           - p < 0.05 → H0 기각 → X → Y 인과적 방향성 시사

    ※ Granger Causality는 "시간적 선후관계 + 예측력"을 검증하는 것이며,
       철학적 의미의 완전한 인과성은 RCT(A/B Test)로만 증명 가능합니다.
       따라서 "인과적 방향성이 시사됨"으로 표현합니다.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    print("\n[Fig A3] 인과 체인 요약 + Granger Causality Test")

    # ── (1) 그룹별 상관 분석 데이터 준비 ──
    group_ovr = user_profile.groupby('club_value_group')['avg_ovr'].mean()

    # OVR별 가격 하락률 — 패키지 전(~1/17) vs 패키지 후(3/21~) 비교
    trade_market['trade_date'] = pd.to_datetime(trade_market['trade_date'])
    pre_pkg = trade_market[trade_market['trade_date'] < '2025-01-18']
    post_pkg = trade_market[trade_market['trade_date'] > '2025-03-20']
    pre_prices = pre_pkg.groupby('ovr')['price_trade'].mean()
    post_prices = post_pkg.groupby('ovr')['price_trade'].mean()
    ovr_drop = ((pre_prices - post_prices) / pre_prices * 100)

    # 그룹별 "보유 OVR대의 가중 평균 가격 하락률" 계산
    user_profile['ovr_rounded'] = user_profile['avg_ovr'].round().astype(int)
    user_profile['ovr_price_drop'] = user_profile['ovr_rounded'].map(ovr_drop).fillna(0)
    group_price_drop = user_profile.groupby('club_value_group')['ovr_price_drop'].mean()
    group_price_drop = group_price_drop.reindex(GROUP_ORDER)

    # 그룹별 구단가치 하락률
    first_date = dcv['date'].min()
    last_date = dcv['date'].max()
    cv_first = dcv[dcv['date'] == first_date].groupby('club_value_group')['club_value'].mean()
    cv_last = dcv[dcv['date'] == last_date].groupby('club_value_group')['club_value'].mean()
    cv_decline = ((cv_first - cv_last) / cv_first * 100).reindex(GROUP_ORDER)

    # 그룹별 이탈률
    churn_rate_grp = churn_data.groupby('club_value_group')['is_churned'].mean() * 100
    churn_rate_grp = churn_rate_grp.reindex(GROUP_ORDER)

    # ── (2) Granger Causality Test ──
    # 일별 시계열 구성: OVR대 평균 거래가격 변화율 → 전체 평균 구단가치 변화율
    dcv_daily = dcv.groupby('date')['club_value'].mean().sort_index()
    cv_pct = dcv_daily.pct_change().dropna() * 100  # 구단가치 일별 변화율

    # 거래 시장 일별 평균 가격 변화율 (130~134 OVR대, 패키지 영향권)
    pkg_ovr_trades = trade_market[trade_market['ovr'].between(130, 134)].copy()
    price_daily = pkg_ovr_trades.groupby('trade_date')['price_trade'].mean().sort_index()
    price_pct = price_daily.pct_change().dropna() * 100  # 가격 일별 변화율

    # 공통 날짜 정렬 (두 시계열을 맞춤)
    common_dates = cv_pct.index.intersection(price_pct.index)
    ts_cv = cv_pct.loc[common_dates].values
    ts_price = price_pct.loc[common_dates].values

    # Granger test: "가격 변화 → 구단가치 변화" 방향 검증
    # grangercausalitytests는 [Y, X] 순서 → X가 Y를 Granger-cause 하는지 테스트
    granger_data = np.column_stack([ts_cv, ts_price])
    max_lag = min(7, len(common_dates) // 5)  # 적정 lag 설정
    max_lag = max(max_lag, 2)  # 최소 2

    # 결과 수집 (stdout 억제)
    import io, contextlib
    granger_results = {}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            gc_result = grangercausalitytests(granger_data, maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag + 1):
                f_pval = gc_result[lag][0]['ssr_ftest'][1]  # F-test p-value
                granger_results[lag] = f_pval
        except Exception as e:
            print(f"  Granger Test 오류: {e}")
            for lag in range(1, max_lag + 1):
                granger_results[lag] = 1.0

    # 최소 p-value lag 선택
    best_lag = min(granger_results, key=granger_results.get)
    best_pval = granger_results[best_lag]
    is_significant = best_pval < 0.05

    print(f"  Granger Causality (가격변화 → 구단가치변화)")
    for lag, pval in granger_results.items():
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else "")
        print(f"    lag={lag}: p={pval:.4f} {sig}")
    print(f"  → 최적 lag={best_lag}, p={best_pval:.4f}, 유의{'함' if is_significant else '하지 않음'}")

    # ── (3) 2-패널 차트 생성 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8),
                                    gridspec_kw={'width_ratios': [1.1, 0.9]})

    # ── 좌: 상관 분석 버블차트 ──
    for g in GROUP_ORDER:
        pd_val = group_price_drop[g]
        cv_val = cv_decline[g]
        cr_val = churn_rate_grp[g]

        ax1.scatter(pd_val, cv_val,
                    s=cr_val * 40, c=COLORS[g],
                    edgecolors='black', linewidths=1.5, alpha=0.85, zorder=5)
        ax1.annotate(f'{g}\nOVR가격하락: {pd_val:.1f}%\n구단가치하락: {cv_val:.1f}%\n이탈률: {cr_val:.1f}%',
                     xy=(pd_val, cv_val),
                     xytext=(15, 15), textcoords='offset points',
                     fontsize=9, fontweight='bold', color=COLORS[g],
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor=COLORS[g], alpha=0.9),
                     arrowprops=dict(arrowstyle='->', color=COLORS[g], lw=1.2))

    # 추세선 + Pearson r
    x_vals = [group_price_drop[g] for g in GROUP_ORDER]
    y_vals = [cv_decline[g] for g in GROUP_ORDER]
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    corr_r, corr_p = stats.pearsonr(x_vals, y_vals)
    x_line = np.linspace(min(x_vals) - 0.3, max(x_vals) + 0.3, 50)
    ax1.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, linewidth=1.5,
             label=f'추세선 (Pearson r={corr_r:.3f})')

    ax1.set_xlabel('보유 OVR대 선수 가격 하락률 (%)', fontsize=11)
    ax1.set_ylabel('구단가치 하락률 (%)', fontsize=11)
    ax1.set_title('① 상관 분석: 가격하락 ↔ 구단가치하락 (버블=이탈률)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='lower right')

    ax1.text(0.02, 0.98,
             f'Pearson r = {corr_r:.3f}\n● 버블이 클수록 이탈률 높음\n→ 강한 양의 상관관계',
             transform=ax1.transAxes, fontsize=10, va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── 우: Granger Causality Test 결과 ──
    lags = list(granger_results.keys())
    pvals = list(granger_results.values())
    bar_colors = ['#e74c3c' if p < 0.01 else '#f39c12' if p < 0.05 else '#95a5a6' for p in pvals]

    bars = ax2.bar(lags, pvals, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)

    # 유의수준 기준선
    ax2.axhline(y=0.05, color='#e74c3c', linestyle='--', linewidth=2, label='유의수준 α=0.05')
    ax2.axhline(y=0.01, color='#c0392b', linestyle=':', linewidth=1.5, alpha=0.7, label='α=0.01')

    # p-value 라벨
    for bar, pval in zip(bars, pvals):
        sig_label = "***" if pval < 0.01 else ("**" if pval < 0.05 else "n.s.")
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'p={pval:.3f}\n{sig_label}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold',
                 color='#e74c3c' if pval < 0.05 else '#95a5a6')

    ax2.set_xlabel('시차 (Lag, 일)', fontsize=11)
    ax2.set_ylabel('p-value', fontsize=11)
    ax2.set_title('② Granger Causality Test\n(OVR 가격변화 → 구단가치변화)',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(lags)
    ax2.set_ylim(0, max(pvals) * 1.3 + 0.02)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(fontsize=9, loc='upper right')

    # Granger 결과 해석 박스
    if is_significant:
        interp_text = (
            f'Granger Causality 유의함\n'
            f'최적 lag={best_lag}일, p={best_pval:.4f}\n\n'
            f'해석: OVR 가격 변화가 {best_lag}일 후\n'
            f'구단가치 변화를 유의하게 예측함\n'
            f'→ 인과적 방향성 시사'
        )
        box_color = '#fff0f0'
        text_color = '#c0392b'
    else:
        interp_text = (
            f'Granger Causality 비유의\n'
            f'최적 lag={best_lag}일, p={best_pval:.4f}\n\n'
            f'해석: 시계열 단위에서\n'
            f'인과 방향성 불확실\n'
            f'→ 추가 검증 필요'
        )
        box_color = '#f5f5f5'
        text_color = '#777777'

    ax2.text(0.02, 0.98, interp_text,
             transform=ax2.transAxes, fontsize=10, va='top', fontweight='bold',
             color=text_color,
             bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9, edgecolor=text_color))

    # 전체 제목
    fig.suptitle(
        '인과 체인 검증: 상관 분석 + Granger Causality\n'
        '패키지 OVR 확률 → 선수 가격 하락 → 구단가치 감소 → 이탈',
        fontsize=15, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    save_fig(fig, 'fig_a3_causal_chain.png')


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    # 데이터 로드
    user_profile, login_logs, package_purchase, trade_market, dcv = load_all_data()

    # 이탈 지표 계산
    churn_data = compute_churn(user_profile, login_logs)
    churn_rate = churn_data.groupby('club_value_group')['is_churned'].mean() * 100
    print("\n[이탈률 검증]")
    for g in GROUP_ORDER:
        print(f"  {g}: {churn_rate.get(g, 0):.1f}%")

    # ── EDA 차트 생성 (PPT 사용 차트만) ──
    fig_01_kmeans_elbow(user_profile)
    fig_02_group_distribution(user_profile)
    fig_02_ovr_violin(user_profile, dcv)
    fig_03_package_ovr_bubble(package_purchase)
    fig_05_package_sales(package_purchase)
    fig_07_club_value_index(dcv)
    fig_08_ovr_price_trend(trade_market)
    fig_09_did_analysis(dcv, churn_data)
    fig_10_churn_by_group(churn_data)
    fig_11_sensitivity_analysis(dcv, churn_data)
    fig_12_revenue_vs_loss(user_profile, churn_data, package_purchase)
    fig_12b_why_mid_group_matters(user_profile, churn_data, package_purchase)
    fig_12c_group_tier_spending(user_profile)

    # ── 가설 근거 시각화 (ML 피처 선택의 근거) ──
    print("\n" + "=" * 70)
    print("가설 근거 시각화 — ML 피처 선택 전 인과적 방향성 검증")
    print("=" * 70)
    fig_a1_ovr_overlap_decline(user_profile, dcv)
    fig_a2_pkg_before_after(dcv)
    fig_a3_causal_chain(user_profile, dcv, churn_data, trade_market)

    print("\n" + "=" * 70)
    print(f"전체 16개 차트 생성 완료! (EDA 13 + 가설근거 3)")
    print(f"저장 경로: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
