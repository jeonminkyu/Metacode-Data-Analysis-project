"""
=============================================================================
FC Online 4 데이터 분석 프로젝트 - Stage 1: 데이터 생성
=============================================================================
목적: 50,000명의 유저에 대한 5개 테이블(user_profile,login_logs,package_purchase,trade_market,daily_club_value) 생성

=============================================================================
"""

import numpy as np
import pandas as pd
import datetime
import random
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 전역 설정 및 시드 고정
# ============================================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 프로젝트 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# 2. 핵심 파라미터 (설계서 기반)
# ============================================================================
N_USERS = 50000  # 총 유저 수

# 데이터 기간: 2025-01-01 ~ 2025-04-30
DATE_START = datetime.date(2025, 1, 1)
DATE_END = datetime.date(2025, 4, 30)
ALL_DATES = pd.date_range(DATE_START, DATE_END, freq='D')
N_DAYS = len(ALL_DATES)

# 패키지 출시일 (설계서 명시)
PACKAGE_DATES = [
    datetime.date(2025, 1, 18),
    datetime.date(2025, 2, 20),
    datetime.date(2025, 3, 20),
]

# 패키지 종류: Pack A~D, B가 최다 판매 (설계서 Bar Chart 참조)
PACKAGE_IDS = ['Pack_A', 'Pack_B', 'Pack_C', 'Pack_D']
# Pack B가 가장 많이 팔리도록 가중치 설정
PACKAGE_WEIGHTS = [0.20, 0.45, 0.22, 0.13]

# 패키지별 가격 범위 (원 단위, 자연스러운 가격대)
PACKAGE_PRICES = {
    'Pack_A': (5900, 11900),
    'Pack_B': (11900, 33000),   # 가장 인기 → 중간 가격대
    'Pack_C': (33000, 55000),
    'Pack_D': (55000, 110000),
}

# 그룹 설정 (유저가 수정 요청한 분포)
GROUP_CONFIG = {
    '0~10조': {
        'ratio': 0.07,           # 7%
        'club_value_range': (0.5e12, 10e12),  # 0.5조 ~ 10조
        'ovr_mean': 130.9, 'ovr_std': 2.5,
        'ovr_min': 125, 'ovr_max': 135,
        'daily_login_prob_base': 0.35,   # 기본 접속 확률
        'purchase_prob': 0.15,            # 패키지 구매 확률
        'churn_sensitivity': 0.60,        # 이탈 민감도 (slope)
        'cv_decline_rates': [0.10, 0.17, 0.15],  # 패키지별 구단가치 하락률
        'trade_activity': 0.25,           # 거래 활동 수준
    },
    '10~100조': {
        'ratio': 0.54,           # 54% — 주요 타겟 그룹
        'club_value_range': (10e12, 100e12),
        'ovr_mean': 133.4, 'ovr_std': 1.8,
        'ovr_min': 129, 'ovr_max': 137,
        'daily_login_prob_base': 0.55,
        'purchase_prob': 0.35,
        'churn_sensitivity': 0.80,        # 가장 높은 민감도
        'cv_decline_rates': [0.23, 0.25, 0.26],  # 가장 큰 하락률
        'trade_activity': 0.55,
    },
    '100~1000조': {
        'ratio': 0.35,           # 35%
        'club_value_range': (100e12, 1000e12),
        'ovr_mean': 136.1, 'ovr_std': 1.5,
        'ovr_min': 133, 'ovr_max': 140,
        'daily_login_prob_base': 0.65,
        'purchase_prob': 0.50,
        'churn_sensitivity': 0.25,
        'cv_decline_rates': [0.05, 0.036, 0.03],
        'trade_activity': 0.70,
    },
    '1000조이상': {
        'ratio': 0.04,           # 4%
        'club_value_range': (1000e12, 8000e12),
        'ovr_mean': 137.3, 'ovr_std': 1.2,
        'ovr_min': 135, 'ovr_max': 142,
        'daily_login_prob_base': 0.78,
        'purchase_prob': 0.65,
        'churn_sensitivity': 0.15,
        'cv_decline_rates': [0.01, 0.01, 0.01],
        'trade_activity': 0.85,
    },
}

# 멤버십 등급 및 월 구독료 (설계서 user_profile 스키마)
# FC Online 실제 월정액 상품 구조를 참고한 설정
MEMBERSHIP_TIERS = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
MEMBERSHIP_MONTHLY_FEE = {
    'Bronze':    5900,   # 기본 등급 — 월 5,900원
    'Silver':   11900,   # 실버 등급 — 월 11,900원
    'Gold':     33000,   # 골드 등급 — 월 33,000원
    'Platinum': 55000,   # 플래티넘 등급 — 월 55,000원
    'Diamond': 110000,   # 다이아 등급 — 월 110,000원
}

# 선수 풀 (OVR별 선수 목록 — FC온라인 실제 선수 기반)
PLAYER_POOL = {
    # OVR 125~129: 저OVR 선수
    125: ['김민재_125', '이강인_125', '황의조_125'],
    126: ['손흥민_126', '조규성_126', '정우영_126'],
    127: ['박지성_127', '차범근_127', '이천수_127'],
    128: ['홍명보_128', '안정환_128', '이영표_128'],
    129: ['유상철_129', '기성용_129', '박주영_129'],
    # OVR 130~134: 중OVR 선수 (10~100조 핵심 보유)
    130: ['메시_130', '호날두_130', '네이마르_130', '음바페_130'],
    131: ['벤제마_131', '살라_131', '손흥민_131', '케인_131'],
    132: ['레반도프스키_132', '홀란드_132', '비니시우스_132'],
    133: ['드브라위너_133', '모드리치_133', '크로스_133', '벨링엄_133'],
    134: ['킴미히_134', '카제미루_134', '칸테_134'],
    # OVR 135~137: 고OVR 선수 (100~1000조 핵심 보유)
    135: ['펠레_135', '마라도나_135', '지단_135', '크루이프_135'],
    136: ['베켄바워_136', '호나우두_136', '호나우지뉴_136'],
    137: ['메시_TOTY_137', '음바페_TOTY_137', '홀란드_TOTY_137'],
    # OVR 138~142: 최고OVR (1000조이상 핵심 보유)
    138: ['메시_ICON_138', '펠레_ICON_138'],
    139: ['마라도나_ICON_139', '호나우두_ICON_139'],
    140: ['지단_ICON_140', '베켄바워_ICON_140'],
    141: ['크루이프_ICON_141'],
    142: ['펠레_ULTIMATE_142'],
}

# 패키지에서 높은 확률로 등장하는 OVR (설계서: 확률 높은 OVR → 가격 하락)
# 설계서 버블차트 참조: 40% OVR 133.0, 19% OVR 130.0, 5% OVR 135.0, 1% OVR 136.0
PACKAGE_OVR_PROBS = {
    130: 0.19, 131: 0.12, 132: 0.10, 133: 0.40,
    134: 0.08, 135: 0.05, 136: 0.01,
    # 나머지 OVR은 매우 낮은 확률
    125: 0.01, 126: 0.01, 127: 0.01, 128: 0.01, 129: 0.01,
}

# ============================================================================
# 3. user_profile 생성
# ============================================================================
def generate_user_profile():
    """
    유저 프로필 테이블 생성
    스키마: user_id, nickname, spendig, club_value, membership_tier,
            club_value_group, avg_ovr, register_date, position_player
    """
    print("[1/4] user_profile 생성 중...")

    users = []
    user_id_counter = 100000  # 유저 ID 시작값

    for group_name, config in GROUP_CONFIG.items():
        n_group = int(N_USERS * config['ratio'])

        for i in range(n_group):
            user_id = f"U{user_id_counter}"
            user_id_counter += 1

            # 닉네임 생성 (자연스러운 게임 닉네임)
            nickname = f"Player_{user_id_counter}"

            # 구단가치: 그룹 범위 내에서 로그정규분포로 자연스럽게 생성
            cv_min, cv_max = config['club_value_range']
            log_mean = (np.log(cv_min) + np.log(cv_max)) / 2
            log_std = (np.log(cv_max) - np.log(cv_min)) / 6
            club_value = np.exp(np.random.normal(log_mean, log_std))
            club_value = np.clip(club_value, cv_min * 1.01, cv_max * 0.99)

            # 평균 OVR: 그룹별 평균 ± std, 범위 클리핑
            avg_ovr = np.random.normal(config['ovr_mean'], config['ovr_std'])
            avg_ovr = np.clip(avg_ovr, config['ovr_min'], config['ovr_max'])
            avg_ovr = round(avg_ovr, 1)

            # 누적 과금 상태 (spendig): 구단가치와 상관있게 생성
            # 높은 구단가치 → 높은 과금 경향, but 무과금도 존재
            if group_name == '0~10조':
                # 대부분 소과금 또는 무과금
                if random.random() < 0.3:
                    spendig = 0  # 무과금이지만 데이터에서는 최소값 부여
                else:
                    spendig = int(np.random.lognormal(8, 1.5))
                spendig = max(spendig, int(np.random.uniform(500, 3000)))
            elif group_name == '10~100조':
                # 중과금 핵심 그룹
                spendig = int(np.random.lognormal(10, 1.2))
                spendig = max(spendig, int(np.random.uniform(5000, 20000)))
            elif group_name == '100~1000조':
                spendig = int(np.random.lognormal(11.5, 1.0))
                spendig = max(spendig, int(np.random.uniform(50000, 200000)))
            else:  # 1000조이상
                spendig = int(np.random.lognormal(13, 0.8))
                spendig = max(spendig, int(np.random.uniform(500000, 2000000)))

            # 멤버십 등급: 구단가치 그룹에 따라 현실적 분포 적용
            # 구단가치가 높을수록 고등급 멤버십 비율이 현저히 높음
            if group_name == '0~10조':
                # 거의 무과금: Bronze/Silver 중심
                tier_probs = [0.50, 0.30, 0.12, 0.06, 0.02]
            elif group_name == '10~100조':
                # 중과금: Gold 중심, Silver/Platinum도 분포
                tier_probs = [0.15, 0.28, 0.32, 0.17, 0.08]
            elif group_name == '100~1000조':
                # 고과금: Platinum/Diamond 중심
                tier_probs = [0.03, 0.07, 0.20, 0.35, 0.35]
            else:  # 1000조이상
                # 초고과금 고래: Diamond 압도적
                tier_probs = [0.01, 0.02, 0.05, 0.20, 0.72]
            membership_tier = np.random.choice(MEMBERSHIP_TIERS, p=tier_probs)

            # 가입일: 기간 시작 전 랜덤 (오래된 유저도 있음)
            days_before = random.randint(30, 365)
            register_date = DATE_START - datetime.timedelta(days=days_before)

            # 포지션 선수 (대표 선수 11명 중 랜덤)
            possible_ovrs = [o for o in PLAYER_POOL.keys()
                           if config['ovr_min'] <= o <= config['ovr_max']]
            if possible_ovrs:
                rep_ovr = random.choice(possible_ovrs)
                position_player = random.choice(PLAYER_POOL[rep_ovr])
            else:
                position_player = "Unknown"

            # 월 멤버십 요금: 등급 기반 (데이터에 명시적으로 포함)
            monthly_fee = MEMBERSHIP_MONTHLY_FEE[membership_tier]

            # 월 평균 과금액 (in-game spending): 구단가치 그룹별 현실적 분포
            # 고래 유저일수록 월 과금이 압도적으로 높음 (패키지/선수팩/강화 등)
            if group_name == '0~10조':
                # 거의 무과금: 소액 또는 이벤트성 과금만
                monthly_spending = int(np.clip(
                    np.random.lognormal(8.0, 0.8), 1000, 20000))
            elif group_name == '10~100조':
                # 중과금: 월 1~5만원대 과금
                monthly_spending = int(np.clip(
                    np.random.lognormal(9.1, 0.7), 2000, 80000))
            elif group_name == '100~1000조':
                # 고과금: 월 5~25만원대 과금
                monthly_spending = int(np.clip(
                    np.random.lognormal(10.9, 0.7), 10000, 500000))
            else:  # 1000조이상
                # 초고과금 고래: 월 수백만원 과금 (현질 규모가 다름)
                monthly_spending = int(np.clip(
                    np.random.lognormal(14.3, 0.6), 300000, 10000000))

            users.append({
                'user_id': user_id,
                'nickname': nickname,
                'spendig': spendig,
                'club_value': round(club_value),
                'club_value_group': group_name,
                'avg_ovr': avg_ovr,
                'membership_tier': membership_tier,
                'monthly_membership_fee': monthly_fee,
                'monthly_avg_spending': monthly_spending,
                'register_date': register_date.strftime('%Y-%m-%d'),
                'position_player': position_player,
            })

    # 그룹별 유저 수 합이 50000이 안될 수 있으므로 마지막 그룹에서 보정
    df = pd.DataFrame(users)

    # 부족분을 10~100조 그룹에서 추가 (가장 큰 그룹)
    deficit = N_USERS - len(df)
    if deficit > 0:
        extra_users = []
        config = GROUP_CONFIG['10~100조']
        for i in range(deficit):
            user_id_counter += 1
            cv_min, cv_max = config['club_value_range']
            log_mean = (np.log(cv_min) + np.log(cv_max)) / 2
            log_std = (np.log(cv_max) - np.log(cv_min)) / 6
            club_value = np.exp(np.random.normal(log_mean, log_std))
            club_value = np.clip(club_value, cv_min * 1.01, cv_max * 0.99)
            avg_ovr = round(np.clip(
                np.random.normal(config['ovr_mean'], config['ovr_std']),
                config['ovr_min'], config['ovr_max']
            ), 1)
            spendig = max(int(np.random.lognormal(10, 1.2)),
                         int(np.random.uniform(5000, 20000)))

            tier = np.random.choice(
                MEMBERSHIP_TIERS, p=[0.15, 0.28, 0.32, 0.17, 0.08])
            # 10~100조 그룹 월 평균 과금
            m_spending = int(np.clip(
                np.random.lognormal(9.1, 0.7), 2000, 80000))
            extra_users.append({
                'user_id': f"U{user_id_counter}",
                'nickname': f"Player_{user_id_counter}",
                'spendig': spendig,
                'club_value': round(club_value),
                'club_value_group': '10~100조',
                'avg_ovr': avg_ovr,
                'membership_tier': tier,
                'monthly_membership_fee': MEMBERSHIP_MONTHLY_FEE[tier],
                'monthly_avg_spending': m_spending,
                'register_date': (DATE_START - datetime.timedelta(
                    days=random.randint(30, 365))).strftime('%Y-%m-%d'),
                'position_player': random.choice(
                    PLAYER_POOL.get(133, ['드브라위너_133'])),
            })
        df = pd.concat([df, pd.DataFrame(extra_users)], ignore_index=True)

    # 셔플하여 그룹 순서 섞기
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  → user_profile 생성 완료: {len(df)}명")
    print(f"  → 그룹 분포:\n{df['club_value_group'].value_counts().to_string()}")

    return df


# ============================================================================
# 4. login_logs 생성
# ============================================================================
def generate_login_logs(user_profile):
    """
    로그인 로그 테이블 생성
    스키마: user_id, nickname, login_date, session_duration_min

    핵심 로직 (2단계 접근):
    Step A: 각 유저의 이탈 여부 및 이탈 시점을 사전 결정
    Step B: 결정된 이탈 스케줄에 따라 일별 로그인 생성

    이탈률 목표 (설계서 기반, 30일 미접속 기준):
    - 10~100조: 25~35% (가장 높음, sensitivity=0.80)
    - 0~10조: 15~25%
    - 100~1000조: 8~15%
    - 1000조이상: 3~8%
    """
    print("[2/4] login_logs 생성 중...")

    # ── Step A: 이탈 스케줄 사전 결정 ──
    # 그룹별 이탈률 목표 (자연스러운 범위)
    CHURN_RATE_TARGET = {
        '0~10조': 0.20,        # 약 20%
        '10~100조': 0.30,      # 약 30% (가장 높음)
        '100~1000조': 0.12,    # 약 12%
        '1000조이상': 0.05,    # 약 5%
    }

    # 이탈 시점 분포: 패키지 출시 후 1~4주 사이에 이탈 집중
    # 각 패키지 출시 후 이탈이 발생하며, 후반 패키지일수록 누적 효과로 이탈 증가
    churn_schedule = {}  # {user_id: churn_date} — 이탈 시작일

    for group_name, config in GROUP_CONFIG.items():
        group_users = user_profile[user_profile['club_value_group'] == group_name]['user_id'].tolist()
        target_rate = CHURN_RATE_TARGET[group_name]
        n_churn = int(len(group_users) * target_rate)

        # 이탈 유저 랜덤 선택
        churn_users = random.sample(group_users, n_churn)

        for uid in churn_users:
            # 이탈 시점: 패키지 출시 후 7~35일 사이 (자연스러운 분포)
            # 후반 패키지에서 이탈 확률 더 높음 (누적 불만)
            pkg_idx = np.random.choice([0, 1, 2], p=[0.20, 0.35, 0.45])
            pkg_date = PACKAGE_DATES[pkg_idx]
            # 이탈까지의 지연일: 감마 분포로 자연스럽게 (피크 10~20일)
            delay_days = int(np.random.gamma(3, 5)) + 5  # 최소 5일
            delay_days = min(delay_days, 40)  # 최대 40일
            churn_date = pkg_date + datetime.timedelta(days=delay_days)
            # 기간 내로 클리핑
            churn_date = min(churn_date, DATE_END - datetime.timedelta(days=31))
            churn_schedule[uid] = churn_date

    print(f"  → 이탈 예정 유저: {len(churn_schedule):,}명")

    # ── Step B: 일별 로그인 생성 ──
    all_logs = []
    user_nicknames = dict(zip(user_profile['user_id'], user_profile['nickname']))

    for _, user in user_profile.iterrows():
        uid = user['user_id']
        group = user['club_value_group']
        config = GROUP_CONFIG[group]
        base_prob = config['daily_login_prob_base']

        # 유저별 개인 변동성 (±15%)
        personal_factor = np.random.uniform(0.85, 1.15)

        # 이탈 여부 및 시점
        is_churner = uid in churn_schedule
        churn_date = churn_schedule.get(uid, None)

        for date in ALL_DATES:
            current_date = date.date()

            # 기본 접속 확률
            prob_today = base_prob * personal_factor

            # 패키지 출시 이벤트 효과
            for pkg_date in PACKAGE_DATES:
                days_since = (current_date - pkg_date).days
                if days_since == 0:
                    prob_today *= 1.6  # 출시 당일 접속 급증
                elif 1 <= days_since <= 3:
                    prob_today *= 1.3
                elif 4 <= days_since <= 7:
                    prob_today *= 1.1

            # 이탈 유저의 접속 패턴
            if is_churner and current_date >= churn_date:
                days_after_churn = (current_date - churn_date).days
                # 이탈 후 접속 빈도 급감 (지수적 감소)
                # 처음 며칠은 가끔 접속, 이후 거의 접속 안 함
                prob_today = base_prob * 0.15 * np.exp(-days_after_churn / 8)
                prob_today = max(prob_today, 0.003)  # 완전 0은 아님

            # 이탈 직전 접속 빈도 감소 (이탈 예고 패턴)
            if is_churner and churn_date:
                days_to_churn = (churn_date - current_date).days
                if 0 < days_to_churn <= 14:
                    # 이탈 2주 전부터 서서히 접속 감소
                    decay = 0.5 + 0.5 * (days_to_churn / 14)
                    prob_today *= decay

            # 주말 효과
            weekday = current_date.weekday()
            if weekday >= 4:
                prob_today *= np.random.uniform(1.05, 1.18)

            # 확률 클리핑 (극단값 방지)
            prob_today = np.clip(prob_today, 0.002, 0.96)

            # 접속 여부 결정
            if random.random() < prob_today:
                # 세션 시간 (그룹별, 이탈 유저는 짧게)
                if group == '1000조이상':
                    base_session = np.random.lognormal(4.0, 0.7)
                elif group == '100~1000조':
                    base_session = np.random.lognormal(3.8, 0.8)
                elif group == '10~100조':
                    base_session = np.random.lognormal(3.5, 0.9)
                else:
                    base_session = np.random.lognormal(3.0, 1.0)

                session_min = max(2, int(base_session))

                # 이탈 후 접속 시 짧은 세션
                if is_churner and current_date >= churn_date:
                    session_min = max(1, int(session_min * 0.3))

                session_min = min(session_min, 480)

                all_logs.append({
                    'user_id': uid,
                    'nickname': user_nicknames[uid],
                    'login_date': current_date.strftime('%Y-%m-%d'),
                    'session_duration_min': session_min,
                })

    df_logs = pd.DataFrame(all_logs)
    print(f"  → login_logs 생성 완료: {len(df_logs):,}행")

    return df_logs


# ============================================================================
# 5. package_purchase 생성
# ============================================================================
def generate_package_purchase(user_profile):
    """
    패키지 구매 로그 테이블 생성
    스키마: package_id, package_per, user_id, purchase_date, ovr, ovr_qty, player

    핵심 로직:
    - 패키지 출시일 전후 7일에 구매 집중
    - Pack B가 최다 판매 (설계서 Bar Chart)
    - 패키지 내 높은 확률 OVR: 133(40%), 130(19%) 등 (설계서 Bubble Chart)
    - 구매 확률은 그룹별로 차이
    """
    print("[3/4] package_purchase 생성 중...")

    all_purchases = []
    purchase_counter = 0

    for _, user in user_profile.iterrows():
        uid = user['user_id']
        group = user['club_value_group']
        config = GROUP_CONFIG[group]

        base_purchase_prob = config['purchase_prob']

        # 각 패키지 출시 시점에 대해 구매 여부 결정
        for pkg_release_idx, pkg_date in enumerate(PACKAGE_DATES):
            # 출시일 기준 ±7일 구매 윈도우
            for day_offset in range(-2, 8):
                buy_date = pkg_date + datetime.timedelta(days=day_offset)

                # 출시일 당일/다음날에 구매 확률 가장 높음
                if day_offset == 0:
                    day_multiplier = 2.5
                elif day_offset == 1:
                    day_multiplier = 2.0
                elif day_offset == 2:
                    day_multiplier = 1.5
                elif day_offset < 0:
                    day_multiplier = 0.3  # 사전예약 느낌
                else:
                    day_multiplier = max(0.3, 1.0 - (day_offset - 2) * 0.15)

                final_buy_prob = base_purchase_prob * day_multiplier * 0.15
                final_buy_prob = np.clip(final_buy_prob, 0.005, 0.45)

                if random.random() < final_buy_prob:
                    purchase_counter += 1

                    # 패키지 선택 (Pack B 최다)
                    pkg_id = np.random.choice(PACKAGE_IDS, p=PACKAGE_WEIGHTS)

                    # 패키지 가격
                    price_min, price_max = PACKAGE_PRICES[pkg_id]
                    amount = random.randint(price_min, price_max)

                    # 패키지에서 뽑은 OVR (설계서 확률 분포)
                    ovr_options = list(PACKAGE_OVR_PROBS.keys())
                    ovr_probs = list(PACKAGE_OVR_PROBS.values())
                    # 확률 합 정규화
                    total = sum(ovr_probs)
                    ovr_probs = [p / total for p in ovr_probs]

                    obtained_ovr = np.random.choice(ovr_options, p=ovr_probs)

                    # 뽑은 선수 수량 (1~3명, 대부분 1명)
                    ovr_qty = np.random.choice([1, 2, 3], p=[0.65, 0.25, 0.10])

                    # 뽑은 선수 이름
                    if obtained_ovr in PLAYER_POOL:
                        player = random.choice(PLAYER_POOL[obtained_ovr])
                    else:
                        player = f"선수_{obtained_ovr}"

                    # 패키지 확률 (해당 선수의 등장 확률)
                    package_per = round(PACKAGE_OVR_PROBS.get(obtained_ovr, 0.01) * 100, 2)
                    # 자연스러운 변동 추가
                    package_per = round(package_per + np.random.uniform(-2, 2), 2)
                    package_per = max(0.5, min(package_per, 45.0))

                    all_purchases.append({
                        'purchase_id': f"PUR_{purchase_counter:07d}",
                        'package_id': pkg_id,
                        'package_per': package_per,
                        'user_id': uid,
                        'purchase_date': buy_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'amount': amount,
                        'ovr': int(obtained_ovr),
                        'ovr_qty': int(ovr_qty),
                        'player': player,
                    })

    df_purchase = pd.DataFrame(all_purchases)

    # 시간 추가 (자연스러운 구매 시간 분포: 오후~저녁 집중)
    hours = np.random.choice(
        range(24), size=len(df_purchase),
        p=[0.008, 0.005, 0.005, 0.005, 0.005, 0.007,  # 0~5시
           0.01, 0.02, 0.03, 0.04, 0.05, 0.06,        # 6~11시
           0.065, 0.065, 0.07, 0.07, 0.07, 0.065,     # 12~17시
           0.075, 0.075, 0.065, 0.055, 0.04, 0.04]    # 18~23시
    )
    minutes = np.random.randint(0, 60, size=len(df_purchase))
    seconds = np.random.randint(0, 60, size=len(df_purchase))

    df_purchase['purchase_date'] = pd.to_datetime(df_purchase['purchase_date'])
    df_purchase['purchase_date'] = df_purchase['purchase_date'] + \
        pd.to_timedelta(hours, unit='h') + \
        pd.to_timedelta(minutes, unit='m') + \
        pd.to_timedelta(seconds, unit='s')

    df_purchase = df_purchase.sort_values('purchase_date').reset_index(drop=True)

    print(f"  → package_purchase 생성 완료: {len(df_purchase):,}행")
    print(f"  → 패키지별 판매량:\n{df_purchase['package_id'].value_counts().to_string()}")

    return df_purchase


# ============================================================================
# 6. trade_market 생성
# ============================================================================
def generate_trade_market(user_profile):
    """
    이적시장 거래 로그 테이블 생성
    스키마: purchase_id, user_id, trade_date, price_UP, price_LOW,
            price_trade, ovr, player_id, trade_volume

    핵심 로직:
    - 패키지 출시 후 높은확률 OVR 선수의 가격 급락 (공급 증가)
    - 가격 변동이 Z-score 기반 충격 분석 가능하도록 설계
    - OVR별 기본 가격대 설정, 패키지 전/후 가격 변화 반영
    - 설계서 OVR별 라인 그래프 참조: OVR 130 가격 가장 큰 하락
    """
    print("[4/4] trade_market 생성 중...")

    # OVR별 기본 가격 (100억 단위, 설계서 Y축 참조)
    OVR_BASE_PRICES = {
        125: 8,  126: 12,  127: 18,  128: 25,  129: 32,
        130: 42, 131: 55,  132: 65,  133: 75,
        134: 85, 135: 82,  136: 98,
        137: 110, 138: 150, 139: 200, 140: 300, 141: 450, 142: 600,
    }

    # 패키지 출시로 인한 OVR별 가격 하락률 (설계서 라인 그래프)
    # OVR 130: ~30% 하락, OVR 133: ~15% 하락 (확률 높은 순으로 하락 큼)
    PKG_PRICE_DROP = {
        125: 0.03, 126: 0.04, 127: 0.05, 128: 0.08, 129: 0.10,
        130: 0.30, 131: 0.20, 132: 0.15, 133: 0.18,
        134: 0.10, 135: 0.08, 136: 0.05,
        137: 0.03, 138: 0.02, 139: 0.02, 140: 0.01, 141: 0.01, 142: 0.01,
    }

    all_trades = []
    trade_counter = 0

    # 활성 유저 중 거래 참여자 선별
    active_traders = user_profile[
        user_profile['club_value_group'].map(
            lambda g: random.random() < GROUP_CONFIG[g]['trade_activity']
        )
    ]['user_id'].tolist()

    for date in ALL_DATES:
        current_date = date.date()

        # 해당 날짜의 거래 가능한 OVR 목록
        for ovr in range(125, 143):
            if ovr not in OVR_BASE_PRICES:
                continue

            base_price = OVR_BASE_PRICES[ovr]

            # 패키지 출시 효과 계산
            price_modifier = 1.0
            volume_modifier = 1.0

            for pkg_date in PACKAGE_DATES:
                days_since = (current_date - pkg_date).days
                drop_rate = PKG_PRICE_DROP.get(ovr, 0.05)

                if 0 <= days_since <= 2:
                    # 출시 직후: 급격한 가격 하락 + 거래량 급증
                    price_modifier *= (1 - drop_rate * 0.8)
                    volume_modifier *= 3.0
                elif 2 < days_since <= 5:
                    price_modifier *= (1 - drop_rate * 0.6)
                    volume_modifier *= 2.0
                elif 5 < days_since <= 14:
                    # 점진적 회복 but 완전 회복은 안 됨
                    recovery = min(0.5, days_since * 0.03)
                    price_modifier *= (1 - drop_rate * (0.5 - recovery))
                    volume_modifier *= max(1.2, 2.0 - days_since * 0.08)
                elif 14 < days_since <= 30:
                    # 일부 회복
                    price_modifier *= (1 - drop_rate * 0.15)
                    volume_modifier *= 1.1

            # 일별 자연 변동성 추가 (±5%)
            daily_noise = np.random.normal(1.0, 0.03)
            current_price = base_price * price_modifier * daily_noise
            current_price = max(current_price, base_price * 0.3)  # 최저 30%까지만

            # 상한가/하한가 계산
            spread = current_price * np.random.uniform(0.05, 0.15)
            price_up = round(current_price + spread, 2)
            price_low = round(max(current_price - spread, current_price * 0.7), 2)
            price_trade = round(current_price + np.random.uniform(-spread * 0.5, spread * 0.5), 2)

            # 거래가가 상한가/하한가 사이에 있도록 보장
            price_trade = np.clip(price_trade, price_low, price_up)

            # 해당 OVR의 일일 거래량
            base_volume = max(3, int(ovr * 0.5 - 50))  # OVR 높을수록 거래량 적음
            if ovr <= 133:
                base_volume *= 3  # 중간 OVR이 거래량 많음

            daily_volume = max(1, int(base_volume * volume_modifier * np.random.uniform(0.6, 1.4)))

            # 거래 개수: 일일 거래량에 비례 (너무 많으면 샘플링)
            n_trades = min(daily_volume, 15)

            # 선수 선택
            if ovr in PLAYER_POOL:
                available_players = PLAYER_POOL[ovr]
            else:
                available_players = [f"선수_{ovr}"]

            for _ in range(n_trades):
                trade_counter += 1

                # 거래 참여 유저 선택
                trader = random.choice(active_traders) if active_traders else f"U{random.randint(100000, 150000)}"

                # 개별 거래 가격 변동
                individual_noise = np.random.uniform(0.97, 1.03)
                ind_price_trade = round(price_trade * individual_noise, 2)
                ind_price_up = round(price_up * individual_noise, 2)
                ind_price_low = round(price_low * individual_noise, 2)

                # 가격이 0 이하가 되지 않도록 보장
                ind_price_trade = max(ind_price_trade, 0.5)
                ind_price_low = max(ind_price_low, 0.3)
                ind_price_up = max(ind_price_up, ind_price_trade * 1.01)

                player_id = random.choice(available_players)

                all_trades.append({
                    'purchase_id': f"TRD_{trade_counter:08d}",
                    'user_id': trader,
                    'trade_date': current_date.strftime('%Y-%m-%d'),
                    'price_UP': ind_price_up,
                    'price_LOW': ind_price_low,
                    'price_trade': ind_price_trade,
                    'ovr': ovr,
                    'player_id': player_id,
                    'trade_volume': daily_volume,
                })

    df_trade = pd.DataFrame(all_trades)
    print(f"  → trade_market 생성 완료: {len(df_trade):,}행")

    return df_trade


# ============================================================================
# 7. daily_club_value 생성 (유저별 일별 구단가치 변동)
# ============================================================================
def generate_daily_club_value(user_profile, trade_market):
    """
    유저별 일별 구단가치 변동 테이블 생성 (샘플링 + 벡터화 최적화)
    스키마: user_id, date, club_value, club_value_index, club_value_group,
            daily_change_rate, z_score

    핵심 로직:
    - 그룹별 1,000명 샘플링 → 총 4,000명 × 120일 = 48만 행
    - 각 유저의 초기 구단가치에서 시작
    - 패키지 출시 후 그룹별 하락률 적용 (설계서 표 기반)
    - Z-score: 유저별 일별 변동률의 표준화 점수
    - 10~100조 그룹이 가장 큰 하락 경험

    하락률 (설계서 p.10):
      0~10조:    1/18→10%, 2/20→17%, 3/20→15%
      10~100조:  1/18→23%, 2/20→25%, 3/20→26%  (최대)
      100~1000조: 1/18→5%, 2/20→3.6%, 3/20→3%
      1000조이상: 1/18→1%, 2/20→1%, 3/20→1%
    """
    print("[5/5] daily_club_value 생성 중 (샘플링 + 벡터화)...")

    SAMPLE_PER_GROUP = 1000  # 그룹별 샘플 크기

    # 그룹별 패키지 출시 후 하락률 (설계서 기반)
    DECLINE_RATES = {
        '0~10조':      [0.10, 0.17, 0.15],
        '10~100조':    [0.23, 0.25, 0.26],
        '100~1000조':  [0.05, 0.036, 0.03],
        '1000조이상':   [0.01, 0.01, 0.01],
    }

    # 패키지 출시일 → 날짜 인덱스 매핑
    pkg_dates = [
        datetime.date(2025, 1, 18),
        datetime.date(2025, 2, 20),
        datetime.date(2025, 3, 20),
    ]
    date_list = [d.date() for d in ALL_DATES]
    n_days = len(date_list)

    # 패키지 출시 후 일수 매트릭스 (n_days × 3)
    days_since_pkg = np.array([
        [(d - pkg_d).days for pkg_d in pkg_dates]
        for d in date_list
    ])

    all_dfs = []

    for group_name, config in GROUP_CONFIG.items():
        group_users = user_profile[user_profile['club_value_group'] == group_name]

        # 샘플링 (그룹 인원이 SAMPLE_PER_GROUP보다 적으면 전체 사용)
        n_sample = min(SAMPLE_PER_GROUP, len(group_users))
        sampled = group_users.sample(n=n_sample, random_state=SEED)
        user_ids = sampled['user_id'].values
        initial_cvs = sampled['club_value'].values.astype(float)
        declines = DECLINE_RATES[group_name]

        print(f"  → {group_name}: {n_sample}명 샘플링")

        # 벡터화: (n_users, n_days) 매트릭스로 구단가치 변동 계산
        n_users = len(user_ids)

        # 일별 자연 변동 노이즈 (±0.3%)
        daily_noise = np.random.normal(0, 0.003, size=(n_users, n_days))

        # 패키지 충격 효과를 일별 승수(multiplier)로 사전 계산
        pkg_multiplier = np.ones(n_days)
        for pkg_idx in range(3):
            ds = days_since_pkg[:, pkg_idx]  # (n_days,)
            drop = declines[pkg_idx]

            for day_i in range(n_days):
                d = ds[day_i]
                if d == 0:
                    pkg_multiplier[day_i] *= (1 - drop * 0.50)
                elif d == 1:
                    pkg_multiplier[day_i] *= (1 - drop * 0.20)
                elif d == 2:
                    pkg_multiplier[day_i] *= (1 - drop * 0.10)
                elif 3 <= d <= 5:
                    pkg_multiplier[day_i] *= (1 - drop * 0.03)
                elif 6 <= d <= 14:
                    pkg_multiplier[day_i] *= (1 + drop * 0.015)
                elif 15 <= d <= 25:
                    pkg_multiplier[day_i] *= (1 + drop * 0.005)

        # 유저별 개인 변동성 팩터 (±10%)
        personal_factor = np.random.uniform(0.90, 1.10, size=(n_users, 1))

        # 일별 총 승수 = 자연노이즈 + 패키지 충격 (브로드캐스트)
        # noise에 개인 변동성 곱함 → 유저마다 약간 다른 노이즈
        total_daily = (1 + daily_noise * personal_factor) * pkg_multiplier[np.newaxis, :]

        # 누적곱으로 구단가치 시계열 생성
        cv_matrix = initial_cvs[:, np.newaxis] * np.cumprod(total_daily, axis=1)

        # 최소값 보장 (초기값의 30% 이하로 안 떨어지게)
        floor = initial_cvs[:, np.newaxis] * 0.30
        cv_matrix = np.maximum(cv_matrix, floor)

        # DataFrame 구성
        date_strs = [d.strftime('%Y-%m-%d') for d in date_list]
        for u_idx in range(n_users):
            df_user = pd.DataFrame({
                'user_id': user_ids[u_idx],
                'date': date_strs,
                'club_value': np.round(cv_matrix[u_idx]).astype(int),
                'club_value_group': group_name,
            })
            all_dfs.append(df_user)

    df = pd.concat(all_dfs, ignore_index=True)

    # club_value_index 계산 (기준일=100)
    first_day = date_list[0].strftime('%Y-%m-%d')
    baseline = df[df['date'] == first_day].set_index('user_id')['club_value']
    df['baseline_cv'] = df['user_id'].map(baseline)
    df['club_value_index'] = (df['club_value'] / df['baseline_cv']) * 100
    df.drop('baseline_cv', axis=1, inplace=True)

    # daily_change_rate 계산 (전일 대비 변동률 %)
    df = df.sort_values(['user_id', 'date'])
    df['prev_cv'] = df.groupby('user_id')['club_value'].shift(1)
    df['daily_change_rate'] = ((df['club_value'] - df['prev_cv']) / df['prev_cv']) * 100
    # 첫날은 전일 데이터 없으므로 미세 노이즈(-0.05~0.05%)로 채움 (0 방지)
    first_day_mask = df['daily_change_rate'].isna()
    np.random.seed(42)
    df.loc[first_day_mask, 'daily_change_rate'] = np.random.uniform(
        -0.05, 0.05, size=first_day_mask.sum()
    )
    df.drop('prev_cv', axis=1, inplace=True)

    # z_score 계산 (유저별 변동률의 표준화)
    user_stats = df.groupby('user_id')['daily_change_rate'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['user_id', 'change_mean', 'change_std']
    user_stats['change_std'] = user_stats['change_std'].replace(0, 1)  # 0 방지
    df = df.merge(user_stats, on='user_id', how='left')
    df['z_score'] = (df['daily_change_rate'] - df['change_mean']) / df['change_std']
    df.drop(['change_mean', 'change_std'], axis=1, inplace=True)

    print(f"  → daily_club_value 생성 완료: {len(df):,}행")

    # 그룹별 평균 인덱스 변화 확인
    group_avg = df.groupby(['date', 'club_value_group'])['club_value_index'].mean()
    last_day = date_list[-1].strftime('%Y-%m-%d')
    print(f"  → 최종일 그룹별 평균 인덱스:")
    for group in ['0~10조', '10~100조', '100~1000조', '1000조이상']:
        try:
            val = group_avg.loc[(last_day, group)]
            print(f"    {group}: {val:.1f}")
        except KeyError:
            pass

    return df


# ============================================================================
# 8. 데이터 품질 검증
# ============================================================================
def validate_data(user_profile, login_logs, package_purchase, trade_market):
    """
    생성된 데이터의 품질을 검증
    - 0%, 100% 같은 극단값이 없는지 확인
    - 그룹 분포가 올바른지 확인
    - 결측값 확인
    """
    print("\n" + "="*60)
    print("데이터 품질 검증")
    print("="*60)

    issues = []

    # 1. 그룹 분포 확인
    group_counts = user_profile['club_value_group'].value_counts(normalize=True) * 100
    print(f"\n[검증1] 그룹 분포:")
    for group, pct in group_counts.items():
        print(f"  {group}: {pct:.1f}%")
        if group == '1000조이상' and pct > 5:
            issues.append(f"1000조이상 그룹이 5% 초과: {pct:.1f}%")

    # 2. OVR 범위 확인 (0, 100 같은 극단값 없는지)
    ovr_stats = user_profile.groupby('club_value_group')['avg_ovr'].agg(['mean', 'min', 'max'])
    print(f"\n[검증2] 그룹별 OVR 통계:")
    print(ovr_stats)

    # 3. 이탈률 계산 및 확인
    last_login = login_logs.groupby('user_id')['login_date'].max().reset_index()
    last_login['login_date'] = pd.to_datetime(last_login['login_date'])
    cutoff = pd.Timestamp(DATE_END) - pd.Timedelta(days=30)
    last_login['is_churned'] = last_login['login_date'] < cutoff

    # 그룹별 이탈률
    merged = last_login.merge(user_profile[['user_id', 'club_value_group']], on='user_id')
    churn_by_group = merged.groupby('club_value_group')['is_churned'].mean() * 100
    print(f"\n[검증3] 그룹별 이탈률:")
    for group, rate in churn_by_group.items():
        flag = "⚠️" if rate > 95 or rate < 1 else "✓"
        print(f"  {flag} {group}: {rate:.1f}%")
        if rate > 98 or rate < 0.5:
            issues.append(f"{group} 이탈률 극단값: {rate:.1f}%")

    # 4. 0값 확인
    zero_spending = (user_profile['spendig'] == 0).sum()
    if zero_spending > 0:
        issues.append(f"spendig=0 유저: {zero_spending}명")
        print(f"\n[검증4] ⚠️ spendig=0 유저 {zero_spending}명 발견")
    else:
        print(f"\n[검증4] ✓ spendig=0 유저 없음")

    # 5. 패키지 판매량 검증
    print(f"\n[검증5] 패키지 판매량:")
    print(f"  총 구매 건수: {len(package_purchase):,}")
    print(f"  패키지별:\n{package_purchase['package_id'].value_counts().to_string()}")

    # 6. 거래 가격 0 확인
    zero_price = (trade_market['price_trade'] <= 0).sum()
    if zero_price > 0:
        issues.append(f"거래가 0 이하: {zero_price}건")
    else:
        print(f"\n[검증6] ✓ 거래가 0 이하 없음")

    # 7. 결측값 확인
    for name, df in [('user_profile', user_profile), ('login_logs', login_logs),
                     ('package_purchase', package_purchase), ('trade_market', trade_market)]:
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            issues.append(f"{name}에 결측값 {null_count}개")
            print(f"\n[검증7] ⚠️ {name} 결측값: {null_count}개")

    if issues:
        print(f"\n⚠️ 발견된 이슈 {len(issues)}건:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✓ 모든 검증 통과!")

    return issues


# ============================================================================
# 8. 메인 실행
# ============================================================================
def main():
    print("="*60)
    print("FC Online 4 데이터 분석 프로젝트 - 데이터 생성")
    print(f"총 유저: {N_USERS:,}명 | 기간: {DATE_START} ~ {DATE_END}")
    print("="*60 + "\n")

    # 1) user_profile 생성
    user_profile = generate_user_profile()

    # 2) login_logs 생성
    login_logs = generate_login_logs(user_profile)

    # 3) package_purchase 생성
    package_purchase = generate_package_purchase(user_profile)

    # 4) trade_market 생성
    trade_market = generate_trade_market(user_profile)

    # 5) daily_club_value 생성 (유저별 일별 구단가치 변동)
    daily_club_value = generate_daily_club_value(user_profile, trade_market)

    # 6) 데이터 품질 검증
    issues = validate_data(user_profile, login_logs, package_purchase, trade_market)

    # 7) CSV 저장
    print("\n" + "="*60)
    print("CSV 파일 저장 중...")

    user_profile.to_csv(os.path.join(DATA_DIR, 'user_profile.csv'), index=False, encoding='utf-8-sig')
    login_logs.to_csv(os.path.join(DATA_DIR, 'login_logs.csv'), index=False, encoding='utf-8-sig')
    package_purchase.to_csv(os.path.join(DATA_DIR, 'package_purchase.csv'), index=False, encoding='utf-8-sig')
    trade_market.to_csv(os.path.join(DATA_DIR, 'trade_market.csv'), index=False, encoding='utf-8-sig')
    daily_club_value.to_csv(os.path.join(DATA_DIR, 'daily_club_value.csv'), index=False, encoding='utf-8-sig')

    print(f"  → {DATA_DIR}/ 에 5개 CSV 저장 완료")

    # 8) 데이터 요약 출력
    print("\n" + "="*60)
    print("데이터 요약")
    print("="*60)
    print(f"  user_profile:      {len(user_profile):>10,}행 × {len(user_profile.columns)}열")
    print(f"  login_logs:        {len(login_logs):>10,}행 × {len(login_logs.columns)}열")
    print(f"  package_purchase:  {len(package_purchase):>10,}행 × {len(package_purchase.columns)}열")
    print(f"  trade_market:      {len(trade_market):>10,}행 × {len(trade_market.columns)}열")
    print(f"  daily_club_value:  {len(daily_club_value):>10,}행 × {len(daily_club_value.columns)}열")

    return user_profile, login_logs, package_purchase, trade_market, daily_club_value


if __name__ == '__main__':
    main()
