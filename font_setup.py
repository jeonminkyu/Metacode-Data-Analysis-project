"""
=============================================================================
FC Online 4 데이터 분석 프로젝트 - 폰트 설정 모듈
=============================================================================
한글 폰트를 OS에 맞게 자동 설정하는 중앙 모듈
- Windows: Malgun Gothic (맑은 고딕)
- macOS: AppleGothic
- Linux: NanumGothic (없으면 자동 설치)
=============================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import subprocess
import warnings

warnings.filterwarnings('ignore')


def setup_korean_font():
    """
    OS에 맞는 한글 폰트를 자동 감지하여 matplotlib에 적용
    Returns: 설정된 폰트 이름 (str)
    """
    system = platform.system()
    font_name = None

    # ── OS별 기본 한글 폰트 탐색 ──
    if system == 'Windows':
        # Windows: 맑은 고딕 → 굴림 → 돋움 순서로 탐색
        for candidate in ['Malgun Gothic', 'Gulim', 'Dotum', 'Batang']:
            if any(candidate.lower() in f.name.lower() for f in fm.fontManager.ttflist):
                font_name = candidate
                break
        # 시스템 폰트 경로에서 직접 탐색
        if font_name is None:
            win_font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
            for fname, ffile in [('Malgun Gothic', 'malgun.ttf'), ('Gulim', 'gulim.ttc')]:
                fpath = os.path.join(win_font_dir, ffile)
                if os.path.exists(fpath):
                    fm.fontManager.addfont(fpath)
                    font_name = fname
                    break

    elif system == 'Darwin':
        # macOS: AppleGothic → Apple SD Gothic Neo
        for candidate in ['AppleGothic', 'Apple SD Gothic Neo']:
            if any(candidate in f.name for f in fm.fontManager.ttflist):
                font_name = candidate
                break

    else:
        # Linux: NanumGothic (없으면 설치 시도)
        font_name = 'NanumGothic'
        if not any(font_name in f.name for f in fm.fontManager.ttflist):
            print("NanumGothic 폰트 설치 중...")
            try:
                subprocess.run(
                    ['apt-get', 'install', '-y', 'fonts-nanum'],
                    capture_output=True, timeout=60
                )
                fm._load_fontmanager(try_read_cache=False)
                print("  → NanumGothic 설치 완료")
            except Exception as e:
                print(f"  → 폰트 설치 실패: {e}")
                font_name = 'DejaVu Sans'

    # 폰트를 찾지 못한 경우 최종 fallback
    if font_name is None:
        # 시스템에서 한글 지원 폰트 자동 탐색
        korean_fonts = [f.name for f in fm.fontManager.ttflist
                        if any(kw in f.name.lower() for kw in
                               ['gothic', 'gulim', 'dotum', 'batang', 'nanum', 'malgun', 'apple'])]
        if korean_fonts:
            font_name = korean_fonts[0]
        else:
            font_name = 'DejaVu Sans'

    # matplotlib 전역 설정
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    return font_name


# 프로젝트 공통 색상 팔레트
COLORS = {
    '0~10조': '#4E79A7',        # 블루
    '10~100조': '#F28E2B',      # 오렌지 (주요 타겟 강조)
    '100~1000조': '#59A14F',    # 그린
    '1000조이상': '#E15759',    # 레드
}

# 순서 (항상 이 순서로 표시)
GROUP_ORDER = ['0~10조', '10~100조', '100~1000조', '1000조이상']

# 차트 스타일 설정
CHART_STYLE = {
    'title_fontsize': 14,
    'label_fontsize': 11,
    'tick_fontsize': 9,
    'legend_fontsize': 10,
    'figsize_single': (10, 6),
    'figsize_double': (14, 6),
    'figsize_large': (14, 10),
}
