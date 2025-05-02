# ===================================================
# PUBG 게임 데이터 분석
# ===================================================
# 이 코드는 PlayerUnknown's Battlegrounds (PUBG) 게임 데이터를 분석합니다.
# Kaggle 대회 데이터를 사용하며, 데이터 전처리, 탐색적 데이터 분석,
# 특성 엔지니어링 및 모델 준비 단계를 포함합니다.

# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

# 시각화 설정
plt.rcParams['figure.figsize'] = (12, 8)  # 그래프 크기 설정
plt.rcParams['font.size'] = 12  # 폰트 크기 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
plt.rcParams['axes.grid'] = False  # 그리드 라인 비활성화

# seaborn 색상 팔레트 및 스타일 설정
color = sns.color_palette()
plt.style.use('fivethirtyeight')

# matplotlib 폰트 설정
import matplotlib.font_manager as fm
parameters = {
    'axes.labelsize': 12,
    'axes.titlesize': 18,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
}
plt.rcParams.update(parameters)

# ===================================================
# 데이터 로드 및 기본 탐색
# ===================================================

# 작업 디렉토리 설정 및 데이터 로드
# 실제 사용 시 본인의 작업 디렉토리로 변경해주세요
try:
    os.chdir(r'c:\Users\ADMIN_PC\Desktop\data-analysis\kaggle-game-battleground')
except:
    print("작업 디렉토리 변경에 실패했습니다. 경로를 확인해주세요.")

# 데이터 로드 
try:
    train = pd.read_csv('dataset/train_V2.csv')
    print("데이터 로드 성공")
except:
    print("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    # 예시 데이터로 진행 (실제 데이터가 없을 경우)
    train = pd.DataFrame()  # 실제 환경에서는 이 라인을 제거하세요

# 데이터 기본 정보 확인
print("데이터셋 상위 5개 행:")
print(train.head())

print("\n데이터셋 크기 (행, 열):")
print(train.shape)

print("\n결측치 확인:")
print(train.isnull().sum())

# ===================================================
# 메모리 최적화 함수
# ===================================================
# 이 함수는 대용량 데이터셋의 메모리 사용량을 줄이기 위해
# 각 열의 데이터 타입을 최적화합니다.
# Credit: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):
    """ 
    데이터프레임의 모든 컬럼을 순회하며 데이터 타입을 메모리 사용량을 
    줄이기 위해 최적화합니다.
    """
    # 최적화 전 메모리 사용량 계산 (실제 환경에선 주석 해제)
    # start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    # 각 열을 순회하며 데이터 타입 최적화
    for col in df.columns:
        col_type = df[col].dtype
        
        # 숫자형 데이터만 최적화
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # 정수형 데이터 최적화
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            # 실수형 데이터 최적화
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    # 최적화 후 메모리 사용량 계산 (실제 환경에선 주석 해제)
    # end_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# 데이터 메모리 최적화 적용
if not train.empty:
    train = reduce_mem_usage(train)
    print("\n메모리 최적화 후 데이터 정보:")
    print(train.info())

# ===================================================
# 탐색적 데이터 분석 (EDA)
# ===================================================

# 데이터가 비어있지 않은 경우에만 시각화 수행
if not train.empty:
    # 킬 수(kills) 분포 시각화
    plt.figure(figsize=(12, 4))
    sns.distplot(train['kills'])
    plt.title('킬 수 분포', fontsize=16)
    plt.xlabel('킬 수')
    plt.ylabel('빈도')
    plt.show()
    
    # 킬 수 통계 출력
    print("\n kills 횟수 평균")
    print(train['kills'].mean())
    
    print("\n kills 횟수 최대값")
    print(train['kills'].max())
    
    # 이동 거리 관련 특성 분석
    plt.figure(figsize=(12, 4))
    
    # 걷기, 운전, 수영 이동 거리 합계 계산
    train['total_distance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']
    
    # 이동 거리 히스토그램
    sns.distplot(train['walkDistance'])
    plt.title('도보 이동 거리 분포', fontsize=16)
    plt.xlabel('도보 이동 거리')
    plt.ylabel('빈도')
    plt.show()
    
    # 데미지와 킬 수 관계 분석
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='damageDealt', y='kills', data=train, alpha=0.5)
    plt.title('데미지와 킬 수 관계', fontsize=16)
    plt.xlabel('가한 데미지')
    plt.ylabel('킬 수')
    plt.show()
    
    # 매치 시간에 따른 생존 분석
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='matchType', y='matchDuration', data=train)
    plt.title('매치 타입별 게임 시간', fontsize=16)
    plt.xlabel('매치 타입')
    plt.ylabel('매치 시간(초)')
    plt.xticks(rotation=90)
    plt.show()

    # ===================================================
    # 상관 관계 분석
    # ===================================================
    
    # 주요 특성 간 상관관계 분석
    important_features = ['kills', 'damageDealt', 'walkDistance', 'rideDistance', 
                          'weaponsAcquired', 'heals', 'boosts', 'winPlacePerc']
    
    correlation_df = train[important_features].corr()
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation_df, cmap='Greens', annot=True, linewidths=0.5, fmt='.3f', cbar=True)
    plt.title('상관 관계 행렬', size=20)
    plt.show()
    
    # ===================================================
    # 특성 중요도 분석 및 모델 준비
    # ===================================================
    
    # 모델링을 위한 특성과 타겟 분리
    # 여기서는 특성 중요도와 모델 준비 단계까지만 구현합니다
    # Random Forest 특성 중요도를 계산하는 예시 코드 (실제로는 주석 처리)
    
    '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # 타겟 변수와 특성 분리
    y = train['winPlacePerc']
    X = train.drop(['winPlacePerc', 'Id', 'groupId', 'matchId', 'matchType'], axis=1)
    
    # 모델 학습을 위한 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 특성 중요도 추출
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Feature Importance': rf.feature_importance_
    }).sort_values(by='Feature Importance', ascending=False)
    
    # 중요 특성만 선택
    important_features = feature_importance[feature_importance['Feature Importance'] > 0.05].index
    X_important = X[important_features]
    
    # 선택된 특성으로 상관관계 분석
    corr = X_important.corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, cmap='Greens', annot=True, linewidths=0.5, fmt='.3f', cbar=True)
    plt.title('중요 특성 간 상관 관계 행렬', size=20)
    plt.show()
    '''
    
    # ===================================================
    # 유용한 함수: 데이터 분할 함수 (실제 모델링에 사용)
    # ===================================================
    
    def split_vals(df, n_trn):
        """
        데이터프레임을 훈련 세트와 검증 세트로 분할하는 함수
        
        파라미터:
            df: 분할할 데이터프레임
            n_trn: 훈련 세트의 크기
            
        반환값:
            훈련 세트와 검증 세트
        """
        trn = df.iloc[:n_trn].copy()
        val = df.iloc[n_trn:].copy()
        return trn, val
    
    # 전체 데이터에 대한 모델 준비 (실제로는 필요에 따라 사용)
    if not train.empty:
        val_perc_full = 0.2  # 검증 세트 비율
        n_valid_full = int(val_perc_full * len(train))
        n_trn_full = len(train) - n_valid_full
        
        y = train['winPlacePerc']  # 타겟 변수
        df_full = train.drop('winPlacePerc', axis=1)  # 특성 변수
        
        # 훈련 세트와 검증 세트 분할
        X_train, X_valid = split_vals(df_full, n_trn_full)
        y_train, y_valid = split_vals(y, n_trn_full)
        
        print("훈련 세트 크기:", X_train.shape, "타겟:", y_train.shape, "검증 세트 크기:", X_valid.shape)

print("분석 완료!")