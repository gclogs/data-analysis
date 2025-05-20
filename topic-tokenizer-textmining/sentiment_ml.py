import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import os
import json
import re
import pickle
import requests
from bs4 import BeautifulSoup
import time
import random

# KNU 감성 사전 로드 함수
def load_knu_sentiment_dict(file_path=None):
    """
    KNU 감성 사전을 로드하여 단어-감성 점수 딕셔너리를 반환
    
    Parameters:
    file_path (str): 감성 사전 파일 경로. 기본값은 None으로, None일 경우 기본 경로 사용
    
    Returns:
    dict: 단어를 키로, 감성 점수를 값으로 하는 딕셔너리
    """
    print("KNU 감성 사전 로드 중...")
    
    # 기본 경로 설정
    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "knu", "SentiWord_info.json")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sentiment_info = json.load(f)
        
        # 단어-감성 점수 딕셔너리 생성
        sentiment_dict = {}
        for item in sentiment_info:
            word = item['word']
            polarity = float(item['polarity'])  # 문자열 값을 숫자로 변환
            sentiment_dict[word] = polarity
            
        print(f"감성 사전 로드 완료: {len(sentiment_dict)}개 단어")
        return sentiment_dict
    except Exception as e:
        print(f"감성 사전 로드 오류: {e}")
        return {}

# 형태소 분석기를 이용한 텍스트 전처리 함수
def preprocess_text(text, mecab, is_political=False):
    # 텍스트 정제
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
    
    # 형태소 분석 - MeCab은 (형태소, 품사) 튜플 리스트를 반환
    pos_tagged = mecab.pos(text)
    # 명사, 동사, 형용사만 추출 (필요에 따라 조정 가능)
    tokens = [word for word, pos in pos_tagged if pos.startswith('N') or pos.startswith('V') or pos.startswith('XR')]
    
    # 불용어 제거 (필요에 따라 확장)
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    tokens = [token for token in tokens if token not in stopwords and len(token) > 1]
    
    # 정치 관련 용어에 가중치 부여 (정치 텍스트 분석 시)
    if is_political:
        political_terms = ['경제', '복지', '정책', '국민', '일자리', '미래', '혁신', '개혁', '안보', '평화', 
                           '민주', '권리', '자유', '평등', '교육', '환경', '기후', '공정', '정의', '세금',
                           '재정', '예산', '국가', '사회', '문화', '기술', '의료', '건강', '외교', '안전']
        
        weighted_tokens = []
        for token in tokens:
            if token in political_terms:
                # 정치 용어 중복 추가로 가중치 부여
                weighted_tokens.extend([token, token])
            else:
                weighted_tokens.append(token)
        tokens = weighted_tokens
    
    return ' '.join(tokens)

# 정책 분야 분류를 위한 함수
def classify_policy_area(text, mecab):
    # 정책 분야별 키워드
    policy_areas = {
        '경제': ['경제', '재정', '세금', '일자리', '투자', '성장', '기업', '물가', '부동산', '주택', '금융', '소득', '무역', '산업'],
        '복지': ['복지', '연금', '건강보험', '의료', '돌봄', '아동', '노인', '장애인', '기초생활', '사회안전망', '사회보장', '출산', '육아'],
        '교육': ['교육', '학교', '대학', '학생', '등록금', '장학금', '사교육', '직업교육', '평생교육', '인재', '학습', '취업'],
        '환경': ['환경', '기후', '탄소', '에너지', '녹색', '재생에너지', '미세먼지', '쓰레기', '자원순환', '지속가능', '생태', '오염'],
        '안보': ['안보', '국방', '군사', '북한', '동맹', '평화', '외교', '국제', '핵무기', '군대', '안전', '방위', '통일']
    }
    
    # 형태소 분석 - 명사만 추출
    nouns = [word for word, pos in mecab.pos(text) if pos.startswith('N')]
    
    # 각 정책 분야별 점수 계산
    scores = {area: 0 for area in policy_areas}
    for noun in nouns:
        for area, keywords in policy_areas.items():
            if noun in keywords:
                scores[area] += 1
    
    # 가장 높은 점수의 정책 분야 반환 (동점이면 사전순 첫 번째)
    if all(score == 0 for score in scores.values()):
        return '기타'
    
    return max(scores.items(), key=lambda x: x[1])[0]

# 대통령 후보 연설문 데이터 로드 또는 생성
def load_or_create_speech_data(sample_file_path=None):
    """
    대통령 후보 연설문 데이터를 로드하거나 샘플 데이터를 생성합니다.
    실제 프로젝트에서는 이 부분을 실제 데이터로 대체해야 합니다.
    """
    if sample_file_path and os.path.exists(sample_file_path):
        return pd.read_csv(sample_file_path)
    
    # 샘플 연설문 데이터 생성 (실제 데이터로 대체 필요)
    speeches = [
        # 경제 관련 연설문 (긍정)
        "우리 경제는 활력을 되찾고 있습니다. 일자리가 늘어나고 투자가 확대되는 선순환 구조를 만들겠습니다.",
        "경제 성장을 위한 규제 혁신과 기업 투자 활성화가 필요합니다. 우리 경제의 미래를 위한 새로운 성장 동력을 발굴하겠습니다.",
        "중소기업과 소상공인이 함께 성장하는 경제 생태계를 조성하겠습니다. 상생의 경제 모델을 만들어 나가겠습니다.",
        
        # 복지 관련 연설문 (긍정)
        "모든 국민이 누릴 수 있는 공정한 복지 정책을 강화하겠습니다. 복지의 사각지대를 없애겠습니다.",
        "어린이부터 어르신까지 생애주기별 맞춤형 복지 체계를 구축하겠습니다. 누구도 소외되지 않는 사회를 만들겠습니다.",
        
        # 환경 관련 연설문 (긍정)
        "기후 변화 대응과 녹색 산업 육성을 최우선 과제로 삼겠습니다. 지속가능한 발전을 추구하겠습니다.",
        "우리 아이들에게 깨끗한 환경을 물려주기 위해 탄소중립 정책을 적극 추진하겠습니다.",
        
        # 교육 관련 연설문 (긍정)
        "교육 불평등 해소를 위해 사교육비 경감 정책을 추진하겠습니다. 공교육의 질을 높이겠습니다.",
        "디지털 시대에 맞는 교육 혁신으로 미래 인재를 양성하겠습니다. 창의적인 교육 시스템을 구축하겠습니다.",
        
        # 안보 관련 연설문 (긍정)
        "국방 개혁과 병사 복지 향상을 통해 강한 군대를 만들겠습니다. 국민이 안심할 수 있는 안보 체계를 구축하겠습니다.",
        "굳건한 한미동맹을 바탕으로 평화와 번영의 한반도를 만들어 나가겠습니다.",
        
        # 부정적 연설문 (다양한 영역)
        "현 정부의 경제 정책은 실패했습니다. 물가는 오르고 일자리는 줄어들었습니다.",
        "무분별한 복지 확대는 재정 건전성을 해치고 미래 세대에게 부담을 줍니다.",
        "환경을 핑계로 한 과도한 규제는 산업 경쟁력을 저해합니다."
    ]
    
    # 각 연설문에 대한 감성과 정책 분야 수동 라벨링
    sentiments = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 1: 긍정, 0: 부정
    policy_areas = ['경제', '경제', '경제', '복지', '복지', '환경', '환경', '교육', '교육', '안보', '안보', '경제', '복지', '환경']
    
    # 데이터프레임 생성
    speech_data = pd.DataFrame({
        'document': speeches,
        'label': sentiments,
        'policy_area': policy_areas
    })
    
    return speech_data

# KNU 감성 사전을 사용한 감성 분석 함수
def analyze_sentiment_with_lexicon(text, mecab, sentiment_dict):
    """
    KNU 감성 사전을 사용하여 텍스트의 감성 점수를 계산
    """
    # MeCab으로 형태소 분석
    pos_tagged = mecab.pos(text)
    # 단어 목록 추출
    tokens = [word for word, _ in pos_tagged]
    
    score = 0
    matched_words = 0
    
    for token in tokens:
        if token in sentiment_dict:
            score += sentiment_dict[token]
            matched_words += 1
    
    # 매칭된 단어가 없으면 중립으로 간주
    if matched_words == 0:
        return "중립", 0.0
    
    # 점수 정규화 (매칭된 단어 수로 나누기)
    normalized_score = score / matched_words
    
    # 감성 레이블 결정
    if normalized_score > 0:
        return "긍정", normalized_score
    elif normalized_score < 0:
        return "부정", abs(normalized_score)
    else:
        return "중립", 0.0

# 메인 함수: 감성 사전 기반 및 ML 모델 결합
def train_sentiment_model(use_speech_data=True, use_sentiword_data=True, policy_area_classification=True, save_model=True):
    # KNU 감성 사전 로드
    sentiment_dict = load_knu_sentiment_dict()
    
    # 형태소 분석기 초기화 - MeCab으로 변경
    try:
        mecab = Mecab()
        print("MeCab 형태소 분석기 초기화 완료")
    except Exception as e:
        print(f"MeCab 초기화 오류: {e}")
        print("MeCab 설치 확인: https://konlpy.org/en/latest/install/")
        return None, None, None, None, sentiment_dict
    
    # 학습 데이터 준비
    training_data = pd.DataFrame()
    
    # 1. 연설문 데이터 로드 (선택적)
    if use_speech_data:
        speech_data = load_or_create_speech_data()
        print(f"연설문 데이터 로드 완료 (샘플 수: {len(speech_data)}개)")
        training_data = pd.concat([training_data, speech_data])
    
    # 2. SentiWord_info.json 데이터에서 학습 데이터 생성 (선택적)
    if use_sentiword_data and sentiment_dict:
        sentiword_data = create_training_data_from_sentiword(sentiment_dict, sample_size=2000)
        if not sentiword_data.empty:
            # 'policy_area' 열이 없으면 추가
            if 'policy_area' not in sentiword_data.columns:
                sentiword_data['policy_area'] = '기타'
            training_data = pd.concat([training_data, sentiword_data])
    
    # 학습 데이터 전처리 및 모델 학습
    sentiment_model = None
    policy_model = None
    tfidf_vectorizer = None
    
    if not training_data.empty:
        print(f"전체 학습 데이터 크기: {len(training_data)}개 샘플")
        print("텍스트 전처리 중...")
        
        # 'document' 열의 널값 확인 및 처리
        training_data = training_data.dropna(subset=['document'])
        
        # 텍스트 전처리 - MeCab 사용
        training_data['processed'] = training_data['document'].apply(
            lambda x: preprocess_text(x, mecab, is_political='policy_area' in training_data.columns)
        )
        
        # 텍스트 벡터화
        print("TF-IDF 벡터화 중...")
        tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
        train_tfidf = tfidf_vectorizer.fit_transform(training_data['processed'])
        
        # 감성 분석 모델 (RandomForest)
        print("감성 분석 모델 학습 중...")
        X_train, X_test, y_train, y_test = train_test_split(
            train_tfidf, training_data['label'], test_size=0.2, random_state=42
        )
        sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
        sentiment_model.fit(X_train, y_train)
        
        # 모델 평가
        y_pred = sentiment_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"감성 분석 모델 정확도: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # 정책 분야 분류 모델 (선택적)
        if policy_area_classification and 'policy_area' in training_data.columns:
            print("정책 분야 분류 모델 학습 중...")
            X_policy_train, X_policy_test, y_policy_train, y_policy_test = train_test_split(
                train_tfidf, training_data['policy_area'], test_size=0.2, random_state=42
            )
            policy_model = RandomForestClassifier(n_estimators=100, random_state=42)
            policy_model.fit(X_policy_train, y_policy_train)
            
            # 모델 평가
            y_policy_pred = policy_model.predict(X_policy_test)
            policy_accuracy = accuracy_score(y_policy_test, y_policy_pred)
            print(f"정책 분야 분류 모델 정확도: {policy_accuracy:.4f}")
        
        # 모델 저장 (선택적)
        if save_model:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # 감성 분석 모델 저장
            with open(os.path.join(model_dir, "sentiment_model.pkl"), "wb") as f:
                pickle.dump(sentiment_model, f)
            
            # 정책 분야 분류 모델 저장 (있는 경우)
            if policy_model is not None:
                with open(os.path.join(model_dir, "policy_model.pkl"), "wb") as f:
                    pickle.dump(policy_model, f)
            
            # TF-IDF 벡터라이저 저장
            with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "wb") as f:
                pickle.dump(tfidf_vectorizer, f)
            
            print(f"모델 저장 완료: {model_dir}")
    else:
        # 학습 데이터가 없을 경우 더미 모델 생성
        print("경고: 학습 데이터가 없어 더미 모델을 생성합니다.")
        tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
        dummy_texts = ["긍정 텍스트", "부정 텍스트"]
        tfidf_vectorizer.fit(dummy_texts)
    
    return sentiment_model, tfidf_vectorizer, mecab, policy_model, sentiment_dict

# 새로운 텍스트에 대한 감성 분석 및 정책 분야 분류 함수
def analyze_text(text, sentiment_model, tfidf_vectorizer, mecab, policy_model, sentiment_dict):
    """
    텍스트에 대한 종합 분석 수행: 
    1. 감성 사전 기반 감성 분석
    2. ML 모델 기반 감성 분석 (모델이 있는 경우)
    3. 정책 분야 분류 (모델이 있는 경우 모델 사용, 없으면 규칙 기반)
    """
    results = {}
    
    # 텍스트 전처리
    processed_text = preprocess_text(text, mecab, is_political=True)
    
    # 1. 감성 사전 기반 감성 분석
    lexicon_sentiment, lexicon_score = analyze_sentiment_with_lexicon(text, mecab, sentiment_dict)
    results['lexicon_sentiment'] = {
        'label': lexicon_sentiment,
        'score': lexicon_score
    }
    
    # 2. ML 모델 기반 감성 분석 (모델이 있는 경우)
    if sentiment_model is not None and tfidf_vectorizer is not None:
        # TF-IDF 벡터화
        text_tfidf = tfidf_vectorizer.transform([processed_text])
        
        # 감성 예측
        sentiment = sentiment_model.predict(text_tfidf)[0]
        probability = sentiment_model.predict_proba(text_tfidf)[0]
        
        sentiment_label = "긍정" if sentiment == 1 else "부정"
        confidence = probability[1] if sentiment == 1 else probability[0]
        
        results['model_sentiment'] = {
            'label': sentiment_label,
            'confidence': confidence
        }
    
    # 3. 정책 분야 분류
    if policy_model is not None and tfidf_vectorizer is not None:
        # TF-IDF 벡터화 (이미 수행되었으면 재사용)
        if 'text_tfidf' not in locals():
            text_tfidf = tfidf_vectorizer.transform([processed_text])
        
        # 정책 분야 예측
        policy_area = policy_model.predict(text_tfidf)[0]
    else:
        # 규칙 기반 분류
        policy_area = classify_policy_area(text, mecab)
    
    results['policy_area'] = policy_area
    
    return results

# 모델 로드 함수 추가
def load_models(model_dir=None):
    """
    저장된 모델 파일 로드
    
    Parameters:
    model_dir (str): 모델 파일이 저장된 디렉토리 경로
    
    Returns:
    tuple: (sentiment_model, tfidf_vectorizer, policy_model) 튜플
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    sentiment_model = None
    tfidf_vectorizer = None
    policy_model = None
    
    try:
        # 감성 분석 모델 로드
        sentiment_model_path = os.path.join(model_dir, "sentiment_model.pkl")
        if os.path.exists(sentiment_model_path):
            with open(sentiment_model_path, "rb") as f:
                sentiment_model = pickle.load(f)
            print("감성 분석 모델 로드 완료")
        
        # TF-IDF 벡터라이저 로드
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                tfidf_vectorizer = pickle.load(f)
            print("TF-IDF 벡터라이저 로드 완료")
        
        # 정책 분야 분류 모델 로드
        policy_model_path = os.path.join(model_dir, "policy_model.pkl")
        if os.path.exists(policy_model_path):
            with open(policy_model_path, "rb") as f:
                policy_model = pickle.load(f)
            print("정책 분야 분류 모델 로드 완료")
        
        return sentiment_model, tfidf_vectorizer, policy_model
    
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return None, None, None

# SentiWord_info.json 데이터를 활용한 학습 데이터 생성
def create_training_data_from_sentiword(sentiment_dict, sample_size=1000):
    """
    KNU 감성 사전에서 학습 데이터 생성
    
    Parameters:
    sentiment_dict (dict): 단어-감성 점수 딕셔너리
    sample_size (int): 생성할 샘플 수
    
    Returns:
    DataFrame: 학습용 데이터프레임
    """
    print(f"감성 사전에서 학습 데이터 생성 중 (목표 샘플 수: {sample_size})...")
    
    # 긍정/부정 단어 분류
    positive_words = [word for word, score in sentiment_dict.items() if score > 0]
    negative_words = [word for word, score in sentiment_dict.items() if score < 0]
    
    # 긍정/부정 단어 수 확인
    print(f"긍정 단어 수: {len(positive_words)}, 부정 단어 수: {len(negative_words)}")
    
    # 각 범주별 목표 샘플 수 계산
    target_per_category = sample_size // 2
    
    # 단어 샘플링 (범주별 목표 샘플 수 만큼, 중복 허용)
    positive_samples = random.choices(positive_words, k=target_per_category)
    negative_samples = random.choices(negative_words, k=target_per_category)
    
    # 학습 데이터 생성
    training_data = []
    
    # 긍정 샘플
    for word in positive_samples:
        training_data.append({
            'document': word,
            'label': 1  # 긍정
        })
    
    # 부정 샘플
    for word in negative_samples:
        training_data.append({
            'document': word,
            'label': 0  # 부정
        })
    
    # 데이터프레임 생성 및 섞기
    df = pd.DataFrame(training_data)
    df = df.sample(frac=1).reset_index(drop=True)  # 데이터 섞기
    
    print(f"학습 데이터 생성 완료: {len(df)}개 샘플")
    return df

# 웹 크롤링을 통한 뉴스 및 정치 관련 데이터 수집 함수
def crawl_news_data(keywords, num_pages=1, save_path=None):
    """
    requests와 BeautifulSoup을 사용한 뉴스 크롤링 함수
    
    Parameters:
    keywords (list): 검색 키워드 리스트
    num_pages (int): 수집할 페이지 수
    save_path (str): 결과를 저장할 경로
    
    Returns:
    DataFrame: 수집된 뉴스 데이터
    """
    news_data = []
    
    print(f"{', '.join(keywords)} 관련 뉴스 데이터 수집 중...")
    
    for keyword in keywords:
        for page in range(1, num_pages + 1):
            try:
                # 네이버 뉴스 검색 URL (예시)
                url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={keyword}&start={(page-1)*10+1}"
                
                # 요청 헤더 설정 (차단 방지)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # 요청 보내기
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 뉴스 아이템 추출 (예시, 실제 선택자는 웹사이트에 따라 다름)
                news_items = soup.select('.news_wrap')
                
                for item in news_items:
                    try:
                        # 제목과 요약 추출 (예시, 실제 선택자는 웹사이트에 따라 다름)
                        title_elem = item.select_one('.news_tit')
                        summary_elem = item.select_one('.dsc_wrap')
                        
                        if title_elem and summary_elem:
                            title = title_elem.text.strip()
                            summary = summary_elem.text.strip()
                            
                            news_data.append({
                                'keyword': keyword,
                                'title': title,
                                'summary': summary,
                                'content': f"{title} {summary}",
                                'label': None  # 라벨은 나중에 수동으로 추가하거나 모델로 예측
                            })
                    except Exception as e:
                        print(f"뉴스 아이템 파싱 오류: {e}")
                
                # 과도한 요청 방지를 위한 대기
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"페이지 크롤링 오류: {e}")
    
    # 데이터프레임 생성
    news_df = pd.DataFrame(news_data)
    
    # 결과 저장
    if save_path and len(news_data) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        news_df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"수집 데이터 저장 완료: {save_path}")
    
    print(f"뉴스 데이터 수집 완료: {len(news_data)}개 항목")
    return news_df

# 샘플 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="대통령 후보 연설문 감성 분석 및 정책 분야 분류")
    parser.add_argument("--train", action="store_true", help="새 모델 학습 여부")
    parser.add_argument("--crawl", action="store_true", help="뉴스 데이터 크롤링 여부")
    parser.add_argument("--keywords", nargs="+", default=["대통령", "후보", "연설"], help="크롤링 키워드")
    parser.add_argument("--pages", type=int, default=3, help="크롤링할 페이지 수")
    
    args = parser.parse_args()
    
    # 형태소 분석기 초기화 - MeCab으로 변경
    try:
        mecab_instance = Mecab()
        print("MeCab 형태소 분석기 초기화 완료")
    except Exception as e:
        print(f"MeCab 초기화 오류: {e}")
        print("MeCab 설치 확인: https://konlpy.org/en/latest/install/")
        exit(1)
    
    # 감성 사전 로드
    sentiment_dict = load_knu_sentiment_dict()
    
    # 크롤링 수행 (선택적)
    if args.crawl:
        crawl_data = crawl_news_data(
            keywords=args.keywords,
            num_pages=args.pages,
            save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "crawled_news.csv")
        )
        print(f"크롤링 결과: {len(crawl_data)}개 항목")
    
    # 모델 학습 또는 로드
    if args.train:
        print("새 모델 학습 중...")
        sentiment_model, vectorizer, _, policy_model, _ = train_sentiment_model(
            use_speech_data=True,
            use_sentiword_data=True,
            policy_area_classification=True,
            save_model=True
        )
    else:
        print("저장된 모델 로드 중...")
        sentiment_model, vectorizer, policy_model = load_models()
        
        # 모델 로드 실패 시 새 모델 학습
        if sentiment_model is None or vectorizer is None:
            print("저장된 모델을 찾을 수 없어 새 모델을 학습합니다.")
            sentiment_model, vectorizer, _, policy_model, _ = train_sentiment_model(
                use_speech_data=True,
                use_sentiword_data=True,
                policy_area_classification=True,
                save_model=True
            )
    
    # 샘플 텍스트 감성 분석
    sample_comments = [
        "경제 성장을 위한 규제 혁신 정책은 정말 기대됩니다!",
        "이 후보의 복지 공약은 실현 가능성이 낮아 보입니다.",
        "교육 정책이 구체적이지 않고 모호합니다.",
        "환경 문제에 대한 강력한 대책, 매우 환영합니다.",
        "안보와 국방 정책은 현실적이고 균형 잡힌 접근법이라고 생각합니다."
    ]
    
    print("\n샘플 텍스트 감성 분석 결과:")
    for comment in sample_comments:
        results = analyze_text(comment, sentiment_model, vectorizer, mecab_instance, policy_model, sentiment_dict)
        print(f"\n댓글: '{comment}'")
        
        # 결과 출력
        lexicon_result = results['lexicon_sentiment']
        print(f"→ 사전 감정: {lexicon_result['label']} (점수: {lexicon_result['score']:.4f})")
        
        if 'model_sentiment' in results:
            model_result = results['model_sentiment']
            print(f"→ 모델 감정: {model_result['label']} (확률: {model_result['confidence']:.4f})")
        
        print(f"→ 정책 분야: {results['policy_area']}")
    
    # 대화형 모드 실행
    print("\n직접 텍스트를 입력하여 분석해보세요 (종료하려면 'exit' 입력):")
    while True:
        user_input = input("\n분석할 텍스트 입력: ")
        if user_input.lower() == 'exit':
            break
        
        results = analyze_text(user_input, sentiment_model, vectorizer, mecab_instance, policy_model, sentiment_dict)
        
        # 결과 출력
        lexicon_result = results['lexicon_sentiment']
        print(f"→ 사전 감정: {lexicon_result['label']} (점수: {lexicon_result['score']:.4f})")
        
        if 'model_sentiment' in results:
            model_result = results['model_sentiment']
            print(f"→ 모델 감정: {model_result['label']} (확률: {model_result['confidence']:.4f})")
        
        print(f"→ 정책 분야: {results['policy_area']}")
