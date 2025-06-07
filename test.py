import os
import json
import random
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from konlpy.tag import Mecab, Okt
from wordcloud import WordCloud

# ───────────────────────────────────────────────────────────
# 1) 감성 사전 로드 & 사전 기반 분석 함수
# ───────────────────────────────────────────────────────────

def load_knu_sentiment_dict(path="SentiWord_info.json"):
    """군산대 KNU 감성사전 로드"""
    with open(path, "r", encoding="utf-8") as f:
        senti = json.load(f)
    # 사전 키 정규화 (소문자 등)
    senti = {k.strip(): float(v) for k, v in senti.items()}
    return senti

def analyze_sentiment_with_lexicon(text, tokenizer, senti_dict):
    """
    텍스트를 형태소 분석 → 단어별 점수 합산 → 최종 레이블/점수 반환
    """
    tokens = tokenizer.morphs(text) if hasattr(tokenizer, "morphs") else tokenizer.pos(text)
    score = 0.0
    for tok in tokens:
        tok = tok.strip()
        if tok in senti_dict:
            score += senti_dict[tok]
    label = "긍정" if score > 0 else "부정" if score < 0 else "중립"
    return label, score

# ───────────────────────────────────────────────────────────
# 2) SentiWord 로부터 학습데이터 생성 (레이블 컬럼 통일)
# ───────────────────────────────────────────────────────────

def create_training_data_from_sentiword(senti_dict, sample_size=1000):
    pos = [w for w,s in senti_dict.items() if s>0]
    neg = [w for w,s in senti_dict.items() if s<0]
    n_each = sample_size // 2
    samples = (
        [(w,1) for w in random.choices(pos, k=n_each)] +
        [(w,0) for w in random.choices(neg, k=n_each)]
    )
    df = pd.DataFrame(samples, columns=["text","sentiment"])
    return df.sample(frac=1).reset_index(drop=True)

# ───────────────────────────────────────────────────────────
# 3) 전처리 함수 (예시: Mecab 기반)
# ───────────────────────────────────────────────────────────

def preprocess_text(text, tokenizer):
    # 간단 예: 소문자, 숫자/특수 제거, 토큰 재결합
    text = text.lower()
    tokens = tokenizer.morphs(text)
    tokens = [t for t in tokens if t.isalpha()]
    return " ".join(tokens)

# ───────────────────────────────────────────────────────────
# 4) 모델 학습 파이프라인
# ───────────────────────────────────────────────────────────

def train_pipeline(
    use_sentiword=True,
    sample_size=2000,
    tfidf_params=None,
    rf_params=None
):
    # 1) 준비
    mecab = Mecab()
    senti_dict = load_knu_sentiment_dict()
    
    # 2) 데이터 생성
    df_senti = create_training_data_from_sentiword(senti_dict, sample_size)
    df_senti["processed"] = df_senti["text"].apply(lambda x: preprocess_text(x, mecab))
    
    # 3) TF-IDF
    tfidf_params = tfidf_params or {"max_features":5000,"min_df":5,"max_df":0.7,"ngram_range":(1,2)}
    vect = TfidfVectorizer(**tfidf_params)
    X = vect.fit_transform(df_senti["processed"])
    y = df_senti["sentiment"]
    
    # 4) 학습/테스트 분리 & 학습
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
    rf_params = rf_params or {"n_estimators":100,"max_depth":10,"random_state":42,n_jobs:-1}
    model = RandomForestClassifier(**rf_params)
    model.fit(Xtr,ytr)
    
    # 5) 평가
    yp = model.predict(Xte)
    print("Accuracy:", accuracy_score(yte, yp))
    print(classification_report(yte, yp))
    
    return mecab, senti_dict, vect, model, df_senti

# ───────────────────────────────────────────────────────────
# 5) 분석 및 시각화 예시
# ───────────────────────────────────────────────────────────

def visualize_wordcloud(df_senti, senti_dict, top_n=100):
    # 긍정/부정 단어별 빈도 계산
    pos_tokens = [w for w,s in senti_dict.items() if s>0]
    text_pos = " ".join(df_senti[df_senti.sentiment==1]["processed"])
    wc = WordCloud(font_path="path/to/your/font.ttf", background_color="white",
                   width=800, height=400).generate(text_pos)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("긍정 단어 워드클라우드")
    plt.show()

# ───────────────────────────────────────────────────────────
# 6) 실행 예시
# ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    mecab, senti_dict, vect, model, df_senti = train_pipeline()
    visualize_wordcloud(df_senti, senti_dict)
