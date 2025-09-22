# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from crawler import crawl_reviews
from preprocess import clean_text
from analysis import load_models, kobart_summarize, predict_sentiment
from visualization import plot_bar_pie, plot_wordcloud

# ====================
# Streamlit 메인 앱
# ====================
def main():
    st.set_page_config(page_title="리뷰 요약 + 감성분석 앱", layout="wide")
    st.title("📝 네이버 리뷰 요약 & 감성분석")

    kobart_tokenizer, kobart_model, koelectra_tokenizer, koelectra_model = load_models()

    # Step 1. 리뷰 수집
    with st.expander("📌 Step 1. 리뷰 수집"):
        url = st.text_input("리뷰 페이지 URL 입력")
        pages = st.slider("수집할 페이지 수", 1, 10, 3)
        if st.button("리뷰 수집 시작"):
            with st.spinner("리뷰 수집 중..."):
                reviews = crawl_reviews(url, pages)
                df = pd.DataFrame(reviews, columns=['review'])
                df.to_csv("naver_reviews_raw.csv", index=False, encoding='utf-8-sig')
                st.success(f"✅ 총 {len(reviews)}개 리뷰 수집 완료!")

    # Step 2. 전처리
    with st.expander("🧹 Step 2. 전처리"):
        try:
            df = pd.read_csv("naver_reviews_raw.csv")
            df['cleaned'] = df['review'].apply(clean_text)
            st.write(df[['review', 'cleaned']].head())
            df.to_csv("naver_reviews_cleaned.csv", index=False, encoding='utf-8-sig')
            st.success("✅ 전처리 완료")
        except:
            st.warning("⚠️ 먼저 리뷰를 수집해주세요.")

    # Step 3. 요약 및 감성분석
    with st.expander("🔍 Step 3. 요약 & 감성분석"):
        try:
            df = pd.read_csv("naver_reviews_cleaned.csv")
            summaries, sentiments = [], []
            progress = st.progress(0)
            for i, review in enumerate(df['cleaned']):
                try:
                    summary = kobart_summarize(review, kobart_tokenizer, kobart_model)
                    sentiment = predict_sentiment(review, koelectra_tokenizer, koelectra_model)
                except:
                    summary, sentiment = "요약 실패", "분석 실패"
                summaries.append(summary)
                sentiments.append(sentiment)
                progress.progress((i+1)/len(df))
            df['summary'], df['sentiment'] = summaries, sentiments
            df.to_csv("naver_reviews_summary_sentiment.csv", index=False, encoding='utf-8-sig')
            st.dataframe(df[['cleaned', 'summary', 'sentiment']].head())
            st.success("✅ 요약 및 감성분석 완료")
        except:
            st.warning("⚠️ 전처리된 리뷰를 먼저 준비해주세요.")

    # Step 4. 시각화
    with st.expander("📊 Step 4. 감성분석 시각화"):
        try:
            df = pd.read_csv("naver_reviews_summary_sentiment.csv")
            sentiment_counts = df['sentiment'].value_counts()
            plot_bar_pie(sentiment_counts, st)
            plot_wordcloud(df, st)
        except:
            st.warning("⚠️ 분석 데이터가 없습니다. 먼저 분석을 완료해주세요.")

if __name__ == "__main__":
    main()

