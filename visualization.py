# visualization.py

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

# 감성분석 결과를 막대그래프와 파이차트로 시각화
def plot_bar_pie(sentiment_counts):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("막대그래프")
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "red"])
        ax.set_title("긍정 vs 부정 리뷰 수")
        st.pyplot(fig)

    with col2:
        st.subheader("파이차트")
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["green", "red"])
        ax.set_title("리뷰 비율")
        st.pyplot(fig)

# 워드클라우드 시각화
def plot_wordcloud(df, font_path="C:/Windows/Fonts/malgun.ttf"):
    try:
        st.subheader("워드클라우드")
        pos_text = " ".join(df[df['sentiment'] == '긍정']['cleaned'])
        neg_text = " ".join(df[df['sentiment'] == '부정']['cleaned'])

        wc = WordCloud(font_path=font_path, background_color='white', width=800, height=400)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👍 긍정 리뷰 WordCloud")
            st.image(wc.generate(pos_text).to_array())
        with col2:
            st.markdown("### 👎 부정 리뷰 WordCloud")
            st.image(wc.generate(neg_text).to_array())
    except Exception as e:
        st.warning("⚠️ 분석 데이터가 없습니다. 먼저 분석을 완료해주세요.")
        st.text(f"에러 내용: {e}")
