# app.py

import streamlit as st
import pandas as pd
import re
import time
import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# ====================
# 1. 모델 로드 (요약 + 감성)
# ====================

@st.cache_resource
def load_models():
    kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
    kobart_model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

    koelectra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
    koelectra_model = ElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-base-discriminator", num_labels=2)
    koelectra_model.eval()
    
    return kobart_tokenizer, kobart_model, koelectra_tokenizer, koelectra_model


# ====================
# 2. 텍스트 처리 함수
# ====================

def clean_text(text):
    text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s]', '', str(text))  # 특수문자 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def kobart_summarize(text, tokenizer, model):
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1).item()
    return "긍정" if predicted == 1 else "부정"


# ====================
# 3. 웹 크롤링 함수
# ====================

def crawl_reviews(url, pages=5):
    chrome_path = "C:/tools/chromedriver-win64/chromedriver.exe"
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    service = Service(chrome_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    review_list = []
    for page in range(pages):
        time.sleep(2)
        try:
            review_items = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div._1kMfD5ErZ6 > span._2L3vDiadT9'))
            )
            for r in review_items:
                text = r.text.strip()
                if text:
                    review_list.append(text)
            print(f"{page + 1}페이지 수집 완료")
        except:
            print(f"{page + 1}페이지 수집 실패")
            break
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, 'a.fAUKmK')
            next_btn.click()
        except:
            break
    driver.quit()
    return review_list


# ====================
# 4. 메인 Streamlit 앱
# ====================

def main():
    st.set_page_config(page_title="리뷰 요약 + 감성분석 앱", layout="wide")
    st.title("📝 네이버 리뷰 요약 & 감성분석")

    kobart_tokenizer, kobart_model, koelectra_tokenizer, koelectra_model = load_models()

    # 1단계: 리뷰 수집
    with st.expander("📌 Step 1. 리뷰 수집"):
        url = st.text_input("리뷰 페이지 URL 입력")
        pages = st.slider("수집할 페이지 수", 1, 10, 3)
        if st.button("리뷰 수집 시작"):
            with st.spinner("리뷰 수집 중..."):
                reviews = crawl_reviews(url, pages)
                df = pd.DataFrame(reviews, columns=['review'])
                df.to_csv("naver_reviews_raw.csv", index=False, encoding='utf-8-sig')
                st.success(f"✅ 총 {len(reviews)}개 리뷰 수집 완료!")

    # 2단계: 전처리
    with st.expander("🧹 Step 2. 전처리"):
        try:
            df = pd.read_csv("naver_reviews_raw.csv")
            df['cleaned'] = df['review'].apply(clean_text)
            st.write(df[['review', 'cleaned']].head())
            df.to_csv("naver_reviews_cleaned.csv", index=False, encoding='utf-8-sig')
            st.success("✅ 전처리 완료")
        except:
            st.warning("⚠️ 먼저 리뷰를 수집해주세요.")

    # 3단계: 요약 및 감성분석
    with st.expander("🔍 Step 3. 요약 & 감성분석"):
        try:
            df = pd.read_csv("naver_reviews_cleaned.csv")
            summaries = []
            sentiments = []
            progress = st.progress(0)
            for i, review in enumerate(df['cleaned']):
                try:
                    summary = kobart_summarize(review, kobart_tokenizer, kobart_model)
                    sentiment = predict_sentiment(review, koelectra_tokenizer, koelectra_model)
                except:
                    summary = "요약 실패"
                    sentiment = "분석 실패"
                summaries.append(summary)
                sentiments.append(sentiment)
                progress.progress((i+1)/len(df))
            df['summary'] = summaries
            df['sentiment'] = sentiments
            df.to_csv("naver_reviews_summary_sentiment.csv", index=False, encoding='utf-8-sig')
            st.dataframe(df[['cleaned', 'summary', 'sentiment']].head())
            st.success("✅ 요약 및 감성분석 완료")
        except:
            st.warning("⚠️ 전처리된 리뷰를 먼저 준비해주세요.")

    # 4단계: 시각화
    with st.expander("📊 Step 4. 감성분석 시각화"):
        try:
            df = pd.read_csv("naver_reviews_summary_sentiment.csv")
            sentiment_counts = df['sentiment'].value_counts()

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

            # WordCloud
            st.subheader("워드클라우드")
            pos_text = " ".join(df[df['sentiment'] == '긍정']['cleaned'])
            neg_text = " ".join(df[df['sentiment'] == '부정']['cleaned'])
            font_path = "C:/Windows/Fonts/malgun.ttf"

            wc = WordCloud(font_path=font_path, background_color='white', width=800, height=400)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 👍 긍정 리뷰 WordCloud")
                st.image(wc.generate(pos_text).to_array())
            with col2:
                st.markdown("### 👎 부정 리뷰 WordCloud")
                st.image(wc.generate(neg_text).to_array())
        except:
            st.warning("⚠️ 분석 데이터가 없습니다. 먼저 분석을 완료해주세요.")

if __name__ == "__main__":
    main()
