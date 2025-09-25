
# naver_review_ai_project
# 네이버 스토어 리뷰 요약 & 감성분석

## 프로젝트 개요
본 프로젝트는 네이버 스토어 상품 리뷰 데이터를 자동으로 요약하고 감성(긍정/부정)을 분석하는 AI 기반 텍스트 분석 시스템입니다.
방대한 리뷰 데이터를 효율적으로 요약하여 핵심 의견만 빠르게 파악할 수 있도록 하며, 감성 분석을 통해 고객의 만족도와 불만 요소를
체계적으로 정리할 수 있습니다.
이는 판매자 및 마케터가 고객 피드백을 효과적으로 반영하여 제품 개선 및 전략 수립에 활용할 수 있도록 돕는 것을 목표로 합니다.

## 주요 기능
- 리뷰 자동 크롤링 ('crawler.py')
- 데이터 전처리 ('preprocess.py')
- 리뷰 요약 (KoBART, 'analysis.py')
- 감성 분석 (KoELECTRA, 'analysis.py')
- 결과 시각화 (WordCloud 및 통계, 'visualization.py')
- Streamlit 웹 대시보드 제공 ('app.py')

## 내 역할
- 데이터 수집: Selenium 라이브러리에서 제공하는 기능을 활용해 네이버 스토어 리뷰 크롤링
- 데이터 전처리: 텍스트 정제, 토큰화, 불용어 처리
- 요약 모델 구현: HuggingFace KoBART 기반 리뷰 요약 모델 적용
- 감성 분석 모델 구축: KoELECTRA를 활용한 감성 분류 모델 학습 및 평가
- 시각화 및 분석: WordCloud, matplotlib을 활용한 리뷰 감정 분포 시각화
- 통합 구현: 수집 -> 전처리 -> 요약 -> 감성분석 -> 시각화까지 하나의 파이프라인으로 통합

## 기술 스택
- 언어/프레임워크: Python, PyTorch, HuggingFace Transformers
- 데이터 처리: Pandas, Numpy, scikit-learn
- 웹 크롤링: Selenium
- 시각화: matplotlib, WordCloud
- 환경: jupyter Notebook
  

## 실행 방법
```bash
git clone https://github.com/qkek033/naver_review_ai_project.git
cd naver-review-ai-project
pip install -r requirements.txt
streamlit run app.py
>>>>>>> ca5f8e9 (project file)
