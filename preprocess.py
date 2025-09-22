# app.py

import re

# ====================
# 2. 텍스트 처리 함수
# ====================

def clean_text(text):
    text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s]', '', str(text))  # 특수문자 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text