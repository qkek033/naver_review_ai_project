# crawler.py

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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
