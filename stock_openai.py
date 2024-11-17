# stock_openai.py
import logging
import json
import os
import time

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_result_from_file():
    """파일에서 결과를 읽는 함수."""
    if os.path.exists('result.json'):
        with open('result.json', 'r') as f:
            data = json.load(f)
            logging.info("수신된 데이터: %s", data)
    else:
        logging.warning("결과 파일이 존재하지 않습니다.")

if __name__ == "__main__":
    while True:
        read_result_from_file()
        time.sleep(5)  # 5초마다 결과 파일 확인
