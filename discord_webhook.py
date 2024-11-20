import logging
import sys
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def receive_stock_data(data):
    """주식 데이터를 수신하고 로그로 출력하는 함수."""
    logging.info(f"수신된 주식 데이터: {data}")

if __name__ == "__main__":
    logging.info("Discord 웹훅 스크립트 실행 중...")

    # 명령줄 인수로 전달된 데이터 수신
    if len(sys.argv) > 1:
        stock_data_json = sys.argv[1]
        stock_data = json.loads(stock_data_json)
        receive_stock_data(stock_data)
    else:
        logging.error("주식 데이터가 전달되지 않았습니다.")
