import sys
import json
import logging
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 인수로 전달된 파일 경로 확인
    if len(sys.argv) < 2:
        logging.error("파일 경로가 제공되지 않았습니다.")
        return

    filename = sys.argv[1]
    logging.info(f"파일 경로: {filename}")

    try:
        # JSON 파일 읽기
        with open(filename, 'r', encoding='utf-8') as f:
            top_stocks = json.load(f)

        logging.info(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 데이터 확인 및 날짜 추가
        for stock in top_stocks:
            code = stock.get('code', '알 수 없음')
            score = stock.get('score', '알 수 없음')

            # 날짜 데이터가 없으면 기본 날짜 추가
            if 'data' in stock:
                for index, record in enumerate(stock['data']):
                    if 'Date' not in record or not record['Date']:
                        record['Date'] = (datetime.today() - timedelta(days=index)).strftime('%Y-%m-%d')
                        logging.info(f"종목 코드: {code} - 날짜 추가: {record['Date']}")

            # 로그 간소화: 데이터의 특정 부분만 출력
            for record in stock.get('data', []):
                date_str = record.get('Date', '없음')
                logging.debug(f"날짜: {date_str}")  # DEBUG 레벨로 변경하여 기본 정보는 INFO로 줄임

    except FileNotFoundError:
        logging.error("파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        logging.error("JSON 파일을 읽는 중 오류 발생.")
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
