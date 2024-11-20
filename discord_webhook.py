import sys
import json
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 인수로 전달된 파일 경로 확인
    if len(sys.argv) < 2:
        logging.error("파일 경로가 제공되지 않았습니다. 사용법: python discord_webhook.py <파일경로>")
        return

    filename = sys.argv[1]
    logging.info(f"전달받은 파일 경로: {filename}")

    try:
        # JSON 파일 읽기
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        top_stocks = data.get('top_stocks', [])
        logging.info(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 데이터 확인을 위한 출력
        for stock in top_stocks:
            # 각 종목의 정보에 접근
            code = stock.get('code', '코드 없음')
            score = stock.get('score', '점수 없음')
            logging.info(f"종목 코드: {code}, 점수: {score}")

            # 필요한 경우 데이터의 특정 부분만 로그로 출력
            for record in stock.get('data', []):  # 'data' 키가 있는 경우만 처리
                date_str = record.get('Date', '날짜 정보 없음')  # 날짜 정보가 없으면 기본값 설정
                logging.info(f"날짜: {date_str}, 데이터: {record}")

    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {filename}")
    except json.JSONDecodeError:
        logging.error(f"JSON 파일을 읽는 중 오류 발생: {filename} - 잘못된 형식일 수 있습니다.")
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
