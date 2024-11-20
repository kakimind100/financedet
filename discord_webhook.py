import sys
import json
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨을 INFO로 변경하여 DEBUG 로그를 숨깁니다.
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
            logging.info(f"종목 코드: {stock['code']}, 컵: {stock['cup']}, 다이버전스: {stock['divergence']}, 원형 바닥: {stock['round_bottom']}")

            # 필요한 경우 데이터의 특정 부분만 로그로 출력
            # 예를 들어, 종목 코드와 마지막 종가만 출력
            last_record = stock['data'][-1]  # 마지막 데이터 포인트
            logging.info(f"종목 코드: {stock['code']}, 마지막 종가: {last_record['Close']}")

    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {filename}")
    except json.JSONDecodeError:
        logging.error(f"JSON 파일을 읽는 중 오류 발생: {filename} - 잘못된 형식일 수 있습니다.")
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
