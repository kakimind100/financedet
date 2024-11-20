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
            top_stocks = json.load(f)

        if not top_stocks:
            logging.error("주식 데이터가 없습니다.")
            return

        logging.info(f"읽어온 데이터 개수: {len(top_stocks)}개")
        logging.debug(f"읽어온 데이터: {top_stocks}")  # 데이터 내용 로그

        # 데이터 확인을 위한 출력
        for stock in top_stocks:
            logging.info(f"종목 코드: {stock['code']}, 점수: {stock['score']}")

    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {filename}")
    except json.JSONDecodeError:
        logging.error(f"JSON 파일을 읽는 중 오류 발생: {filename} - 잘못된 형식일 수 있습니다.")
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
