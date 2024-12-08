import logging
import requests
import openai
import os
import pandas as pd
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨로 설정하여 모든 로그를 출력하도록 변경
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def send_discord_message(webhook_url, message):
    """Discord 웹훅으로 메시지를 전송하는 함수."""
    data = {
        "content": message
    }
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()  # HTTP 오류 확인
        logging.info("메시지를 성공적으로 Discord에 전송했습니다.")
    except Exception as e:
        logging.error(f"메시지 전송 실패: {e}")

def get_ai_response(api_key, prompt):
    """AI에게 질문을 하고 응답을 받는 함수."""
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 투자 전문가로, 시장의 다양한 기술적 지표를 분석하여 투자 결정을 돕는 역할을 합니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        logging.info("AI로부터 응답을 성공적으로 받았습니다.")
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"AI 응답 오류: {str(e)}")
        return None

def main():
    logging.info("Discord 웹훅 스크립트 실행 중...")
    # 환경 변수에서 설정 로드
    discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not discord_webhook_url or not openai_api_key:
        logging.error("환경 변수가 설정되지 않았습니다.")
        return

    # 파일 경로 확인
    filename = 'data/top_20_stocks_all_dates.csv'
    logging.debug(f"파일 경로: {filename}")

    try:
        # CSV 파일 읽기
        logging.info("CSV 파일을 읽는 중...")
        top_stocks = pd.read_csv(filename, dtype={'Code': 'object'})

        if top_stocks.empty:
            logging.error("읽어온 데이터가 비어 있습니다.")
            return
        
        logging.debug(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 전체 데이터 프레임 로그
        logging.info("전체 데이터 프레임:")
        logging.debug(top_stocks.to_string())

        # 현재 날짜를 가져오는 부분
        current_date = datetime.today()
        logging.info(f"현재 날짜: {current_date.strftime('%Y-%m-%d')}")

        # 상위 5개 종목만 필터링하여 상위 5개 데이터 준비
        top_5_stocks = top_stocks.nlargest(5, 'Gap')  # 'Gap' 컬럼 기준으로 상위 5개 종목 추출
        logging.info(f"상위 5개 종목: {top_5_stocks}")

        # 중복 제거
        top_5_stocks = top_5_stocks.drop_duplicates(subset=['Code'], keep='first')

        # 상위 5종목을 Discord 메시지로 전송
        message = "\n\n".join([
            f"종목코드: {row['Code']}, 상승률: {row['Gap']:.2%}\n매수 시점: {row['Buy Date']}\n매도 시점: {row['Sell Date']}\n매수가: {row['Buy Price']:.2f}, 매도가: {row['Sell Price']:.2f}, 현재가: {row['Current Price']:.2f}"
            for idx, row in top_5_stocks.iterrows()
        ])
        send_discord_message(discord_webhook_url, message)

        # 전체 종목 분석 프롬프트
        analysis_prompt = (
            f"주식 데이터는 다음과 같습니다:\n{top_stocks.to_json(orient='records', force_ascii=False)}\n"
            f"현재 날짜는 {current_date.strftime('%Y-%m-%d')}입니다. "
            f"오늘 시간외 거래에 매수하기에 적절한 종목 코드 세개를 추천해 주세요. "
            f"추천 시 다음 조건을 고려해 주세요:\n"
            f"1. 모든 기술적 지표가 긍정적인 신호를 나타내는지 검토\n"
            f"2. 추천 종목에 대한 확신을 가지고 이유를 50자 내외로 설명\n"
            f"AI는 가장 적합한 매수 조건을 판단하여 종목을 선택해 주세요. "
            f"추천 종목이 적합하지 않을 경우, '추천할 종목이 없습니다.'라고 답변해 주세요."
        )

        logging.info("AI에게 분석 요청을 보내는 중...")
        ai_response = get_ai_response(openai_api_key, analysis_prompt)

        if ai_response:
            logging.info(f"AI 응답: {ai_response}")
            # AI의 응답을 Discord 웹훅으로 전송
            send_discord_message(discord_webhook_url, f"상승 가능성 예측:\n{ai_response}")

    except FileNotFoundError:
        logging.error("파일을 찾을 수 없습니다.")
    except pd.errors.EmptyDataError:
        logging.error("CSV 파일이 비어 있습니다.")
    except pd.errors.ParserError:
        logging.error("CSV 파일을 읽는 중 오류 발생.")
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")
    finally:
        logging.info("Discord 웹훅 스크립트 실행 완료.")

if __name__ == "__main__":
    main()
