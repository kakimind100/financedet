import sys
import logging
import requests
import openai
import os
import pandas as pd
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def send_discord_message(webhook_url, message):
    """Discord 웹훅으로 메시지를 전송하는 함수."""
    data = {"content": message}
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()
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
                {"role": "system", "content": "당신은 투자 전문가로, 주식 데이터를 분석해 최적의 종목을 추천하는 역할을 합니다."},
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

    # CSV 파일 경로
    filename = 'data/top_20_stocks_all_dates.csv'
    logging.debug(f"파일 경로: {filename}")

    try:
        # CSV 파일 읽기
        logging.info("CSV 파일을 읽는 중...")
        top_stocks = pd.read_csv(filename, dtype={'Code': 'object'})
        logging.debug(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 데이터 확인
        if top_stocks.empty:
            logging.error("읽어온 데이터가 비어 있습니다.")
            return

        # 현재 날짜
        current_date = datetime.today()
        logging.info(f"현재 날짜: {current_date.strftime('%Y-%m-%d')}")

        # AI 분석 프롬프트
        analysis_prompt = (
            f"주식 데이터는 다음과 같습니다:\n{top_stocks.to_json(orient='records', force_ascii=False)}\n"
            f"현재 날짜는 {current_date.strftime('%Y-%m-%d')}입니다. "
            f"오늘 시간외 거래에 매수하기에 적절한 종목 코드 3개를 추천하고 각각 상승 이유를 간단히 설명해 주세요."
        )

        logging.info("AI에게 분석 요청을 보내는 중...")
        ai_response = get_ai_response(openai_api_key, analysis_prompt)

        if ai_response:
            logging.info(f"AI 응답: {ai_response}")

            try:
                # AI 응답 파싱 (예: JSON 형태 또는 리스트)
                recommended_stocks = eval(ai_response)  # AI 응답을 리스트로 변환
                if not isinstance(recommended_stocks, list):
                    raise ValueError("AI 응답 형식이 올바르지 않습니다.")

                # 종목별 정보 구성
                message = "AI가 추천한 상승 가능성 종목 분석:\n"
                for stock in recommended_stocks:
                    stock_code = stock.get('종목코드')
                    reason = stock.get('이유')

                    # 해당 종목 데이터를 필터링
                    stock_data = top_stocks[top_stocks['Code'] == stock_code]

                    if stock_data.empty:
                        logging.warning(f"추천 종목 {stock_code}에 대한 데이터가 없습니다.")
                        continue

                    # 매수/매도 시점, 가격, 예상 상승률 추출
                    buy_time = stock_data['BuyTime'].iloc[-1]
                    sell_time = stock_data['SellTime'].iloc[-1]
                    buy_price = stock_data['BuyPrice'].iloc[-1]
                    sell_price = stock_data['SellPrice'].iloc[-1]
                    estimated_return = ((sell_price - buy_price) / buy_price) * 100

                    # 메시지 작성
                    message += (
                        f"\n종목 코드: {stock_code}\n"
                        f"추천 이유: {reason}\n"
                        f"매수 시점: {buy_time}\n"
                        f"매도 시점: {sell_time}\n"
                        f"매수 가격: {buy_price}\n"
                        f"매도 가격: {sell_price}\n"
                        f"예상 상승률: {estimated_return:.2f}%\n"
                    )
                logging.info("Discord 메시지를 준비했습니다.")

            except Exception as e:
                logging.error(f"AI 응답 처리 중 오류 발생: {e}")
                message = "AI 응답 처리에 실패했습니다."

            # 메시지를 Discord로 전송
            send_discord_message(discord_webhook_url, message)

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
