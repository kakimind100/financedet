import sys
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
            model="gpt-4o-mini",  # 모델을 GPT-4o-mini로 설정
            messages=[
                {"role": "system", "content": "당신은 투자 전문가로, 시장의 다양한 기술적 지표를 분석하여 투자 결정을 돕는 역할을 합니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  # 최대 토큰 수 설정
            temperature=0.3   # 온도를 0.3으로 설정
        )
        logging.info("AI로부터 응답을 성공적으로 받았습니다.")
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"AI 응답 오류: {str(e)}")
        return None

def main():
    logging.info("Discord 웹훅 스크립트 실행 중...")  # 시작 로그 추가
    # 환경 변수에서 설정 로드
    discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not discord_webhook_url or not openai_api_key:
        logging.error("환경 변수가 설정되지 않았습니다.")
        return

    # 파일 경로 확인
    filename = 'data/top_20_stocks_all_dates.csv'  # 파일 경로를 직접 설정
    logging.debug(f"파일 경로: {filename}")

    try:
        # CSV 파일 읽기 (종목 코드를 object 타입으로 읽기)
        logging.info("CSV 파일을 읽는 중...")
        top_stocks = pd.read_csv(filename, dtype={'Code': 'object'})  # 종목 코드를 object로 지정
        logging.debug(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 데이터 확인
        if top_stocks.empty:
            logging.error("읽어온 데이터가 비어 있습니다.")
            return
        
        # 전체 데이터 프레임 로그
        logging.info("전체 데이터 프레임:")
        logging.debug(top_stocks.to_string())  # 전체 데이터 프레임 출력

        # 현재 날짜를 가져오는 부분
        current_date = datetime.today()
        logging.info(f"현재 날짜: {current_date.strftime('%Y-%m-%d')}")

        # 최근 10 거래일 데이터만 남기기
        filtered_stocks = []
        for code in top_stocks['Code'].unique():
            stock_data = top_stocks[top_stocks['Code'] == code]
            logging.debug(f"{code}의 전체 데이터 개수: {len(stock_data)}개")

            # 최근 10일치 데이터만 남기기
            if len(stock_data) > 10:
                recent_data = stock_data.tail(10)  # 마지막 10일 데이터
            else:
                recent_data = stock_data  # 데이터가 10일 미만이면 전체 데이터 사용

            filtered_stocks.extend(recent_data.to_dict(orient='records'))  # 목록에 추가

        # 필터링된 데이터로 DataFrame 생성
        filtered_df = pd.DataFrame(filtered_stocks)
        logging.info(f"필터링된 데이터 개수: {len(filtered_df)}개")

        # 거래량 열 제거
        if 'Volume' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['Volume'])
            logging.info("거래량 열을 제거했습니다.")

        # AI에게 전달할 분석 프롬프트 (하나의 종목 추천으로 변경)
        analysis_prompt = (
            f"주식 데이터는 다음과 같습니다:\n{filtered_df.to_json(orient='records', force_ascii=False)}\n"
            f"현재 날짜는 {current_date.strftime('%Y-%m-%d')}입니다. "
            f"오늘 시간외 거래에 매수하기에 적절한 종목 코드 하나를 추천해 주세요. "
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
        logging.info("Discord 웹훅 스크립트 실행 완료.")  # 종료 로그

if __name__ == "__main__":
    main()
