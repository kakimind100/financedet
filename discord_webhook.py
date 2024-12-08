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
    discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not discord_webhook_url or not openai_api_key:
        logging.error("환경 변수가 설정되지 않았습니다.")
        return

    filename = 'data/top_20_stocks_all_dates.csv'
    try:
        logging.info("CSV 파일을 읽는 중...")
        top_stocks = pd.read_csv(filename, dtype={'Code': 'object'})
        logging.debug(f"읽어온 데이터 개수: {len(top_stocks)}개")

        if top_stocks.empty:
            logging.error("읽어온 데이터가 비어 있습니다.")
            return
        
        current_date = datetime.today()
        logging.info(f"현재 날짜: {current_date.strftime('%Y-%m-%d')}")

        # 최근 26거래일 데이터만 필터링
        filtered_stocks = []
        for code in top_stocks['Code'].unique():
            stock_data = top_stocks[top_stocks['Code'] == code]
            logging.debug(f"{code}의 전체 데이터 개수: {len(stock_data)}개")

            # 최근 26거래일만 필터링
            if len(stock_data) > 26:
                recent_data = stock_data.tail(26)  # 마지막 26거래일 데이터
            else:
                recent_data = stock_data  # 데이터가 26거래일 미만이면 전체 데이터 사용

            filtered_stocks.extend(recent_data.to_dict(orient='records'))  # 목록에 추가

        # 필터링된 데이터로 DataFrame 생성
        filtered_df = pd.DataFrame(filtered_stocks)
        logging.info(f"필터링된 데이터 개수: {len(filtered_df)}개")

        # AI에게 전달할 분석 프롬프트
        analysis_prompt = (
            f"주식 데이터는 다음과 같습니다:\n{filtered_df.to_json(orient='records', force_ascii=False)}\n"
            f"현재 날짜는 {current_date.strftime('%Y-%m-%d')}입니다. "
            f"오늘 시간외 거래에 매수하기에 적절한 종목 코드 3개를 추천해 주세요. "
            f"추천 시 간단한 이유를 함께 작성해 주세요."
        )

        logging.info("AI에게 분석 요청을 보내는 중...")
        ai_response = get_ai_response(openai_api_key, analysis_prompt)

        if ai_response:
            logging.info(f"AI 응답: {ai_response}")
            recommendations = ai_response.split("\n")
            
            message = "AI가 추천한 상승 가능성 종목 분석:\n"
            for recommendation in recommendations:
                try:
                    stock_code, reason = recommendation.split(":")
                    stock_code, reason = stock_code.strip(), reason.strip()
                    
                    # CSV에서 해당 종목 코드 데이터 추출
                    stock_data = filtered_df[filtered_df['Code'] == stock_code].iloc[0]
                    
                    # 데이터 파싱
                    buy_date = stock_data['Buy Date']
                    sell_date = stock_data['Sell Date']
                    buy_price = stock_data['Buy Price']
                    sell_price = stock_data['Sell Price']
                    current_price = stock_data['Current Price']
                    estimated_return = stock_data['Gap'] * 100

                    # 메시지 작성
                    message += (
                        f"종목 코드: {stock_code}, 추천 이유: {reason}, "
                        f"매수 날짜: {buy_date}, 매도 날짜: {sell_date}, "
                        f"매수 가격: {buy_price:.2f}, 매도 가격: {sell_price:.2f}, "
                        f"현재 가격: {current_price:.2f}, 예상 상승률: {estimated_return:.2f}%\n"
                    )
                except Exception as e:
                    logging.error(f"추천 항목 처리 중 오류 발생: {e}")
            
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
