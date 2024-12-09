import sys
import logging
import requests
import os
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from textblob import TextBlob

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
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    payload = {
        "model": "gpt-4o-mini",  # 모델을 GPT-4o-mini로 설정
        "messages": [
            {"role": "system", "content": "당신은 투자 전문가로, 시장의 다양한 기술적 지표를 분석하여 투자 결정을 돕는 역할을 합니다."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,  # 최대 토큰 수 설정
        "temperature": 0.3   # 온도를 0.3으로 설정
    }
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        logging.info("AI로부터 응답을 성공적으로 받았습니다.")
        return data['choices'][0]['message']['content']
    except requests.RequestException as e:
        logging.error(f"AI 응답 오류: {str(e)}")
        return None

def fetch_blog_posts():
    """이블로그에서 최신 글을 파싱하는 함수."""
    url = 'https://investqq.wordpress.com/feed'
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'xml')  # XML로 파싱
        posts = soup.find_all('item')  # RSS 피드에서 글 찾기
        logging.info("이블로그에서 최신 글 파싱 완료.")

        blog_texts = [post.get_text() for post in posts]
        return blog_texts
    
    except requests.RequestException as e:
        logging.error(f"이블로그에서 최신 글을 가져오는 중 오류 발생: {e}")
        return []

def perform_sentiment_analysis(texts):
    """텍스트를 입력 받아 감성 분석을 수행하는 함수."""
    sentiments = []
    for i, text in enumerate(texts):
        try:
            analysis = TextBlob(text)
            sentiment_score = analysis.sentiment.polarity  # 감성 점수 추출
            sentiments.append(sentiment_score)
            logging.info(f"감성 분석 결과 [{i+1}/{len(texts)}]: {sentiment_score} for text snippet: {text[:50]}...")  # 일부 로그 추가
        except Exception as e:
            logging.error(f"감성 분석 중 오류 발생: {e} for text snippet: {text[:50]}...")
            sentiments.append(None)  # 오류 발생 시 None 처리
    
    return sentiments

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
        # CSV 파일 읽기 (종목 코드를 object 타입으로 읽기)
        logging.info("CSV 파일을 읽는 중...")
        top_stocks = pd.read_csv(filename, dtype={'Code': 'object'})
        logging.debug(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 데이터 확인
        if top_stocks.empty:
            logging.error("읽어온 데이터가 비어 있습니다.")
            return
        
        # 전체 데이터 프레임 로그
        logging.info("전체 데이터 프레임:")
        logging.debug(top_stocks.to_string())

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

        # 블로그에서 가져온 감성 점수
        blog_texts = fetch_blog_posts()
        if not blog_texts:
            logging.warning("블로그 감성 점수 수집에 실패했습니다.")
            return

        # 감성 분석 수행
        sentiment_scores = perform_sentiment_analysis(blog_texts)

        # 감성 점수를 분석 프롬프트에 추가하여 AI에게 전달
        analysis_prompt = (
            f"주식 데이터는 다음과 같습니다:\n{filtered_df.to_json(orient='records', force_ascii=False)}\n"
            f"현재 전체 주식 시장의 감성 점수는 {overall_sentiment_score}입니다.\n"
            f"이 점수는 -1에서 +1 사이의 값으로, +1에 가까울수록 시장이 긍정적이고, "
            f"-1에 가까울수록 시장이 부정적임을 의미합니다.\n"
            f"현재 날짜는 {current_date.strftime('%Y-%m-%d')}입니다. "
            f"오늘 시간외 거래에 매수하기에 적절한 종목 코드 하나를 추천해 주세요. "
            f"추천 시 다음 조건을 고려해 주세요:\n"
            f"1. 기술적 지표가 긍정적인 신호를 나타내는 종목을 선택\n"
            f"2. 시장 감성 점수를 고려하여:\n"
            f"   - 감성 점수가 0 이상일 경우: 상승 가능성이 있는 종목을 추천\n"
            f"   - 감성 점수가 0 미만일 경우: 추천하지 않거나 보수적으로 접근\n"
            f"3. 추천 종목에 대한 확신을 가지고 이유를 50자 내외로 설명\n"
            f"4. 추천이 불가능한 경우, '추천할 종목이 없습니다.'라고 답변하며 그 이유를 명확히 설명 "
            f"(예: 기술적 지표가 부정적, 시장 감성 점수가 낮음 등).\n"
            f"AI는 가장 적합한 매수 조건을 판단하여 종목을 선택해 주세요."
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
