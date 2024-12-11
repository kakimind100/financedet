import sys
import logging
import requests
import openai
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
    try:
        response = requests.post(webhook_url, json={"content": message})
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
                {"role": "system", "content": (
                    "당신은 워렌 버핏이다. "
                    "가치 투자를 중시하며, 장기적인 관점에서 기업의 본질적인 가치를 분석하여 투자 결정을 내린다. "
                    "안정적이고 지속 가능한 수익을 창출할 가능성이 높은 기업을 선호한다."
                )},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.4,  # 낮은 온도로 안정적이고 현실적인 응답 생성
            top_p=1.0,        # 전체 확률 분포를 사용
            presence_penalty=0.0,  # 새로운 아이디어를 유도하지 않음
            frequency_penalty=0.2  # 반복 방지를 위한 페널티
        )
        logging.info("AI로부터 응답을 성공적으로 받았습니다.")
        return response['choices'][0]['message']['content']
    except Exception as e:
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

        return [post.get_text() for post in posts]
    
    except requests.RequestException as e:
        logging.error(f"이블로그에서 최신 글을 가져오는 중 오류 발생: {e}")
        return []

def perform_sentiment_analysis(texts, score_adjustment_factor=0.0):
    """텍스트를 입력 받아 감성 분석을 수행하는 함수.

    Args:
        texts (list): 감성 분석할 텍스트 리스트.
        score_adjustment_factor (float): 감성 점수를 낮출 비율 (0 ~ 1 사이의 값).

    Returns:
        list: 감성 점수 리스트.
    """
    sentiments = []
    for i, text in enumerate(texts):
        try:
            analysis = TextBlob(text)
            sentiment_score = analysis.sentiment.polarity  # 기본 감성 점수 추출
            adjusted_score = sentiment_score - (score_adjustment_factor * sentiment_score)
            sentiments.append(adjusted_score)
            logging.info(f"감성 분석 결과 [{i+1}/{len(texts)}]: {adjusted_score} for text snippet: {text[:50]}...")  # 일부 로그 추가
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
            recent_data = stock_data.tail(10) if len(stock_data) > 10 else stock_data
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
        score_adjustment_factor = 0.2  # 점수를 20% 낮추기 위해 조정
        sentiment_scores = perform_sentiment_analysis(blog_texts, score_adjustment_factor=score_adjustment_factor)


        # 감성 점수를 분석 프롬프트에 추가하여 AI에게 전달
        analysis_prompt = (
            f"현재 주식 데이터는 다음과 같습니다:\n{filtered_df.to_json(orient='records', force_ascii=False)}\n"
            f"전체 주식 시장의 감성 점수는 {sentiment_scores[-1]}입니다. 이 점수는 -1에서 +1 사이의 값으로, "
            f"+1에 가까울수록 긍정적이고, -1에 가까울수록 부정적입니다.\n"
            f"현재 날짜는 {current_date.strftime('%Y-%m-%d')}입니다.\n\n"
            f"당신은 워렌 버핏이다. "
            f"가치 투자를 중시하며 기업의 본질적인 가치를 분석해 투자 결정을 내린다. "
            f"오늘 시간외 거래에 매수하기 적합한 종목 3개를 추천하라. "
            f"추천 조건은 다음과 같다:\n"
            f"1. 강력한 경쟁 우위를 가진 기업.\n"
            f"2. 재무 상태가 안정적이고 부채 비율이 낮은 기업.\n"
            f"3. 배당을 지속적으로 지급하며 신뢰할 수 있는 실적 기록이 있는 기업.\n"
            f"4. 시장 감성 점수를 참고해 상승 가능성이 있는 종목.\n\n"
            f"각 종목의 추천 사유를 100자 내외로 작성하고, "
            f"추천이 불가능한 경우 그 이유를 명확히 설명하라."
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
