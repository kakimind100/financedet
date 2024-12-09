import logging
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import openai
import os
import pandas as pd
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_latest_blog_posts(blog_url, num_posts=5):
    """블로그에서 최신 글 가져오기"""
    try:
        response = requests.get(blog_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # 블로그에서 글 제목과 링크 추출
        articles = soup.find_all('article', limit=num_posts)
        posts = []
        for article in articles:
            title = article.find('h2').text.strip() if article.find('h2') else "제목 없음"
            link = article.find('a')['href'] if article.find('a') else "링크 없음"
            content = article.find('div', class_='entry-content').text.strip() if article.find('div', class_='entry-content') else "내용 없음"
            posts.append({'title': title, 'link': link, 'content': content})
        return posts
    except Exception as e:
        logging.error(f"블로그 데이터를 가져오는 중 오류 발생: {e}")
        return []

def analyze_sentiment(text):
    """텍스트 감성 분석 수행"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    return polarity, subjectivity

def generate_prompt_with_sentiment(stock_data, blog_data):
    """주식 데이터와 감성 분석 데이터를 통합한 프롬프트 생성"""
    sentiment_summary = "\n".join([
        f"제목: {post['title']}\n내용 감성 점수: {analyze_sentiment(post['content'])[0]:.2f} "
        f"(링크: {post['link']})"
        for post in blog_data
    ])

    prompt = (
        f"주식 데이터는 다음과 같습니다:\n{stock_data.to_json(orient='records', force_ascii=False)}\n"
        f"블로그 최신 글 감성 분석 결과:\n{sentiment_summary}\n"
        f"현재 날짜는 {datetime.today().strftime('%Y-%m-%d')}입니다. "
        f"오늘 시간외 거래에 매수하기에 적절한 종목 코드 세개를 추천해 주세요. "
        f"추천 종목이 적합하지 않을 경우, '추천할 종목이 없습니다.'라고 답변해 주세요."
    )
    return prompt

def main():
    logging.info("스크립트 실행 중...")
    
    # 환경 변수 로드
    discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    blog_url = "https://investqq.wordpress.com"

    if not discord_webhook_url or not openai_api_key:
        logging.error("환경 변수가 설정되지 않았습니다.")
        return

    # 데이터 로드
    try:
        stock_filename = 'data/top_20_stocks_all_dates.csv'
        stock_data = pd.read_csv(stock_filename, dtype={'Code': 'object'})

        if stock_data.empty:
            logging.error("주식 데이터가 비어 있습니다.")
            return

        # 블로그 데이터 수집 및 감성 분석
        logging.info("블로그 데이터 수집 중...")
        blog_posts = fetch_latest_blog_posts(blog_url)
        if not blog_posts:
            logging.warning("블로그 글이 없습니다. 감성 분석을 건너뜁니다.")

        # 프롬프트 생성
        prompt = generate_prompt_with_sentiment(stock_data, blog_posts)

        # AI 응답 요청
        logging.info("AI 분석 요청 중...")
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        ai_response = response['choices'][0]['message']['content']
        logging.info(f"AI 응답: {ai_response}")

        # Discord로 전송
        send_discord_message(discord_webhook_url, f"AI 분석 결과:\n{ai_response}")

    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
