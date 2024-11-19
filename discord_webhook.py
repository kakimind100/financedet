import os
import logging
import requests
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import xmlrpc.client
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경변수에서 비밀 키 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WP_URL = os.getenv('WP_URL')
WP_USERNAME = os.getenv('WP_USERNAME')
WP_PASSWORD = os.getenv('WP_PASSWORD')
DISCORD_WEBHOOK_URL_BLOG = os.getenv('DISCORD_WEBHOOK_URL_BLOG')  # 블로그 디스코드 웹훅 URL
NEWS_RSS_FEED_URL = os.getenv('NEWS_RSS_FEED_URL')  # 뉴스 RSS 피드 URL
BLOG_RSS_FEED_URL = os.getenv('BLOG_RSS_FEED_URL')  # 블로그 RSS 피드 URL

# API 키가 설정되지 않은 경우 오류 메시지 출력
if OPENAI_API_KEY is None:
    logging.error("OPENAI_API_KEY가 설정되지 않았습니다. 환경 변수를 확인하세요.")
    exit(1)

if NEWS_RSS_FEED_URL is None:
    logging.error("NEWS_RSS_FEED_URL이 설정되지 않았습니다. 환경 변수를 확인하세요.")
    exit(1)

if BLOG_RSS_FEED_URL is None:
    logging.error("BLOG_RSS_FEED_URL이 설정되지 않았습니다. 환경 변수를 확인하세요.")
    exit(1)

def send_discord_message(content):
    """디스코드 웹훅을 통해 메시지를 보냅니다."""
    if DISCORD_WEBHOOK_URL_BLOG is None:
        logging.error("DISCORD_WEBHOOK_URL_BLOG이 설정되지 않았습니다.")
        return

    data = {
        "content": content
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL_BLOG, json=data)
        response.raise_for_status()
        logging.info("디스코드에 메시지를 성공적으로 보냈습니다.")
    except requests.exceptions.RequestException as e:
        logging.error(f"디스코드 메시지 전송 중 오류 발생: {e}")

def fetch_latest_analysis_article(feed_url):
    logging.info(f"RSS 피드 URL: {feed_url}에서 최신 분석 기사 가져오는 중...")
    try:
        response = requests.get(feed_url)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for item in root.findall('.//item'):
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text

            logging.info(f"발행일: {pub_date}, 제목: {title}, 링크: {link}")

            if 'analysis' in link:
                logging.info(f"분석 기사 발견: {title}")
                return title, link, pub_date

        logging.warning("분석 기사를 찾을 수 없습니다.")
        return None, None, None
    except requests.exceptions.RequestException as e:
        logging.error(f"페이지를 가져오는 중 오류 발생: {e}")
        return None, None, None
    except ET.ParseError as e:
        logging.error(f"XML 파싱 중 오류 발생: {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"알 수 없는 오류 발생: {e}")
        return None, None, None

def fetch_latest_blog_post(feed_url):
    """블로그 RSS 피드에서 최신 포스트를 가져옵니다."""
    logging.info(f"블로그 RSS 피드 URL: {feed_url}에서 최신 포스트 가져오는 중...")
    try:
        response = requests.get(feed_url)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for item in root.findall('.//item'):
            title = item.find('title').text
            link = item.find('link').text
            pub_date = item.find('pubDate').text

            logging.info(f"블로그 발행일: {pub_date}, 제목: {title}, 링크: {link}")
            return title, link, pub_date  # 블로그 포스트를 반환

        logging.warning("블로그 포스트를 찾을 수 없습니다.")
        return None, None, None
    except requests.exceptions.RequestException as e:
        logging.error(f"페이지를 가져오는 중 오류 발생: {e}")
        return None, None, None
    except ET.ParseError as e:
        logging.error(f"XML 파싱 중 오류 발생: {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"알 수 없는 오류 발생: {e}")
        return None, None, None

def fetch_article_content(article_url):
    logging.info(f"기사 내용을 가져오는 중: {article_url}")
    try:
        response = requests.get(article_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content_div = soup.find('div', class_='article_WYSIWYG__O0uhw')  # 주어진 클래스 이름으로 수정

        if content_div:
            logging.debug("기사 내용 가져오기 성공.")
            return clean_article_content(content_div)
        else:
            logging.error("기사 내용을 찾을 수 없습니다. content_div가 None입니다.")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"기사 내용을 가져오는 중 오류 발생: {e}")
        return None

def clean_article_content(content_div):
    """기사 내용에서 불필요한 HTML 태그 및 스크립트 제거"""
    for script in content_div(['script', 'style']):
        script.decompose()

    text = content_div.get_text(separator='\n', strip=True)
    clean_text = re.sub(r'\n+', '\n', text)  # 여러 줄을 한 줄로
    logging.debug("기사 내용 정리 완료.")
    return clean_text.strip()

def is_within_12_hours(pub_date_str):
    """발행일이 현재 시간과 12시간 이내인지 확인합니다."""
    try:
        pub_date = datetime.strptime(pub_date_str, '%b %d, %Y %H:%M %Z')  # 포맷 수정
        current_time = datetime.utcnow()  # UTC 기준으로 현재 시간 가져오기
        time_difference = current_time - pub_date
        logging.info(f"발행일과 현재 시간 차이: {time_difference}")
        return time_difference <= timedelta(hours=12)  # 12시간 이내
    except ValueError as e:
        logging.error(f"발행일 포맷 오류: {e}")
        return False

def generate_prediction_content(content):
    logging.info("AI가 글 생성 중...")
    
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }

    system_message = (
        '당신은 대한민국 최고의 투자 전문 미래예측 전문가입니다. '
        '다음 기사를 바탕으로 향후 시장 경향과 예측을 작성하십시오. '
        '내용은 400토큰 내외로 요약하되, 논리적인 흐름이 있도록 자연스럽게 연결해야 합니다. '
        '예를 들어, "이러한 경향은 다음과 같은 이유로 발생할 것으로 예상됩니다..."와 같은 형식을 따르세요.'
    )

    data = {
        'model': 'gpt-4',
        'messages': [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': content}  # 기사 내용만 사용
        ],
        'max_tokens': 600,  # 토큰 수를 600으로 설정
        'temperature': 0.7  # 약간의 변화와 창의성을 위해 온도를 높임
    }

    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()
        generated_content = response.json()['choices'][0]['message']['content']
        logging.info("AI가 글을 성공적으로 생성했습니다.")
        return generated_content.strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"미래 예측 글 생성 중 오류 발생: {e}")
        return None

def generate_caption(content):
    logging.info("AI가 캡션 생성 중...")
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': 'gpt-4',
        'messages': [
            {'role': 'system', 'content': '당신은 최고의 캡션 전문입니다. 주어진 내용을 바탕으로 캡션을 생성하세요. 70토큰 이내로 문장이 완성되게 요약해서 작성해줘'},
            {'role': 'user', 'content': f'내용: {content}'}
        ],
        'max_tokens': 70,  # 토큰 수를 70으로 설정
        'temperature': 0.5
    }

    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()
        caption = response.json()['choices'][0]['message']['content']
        logging.info("AI가 캡션을 성공적으로 생성했습니다.")
        return caption.strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"캡션 생성 중 오류 발생: {e}")
        return None

def post_to_wordpress(title, content, link):
    logging.info(f"워드프레스에 포스팅 중: {title}")
    wp = xmlrpc.client.ServerProxy(WP_URL)
    try:
        final_body = f"""
{content}
        """
        
        # 포스팅할 내용을 확인
        logging.info(f"포스팅될 내용: {final_body.strip()}")
        
        wp.metaWeblog.newPost('1', WP_USERNAME, WP_PASSWORD, {
            'title': title,  # 생성된 캡션을 제목으로 사용
            'description': final_body.strip(),
            'mt_keywords': '',
            'categories': ['투자', '예측']
        }, True)
        logging.info("포스팅이 성공적으로 완료되었습니다.")

    except Exception as e:
        logging.error(f"워드프레스 포스팅 중 오류 발생: {e}")

def main():
    logging.info("최신 분석 뉴스에서 콘텐츠 전송을 시작합니다.")

    # 뉴스 RSS 피드에서 최신 분석 기사 가져오기
    title, link, pub_date = fetch_latest_analysis_article(NEWS_RSS_FEED_URL)

    if title and link:
        if is_within_12_hours(pub_date):  # 발행일이 12시간 이내인지 확인
            article_content = fetch_article_content(link)
            if article_content:
                logging.info(f"기사 내용을 성공적으로 가져왔습니다: {title}")

                # AI에게 본문 내용만 요청하여 콘텐츠 생성
                generated_content = generate_prediction_content(article_content)
                if generated_content:
                    caption = generate_caption(generated_content)
                    if caption:
                        post_to_wordpress(caption, generated_content, link)  # 캡션과 AI 생성 글을 포스팅
                        # 포스팅이 완료된 후 디스코드 알림 전송
                        send_discord_message(f"새 포스트가 작성되었습니다: {caption}\n링크: {link}")
                    else:
                        logging.error(f"{title}의 캡션 생성에 실패했습니다.")
                else:
                    logging.error(f"{title}의 미래 예측 글 생성에 실패했습니다.")
            else:
                logging.error(f"{title}의 기사 내용을 가져오는 데 실패했습니다.")
        else:
            logging.info(f"{pub_date}의 기사는 12시간 이상 경과하여 글 작성을 건너뜁니다.")
    else:
        logging.warning("최신 분석 기사를 찾을 수 없습니다.")

    # 블로그 RSS 피드에서 최신 포스트 가져오기
    blog_title, blog_link, blog_pub_date = fetch_latest_blog_post(BLOG_RSS_FEED_URL)

    if blog_title and blog_link:
        logging.info(f"블로그에서 최신 포스트를 가져왔습니다: {blog_title}")
        # 블로그 포스트가 성공적으로 가져와졌을 때만 디스코드 알림 전송
        send_discord_message(f"새 블로그 포스트가 작성되었습니다: {blog_title}\n링크: {blog_link}")

if __name__ == '__main__':
    main()

