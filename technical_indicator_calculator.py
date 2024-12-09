import pandas as pd
import logging
import os
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'technical_indicator_calculator.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # DEBUG로 변경하여 모든 로그를 출력하도록 설정
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def fetch_blog_posts():
    """이블로그에서 최신 글을 파싱하는 함수."""
    url = 'https://example-blog.com/recent-posts'
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        posts = soup.find_all('div', class_='post')
        logging.info("이블로그에서 최신 글 파싱 완료.")

        blog_texts = []
        for post in posts:
            text = post.get_text()
            blog_texts.append(text)
        
        return blog_texts
    
    except requests.RequestException as e:
        logging.error(f"이블로그에서 최신 글을 가져오는 중 오류 발생: {e}")
        return []

def perform_sentiment_analysis(texts):
    """텍스트를 입력 받아 감성 분석을 수행하는 함수."""
    sentiments = []
    for text in texts:
        try:
            analysis = TextBlob(text)
            sentiment_score = analysis.sentiment.polarity  # 감성 점수 추출
            sentiments.append(sentiment_score)
            logging.info(f"감성 분석 결과: {sentiment_score} for text: {text[:50]}...")  # 일부 로그 추가
        except Exception as e:
            logging.error(f"감성 분석 중 오류 발생: {e}")
            sentiments.append(None)  # 오류 발생 시 None 처리
    
    return sentiments

def calculate_technical_indicators(target_code):
    """기술적 지표를 계산하는 함수."""
    data_dir = 'data'
    dtype = {
        'Date': 'str',
        'Open': 'float',
        'High': 'float',
        'Low': 'float',
        'Close': 'float',
        'Volume': 'float',
        'Change': 'float',
        'Code': 'object'
    }

    # 데이터 로딩
    try:
        df = pd.read_csv(os.path.join(data_dir, 'stock_data.csv'), dtype=dtype)
        logging.debug(f"CSV 파일 '{os.path.join(data_dir, 'stock_data.csv')}'을(를) 성공적으로 읽었습니다.")
        logging.info(f"데이터프레임의 첫 5행:\n{df.head()}")  # 첫 5행 로그

        # 날짜 열을 datetime 형식으로 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.set_index(['Code', 'Date'], inplace=True)

        # 중복된 데이터 처리: 종목 코드와 날짜로 그룹화하여 평균값으로 대체
        df = df.groupby(['Code', df.index.get_level_values('Date')]).mean()
        logging.info("중복 데이터 처리 완료.")

    except FileNotFoundError:
        logging.error(f"파일 '{os.path.join(data_dir, 'stock_data.csv')}'을(를) 찾을 수 없습니다.")
        return
    except pd.errors.EmptyDataError:
        logging.error("CSV 파일이 비어 있습니다.")
        return
    except Exception as e:
        logging.error(f"CSV 파일 읽기 중 오류 발생: {e}")
        return

    # 이동 평균 계산
    try:
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        logging.debug("이동 평균(MA5, MA20)을 계산했습니다.")
    except Exception as e:
        logging.error(f"이동 평균 계산 중 오류 발생: {e}")
        return

    # MACD 계산
    try:
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDh_12_26_9']
        df['MACD_Hist'] = macd['MACD_12_26_9'] - macd['MACDh_12_26_9']
        logging.info("MACD 계산 완료.")
    except Exception as e:
        logging.error(f"MACD 계산 중 오류 발생: {e}")
        return

    # Bollinger Bands 계산
    try:
        bollinger_bands = ta.bbands(df['Close'], length=20, std=2)
        df['Bollinger_High'] = bollinger_bands['BBM_20_2.0']
        df['Bollinger_Low'] = bollinger_bands['BBL_20_2.0']
        logging.info("Bollinger Bands 계산 완료.")
    except Exception as e:
        logging.error(f"Bollinger Bands 계산 중 오류 발생: {e}")
        return

    # Stochastic Oscillator 추가
    try:
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch'] = stoch['STOCHk_14_3_3']
        logging.info("Stochastic Oscillator 계산 완료.")
    except Exception as e:
        logging.error(f"Stochastic Oscillator 계산 중 오류 발생: {e}")
        return

    # 기술적 지표 추가
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)

        # 추가 기술적 지표
        df['Momentum'] = df['Close'].diff(4)
        df['Williams %R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['ROC'] = ta.roc(df['Close'], length=12)

        # CMF 및 OBV 계산 추가
        df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
        df['OBV'] = ta.obv(df['Close'], df['Volume'])

        logging.info("추가 기술적 지표(Momentum, Williams %R, ADX, Volume MA, ROC, CMF, OBV)를 계산했습니다.")
    except Exception as e:
        logging.error(f"기술적 지표 계산 중 오류 발생: {e}")
        return

    # NaN 값이 있는 행 제거
    df.dropna(inplace=True)
    logging.info(f"NaN 값이 제거된 후 데이터프레임의 크기: {df.shape}")

    # 특정 종목 코드의 데이터 로그하기
    if target_code in df.index.levels[0]:
        target_data = df.loc[target_code]
        logging.info(f"{target_code} 종목 코드의 계산된 데이터:\n{target_data}")
    else:
        logging.warning(f"{target_code} 종목 코드는 데이터에 존재하지 않습니다.")

    # 계산된 데이터프레임을 CSV로 저장
    output_file = os.path.join(data_dir, 'stock_data_with_indicators.csv')
    df.to_csv(output_file)
    logging.info("기술적 지표와 감성 분석 결과가 'stock_data_with_indicators.csv'로 저장되었습니다.")
    logging.debug(f"저장된 데이터프레임 정보:\n{df.info()}")
    
if __name__ == "__main__":
    target_code = '004980'  # 특정 종목 코드를 입력하세요.
    logging.info("기술 지표 계산 스크립트 실행 중...")

    # 이블로그에서 최신 글 파싱 및 감성 분석 수행
    blog_texts = fetch_blog_posts()
    if blog_texts:
        sentiment_scores = perform_sentiment_analysis(blog_texts)
        logging.info(f"감성 분석 결과: {sentiment_scores}")

    calculate_technical_indicators(target_code)
    logging.info("기술 지표 계산 스크립트 실행 완료.")
