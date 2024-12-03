import pandas as pd
import numpy as np
import logging
import os
import pandas_ta as ta  # pandas_ta 임포트 추가
from sklearn.ensemble import IsolationForest  # Isolation Forest 임포트 추가

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
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

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
        df.set_index(['Code', 'Date'], inplace=True)  # 종목 코드와 날짜를 인덱스로 설정

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
        df['MACD_Hist'] = macd['MACD_12_26_9'] - macd['MACDh_12_26_9']  # MACD 히스토그램 추가
        logging.info("MACD 계산 완료.")
    except Exception as e:
        logging.error(f"MACD 계산 중 오류 발생: {e}")
        return

    # Bollinger Bands 계산
    try:
        bollinger_bands = ta.bbands(df['Close'], length=20, std=2)
        df['Bollinger_High'] = bollinger_bands['BBM_20_2.0']  # 중간선
        df['Bollinger_Low'] = bollinger_bands['BBL_20_2.0']  # 하한선
        logging.info("Bollinger Bands 계산 완료.")
    except Exception as e:
        logging.error(f"Bollinger Bands 계산 중 오류 발생: {e}")
        return

    # Stochastic Oscillator 추가
    try:
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch'] = stoch['STOCHk_14_3_3']  # 올바른 열 이름 사용
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
        df['Momentum'] = df['Close'].diff(4)  # 4일 전과의 가격 차이
        df['Williams %R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()  # 20일 거래량 이동 평균
        df['ROC'] = ta.roc(df['Close'], length=12)  # Rate of Change 추가

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

    # 조정 상태 레이블 생성
    # 기본적으로 가격이 하락하는 경우를 조정으로 설정하되,
    # RSI가 70 이상일 때 하락하는 경우를 추가하여 조정으로 판단
    df['Adjustment'] = np.where(
        (df['Close'] < df['Close'].shift(1)) | 
        ((df['RSI'] > 70) & (df['Close'] < df['Close'].shift(1))),
        1,  # 조정
        0   # 정상
    )

    # 특징 및 레이블 설정
    features = ['Close', 'MA5', 'MA20', 'MACD', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'Stoch']
    X = df[features]

    # Isolation Forest 모델을 사용하여 조정 상태 탐지
    try:
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        df['Anomaly'] = isolation_forest.fit_predict(X)
        df['Adjustment'] = np.where(df['Anomaly'] == -1, '조정', '정상')  # 조정 상태 해석
        logging.info("Isolation Forest를 사용한 조정 상태 탐지 완료.")
    except Exception as e:
        logging.error(f"Isolation Forest 모델 학습 중 오류 발생: {e}")
        return

    # 특정 종목 코드의 데이터 로그하기
    if target_code in df.index.levels[0]:
        target_data = df.loc[target_code]
        logging.info(f"{target_code} 종목 코드의 계산된 데이터:\n{target_data}")
    else:
        logging.warning(f"{target_code} 종목 코드는 데이터에 존재하지 않습니다.")

    # 계산된 데이터프레임을 CSV로 저장
    output_file = os.path.join(data_dir, 'stock_data_with_indicators.csv')
    df.to_csv(output_file)
    logging.info("기술적 지표가 'stock_data_with_indicators.csv'로 저장되었습니다.")
    logging.debug(f"저장된 데이터프레임 정보:\n{df.info()}")  # 저장된 데이터프레임 정보 로그

if __name__ == "__main__":
    target_code = '006280'  # 특정 종목 코드를 입력하세요.
    logging.info("기술 지표 계산 스크립트 실행 중...")  # 실행 시작 메시지
    calculate_technical_indicators(target_code)
    logging.info("기술 지표 계산 스크립트 실행 완료.")
