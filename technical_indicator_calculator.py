import pandas as pd
import numpy as np
import logging
import os
import pandas_ta as ta
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold

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
        logging.info(f"데이터프레임의 첫 5행:\n{df.head()}")

        # 날짜 열을 datetime 형식으로 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.set_index(['Code', 'Date'], inplace=True)

        # 중복된 데이터 처리
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

    # 기술적 지표 계산
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

    # 추가 기술적 지표 계산
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        logging.info("추가 기술적 지표 계산 완료.")
    except Exception as e:
        logging.error(f"기술적 지표 계산 중 오류 발생: {e}")
        return

    # Price Change 계산
    df['Price_Change'] = df['Close'].pct_change() * 100
    logging.info("가격 변화율(Price Change) 계산 완료.")

    # NaN 값 제거 (모든 기술적 지표 계산 후에)
    df.dropna(inplace=True)
    logging.info(f"NaN 값이 제거된 후 데이터프레임의 크기: {df.shape}")

    # 가격 조정 및 기간 조정 조건 설정
    # 가격 조정: 현재 가격이 이전 가격보다 낮은 경우
    df['Price_Adjustment'] = np.where(df['Close'] < df['Close'].shift(1), 1, 0)

    # 기간 조정: 가격 변동성이 낮고 거래량 변화가 적은 경우
    df['Volume_Change'] = df['Volume'].pct_change()

    # 가격 변동이 1% 이하이고 거래량 변화가 5% 이하인 경우를 기간 조정으로 설정
    df['Period_Adjustment'] = np.where(
        (df['Price_Change'].abs() < 0.01) & (df['Volume_Change'].abs() < 0.05), 1, 0
    )

    # 조정 상태 레이블 생성
    df['Adjustment'] = np.where(
        (df['Price_Adjustment'] == 1) | (df['Period_Adjustment'] == 1), 1, 0
    )

    # 특징 및 레이블 설정
    features = ['Price_Change', 'RSI']
    X = df[features]
    y = df['Adjustment']

    # KFold 교차 검증 설정
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # Isolation Forest 모델을 사용하여 조정 상태 탐지
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        try:
            isolation_forest = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
            isolation_forest.fit(X_train)
            y_pred = isolation_forest.predict(X_test)
            results.append(y_pred)
            logging.info("Isolation Forest를 사용한 조정 상태 탐지 완료.")
        except Exception as e:
            logging.error(f"Isolation Forest 모델 학습 중 오류 발생: {e}")
            return

    # 예측 결과를 통합하여 최종 결과 생성
    final_predictions = np.concatenate(results)

    # 특정 종목 코드의 데이터 로그하기
    if target_code in df.index.levels[0]:
        target_data = df.loc[target_code]
        logging.info(f"{target_code} 종목 코드의 계산된 데이터:\n{target_data}")
    else:
        logging.warning(f"{target_code} 종목 코드는 데이터에 존재하지 않습니다.")

    # 최종 예측 결과를 데이터프레임에 추가 (1과 -1로 저장)
    df['Anomaly'] = np.where(final_predictions == -1, -1, 1)  # '조정'을 -1로, '정상'을 1로 변경

    # 계산된 데이터프레임을 CSV로 저장
    output_file = os.path.join(data_dir, 'stock_data_with_indicators.csv')
    df.to_csv(output_file)
    logging.info("기술적 지표가 'stock_data_with_indicators.csv'로 저장되었습니다.")
    logging.debug(f"저장된 데이터프레임 정보:\n{df.info()}")

if __name__ == "__main__":
    target_code = '006280'  # 특정 종목 코드를 입력하세요.
    logging.info("기술 지표 계산 스크립트 실행 중...")
    calculate_technical_indicators(target_code)
    logging.info("기술 지표 계산 스크립트 실행 완료.")
