import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import holidays  # 한국 공휴일 정보 가져오기

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    logging.debug("주식 데이터를 가져오는 중...")
    try:
        file_path = os.path.join('data', 'stock_data_with_indicators.csv')
        logging.debug(f"CSV 파일 경로: {file_path}")

        dtype = {
            'Code': 'object',
            'Date': 'str',
            'Open': 'float',
            'High': 'float',
            'Low': 'float',
            'Close': 'float',
            'Volume': 'float',
            'Change': 'float',
            'MA5': 'float',
            'MA20': 'float',
            'MACD': 'float',
            'MACD_Signal': 'float',
            'Bollinger_High': 'float',
            'Bollinger_Low': 'float',
            'Stoch': 'float',
            'RSI': 'float',
            'ATR': 'float',
            'CCI': 'float',
            'EMA20': 'float',
            'EMA50': 'float',
            'Momentum': 'float',
            'Williams %R': 'float',
            'ADX': 'float',
            'Volume_MA20': 'float',
            'ROC': 'float',
            'CMF': 'float',
            'OBV': 'float',
        }

        df = pd.read_csv(file_path, dtype=dtype)
        logging.info(f"주식 데이터를 '{file_path}'에서 성공적으로 가져왔습니다.")
        logging.debug(f"데이터프레임 정보:\n{df.info()}")
        logging.debug(f"가져온 데이터 길이: {len(df)}")  # 데이터 길이 로그 추가

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        logging.debug(f"가져온 데이터의 첫 5행:\n{df.head()}")
        return df
    except Exception as e:
        logging.error(f"주식 데이터 가져오기 중 오류 발생: {e}")
        return None

def is_business_day(date):
    """주어진 날짜가 거래일인지 확인하는 함수."""
    kr_holidays = holidays.KR()  # 한국의 공휴일 정보 가져오기
    return date.weekday() < 5 and date not in kr_holidays  # 주말(토요일: 5, 일요일: 6) 및 공휴일 확인

def prepare_data(df):
    """데이터를 준비하고 분할하는 함수."""
    logging.debug("데이터 준비 및 분할 시작...")
    
    features = [
        'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
        'EMA20', 'EMA50', 'ATR', 'Volume'
    ]

    X = []
    y = []
    stock_codes = []

    # 각 종목에 대해 6개월치 데이터를 가지고 훈련
    for stock_code in df['Code'].unique():
        stock_data = df[df['Code'] == stock_code]
        stock_data = stock_data.sort_values(by='Date')  # 날짜 기준으로 정렬

        # 6개월 데이터 필터링
        recent_data = stock_data[stock_data['Date'] >= (pd.Timestamp.now() - pd.DateOffset(months=6))]

        # 마지막 날의 피처와 타겟을 함께 추가
        for i in range(len(recent_data) - 26):  # 26 거래일 예측
            # 향후 26 거래일 동안의 평균 종가가 오늘 종가보다 높은지 확인
            target_future = 1 if recent_data['Close'].iloc[i + 26:i + 52].mean() > recent_data['Close'].iloc[i] else 0
            
            # 피처 추가
            X.append(recent_data[features].iloc[i].values)  # 현재 피처
            y.append(target_future)  # 향후 26 거래일의 평균 상승 여부
            stock_codes.append(stock_code)  # 종목 코드 추가

    X = np.array(X)
    y = np.array(y)

    # 클래스 분포 확인
    logging.info(f"타겟 클래스 분포: {np.bincount(y)}")

    # SMOTE 적용
    if len(np.unique(y)) > 1:
        logging.debug("SMOTE를 적용하여 클래스 불균형 문제 해결 중...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info("SMOTE를 적용하여 데이터 샘플 수를 조정했습니다.")
    else:
        logging.warning("타겟 클래스가 1개만 존재합니다. SMOTE를 적용하지 않습니다.")
        X_resampled, y_resampled = X, y

    # 데이터 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )
    
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    logging.debug("데이터 준비 및 분할 완료.")
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_model_with_hyperparameter_tuning():
    """모델을 훈련시키고 하이퍼파라미터를 튜닝하는 함수."""
    logging.info("모델 훈련 시작...")
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 모델 훈련을 중단합니다.")
        return None

    # 데이터 준비 및 분할
    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_data(df)

    # 하이퍼파라미터 튜닝을 위한 GridSearchCV 설정
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
    
    logging.info("하이퍼파라미터 튜닝 시작...")
    grid_search.fit(X_train, y_train)  # 하이퍼파라미터 튜닝
    
    # 최적의 하이퍼파라미터 출력
    logging.info(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
    print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")

    # 최적의 모델로 재훈련
    best_model = grid_search.best_estimator_

    # 모델 평가
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info(f"모델 성능 보고서:\n{report}")
    print(report)

    logging.info("모델 훈련 완료.")
    return best_model  # 최적 모델 반환

def predict_future_trading_days(model):
    """향후 26 거래일의 상승 여부를 예측하는 함수."""
    logging.info("향후 26 거래일 예측 시작...")
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 예측을 중단합니다.")
        return None
        
    # features 리스트 정의
    features = [
        'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
        'EMA20', 'EMA50', 'ATR', 'Volume'
    ]

    # 오늘 데이터 가져오기
    today_data = df.tail(1)
    if today_data.empty:
        logging.warning("오늘 데이터가 없습니다. 예측을 중단합니다.")
        return

    future_predictions = []
    
    # 오늘 날짜를 기준으로 향후 거래일 예측
    today_date = today_data['Date'].values[0]
    logging.info(f"오늘 날짜: {today_date}")

    future_date = today_date
    for _ in range(26):  # +1 거래일부터 +26 거래일까지
        future_date += pd.DateOffset(days=1)  # 하루씩 더하기
        while not is_business_day(future_date):  # 거래일이 아닐 경우 다음 날로 이동
            logging.debug(f"{future_date}는 거래일이 아닙니다. 다음 날로 이동합니다.")
            future_date += pd.DateOffset(days=1)

        logging.info(f"유효한 거래일: {future_date}")

        # future_date에 대한 피처 데이터 준비
        future_data = today_data.copy()
        future_data['Date'] = future_date  # 날짜를 미래 날짜로 설정

        # 예측을 위한 피처 데이터 준비
        future_features = future_data[features].values
        logging.debug(f"예측에 사용될 피처 데이터: {future_features}")

        # 예측 수행
        prediction = model.predict(future_features)
        future_predictions.append((future_date, prediction[0]))  # 날짜와 예측값 저장
        logging.info(f"{future_date}에 대한 예측 결과: {'상승' if prediction[0] == 1 else '하락'}")

    logging.info("향후 26 거래일 예측 완료.")
    return future_predictions  # 날짜와 예측값 리스트 반환

def identify_buy_signals(future_predictions):
    """매수 신호를 식별하는 함수."""
    buy_signals = []
    for date, prediction in future_predictions:
        if prediction == 1:  # 상승 예측인 경우
            buy_signals.append(date)

    return buy_signals

def calculate_returns(future_predictions, df):
    """매수 및 매도 시점에서의 상승률을 계산하는 함수."""
    returns = []

    # 예측된 미래 종가를 가져오기 위한 데이터프레임 생성
    future_dates = [date for date, _ in future_predictions]
    future_prices = []  # 예측된 가격 저장

    # 각 미래 날짜에 대한 Close 가격을 가져옵니다.
    for date in future_dates:
        if date in df['Date'].values:
            future_prices.append(df.loc[df['Date'] == date, 'Close'].values[0])
        else:
            logging.warning(f"{date}에 대한 데이터가 존재하지 않습니다.")

    # 매수 시점과 매도 시점 정의
    if len(future_prices) > 0:
        min_price_index = np.argmin(future_prices)  # 가장 낮은 가격의 인덱스
        max_price_index = np.argmax(future_prices)  # 가장 높은 가격의 인덱스

        # 매수 및 매도 시점 설정
        buy_date = future_dates[min_price_index]
        sell_date = future_dates[max_price_index]

        # 매수 및 매도 가격
        buy_price = future_prices[min_price_index]
        sell_price = future_prices[max_price_index]

        if sell_date > buy_date:  # 매도 시점이 매수 시점 이후인지 확인
            # 상승률 계산
            return_rate = (sell_price - buy_price) / buy_price * 100  # 상승률(%)
            returns.append((buy_date, sell_date, return_rate))

    # 상승률 기준으로 정렬
    returns.sort(key=lambda x: x[2], reverse=True)  # 높은 순으로 정렬
    return returns[:20]  # 상위 20개 종목 반환

def plot_predictions_and_signals(future_predictions, buy_signals):
    """예측 결과와 매수 신호를 그래프로 시각화하는 함수."""
    dates, predictions = zip(*future_predictions)  # 날짜와 예측 결과 분리
    dates = pd.to_datetime(dates)  # 날짜 형식 변환

    plt.figure(figsize=(12, 6))
    plt.plot(dates, predictions, marker='o', linestyle='-', color='b', label='예측된 상승 여부')
    plt.axhline(y=0.5, color='r', linestyle='--', label='매수 기준선')
    
    # 매수 신호 표시
    for signal in buy_signals:
        plt.axvline(x=signal, color='g', linestyle='--', label='매수 신호')

    plt.title('향후 26 거래일 주가 상승 예측 및 매수 신호')
    plt.xlabel('날짜')
    plt.ylabel('예측 결과 (1: 상승, 0: 하락)')
    plt.xticks(rotation=45)
    plt.yticks([0, 1], ['하락', '상승'])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    # 모델 훈련 및 예측 실행
    best_model = train_model_with_hyperparameter_tuning()
    if best_model:
        future_predictions = predict_future_trading_days(best_model)
        if future_predictions is not None:
            for date, prediction in future_predictions:
                logging.info(f"{date.date()}: 예측된 상승 여부: {'상승' if prediction == 1 else '하락'}")
            
            # 매수 신호 식별
            buy_signals = identify_buy_signals(future_predictions)
            logging.info(f"매수 신호: {buy_signals}")

            # 상승률 계산
            df = fetch_stock_data()  # 주식 데이터 가져오기
            if df is not None:
                top_stocks = calculate_returns(future_predictions, df)
                print("상승률이 가장 높은 종목 20개:")
                for stock in top_stocks:
                    logging.info(f"매수 시점: {stock[0].date()}, 매도 시점: {stock[1].date()}, 상승률: {stock[2]:.2f}%")
                    print(f"매수 시점: {stock[0].date()}, 매도 시점: {stock[1].date()}, 상승률: {stock[2]:.2f}%")

            # 예측 결과 및 매수 신호 시각화
            plot_predictions_and_signals(future_predictions, buy_signals)

# 메인 함수 실행
main()

