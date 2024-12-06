import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging
from datetime import datetime
import os

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    logging.debug("주식 데이터를 가져오는 중...")
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    logging.info("주식 데이터를 성공적으로 가져왔습니다.")
    return df

def prepare_data(df, cutoff_date):
    """데이터를 준비하고 훈련/검증 세트로 분할하는 함수."""
    df_train = df[df['Date'] < cutoff_date]
    df_train = pd.get_dummies(df_train, columns=['Code'], drop_first=True)
    df_train['Weekday'] = df_train['Date'].dt.weekday
    df_train['Month'] = df_train['Date'].dt.month

    X = df_train.drop(columns=['Date', 'Close'])
    y = (df_train['Close'].shift(-1) > df_train['Close']).astype(int)

    return X, y, df[df['Date'] >= cutoff_date]

def train_model(X, y):
    """모델을 훈련시키는 함수."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    logging.info("모델 훈련 완료.")
    return model

def predict_future_trading_days(model, future_data):
    """향후 거래일의 상승 여부를 예측하는 함수."""
    features = ['RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 'EMA20', 'EMA50', 'ATR', 'Volume', 'Weekday', 'Month']
    future_predictions = future_data[features].copy()
    future_predictions['Predicted'] = model.predict(future_predictions[features])
    future_predictions['Close'] = future_data['Close'].values
    return future_predictions

def calculate_buy_sell_signals(predictions):
    """매수 및 매도 시점을 계산하는 함수."""
    predictions['Buy_Signal'] = (predictions['Predicted'] == 1) & (predictions['Close'] < predictions['Close'].shift(1))
    predictions['Sell_Signal'] = (predictions['Predicted'] == 0) & (predictions['Close'] > predictions['Close'].shift(1))
    return predictions

def calculate_return(predictions):
    """상승률을 계산하는 함수."""
    predictions['Future_Close'] = predictions['Close'].shift(-1)
    predictions['Return'] = (predictions['Future_Close'] - predictions['Close']) / predictions['Close'] * 100
    return predictions

def main():
    # 데이터 로드
    df = fetch_stock_data()
    cutoff_date = datetime.now()  # 오늘 날짜로 컷오프 설정

    # 데이터 준비
    X, y, future_data = prepare_data(df, cutoff_date)

    # 모델 훈련
    model = train_model(X, y)

    # 예측
    future_predictions = predict_future_trading_days(model, future_data)

    # 매수 및 매도 신호 계산
    future_predictions = calculate_buy_sell_signals(future_predictions)

    # 상승률 계산
    future_predictions = calculate_return(future_predictions)

    # 상승률이 가장 높은 20개 종목 추출
    top_stocks = future_predictions.groupby('Code').mean().nlargest(20, 'Return')

    logging.info("상승률이 가장 높은 20개 종목:")
    print(top_stocks[['Return']])

if __name__ == "__main__":
    main()
