import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path, dtype={'Code': str})  # 'Code' 열을 문자열로 읽어오기
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print("데이터 로드 완료. 열 목록:")
    print(df.columns.tolist())  # 로드된 데이터의 열 목록 출력
    return df


def prepare_data(df):
    """데이터를 준비하는 함수 (최근 60일 데이터를 학습, 향후 26일 예측)."""
    df = df[['Close', 'Change', 'EMA20', 'EMA50', 'RSI', 'MACD',
             'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Stoch',
             'Momentum', 'ADX']].dropna()

    x_data, y_data = [], []
    for i in range(60, len(df) - 26):
        x_data.append(df.iloc[i - 60:i].values)  # 이전 60일 데이터
        y_data.append(df.iloc[i + 25]['Close'])  # 26일 후의 종가

    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data


def create_and_train_model(X_train, y_train):
    """모델을 생성하고 훈련하는 함수."""
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,
                              colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_jobs=-1)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model


def predict_future_prices(model, last_60_days):
    """모델을 사용하여 향후 26일 가격을 예측하는 함수."""
    predictions = []
    input_data = last_60_days.copy()

    for _ in range(26):
        pred = model.predict(input_data.reshape(1, -1))[0]
        predictions.append(pred)
        # 새로 예측된 값을 입력 데이터에 추가하고, 맨 앞의 데이터를 제거
        input_data = np.roll(input_data, -1, axis=0)
        input_data[-1, 0] = pred  # 'Close'에 해당하는 값 업데이트

    return predictions


def generate_signals(predictions, start_date):
    """예측 결과를 기반으로 매수 및 매도 신호를 생성하는 함수."""
    buy_index = np.argmin(predictions)  # 최저점 인덱스
    remaining_predictions = predictions[buy_index + 1:]
    
    if len(remaining_predictions) > 0:
        sell_index = buy_index + np.argmax(remaining_predictions) + 1  # 매수 후 최고점 인덱스
    else:
        sell_index = buy_index  # 매수 인덱스를 그대로 매도 인덱스로 설정

    if sell_index >= len(predictions):
        sell_index = buy_index  # 매도 인덱스를 매수 인덱스로 설정

    buy_date = start_date + pd.Timedelta(days=buy_index)
    sell_date = start_date + pd.Timedelta(days=sell_index)

    print(f"매수 신호 날짜: {buy_date}, 매도 신호 날짜: {sell_date}")
    return buy_index, sell_index, buy_date, sell_date


def save_and_merge_top_20(df_top_20, original_data_path):
    """
    상위 20개 종목 데이터를 기존 데이터와 결합하여 저장.
    """
    try:
        # 기존 데이터 로드
        original_data = pd.read_csv(original_data_path, dtype={'Code': str})
        original_data['Date'] = pd.to_datetime(original_data['Date'])

        # 상위 20개 종목 데이터와 결합
        merged_data = pd.merge(
            df_top_20,
            original_data,
            on='Code',
            how='left'
        )

        # 결합된 데이터 저장
        output_path = 'data/top_20_stocks_all_dates.csv'
        merged_data.to_csv(output_path, index=False)
        print(f"결합된 데이터가 {output_path}에 저장되었습니다.")
    except Exception as e:
        print(f"데이터 병합 중 오류 발생: {e}")
        raise


def main():
    # 데이터 로드
    df = fetch_stock_data()

    # 결과 저장 리스트
    results = []

    # 종목 코드별로 데이터 처리
    for code in df['Code'].unique():
        stock_data = df[df['Code'] == code]
        current_price = stock_data['Close'].iloc[-1]

        # 데이터 준비
        try:
            x_data, y_data = prepare_data(stock_data)
        except Exception as e:
            print(f"종목 코드 {code}의 데이터 준비 중 오류: {e}")
            continue

        if len(x_data) < 60:
            print(f"종목 코드 {code}의 데이터가 충분하지 않습니다.")
            continue

        # 훈련 및 테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        # 모델 생성 및 훈련
        model = create_and_train_model(X_train, y_train)

        # 향후 26일 가격 예측
        last_60_days = x_data[-1]
        future_predictions = predict_future_prices(model, last_60_days)

        # 매수 및 매도 신호 생성
        start_date = stock_data.index[-1]
        buy_index, sell_index, buy_date, sell_date = generate_signals(future_predictions, start_date)

        buy_price = future_predictions[buy_index]
        sell_price = future_predictions[sell_index]
        price_increase_ratio = (sell_price - buy_price) / buy_price  # 상승률 계산

        results.append((code, price_increase_ratio, buy_date, sell_date, buy_price, sell_price, current_price))

    # 상위 20 종목 추출
    results.sort(key=lambda x: x[1], reverse=True)
    df_top_20 = pd.DataFrame(results[:20], columns=[
        'Code', 'Gap', 'Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Current Price'
    ])

    # 상위 20 종목 저장 및 병합
    original_data_path = 'data/stock_data_with_indicators.csv'
    save_and_merge_top_20(df_top_20, original_data_path)


# 실행
main()
