import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization


def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path, dtype={'Code': str})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print("데이터 로드 완료. 열 목록:")
    print(df.columns.tolist())
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


def optimize_hyperparameters_bayes(X_train, y_train):
    """Bayesian Optimization을 사용하여 XGBoost 하이퍼파라미터를 최적화."""
    def xgb_evaluate(n_estimators, learning_rate, max_depth, subsample, colsample_bytree, alpha):
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            alpha=alpha,
            n_jobs=-1,
            tree_method='hist'  # 빠른 학습을 위한 설정
        )
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        model.fit(X_train_reshaped, y_train)
        mse = -model.score(X_train_reshaped, y_train)  # 부호 반전
        return mse

    param_bounds = {
        'n_estimators': (50, 100),  # 범위 축소
        'learning_rate': (0.05, 0.2),
        'max_depth': (3, 6),
        'subsample': (0.7, 1),
        'colsample_bytree': (0.5, 1),
        'alpha': (0, 10),
    }

    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=param_bounds,
        random_state=42,
    )
    optimizer.maximize(init_points=3, n_iter=10)

    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    print("Bayesian Optimization 최적의 하이퍼파라미터:", best_params)

    best_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, **best_params)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    best_model.fit(X_train_reshaped, y_train)
    return best_model


def create_and_train_model(X_train, y_train):
    """모델 생성 및 훈련."""
    print("하이퍼파라미터 최적화 중...")
    model = optimize_hyperparameters_bayes(X_train, y_train)
    print("최적화된 모델 생성 완료.")
    return model


def predict_future_prices(model, last_60_days):
    """모델을 사용하여 향후 26일 가격을 예측하는 함수."""
    predictions = []
    input_data = last_60_days.copy()

    for _ in range(26):
        input_data_reshaped = input_data.reshape(1, -1)
        pred = model.predict(input_data_reshaped)[0]
        predictions.append(pred)
        input_data = np.roll(input_data, -1, axis=0)
        input_data[-1, 0] = pred  # 'Close' 값 업데이트

    return predictions


def generate_signals(predictions, start_date):
    """예측 결과를 기반으로 매수 및 매도 신호를 생성하는 함수."""
    buy_index = np.argmin(predictions)
    remaining_predictions = predictions[buy_index + 1:]
    sell_index = buy_index + np.argmax(remaining_predictions) + 1 if remaining_predictions else buy_index

    buy_date = start_date + pd.Timedelta(days=buy_index)
    sell_date = start_date + pd.Timedelta(days=sell_index)

    print(f"매수 신호 날짜: {buy_date}, 매도 신호 날짜: {sell_date}")
    return buy_index, sell_index, buy_date, sell_date


def save_and_merge_top_20(df_top_20, original_data_path):
    """상위 20개 종목 데이터를 기존 데이터와 결합하여 저장."""
    try:
        original_data = pd.read_csv(original_data_path, dtype={'Code': str})
        original_data['Date'] = pd.to_datetime(original_data['Date'])
        merged_data = pd.merge(df_top_20, original_data, on='Code', how='left')
        output_path = 'data/top_20_stocks_all_dates.csv'
        merged_data.to_csv(output_path, index=False)
        print(f"결합된 데이터가 {output_path}에 저장되었습니다.")
    except Exception as e:
        print(f"데이터 병합 중 오류 발생: {e}")
        raise


def main():
    df = fetch_stock_data()
    results = []

    for code in df['Code'].unique():
        stock_data = df[df['Code'] == code]
        current_price = stock_data['Close'].iloc[-1]

        try:
            x_data, y_data = prepare_data(stock_data)
        except Exception as e:
            print(f"종목 코드 {code}의 데이터 준비 중 오류: {e}")
            continue

        if len(x_data) < 60:
            print(f"종목 코드 {code}의 데이터가 충분하지 않습니다.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        model = create_and_train_model(X_train, y_train)
        last_60_days = x_data[-1]
        future_predictions = predict_future_prices(model, last_60_days)

        start_date = stock_data.index[-1]
        buy_index, sell_index, buy_date, sell_date = generate_signals(future_predictions, start_date)

        buy_price = future_predictions[buy_index]
        sell_price = future_predictions[sell_index]
        price_increase_ratio = (sell_price - buy_price) / buy_price

        results.append((code, price_increase_ratio, buy_date, sell_date, buy_price, sell_price, current_price))

    # 상위 20 종목 정렬
    results.sort(key=lambda x: x[1], reverse=True)
    df_top_20 = pd.DataFrame(results[:20], columns=[
        'Code', 'Gap', 'Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Current Price'
    ])

    # 상위 20 종목 로그 출력
    print("\n===== 상위 20 종목 =====")
    for idx, row in df_top_20.iterrows():
        print(f"종목코드: {row['Code']}, 상승률: {row['Gap']:.2%}")
        print(f"매수 시점: {row['Buy Date']}, 매도 시점: {row['Sell Date']}")
        print(f"매수가: {row['Buy Price']:.2f}, 매도가: {row['Sell Price']:.2f}, 현재가: {row['Current Price']:.2f}")
        print("------------------------")

    save_and_merge_top_20(df_top_20, 'data/stock_data_with_indicators.csv')


# 실행
main()
