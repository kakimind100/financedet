import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
import os


def fetch_stock_data():
    """주식 데이터를 가져오는 함수."""
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path, dtype={'Code': str})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print("데이터 로드 완료. 열 목록:", df.columns.tolist())
    return df


def prepare_data(df):
    """데이터를 준비하는 함수."""
    df = df[['Close', 'Change', 'EMA20', 'EMA50', 'RSI', 'MACD',
             'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Stoch',
             'Momentum', 'ADX']].dropna()

    x_data, y_data = [], []
    for i in range(60, len(df) - 26):
        x_data.append(df.iloc[i - 60:i].values)
        y_data.append(df.iloc[i + 25]['Close'])

    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data


def optimize_hyperparameters_bayes(X_train, y_train):
    """Bayesian Optimization을 사용하여 하이퍼파라미터 최적화."""
    def xgb_evaluate(n_estimators, learning_rate, max_depth, subsample, colsample_bytree, alpha):
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            alpha=alpha,
            n_jobs=-1
        )
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        mse = -model.score(X_train.reshape(X_train.shape[0], -1), y_train)  # 부호 반전
        return mse

    param_bounds = {
        'n_estimators': (50, 200),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 7),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.3, 1),
        'alpha': (0, 50),
    }

    optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=10)

    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    best_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, **best_params)
    best_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return best_model


def process_stock_data(stock_data, code):
    """개별 종목 데이터 처리."""
    try:
        x_data, y_data = prepare_data(stock_data)
    except Exception as e:
        print(f"종목 코드 {code}: 데이터 준비 중 오류 - {e}")
        return None

    if len(x_data) < 60:
        print(f"종목 코드 {code}: 데이터가 충분하지 않음.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = optimize_hyperparameters_bayes(X_train, y_train)

    last_60_days = x_data[-1]
    predictions = []
    input_data = last_60_days.copy()

    for _ in range(26):
        pred = model.predict(input_data.reshape(1, -1))[0]
        predictions.append(pred)
        input_data = np.roll(input_data, -1, axis=0)
        input_data[-1, 0] = pred

    buy_index = np.argmin(predictions)
    sell_index = buy_index + np.argmax(predictions[buy_index + 1:]) + 1

    start_date = stock_data.index[-1]
    buy_date = start_date + pd.Timedelta(days=buy_index)
    sell_date = start_date + pd.Timedelta(days=sell_index)

    buy_price = predictions[buy_index]
    sell_price = predictions[sell_index]
    price_increase_ratio = (sell_price - buy_price) / buy_price
    current_price = stock_data['Close'].iloc[-1]

    return code, price_increase_ratio, buy_date, sell_date, buy_price, sell_price, current_price


def main():
    df = fetch_stock_data()
    results = Parallel(n_jobs=10, backend='loky')(
        delayed(process_stock_data)(df[df['Code'] == code], code) for code in df['Code'].unique()
    )

    results = [res for res in results if res is not None]
    results.sort(key=lambda x: x[1], reverse=True)

    df_top_20 = pd.DataFrame(results[:20], columns=[
        'Code', 'Gap', 'Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Current Price'
    ])
    output_path = 'data/top_20_stocks.csv'
    df_top_20.to_csv(output_path, index=False)
    print(f"상위 20개 종목 결과가 {output_path}에 저장되었습니다.")


# 실행
main()
