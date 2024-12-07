import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
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
    """Bayesian Optimization과 Cross-Validation을 사용하여 XGBoost 하이퍼파라미터를 최적화."""
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
        # Cross-Validation을 통한 평가
        cv_scores = cross_val_score(model, X_train_reshaped, y_train, cv=3, scoring='neg_mean_squared_error')
        return np.mean(cv_scores)  # 평균 CV 점수 반환 (음수 부호는 반전)

    param_bounds = {
        'n_estimators': (50, 100),
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
    optimizer.maximize(init_points=5, n_iter=15)  # 초기 탐색 및 최적화 반복 횟수

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

# 나머지 코드는 그대로 유지 (변경 사항 없음)

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
