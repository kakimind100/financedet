import pandas as pd
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib  # 추가

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
    try:
        file_path = os.path.join('data', 'stock_data_with_indicators.csv')
        
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
            'OBV': 'float'
        }

        df = pd.read_csv(file_path, dtype=dtype)
        logging.info(f"주식 데이터를 '{file_path}'에서 성공적으로 가져왔습니다.")
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        return df
    except Exception as e:
        logging.error(f"주식 데이터 가져오기 중 오류 발생: {e}")
        return None

def prepare_data(df):
    """데이터를 준비하고 분할하는 함수."""
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
                'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50', 'Momentum', 
                'Williams %R', 'ADX', 'Volume_MA20', 'ROC', 'CMF', 'OBV']

    X = []
    y = []
    stock_codes = []

    for stock_code in df['Code'].unique():
        stock_data = df[df['Code'] == stock_code].tail(11)

        if len(stock_data) == 11:
            low_price = stock_data['Low'].min()
            high_price = stock_data['High'].max()
            target_today = 1 if high_price > low_price * 1.2 else 0

            X.append(stock_data[features].values[-1])
            y.append(target_today)
            stock_codes.append(stock_code)

    X = np.array(X)
    y = np.array(y)

    logging.info(f"타겟 클래스 분포: {np.bincount(y)}")

    if len(np.unique(y)) > 1:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        stock_codes_resampled = [stock_codes[i % len(stock_codes)] for i in range(len(y_resampled))]
    else:
        logging.warning("타겟 클래스가 1개만 존재합니다. SMOTE를 적용하지 않습니다.")
        X_resampled, y_resampled = X, y
        stock_codes_resampled = stock_codes

    X_train, X_temp, y_train, y_temp, stock_codes_train, stock_codes_temp = train_test_split(
        X_resampled, y_resampled, stock_codes_resampled, test_size=0.3, random_state=42
    )

    logging.info(f"훈련에 사용된 종목 코드: {stock_codes_train}")

    X_valid, X_test, y_valid, y_test, stock_codes_valid, stock_codes_test = train_test_split(
        X_temp, y_temp, stock_codes_temp, test_size=0.5, random_state=42
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test, stock_codes_train, stock_codes_valid, stock_codes_test

def train_model_with_hyperparameter_tuning():
    """모델을 훈련시키고 하이퍼파라미터를 튜닝하는 함수."""
    df = fetch_stock_data()
    if df is None:
        logging.error("데이터프레임이 None입니다. 모델 훈련을 중단합니다.")
        return None, None

    X_train, X_valid, X_test, y_train, y_valid, y_test, stock_codes_train, stock_codes_valid, stock_codes_test = prepare_data(df)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)

    logging.info(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
    print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info(f"모델 성능 보고서:\n{report}")
    print(report)

    return best_model, stock_codes_test

if __name__ == "__main__":
    logging.info("모델 훈련 스크립트 실행 중...")
    model, stock_codes_test = train_model_with_hyperparameter_tuning()
    logging.info("모델 훈련 스크립트 실행 완료.")
    joblib.dump(model, 'best_model.pkl')  # 모델 저장
