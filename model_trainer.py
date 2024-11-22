# model_trainer.py
import pandas as pd
import joblib
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'model_trainer.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model():
    """모델을 훈련시키고 저장하는 함수."""
    df = pd.read_csv('stock_data_with_indicators.csv')
    features = ['MA5', 'MA20', 'RSI']
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 다음 날 종가 상승 여부

    # NaN 제거
    df.dropna(subset=features + ['Target'], inplace=True)

    X = df[features]
    y = df['Target']

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 훈련
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 모델 저장
    joblib.dump(model, 'stock_model.pkl')
    logging.info("모델 훈련 완료 및 'stock_model.pkl'로 저장되었습니다.")

    # 모델 평가
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info(f"모델 성능 보고서:\n{report}")
    print(report)

if __name__ == "__main__":
    train_model()
