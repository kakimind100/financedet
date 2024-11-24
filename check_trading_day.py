import os
import datetime
import requests

def is_trading_day():
    today = datetime.datetime.now()
    
    # 주말 확인 (토요일(5) 또는 일요일(6))
    if today.weekday() >= 5:
        return False

    # 공휴일 확인
    api_key = os.getenv('KOREA_HOLIDAY_API_KEY')  # 환경 변수에서 API 키 가져오기
    url = f"http://open.neis.go.kr/hub/holidays?KEY={api_key}&Type=json&SOL_YEAR={today.year}&SOL_MONTH={today.month}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        holidays = data.get('holidays', [])
        for holiday in holidays:
            if holiday['date'] == today.strftime("%Y%m%d"):
                return False  # 오늘이 공휴일이면 거래일 아님

        return True  # 거래일
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

if __name__ == "__main__":
    if not is_trading_day():
        print("오늘은 거래일이 아닙니다.")
        exit(1)  # 비정상 종료
    else:
        print("오늘은 거래일입니다.")
