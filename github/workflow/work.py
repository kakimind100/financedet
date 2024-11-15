name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Check out the repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'  # 필요한 Python 버전으로 변경 가능

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install FinanceDataReader pandas

    - name: Run script
      run: |
        python stock_analysis.py  # 본인의 파일 이름으로 변경

