
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. 페이지 설정 및 디자인
# --------------------------------------------------
st.set_page_config(
    page_title="기온 기반 의류 판매량 분석",
    page_icon="🧥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e5eaf2;
        padding: 14px 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧥 기온 기반 의류 판매량 분석 대시보드")
st.caption("기온·날짜·의류 종류·판매 채널에 따른 판매 현황을 분석하고 판매량을 예측합니다.")
st.caption("※ 의류 판매량은 프로젝트 시연을 위해 생성한 가상(모의) 데이터이며, 실제 매출 자료가 아닙니다.")


# --------------------------------------------------
# 2. 데이터 불러오기 및 정리
# --------------------------------------------------
DATA_PATHS = [
    Path("data/clothing_sales_weather.csv"),
    Path("clothing_sales_weather.csv"),
    Path(__file__).resolve().parent / "data" / "clothing_sales_weather.csv",
]


@st.cache_data
def load_data(path_string: str) -> pd.DataFrame:
    df = pd.read_csv(path_string)
    df.columns = df.columns.str.strip()

    required_columns = {
        "date", "channel", "clothing_category", "units_sold", "avg_temperature"
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "CSV 파일에 필요한 열이 없습니다: " + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["avg_temperature"] = pd.to_numeric(df["avg_temperature"], errors="coerce")
    df["channel"] = df["channel"].astype("string").str.strip()
    df["clothing_category"] = df["clothing_category"].astype("string").str.strip()

    df = df.dropna(
        subset=[
            "date", "channel", "clothing_category",
            "units_sold", "avg_temperature"
        ]
    ).copy()

    df = df[
        (df["units_sold"] >= 0)
        & df["channel"].ne("")
        & df["clothing_category"].ne("")
    ].copy()

    df["units_sold"] = df["units_sold"].astype(float)
    df["avg_temperature"] = df["avg_temperature"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


data_path = next((p for p in DATA_PATHS if p.exists()), None)

if data_path is None:
    st.error(