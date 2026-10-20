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

st.set_page_config(page_title="2023 의류 판매량 분석", page_icon="🧥", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background: #f7f9fc;
    border: 1px solid #dce3ed;
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 112px;
    box-shadow: 0 2px 5px rgba(30, 50, 80, 0.06);
}
div[data-testid="stMetricLabel"] {
    font-size: 1rem;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.title("🧥 기온 기반 의류 판매량 분석 대시보드")
st.caption("※ 의류 판매량 데이터는 분석·학습을 위해 가상으로 생성한 모의 데이터이며, 실제 시장 판매량과 다를 수 있습니다.")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "clothing_sales_weather_2023.csv"


@st.cache_data
def read_sales_csv(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    required = ["date", "channel", "clothing_category", "units_sold", "avg_temperature"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {missing}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["units_sold"] = pd.to_numeric(data["units_sold"], errors="coerce")
    data["avg_temperature"] = pd.to_numeric(data["avg_temperature"], errors="coerce")
    data["channel"] = data["channel"].astype(str).str.strip()
    data["clothing_category"] = data["clothing_category"].astype(str).str.strip()
    data = data.dropna(subset=required)
    data = data[(data["date"].dt.year == 2023) & (data["units_sold"] >= 0)].copy()
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    return data.sort_values("date").reset_index(drop=True)


if not DATA_PATH.is_file():
    st.error("데이터 파일을 찾을 수 없습니다. data 폴더에 clothing_sales_weather_2023.csv 파일이 있는지 확인해 주세요.")
    st.stop()

try:
    df = read_sales_csv(str(DATA_PATH))
except Exception as error:
    st.error("데이터를 읽는 중 문제가 발생했습니다.")
    st.exception(error)
    st.stop()

if df.empty:
    st.error("2023년 데이터가 없습니다. CSV 파일을 확인해 주세요.")
    st.stop()

st.sidebar.header("🔎 분석 조건")
min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "분석 기간", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

categories = sorted(df["clothing_category"].unique().tolist())
channels = sorted(df["channel"].unique().tolist())
selected_categories = st.sidebar.multiselect("의류 종류", categories, default=categories)
selected_channels = st.sidebar.multiselect("판매 채널", channels, default=channels)

filtered = df[
    df["date"].dt.date.between(start_date, end_date)
    & df["clothing_category"].isin(selected_categories)
    & df["channel"].isin(selected_channels)
].copy()

if filtered.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다. 왼쪽 필터를 변경해 주세요.")
    st.stop()

daily = (
    filtered.groupby("date", as_index=False)
    .agg(daily_sales=("units_sold", "sum"), avg_temperature=("avg_temperature", "mean"))
    .sort_values("date")
)
daily["sales_7day_average"] = daily["daily_sales"].rolling(7, min_periods=1).mean()

total_sales = filtered["units_sold"].sum()
mean_daily_sales = daily["daily_sales"].mean()
mean_temp = filtered["avg_temperature"].mean()
days_count = filtered["date"].dt.date.nunique()

# 네모 박스 형태의 핵심 지표
m1, m2, m3, m4 = st.columns(4)
m1.metric("총 판매량", f"{total_sales:,.0f}개")
m2.metric("일평균 판매량", f"{mean_daily_sales:,.1f}개")
m3.metric("평균 기온", f"{mean_temp:.1f}°C")
m4.metric("분석 일수", f"{days_count}일")

st.divider()
st.subheader("📈 날짜별 판매량 추세")
st.caption("이동평균은 하루 단위의 급격한 변동을 줄여 전체 흐름을 보기 쉽게 합니다.")
trend = go.Figure()
trend.add_trace(go.Scatter(
    x=daily["date"], y=daily["daily_sales"],
    mode="lines", name="일별 판매량",
    line=dict(color="#8CC8E8", width=1.3)
))
trend.add_trace(go.Scatter(
    x=daily["date"], y=daily["sales_7day_average"],
    mode="lines", name="7일 이동평균",
    line=dict(color="#E4572E", width=3.2)
))
trend.update_layout(
    xaxis_title="날짜",
    yaxis_title="판매량(개)",
    hovermode="x unified",
    legend_title="",
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(trend, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("🌡️ 기온과 판매량")
    st.caption("추세선을 함께 표시해 전체적인 관계를 확인합니다.")
    correlation = daily["avg_temperature"].corr(daily["daily_sales"])
    st.caption(f"기온과 일별 판매량의 상관계수: {correlation:.3f}")
    scatter = px.scatter(
        daily, x="avg_temperature", y="daily_sales", trendline="ols",
        labels={"avg_temperature": "평균 기온(°C)", "daily_sales": "일별 판매량(개)"},
        title="기온별 일판매량"
    )
    st.plotly_chart(scatter, use_container_width=True)

with right:
    st.subheader("🧊️ 기온 구간별 평균 판매량")
    st.caption("기온 구간별로 하루 평균 판매량을 비교합니다.")
    temp_daily = daily.copy()
    temp_daily["temp_range"] = pd.cut(
        temp_daily["avg_temperature"],
        bins=[-np.inf, 0, 10, 20, 30],
        labels=["0°C 미만", "0~10°C", "10~20°C", "20~30°C"],
        right=False
    )
    temp_summary = (
        temp_daily.dropna(subset=["temp_range"])
        .groupby("temp_range", observed=False)["daily_sales"]
        .mean().reindex(["0°C 미만", "0~10°C", "10~20°C", "20~30°C"])
        .reset_index(name="average_sales")
    )
    temp_chart = px.bar(
        temp_summary, x="temp_range", y="average_sales",
        labels={"temp_range": "기온 구간", "average_sales": "평균 일판매량(개)"},
        title="기온 구간별 평균 일판매량",
        text=temp_summary["average_sales"].map(lambda x: f"{x:,.1f}개" if pd.notna(x) else ""),
        color="temp_range",
        color_discrete_sequence=["#7DB7D8", "#7DB7D8", "#7DB7D8", "#7DB7D8"]
    )
    temp_chart.update_traces(textposition="outside", cliponaxis=False)
    temp_chart.update_layout(showlegend=False, xaxis_title="기온 구간", yaxis_title="평균 일판매량(개)")
    st.plotly_chart(temp_chart, use_container_width=True)

st.subheader("👕 의류 종류별 판매량")
st.caption("선택한 기간과 필터를 기준으로 의류 종류별 총 판매량을 비교합니다.")
category_summary = (
    filtered.groupby("clothing_category", as_index=False)["units_sold"]
    .sum().sort_values("units_sold", ascending=False)
)
category_summary["sales_label"] = category_summary["units_sold"].map(lambda value: f"{value:,.0f}개")
category_chart = px.bar(
    category_summary,
    x="units_sold",
    y="clothing_category",
    orientation="h",
    text="sales_label",
    labels={"clothing_category": "의류 종류", "units_sold": "총 판매량(개)"},
    title="의류 종류별 총 판매량",
    color="clothing_category",
    color_discrete_sequence=["#376996", "#E07A5F", "#81B29A", "#F2CC8F", "#8D80AD"]
)
category_chart.update_traces(textposition="outside", cliponaxis=False)
category_chart.update_layout(
    showlegend=False,
    xaxis_title="총 판매량(개)",
    yaxis_title="의류 종류",
    yaxis={"categoryorder": "total ascending"},
    margin=dict(l=10, r=45, t=45, b=10)
)
st.plotly_chart(category_chart, use_container_width=True)

st.divider()
st.subheader("🔮 판매량 예측 모델")
st.caption("예측 결과는 선택한 날짜·의류 종류·판매 채널 조합의 판매량입니다. 학습 데이터는 가상으로 만든 합성/모의 데이터이므로 실제 시장 판매량을 보장하지 않습니다.")

features = [
    "avg_temperature", "month", "day_of_week", "day_of_month",
    "is_weekend", "clothing_category", "channel"
]
numeric_features = ["avg_temperature", "month", "day_of_week", "day_of_month", "is_weekend"]
categorical_features = ["clothing_category", "channel"]
ordered = df.sort_values("date").reset_index(drop=True)
split_at = max(1, int(len(ordered) * 0.8))
train = ordered.iloc[:split_at]
test = ordered.iloc[split_at:]

if len(test) > 0:
    prep = ColumnTransformer([
        ("numbers", "passthrough", numeric_features),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    model = Pipeline([
        ("prep", prep),
        ("forest", RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2))
    ])
    model.fit(train[features], train["units_sold"])
    predicted_test = model.predict(test[features])

    score1, score2, score3 = st.columns(3)
    score1.metric("MAE", f"{mean_absolute_error(test['units_sold'], predicted_test):.2f}")
    score2.metric("RMSE", f"{np.sqrt(mean_squared_error(test['units_sold'], predicted_test)):.2f}")
    score3.metric("R²", f"{r2_score(test['units_sold'], predicted_test):.3f}")

    st.markdown("#### 예측 조건")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        pred_date = st.date_input(
            "예측 날짜", value=max_date, min_value=min_date, max_value=max_date, key="pred_date"
        )
    with p2:
        pred_temp = st.number_input(
            "평균 기온(°C)",
            min_value=-30.0,
            max_value=45.0,
            value=15.0,
            step=1.0
        )
    with p3:
        pred_category = st.selectbox("의류 종류", categories, key="pred_category")
    with p4:
        pred_channel = st.selectbox("판매 채널", channels, key="pred_channel")

    if st.button("판매량 예측", type="primary"):
        row = pd.DataFrame([{
            "avg_temperature": float(pred_temp),
            "month": pred_date.month,
            "day_of_week": pred_date.weekday(),
            "day_of_month": pred_date.day,
            "is_weekend": int(pred_date.weekday() >= 5),
            "clothing_category": pred_category,
            "channel": pred_channel
        }])
        estimate = max(0.0, float(model.predict(row[features])[0]))
        st.success(f"예상 판매량: **{estimate:.0f}개**")
else:
    st.warning("모델 평가를 위한 데이터가 충분하지 않습니다.")

with st.expander("📋 데이터 미리보기 및 다운로드"):
    st.dataframe(
        filtered[["date", "channel", "clothing_category", "units_sold", "avg_temperature"]],
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "현재 데이터 CSV 다운로드",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="clothing_sales_2023_filtered.csv",
        mime="text/csv"
    )
