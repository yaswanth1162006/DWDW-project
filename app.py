import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- 1. PAGE CONFIGURATION & LAVENDER THEME CSS ---
st.set_page_config(
    page_title="Urban Air Quality Analyzer",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Main Dashboard Lavender Background */
    [data-testid="stAppViewContainer"] {
        background-color: #E6E6FA !important;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }
    
    /* Sidebar Lavender Styling */
    [data-testid="stSidebar"] {
        background-color: #DCD0FF !important;
    }
    
    /* High-Contrast Deep Purple/Plum Metric Cards */
    .metric-card {
        background-color: #2D233D;
        border-radius: 10px;
        padding: 16px 20px;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-card-alert {
        background-color: #3D1E2A;
        border-radius: 10px;
        padding: 16px 20px;
        color: white;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        color: #D8C2FF;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
        color: #FFFFFF;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- 2. DATA GENERATOR & CACHING ---
@st.cache_data
def load_air_quality_data():
    np.random.seed(42)
    cities_data = [
        ("Byrnihat", "Meghalaya", 26.12, 91.87),
        ("Delhi", "Delhi", 28.61, 77.20),
        ("Patna", "Bihar", 25.59, 85.13),
        ("Mumbai", "Maharashtra", 19.07, 72.87),
        ("Kolkata", "West Bengal", 22.57, 88.36),
        ("Bengaluru", "Karnataka", 12.97, 77.59),
        ("Guwahati", "Assam", 26.14, 91.73),
    ]

    start_date = datetime.date(2022, 4, 1)
    end_date = datetime.date(2025, 4, 30)
    total_days = (end_date - start_date).days

    records = []
    for _ in range(2500):
        city_info = cities_data[np.random.randint(0, len(cities_data))]
        random_day = np.random.randint(0, total_days)
        record_date = start_date + datetime.timedelta(days=random_day)

        if city_info[0] == "Byrnihat":
            aqi = int(np.random.normal(250, 70))
        else:
            aqi = int(np.random.normal(120, 50))

        aqi = max(3, min(500, aqi))

        if aqi <= 50:
            category = "Good"
        elif aqi <= 100:
            category = "Moderate"
        elif aqi <= 200:
            category = "Unhealthy"
        elif aqi <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"

        records.append(
            {
                "Date": record_date,
                "City": city_info[0],
                "State": city_info[1],
                "lat": city_info[2],
                "lon": city_info[3],
                "AQI": aqi,
                "AQI_Category": category,
                "Pollutant": np.random.choice(
                    ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
                ),
                "Alert_Triggered": 1 if aqi > 200 else 0,
            }
        )

    df_out = pd.DataFrame(records)
    return df_out.sort_values("Date")


df = load_air_quality_data()

# --- 3. SIDEBAR: CONTROL PANEL & FILTERS ---
st.sidebar.markdown("## 🎲 Control Panel")

module = st.sidebar.radio(
    "Select Module",
    options=[
        "📊 Executive Analytics & Map",
        "📈 Visual Analytics & Trends",
        "🚨 Risk & Alert Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Interactive Filters")

min_date = datetime.date(2022, 4, 1)
max_date = datetime.date(2025, 4, 30)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

aqi_range = st.sidebar.slider(
    "AQI Range", min_value=3, max_value=500, value=(3, 500)
)

all_states = sorted(df["State"].unique().tolist())
selected_states = st.sidebar.multiselect(
    "Filter States", options=all_states, default=[]
)

# Apply Filter Logic
filtered_df = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["Date"] >= date_range[0])
        & (filtered_df["Date"] <= date_range[1])
    ]

filtered_df = filtered_df[
    (filtered_df["AQI"] >= aqi_range[0]) & (filtered_df["AQI"] <= aqi_range[1])
]

if selected_states:
    filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]

# --- SIDEBAR: REPORT DOWNLOAD BUTTON ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download Report")

csv_report = filtered_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="📄 Download Air Quality CSV Report",
    data=csv_report,
    file_name="air_quality_filtered_report.csv",
    mime="text/csv",
    use_container_width=True,
)

# Shared KPI Calculation & Render
if not filtered_df.empty:
    avg_aqi = round(filtered_df["AQI"].mean(), 1)
    max_aqi = filtered_df["AQI"].max()
    alert_count = filtered_df["Alert_Triggered"].sum()
    most_polluted = filtered_df.groupby("City")["AQI"].mean().idxmax()
else:
    avg_aqi, max_aqi, alert_count, most_polluted = 0.0, 0, 0, "N/A"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">AVERAGE AQI</div><div class="metric-value">{avg_aqi}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">MAX AQI VALUE</div><div class="metric-value">{max_aqi}</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card-alert"><div class="metric-label">ALERT TRIGGERED</div><div class="metric-value">{alert_count}</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">MOST POLLUTED CITY</div><div class="metric-value">{most_polluted}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. MODULE CONTROLLERS ---

# --- MODULE 1: EXECUTIVE ANALYTICS & MAP ---
if module == "📊 Executive Analytics & Map":
    tab1, tab2, tab3 = st.tabs(
        [
            "🗺️ Geospatial Alert Map",
            "📊 Rankings & Breakdown",
            "🚨 Alert Logs & Raw Data",
        ]
    )

    with tab1:
        st.subheader("🗺️ City Hazard Mapping")

        if not filtered_df.empty:
            city_map_df = (
                filtered_df.groupby(["City", "State", "lat", "lon"])
                .agg(AQI=("AQI", "mean"), Alerts=("Alert_Triggered", "sum"))
                .reset_index()
            )
            city_map_df["AQI"] = city_map_df["AQI"].round(1)

            # High-visibility red marker map configuration
            red_scale = ["#FF8080", "#FF0000", "#800000"]

            if hasattr(px, "scatter_map"):
                fig_map = px.scatter_map(
                    city_map_df,
                    lat="lat",
                    lon="lon",
                    color="AQI",
                    size="AQI",
                    size_max=32,  # Significantly increases dot sizes for visibility
                    hover_name="City",
                    hover_data=["State", "AQI", "Alerts"],
                    color_continuous_scale=red_scale,
                    range_color=[
                        30,
                        400,
                    ],  # Prevents low AQI dots from washing out
                    zoom=3.8,
                    center={"lat": 22.5937, "lon": 78.9629},
                    map_style="open-street-map",
                )
            else:
                fig_map = px.scatter_mapbox(
                    city_map_df,
                    lat="lat",
                    lon="lon",
                    color="AQI",
                    size="AQI",
                    size_max=32,
                    hover_name="City",
                    hover_data=["State", "AQI", "Alerts"],
                    color_continuous_scale=red_scale,
                    range_color=[30, 400],
                    zoom=3.8,
                    center={"lat": 22.5937, "lon": 78.9629},
                    mapbox_style="open-street-map",
                )

            # Ensure markers are semi-opaque and pop clearly
            fig_map.update_traces(marker=dict(opacity=0.92))
            fig_map.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=520
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No records found matching current filter values.")

    with tab2:
        st.subheader("📊 AQI Rankings by City")
        if not filtered_df.empty:
            city_ranking = (
                filtered_df.groupby("City")["AQI"]
                .mean()
                .reset_index()
                .sort_values("AQI", ascending=False)
            )
            fig_bar = px.bar(
                city_ranking,
                x="City",
                y="AQI",
                color="AQI",
                color_continuous_scale="Reds",
                title="Average AQI by City",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("🚨 Raw Data Logs")
        st.dataframe(filtered_df, use_container_width=True)

# --- MODULE 2: VISUAL ANALYTICS & TRENDS ---
elif module == "📈 Visual Analytics & Trends":
    st.subheader("📈 AQI Trend & Visual Analytics")

    if not filtered_df.empty:
        # AQI Trend Line Chart
        daily_trend = (
            filtered_df.groupby("Date")["AQI"]
            .mean()
            .reset_index()
            .sort_values("Date")
        )
        fig_trend = px.line(
            daily_trend,
            x="Date",
            y="AQI",
            title="📈 AQI Trend Over Time",
            line_shape="linear",
        )
        fig_trend.update_traces(line_color="#7B2CBF", line_width=2)
        st.plotly_chart(fig_trend, use_container_width=True)

        col_left, col_right = st.columns(2)

        with col_left:
            # Bar Chart: Average AQI by City
            city_avg = (
                filtered_df.groupby("City")["AQI"]
                .mean()
                .reset_index()
                .sort_values("AQI", ascending=False)
            )
            fig_city_bar = px.bar(
                city_avg,
                x="City",
                y="AQI",
                color="AQI",
                color_continuous_scale="Purples",
                title="📊 Average AQI by City (Bar Chart)",
            )
            st.plotly_chart(fig_city_bar, use_container_width=True)

        with col_right:
            # Pie Chart: Pollutant Breakdown
            pollutant_counts = (
                filtered_df["Pollutant"].value_counts().reset_index()
            )
            pollutant_counts.columns = ["Pollutant", "Count"]
            fig_pie = px.pie(
                pollutant_counts,
                names="Pollutant",
                values="Count",
                title="🍕 Primary Pollutant Distribution (Pie Chart)",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Purp,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No data available to display charts.")

# --- MODULE 3: RISK & ALERT ANALYSIS ---
elif module == "🚨 Risk & Alert Analysis":
    st.subheader("🚨 Pollution Alert & Risk Category Breakdown")

    if not filtered_df.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            # Bar Chart: Total Alerts Triggered
            alert_cities = (
                filtered_df.groupby("City")["Alert_Triggered"]
                .sum()
                .reset_index()
                .sort_values("Alert_Triggered", ascending=False)
            )
            fig_alerts_bar = px.bar(
                alert_cities,
                x="City",
                y="Alert_Triggered",
                color="Alert_Triggered",
                color_continuous_scale="Reds",
                title="🚨 High-Risk Alerts Triggered by City (Bar Chart)",
            )
            st.plotly_chart(fig_alerts_bar, use_container_width=True)

        with col_b:
            # Pie Chart: AQI Severity Categories
            cat_counts = (
                filtered_df["AQI_Category"].value_counts().reset_index()
            )
            cat_counts.columns = ["Category", "Count"]
            fig_cat_pie = px.pie(
                cat_counts,
                names="Category",
                values="Count",
                title="🍕 AQI Severity Breakdown (Pie Chart)",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            st.plotly_chart(fig_cat_pie, use_container_width=True)

        # Alert Trend Line Chart
        alert_trend = (
            filtered_df.groupby("Date")["Alert_Triggered"].sum().reset_index()
        )
        fig_alert_trend = px.area(
            alert_trend,
            x="Date",
            y="Alert_Triggered",
            title="⚠️ Daily High-Risk Alert Spike Trend",
            color_discrete_sequence=["#9D4EDD"],
        )
        st.plotly_chart(fig_alert_trend, use_container_width=True)
    else:
        st.warning("No data available to display risk metrics.")