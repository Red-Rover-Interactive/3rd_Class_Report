import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Load Mixpanel credentials from .env
# -------------------------------
load_dotenv()
USERNAME = os.getenv("MIXPANEL_USER", "your_service_account_username")
PASSWORD = os.getenv("MIXPANEL_PASS", "your_service_account_secret")
PROJECT_ID = os.getenv("MIXPANEL_PROJECT_ID", "your_project_id")

# -------------------------------
# Date range setup
# -------------------------------
today = datetime.today().date()
lowest_day = date(2025, 1, 1)

max_date = today
default_to = min(max_date, today)
default_from = max(lowest_day, default_to - timedelta(days=28))

from_date = st.sidebar.date_input(
    "From date", value=default_from, min_value=lowest_day, max_value=default_to
)

to_date = st.sidebar.date_input(
    "To date", value=default_to, min_value=from_date, max_value=max_date
)

if from_date > to_date:
    st.sidebar.error("⚠️ 'From date' must be before 'To date'.")
    st.stop()

# -------------------------------
# Query Mixpanel JQL API
# -------------------------------
def query_mixpanel_daily_uniques(from_date, to_date, username, password, project_id, event_selectors):
    event_selectors_js = json.dumps(event_selectors)
    jql_query = f"""
    function main() {{
      return Events({{
        from_date: "{from_date}",
        to_date: "{to_date}",
        event_selectors: {event_selectors_js}
      }});
    }}
    """

    response = requests.post(
        "https://eu.mixpanel.com/api/2.0/jql",
        auth=HTTPBasicAuth(username, password),
        data={"script": jql_query, "project_id": project_id}
    )

    if response.status_code != 200:
        st.error(f"Mixpanel Error: {response.status_code} — {response.text}")
        return pd.DataFrame()

    return pd.DataFrame(response.json())

# -------------------------------
# Load and Filter Playtest Table
# -------------------------------
try:
    df_playtest = pd.read_csv("csv_data/PlaytestTable.csv")
    df_playtest.columns = df_playtest.columns.str.strip()
    df_playtest["Start"] = pd.to_datetime(df_playtest["Start"], unit='s')
    df_playtest["End"] = pd.to_datetime(df_playtest["End"], unit='s')
    df_playtest.rename(columns={"Type of Playtest": "Playtest"}, inplace=True)

    playtest_window_start = pd.to_datetime(from_date)
    playtest_window_end = pd.to_datetime(to_date)

    df_playtest_filtered = df_playtest[
        (df_playtest["End"] >= playtest_window_start) &
        (df_playtest["Start"] <= playtest_window_end)
    ]

    if df_playtest_filtered.empty:
        st.warning("⚠️ No playtests fall within the selected date range.")
    else:
        st.subheader("Filtered Playtests")

        # -------------------------------
        # Query player_train_join
        # -------------------------------
        event_selector = [{"event": "player_train_join"}]
        df_mixpanel = query_mixpanel_daily_uniques(
            from_date.isoformat(), to_date.isoformat(),
            USERNAME, PASSWORD, PROJECT_ID, event_selector
        )

        timestamp_col = next((col for col in df_mixpanel.columns if 'time' in col.lower()), None)
        if timestamp_col:
            df_mixpanel[timestamp_col] = pd.to_datetime(df_mixpanel[timestamp_col], unit='ms', errors='coerce')
            df_mixpanel = df_mixpanel[df_mixpanel[timestamp_col] >= playtest_window_start]

            if 'distinct_id' not in df_mixpanel.columns:
                df_mixpanel['distinct_id'] = df_mixpanel.get('properties', {}).apply(
                    lambda x: x.get('distinct_id') if isinstance(x, dict) else None
                )

            results = []
            for _, row in df_playtest_filtered.iterrows():
                start, end = row["Start"], row["End"]
                mask = (df_mixpanel[timestamp_col] >= start) & (df_mixpanel[timestamp_col] <= end)
                unique_ids = df_mixpanel.loc[mask, 'distinct_id'].dropna().unique()
                results.append({
                    "Playtest": row["Playtest"],
                    "Start": start,
                    "End": end,
                    "Unique Players": len(unique_ids)
                })

            df_summary = pd.DataFrame(results)
            st.subheader("Playtest Unique Player Summary")
            st.dataframe(df_summary)
        else:
            st.error("⚠️ No timestamp column found in Mixpanel data.")

except Exception as e:
    st.error(f"❌ Failed to process playtest table: {e}")

# -------------------------------
# Class Registration Funnel
# -------------------------------
event_selector = [{"event": "class_registration"}]
df_reg_mixpanel = query_mixpanel_daily_uniques(
    from_date.isoformat(), to_date.isoformat(), USERNAME, PASSWORD, PROJECT_ID, event_selector
)

df_reg_mixpanel["time"] = pd.to_datetime(df_reg_mixpanel["time"], unit="ms", errors="coerce")
df_reg_mixpanel["class"] = df_reg_mixpanel["properties"].apply(
    lambda x: str(x.get("class")) if isinstance(x, dict) else None
)

funnel_results = []

for _, row in df_summary.iterrows():
    start = pd.to_datetime(row["Start"])
    end = pd.to_datetime(row["End"])
    playtest_name = row["Playtest"]

    df_filtered = df_reg_mixpanel[
        (df_reg_mixpanel["time"] >= start) & (df_reg_mixpanel["time"] <= end)
    ]

    class4 = set(df_filtered[df_filtered["class"] == "4"]["distinct_id"])
    class3 = set(df_filtered[df_filtered["class"] == "3"]["distinct_id"])
    class2 = set(df_filtered[df_filtered["class"] == "2"]["distinct_id"])
    class1 = set(df_filtered[df_filtered["class"] == "1"]["distinct_id"])
    class0 = set(df_filtered[df_filtered["class"] == "0"]["distinct_id"])

    funnel_results.append({
        "Playtest": playtest_name,
        "Registered Class 4": len(class4),
        "→ Class 3": len(class4 & class3),
        "→ Class 2": len(class4 & class2),
        "→ Class 1": len(class4 & class1),
        "→ Class 0": len(class4 & class0),
    })

df_funnel = pd.DataFrame(funnel_results)

st.subheader("Class Registration Funnel")
st.dataframe(df_funnel)

# Conversion %
df_funnel["Class 4 %"] = 1.0
df_funnel["Class 3 %"] = df_funnel["→ Class 3"] / df_funnel["Registered Class 4"]
df_funnel["Class 2 %"] = df_funnel["→ Class 2"] / df_funnel["Registered Class 4"]
df_funnel["Class 1 %"] = df_funnel["→ Class 1"] / df_funnel["Registered Class 4"]
df_funnel["Class 0 %"] = df_funnel["→ Class 0"] / df_funnel["Registered Class 4"]

df_funnel_pct = df_funnel[["Playtest","Class 4 %", "Class 3 %", "Class 2 %", "Class 1 %", "Class 0 %"]].melt(
    id_vars="Playtest", var_name="Stage", value_name="Conversion Rate"
)
for playtest in df_funnel_pct["Playtest"].unique():
    df_plot = df_funnel_pct[df_funnel_pct["Playtest"] == playtest]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(df_plot["Stage"], df_plot["Conversion Rate"] * 100)

    ax.set_title(f"Class Registration Funnel — {playtest}")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')

    st.pyplot(fig)

event_selector = [{"event": "player_death"}]
df_death_mixpanel = query_mixpanel_daily_uniques(
    from_date.isoformat(), to_date.isoformat(), USERNAME, PASSWORD, PROJECT_ID, event_selector
)
if "properties" in df_death_mixpanel.columns:
    properties_df = pd.json_normalize(df_death_mixpanel["properties"])
    df_death_mixpanel = pd.concat([df_death_mixpanel.drop(columns=["properties"]), properties_df], axis=1)

# Global filters
cause_options = ['All'] + sorted(df_death_mixpanel['cause'].dropna().unique().tolist())
selected_cause = st.selectbox("Select Cause to Display", cause_options)

server_options = ['All'] + sorted(df_death_mixpanel['server_id'].dropna().unique().tolist())
selected_server = st.selectbox("Select Server to Display", server_options)
def plot_death_scatter_3d(df, playtest_name, carriage_id, start_time, end_time, selected_cause, selected_server):
    df_filtered = df.copy()

    # Ensure datetime
    df_filtered["time"] = pd.to_datetime(df_filtered["time"], unit="ms", errors="coerce")

    # Filter by playtest time window
    df_filtered = df_filtered[
        (df_filtered["time"] >= pd.to_datetime(start_time)) &
        (df_filtered["time"] <= pd.to_datetime(end_time))
    ]

    # Apply cause, carriage, and server filters
    if selected_cause != 'All':
        df_filtered = df_filtered[df_filtered['cause'] == selected_cause]
    if carriage_id != 'All':
        df_filtered = df_filtered[df_filtered['carriage_id'] == carriage_id]
    if selected_server != 'All':
        df_filtered = df_filtered[df_filtered['server_id'] == selected_server]

    if df_filtered.empty:
        st.warning(f"No death data for {playtest_name} (Carriage {carriage_id}) during that window.")
        return

    fig = px.scatter_3d(
        df_filtered,
        x='loc_x', y='loc_y', z='loc_z',
        color='cause',
        size_max=6,
        title=f'Death Locations — {playtest_name} (Carriage {carriage_id})',
        labels={'loc_x': 'X Axis', 'loc_y': 'Y Axis', 'loc_z': 'Z Axis', 'cause': 'Cause'}
    )
    st.plotly_chart(fig)

event_selector = [{"event": "building_placed"}]
df_building_mixpanel = query_mixpanel_daily_uniques(
    from_date.isoformat(), to_date.isoformat(), USERNAME, PASSWORD, PROJECT_ID, event_selector
)
if "properties" in df_building_mixpanel.columns:
    properties_df = pd.json_normalize(df_building_mixpanel["properties"])
    df_building_mixpanel = pd.concat([df_building_mixpanel.drop(columns=["properties"]), properties_df], axis=1)

# Global filters
building_id_options = ['All'] + sorted(df_building_mixpanel['building_id'].dropna().unique().tolist())
selected_build = st.selectbox("Building to Display", building_id_options)

server_options = ['All'] + sorted(df_building_mixpanel['server_id'].dropna().unique().tolist())
selected_server = st.selectbox("Select Server to Display", server_options)

def plot_building_scatter_3d(df, playtest_name, carriage_id, start_time, end_time, selected_build, selected_server):
    df_filtered = df.copy()
    st.dataframe(df_filtered)

    # Ensure datetime
    df_filtered["time"] = pd.to_datetime(df_filtered["time"], unit="ms", errors="coerce")

    # Filter by playtest time window
    df_filtered = df_filtered[
        (df_filtered["time"] >= pd.to_datetime(start_time)) &
        (df_filtered["time"] <= pd.to_datetime(end_time))
    ]

    # Apply cause, carriage, and server filters
    if selected_build != 'All':
        df_filtered = df_filtered[df_filtered['building_id'] == selected_build]
    if carriage_id != 'All':
        df_filtered = df_filtered[df_filtered['carriage_id'] == carriage_id]
    if selected_server != 'All':
        df_filtered = df_filtered[df_filtered['server_id'] == selected_server]

    if df_filtered.empty:
        st.warning(f"No building data for {playtest_name} (Carriage {carriage_id}) during that window.")
        return

    fig = px.scatter_3d(
        df_filtered,
        x='loc_x', y='loc_y', z='loc_z',
        color='building_id',
        size_max=6,
        title=f'Building Locations — {playtest_name} (Carriage {carriage_id})',
        labels={'loc_x': 'X Axis', 'loc_y': 'Y Axis', 'loc_z': 'Z Axis', 'building_id': 'building_id'}
    )
    st.plotly_chart(fig)
# ------------------------------
# Death Event 3D Visualizations
# -------------------------------

for _, row in df_summary.iterrows():
    start_time = pd.to_datetime(row["Start"])
    end_time = pd.to_datetime(row["End"])
    playtest_name = row["Playtest"]
    plot_death_scatter_3d(df_death_mixpanel, playtest_name, "7", start_time, end_time, selected_cause, selected_server)
    plot_death_scatter_3d(df_death_mixpanel, playtest_name, "8", start_time, end_time, selected_cause, selected_server)
    # 3d building
    plot_building_scatter_3d(df_building_mixpanel, playtest_name, "7", start_time, end_time, selected_cause, selected_server)
    plot_building_scatter_3d(df_building_mixpanel, playtest_name, "8", start_time, end_time, selected_cause, selected_server)







