# call_center_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -------------------------
# 1. Dashboard Configuration
# -------------------------

st.set_page_config(
    page_title="Call Center Operations Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# 2. Load and Preprocess Data
# -------------------------

@st.cache_data
def load_data(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Convert 'called_at' and 'sign_up_date' columns to datetime, handling errors
    df['called_at'] = pd.to_datetime(df['called_at'], errors='coerce', utc=True)
    df['sign_up_date'] = pd.to_datetime(df['sign_up_date'], errors='coerce', utc=True)

    # Ensure both columns are in UTC
    df['called_at'] = df['called_at'].dt.tz_convert('UTC')
    df['sign_up_date'] = df['sign_up_date'].dt.tz_convert('UTC')

    # Handle string columns and missing values
    df['direction'] = df['direction'].str.capitalize().fillna('Unknown')
    df['category'] = df['category'].astype(str).fillna('Unknown')
    df['reason'] = df['reason'].astype(str).fillna('Unknown')
    df['agent_id'] = df['agent_id'].astype(str).fillna('Unknown')

    # Add date and time-related columns
    df['call_date'] = df['called_at'].dt.date
    df['call_hour'] = df['called_at'].dt.hour
    df['day_of_week'] = df['called_at'].dt.day_name()
    df['month'] = df['called_at'].dt.to_period('M').astype(str)

    # Compute talk time in minutes
    df['talk_time_minutes'] = df['talk_time'] / 60

    # Add cohort column based on 'sign_up_date'
    df['cohort'] = df['sign_up_date'].dt.to_period('M').astype(str)

    # Calculate time on supply (difference in whole days)
    df['time_on_supply'] = (df['called_at'] - df['sign_up_date']).dt.days


    return df


# -------------------------
# 3. Load Data
# -------------------------
# data/one_big_table.csv
csv_file_path = 'one_big_table.csv'  # Replace with your actual file path
df = load_data(csv_file_path)

# -------------------------
# 4. Sidebar Configuration
# -------------------------

st.sidebar.title("Menu")

# Date Range Selector
min_date = df['call_date'].min()
max_date = df['call_date'].max()

selected_start_date = st.sidebar.date_input(
    "Start Date",
    min_value=min_date,
    max_value=max_date,
    value=min_date
)

selected_end_date = st.sidebar.date_input(
    "End Date",
    min_value=min_date,
    max_value=max_date,
    value=max_date
)

if selected_start_date > selected_end_date:
    st.sidebar.error("Error: End date must fall after start date.")

# Filter data based on selected date range
filtered_df = df[
    (df['call_date'] >= selected_start_date) &
    (df['call_date'] <= selected_end_date)
]
# ... existing code ...

# Category Selector (Custom Dropdown)
categories = sorted(filtered_df['category'].unique())
categories_options = ['All Categories'] + categories
selected_category = st.sidebar.selectbox(
    'Select Category',
    options=categories_options,
    key='category_dropdown'
)

if selected_category == 'All Categories':
    selected_categories = categories
else:
    selected_categories = [selected_category]

# Reason Selector (Custom Dropdown)
reasons = sorted(filtered_df[filtered_df['category'].isin(selected_categories)]['reason'].unique())
reasons_options = ['All Reasons'] + reasons
selected_reason = st.sidebar.selectbox(
    'Select Reason',
    options=reasons_options,
    key='reason_dropdown'
)

if selected_reason == 'All Reasons':
    selected_reasons = reasons
else:
    selected_reasons = [selected_reason]

# Further filter data based on selected categories and reasons
filtered_df = filtered_df[
    (filtered_df['category'].isin(selected_categories)) &
    (filtered_df['reason'].isin(selected_reasons))
]

# Agent Selector (Custom Dropdown)
agents = sorted(filtered_df['agent_id'].unique())
agents_options = ['All Agents'] + agents
selected_agent = st.sidebar.selectbox(
    'Select Agent for Performance Metrics',
    options=agents_options,
    key='agent_dropdown'
)

if selected_agent == 'All Agents':
    selected_agents_perf = agents
else:
    selected_agents_perf = [selected_agent]

# Filter agent performance data
agent_perf_df = filtered_df[filtered_df['agent_id'].isin(selected_agents_perf)]



# -------------------------
# 5. KPI Calculations
# -------------------------

# Header (General Metrics Overview)
total_calls = filtered_df['call_id'].nunique()
average_talk_time = filtered_df['talk_time_minutes'].mean()
unique_agents = filtered_df['agent_id'].nunique()
avg_calls_per_agent = total_calls / unique_agents if unique_agents else 0

# Section 1: Call Metrics for Selected Period
call_distribution = filtered_df.groupby('call_date')['call_id'].count().reset_index()
call_distribution.rename(columns={'call_id': 'total_calls'}, inplace=True)

avg_talk_time_selected = average_talk_time
unique_agents_selected = unique_agents

hourly_metrics = filtered_df.groupby('call_hour').agg(
    total_calls=('call_id', 'count'),
    avg_talk_time_minutes=('talk_time_minutes', 'mean')
).reset_index()

daily_metrics = filtered_df.groupby('day_of_week').agg(
    total_calls=('call_id', 'count'),
    avg_talk_time_minutes=('talk_time_minutes', 'mean')
).reset_index()

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_metrics['day_of_week'] = pd.Categorical(daily_metrics['day_of_week'], categories=days_order, ordered=True)
daily_metrics = daily_metrics.sort_values('day_of_week')

peak_hours = hourly_metrics.sort_values(by='total_calls', ascending=False).head(5)
peak_days = daily_metrics.sort_values(by='total_calls', ascending=False).head(5)

# Section 2: Category Performance Metrics
calls_by_category = filtered_df.groupby('category')['call_id'].count().reset_index()
calls_by_category.rename(columns={'call_id': 'total_calls'}, inplace=True)

avg_talk_time_category = filtered_df.groupby('category')['talk_time_minutes'].mean().reset_index()
avg_talk_time_category.rename(columns={'talk_time_minutes': 'avg_talk_time_minutes'}, inplace=True)

unique_agents_category = filtered_df.groupby('category')['agent_id'].nunique().reset_index()
unique_agents_category.rename(columns={'agent_id': 'unique_agents'}, inplace=True)

# Section 3: Reason Performance Breakdown
calls_by_reason = filtered_df.groupby('reason')['call_id'].count().reset_index()
calls_by_reason.rename(columns={'call_id': 'total_calls'}, inplace=True)

avg_talk_time_reason = filtered_df.groupby('reason')['talk_time_minutes'].mean().reset_index()
avg_talk_time_reason.rename(columns={'talk_time_minutes': 'avg_talk_time_minutes'}, inplace=True)

unique_agents_reason = filtered_df.groupby('reason')['agent_id'].nunique().reset_index()
unique_agents_reason.rename(columns={'agent_id': 'unique_agents'}, inplace=True)

# Section 4: Sales Channels Performance Metrics
calls_by_channel = filtered_df.groupby('sales_channel')['call_id'].count().reset_index()
calls_by_channel.rename(columns={'call_id': 'total_calls'}, inplace=True)

avg_talk_time_channel = filtered_df.groupby('sales_channel')['talk_time_minutes'].mean().reset_index()
avg_talk_time_channel.rename(columns={'talk_time_minutes': 'avg_talk_time_minutes'}, inplace=True)

unique_agents_channel = filtered_df.groupby('sales_channel')['agent_id'].nunique().reset_index()
unique_agents_channel.rename(columns={'agent_id': 'unique_agents'}, inplace=True)

# Section 5: Call Direction Analysis
calls_by_direction = filtered_df.groupby('direction')['call_id'].count().reset_index()
calls_by_direction.rename(columns={'call_id': 'total_calls'}, inplace=True)

avg_talk_time_direction = filtered_df.groupby('direction')['talk_time_minutes'].mean().reset_index()
avg_talk_time_direction.rename(columns={'talk_time_minutes': 'avg_talk_time_minutes'}, inplace=True)

unique_agents_direction = filtered_df.groupby('direction')['agent_id'].nunique().reset_index()
unique_agents_direction.rename(columns={'agent_id': 'unique_agents'}, inplace=True)

categories_summary = filtered_df.groupby(['direction', 'category'])['call_id'].count().reset_index()
categories_summary.rename(columns={'call_id': 'total_calls'}, inplace=True)

# Agent Performance Metrics
calls_by_agent_perf = agent_perf_df.groupby('agent_id')['call_id'].count().reset_index()
calls_by_agent_perf.rename(columns={'call_id': 'calls_handled'}, inplace=True)

avg_talk_time_agent_perf = agent_perf_df.groupby('agent_id')['talk_time_minutes'].mean().reset_index()
avg_talk_time_agent_perf.rename(columns={'talk_time_minutes': 'avg_talk_time_minutes'}, inplace=True)

calls_by_category_agent = agent_perf_df.groupby(['agent_id', 'category'])['call_id'].count().reset_index()
calls_by_category_agent.rename(columns={'call_id': 'total_calls'}, inplace=True)

# Calculate agent seniority
agent_first_call = df.groupby('agent_id')['called_at'].min().reset_index()
agent_first_call.columns = ['agent_id', 'first_call_date']
agent_last_call = df.groupby('agent_id')['called_at'].max().reset_index()
agent_last_call.columns = ['agent_id', 'last_call_date']
agent_seniority = pd.merge(agent_first_call, agent_last_call, on='agent_id')
agent_seniority['seniority_days'] = (agent_seniority['last_call_date'] - agent_seniority['first_call_date']).dt.days

# Combine all agent metrics
agent_performance = pd.merge(calls_by_agent_perf, avg_talk_time_agent_perf, on='agent_id')
agent_performance = pd.merge(agent_performance, agent_seniority[['agent_id', 'seniority_days']], on='agent_id')

#Client Cohort Analysis (using the full dataset)
cohort_df = df.copy()  # Use the full dataset for cohort analysis
# i think i need to cast
total_calls_per_cohort = cohort_df.groupby('cohort')['call_id'].nunique().reset_index().rename(columns={'call_id': 'total_calls'})
total_calls_per_cohort = total_calls_per_cohort.sort_values('cohort')

avg_talk_time_per_cohort = cohort_df.groupby('cohort')['talk_time_minutes'].mean().reset_index().rename(columns={'talk_time_minutes': 'avg_talk_time'})
avg_talk_time_per_cohort = avg_talk_time_per_cohort.sort_values('cohort')

time_on_supply_per_cohort = cohort_df.groupby('cohort')['time_on_supply'].mean().reset_index()
time_on_supply_per_cohort = time_on_supply_per_cohort.sort_values('cohort')

cohort_analysis = pd.merge(total_calls_per_cohort, avg_talk_time_per_cohort, on='cohort')
cohort_analysis = pd.merge(cohort_analysis, time_on_supply_per_cohort, on='cohort')



# -------------------------
# 6. Dashboard Layout
# -------------------------

# Title
st.title("🐙 Octopus Energy: Call Dashboard")

# Reporting Period Indicator
st.markdown(f"**Reporting Period:** {selected_start_date.strftime('%B %d, %Y')} to {selected_end_date.strftime('%B %d, %Y')}")

st.markdown("---")

# Header (General Metrics Overview)
st.subheader("Metrics Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📞 Total Number of Calls", f"{total_calls}")

with col2:
    st.metric("🗣️ Average Talk Time (Minutes)", f"{average_talk_time:.2f}")

with col3:
    st.metric("👩‍🚀 Number of Unique Agents", f"{unique_agents}")

with col4:
    st.metric("🔢 Average Number of Calls per Agent", f"{avg_calls_per_agent:.2f}")

st.markdown("---")

# Section 1: Call Metrics for Selected Period
st.subheader("📊 Section 1: Call Metrics")

# Call Distribution Over Time
st.plotly_chart(
    px.bar(
        call_distribution,
        x='call_date',
        y='total_calls',
        title='Call Distribution Over Time',
        labels={'call_date': 'Date', 'total_calls': 'Total Calls'},
        text_auto=True,
        color='total_calls',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Number of Calls per Day of the Week
st.plotly_chart(
    px.bar(
        daily_metrics.sort_values('day_of_week'),
        x='day_of_week',
        y='total_calls',
        title='Number of Calls per Day of the Week',
        labels={'day_of_week': 'Day of Week', 'total_calls': 'Total Calls'},
        text_auto=True,
        color='total_calls',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Average Talk Time per Day of the Week
st.plotly_chart(
    px.bar(
        daily_metrics.sort_values('day_of_week'),
        x='day_of_week',
        y='avg_talk_time_minutes',
        title='Average Talk Time per Day of the Week (Minutes)',
        labels={'avg_talk_time_minutes': 'Avg Talk Time (Minutes)', 'day_of_week': 'Day of Week'},
        text_auto='.2f',
        color='avg_talk_time_minutes',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Peak Days: Highest Call Volume and Longest Talk Time


st.plotly_chart(
    px.scatter(
        peak_days,
        x='day_of_week',
        y='total_calls',
        size='avg_talk_time_minutes',
        color='avg_talk_time_minutes',
        title='Peak Days with Highest Call Volume and Longest Talk Time',
        labels={'day_of_week': 'Day of Week', 'total_calls': 'Total Calls', 'avg_talk_time_minutes': 'Avg Talk Time (Minutes)'},
        hover_data=['avg_talk_time_minutes'],
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)
# Number of Calls per Hour
st.plotly_chart(
    px.bar(
        hourly_metrics.sort_values(by='call_hour'),
        x='call_hour',
        y='total_calls',
        title='Number of Calls per Hour',
        labels={'call_hour': 'Call Hour (24h)', 'total_calls': 'Total Calls'},
        text_auto=True,
        color='total_calls',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Average Talk Time per Hour (Line Chart)
st.plotly_chart(
    px.bar(
        hourly_metrics.sort_values(by='call_hour'),
        x='call_hour',
        y='avg_talk_time_minutes',
        title='Average Talk Time per Hour (Minutes)',
        labels={'call_hour': 'Call Hour (24h)', 'avg_talk_time_minutes': 'Avg Talk Time (Minutes)'},
        text_auto='.2f',
        color='avg_talk_time_minutes',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Peak Call Hours with Longest Talk Time
st.plotly_chart(
    px.scatter(
        peak_hours,
        x='call_hour',
        y='total_calls',
        size='avg_talk_time_minutes',
        color='avg_talk_time_minutes',
        title='Peak Call Hours with Longest Talk Time',
        labels={'call_hour': 'Call Hour (24h)', 'total_calls': 'Total Calls', 'avg_talk_time_minutes': 'Avg Talk Time (Minutes)'},
        hover_data=['avg_talk_time_minutes'],
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)


st.markdown("---")

# Section 2: Category Performance Metrics
st.subheader("🏷️ Section 2: Category Performance Metrics")
# Calls by Category
st.plotly_chart(
    px.bar(
        calls_by_category.sort_values(by='total_calls', ascending=False),
        x='category',
        y='total_calls',
        title='Calls by Category',
        labels={'category': 'Category', 'total_calls': 'Total Calls'},
        text_auto=True,
        color='total_calls',
        color_continuous_scale='bupu' #'Viridis'
    ),
    use_container_width=True
)

# Average Talk Time by Category
st.plotly_chart(
    px.bar(
        avg_talk_time_category.sort_values(by='avg_talk_time_minutes', ascending=False),
        x='category',
        y='avg_talk_time_minutes',
        title='Average Talk Time by Category (Minutes)',
        labels={'avg_talk_time_minutes': 'Avg Talk Time (Minutes)', 'category': 'Category'},
        text_auto='.2f',
        color='avg_talk_time_minutes',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Unique Agents by Category
st.plotly_chart(
    px.bar(
        unique_agents_category.sort_values(by='unique_agents', ascending=False),
        x='category',
        y='unique_agents',
        title='Unique Agents by Category',
        labels={'unique_agents': 'Unique Agents', 'category': 'Category'},
        text_auto=True,
        color='unique_agents',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

st.markdown("---")

# Section 3: Reason Performance Breakdown
st.subheader("🔍 Section 3: Reason Performance Breakdown")

# Calls by Reason
st.plotly_chart(
    px.bar(
        calls_by_reason.sort_values(by='total_calls', ascending=False),
        x='reason',
        y='total_calls',
        title='Calls by Reason',
        labels={'reason': 'Reason', 'total_calls': 'Total Calls'},
        text_auto=True,
        color='total_calls',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Average Talk Time by Reason
st.plotly_chart(
    px.bar(
        avg_talk_time_reason.sort_values(by='avg_talk_time_minutes', ascending=False),
        x='reason',
        y='avg_talk_time_minutes',
        title='Average Talk Time by Reason (Minutes)',
        labels={'avg_talk_time_minutes': 'Avg Talk Time (Minutes)', 'reason': 'Reason'},
        text_auto='.2f',
        color='avg_talk_time_minutes',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Unique Agents by Reason
st.plotly_chart(
    px.bar(
        unique_agents_reason.sort_values(by='unique_agents', ascending=False),
        x='reason',
        y='unique_agents',
        title='Unique Agents by Reason',
        labels={'unique_agents': 'Unique Agents', 'reason': 'Reason'},
        text_auto=True,
        color='unique_agents',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

st.markdown("---")

# Section 4: Sales Channels Performance Metrics
st.subheader("💼 Section 4: Sales Channels Performance Metrics")
# Calls by Channel
st.plotly_chart(
    px.bar(
        calls_by_channel.sort_values(by='total_calls', ascending=False),
        x='sales_channel',
        y='total_calls',
        title='Calls by Sales Channel',
        labels={'sales_channel': 'Sales Channel', 'total_calls': 'Total Calls'},
        text_auto=True,
        color='total_calls',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Average Talk Time by Channel
st.plotly_chart(
    px.bar(
        avg_talk_time_channel.sort_values(by='avg_talk_time_minutes', ascending=False),
        x='sales_channel',
        y='avg_talk_time_minutes',
        title='Average Talk Time by Sales Channel (Minutes)',
        labels={'avg_talk_time_minutes': 'Avg Talk Time (Minutes)', 'sales_channel': 'Sales Channel'},
        text_auto='.2f',
        color='avg_talk_time_minutes',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Unique Agents by Channel
st.plotly_chart(
    px.bar(
        unique_agents_channel.sort_values(by='unique_agents', ascending=False),
        x='sales_channel',
        y='unique_agents',
        title='Unique Agents by Sales Channel',
        labels={'unique_agents': 'Unique Agents', 'sales_channel': 'Sales Channel'},
        text_auto=True,
        color='unique_agents',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

st.markdown("---")

# Section 5: Client Cohort Analysis
st.subheader("👥 Section 5: Client Cohort Analysis")
# Create a figure with secondary y-axis
fig_combined = make_subplots(specs=[[{"secondary_y": True}]])

# Add bar trace for total calls
fig_combined.add_trace(
    go.Bar(
        x=cohort_analysis['cohort'],
        y=cohort_analysis['total_calls'],
        name='Total Calls',
        marker_color='indigo'
    ),
    secondary_y=False,
)

# Add line trace for average talk time
fig_combined.add_trace(
    go.Scatter(
        x=cohort_analysis['cohort'],
        y=cohort_analysis['avg_talk_time'],
        name='Average Talk Time',
        mode='lines+markers',
        line=dict(color='orange')
    ),
    secondary_y=True,
)

# Add titles and labels
fig_combined.update_layout(
    title_text="Cohort Analysis: Total Calls and Average Talk Time",
    xaxis_title="Cohort (Year-Month)",
    legend=dict(x=0.01, y=0.99),
)

# Set y-axes titles
fig_combined.update_yaxes(title_text="Total Calls", secondary_y=False)
fig_combined.update_yaxes(title_text="Average Talk Time (Minutes)", secondary_y=True)

st.plotly_chart(fig_combined, use_container_width=True)

st.plotly_chart(
    px.bar(
        time_on_supply_per_cohort,
        x='cohort',
        y='time_on_supply',
        title='Average Time on Supply by Cohort (Days)',
        labels={'time_on_supply': 'Avg Time on Supply (Days)', 'cohort': 'Cohort (Year-Month)'},
        text_auto='.2f',
        color='time_on_supply',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Section 6: Call Direction Analysis
st.subheader("↔️ Section 6: Call Direction Analysis")

# Total Calls by Direction
st.plotly_chart(
    px.bar(
        calls_by_direction.sort_values(by='total_calls', ascending=False),
        x='direction',
        y='total_calls',
        title='Total Calls by Direction',
        labels={'direction': 'Direction', 'total_calls': 'Total Calls'},
        text_auto=True,
        color='total_calls',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Average Talk Time by Direction
st.plotly_chart(
    px.bar(
        avg_talk_time_direction.sort_values(by='avg_talk_time_minutes', ascending=False),
        x='direction',
        y='avg_talk_time_minutes',
        title='Average Talk Time by Direction (Minutes)',
        labels={'avg_talk_time_minutes': 'Avg Talk Time (Minutes)', 'direction': 'Direction'},
        text_auto='.2f',
        color='avg_talk_time_minutes',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Unique Agents by Direction
st.plotly_chart(
    px.bar(
        unique_agents_direction.sort_values(by='unique_agents', ascending=False),
        x='direction',
        y='unique_agents',
        title='Unique Agents by Direction',
        labels={'unique_agents': 'Unique Agents', 'direction': 'Direction'},
        text_auto=True,
        color='unique_agents',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Categories Summary by Call Direction
st.plotly_chart(
    px.bar(
        categories_summary,
        x='direction',
        y='total_calls',
        color='category',
        title='Categories Summary by Call Direction',
        labels={'direction': 'Direction', 'total_calls': 'Total Calls', 'category': 'Category'},
        text_auto=True,
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Vivid
    ),
    use_container_width=True
)

st.markdown("---")

# Section 7: Agent Performance Metrics
st.subheader("🏆 Section 7: Agent Performance Metrics")

# Agent Seniority Metric
if selected_agent != 'All Agents':
    agent_seniority = agent_performance[agent_performance['agent_id'] == selected_agent]['seniority_days'].values[0]
    seniority_label = f"{selected_agent} Seniority (Days)"
else:
    agent_seniority = agent_performance['seniority_days'].mean()
    seniority_label = "Avg Agent Seniority (Days)"

# Create columns to center the metric
left_spacer, center_col, right_spacer = st.columns([1, 2, 1])

# Display the metric in the center column
with center_col:
    st.metric("👨‍💼 " + seniority_label, f"{agent_seniority:.0f}")

# Add some vertical space after the metric
st.markdown("<br>", unsafe_allow_html=True)

# Calls Handled by Agent
st.plotly_chart(
    px.bar(
        calls_by_agent_perf.sort_values(by='calls_handled', ascending=False),
        x='agent_id',
        y='calls_handled',
        title='Calls Handled by Agent',
        labels={'agent_id': 'Agent ID', 'calls_handled': 'Calls Handled'},
        text_auto=True,
        color='calls_handled',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Average Talk Time by Agent
st.plotly_chart(
    px.bar(
        avg_talk_time_agent_perf.sort_values(by='avg_talk_time_minutes', ascending=False),
        x='agent_id',
        y='avg_talk_time_minutes',
        title='Average Talk Time by Agent (Minutes)',
        labels={'avg_talk_time_minutes': 'Avg Talk Time (Minutes)', 'agent_id': 'Agent ID'},
        text_auto='.2f',
        color='avg_talk_time_minutes',
        color_continuous_scale='bupu'
    ),
    use_container_width=True
)

# Distribution of Calls per Category by Agent
st.plotly_chart(
    px.bar(
        calls_by_category_agent,
        x='category',
        y='total_calls',
        color='category',
        facet_col='agent_id',
        title='Distribution of Calls per Category by Agent',
        labels={'category': 'Category', 'total_calls': 'Total Calls'},
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Vivid
    ),
    use_container_width=True
)



st.markdown("---")


# -------------------------
# 7. Footer
# -------------------------

st.markdown("""
---
**Data Source:** `one_big_table.csv`
**Last Updated:** {0}
**Contact:** chinta.vincent@gmail.com
""".format(datetime.now().strftime("%B %d, %Y")))
