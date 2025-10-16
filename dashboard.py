import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import openpyxl

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Comprehensive Sales Dashboard")

# --- Caching Functions for Performance ---
@st.cache_data
def load_data(uploaded_file):
    """Loads and does initial cleaning of the uploaded Excel file."""
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def preprocess_data(df, date_col, quantity_col, customer_col):
    """Core data preprocessing, run once after column mapping."""
    df_processed = df.copy()
    try:
        df_processed[date_col] = pd.to_datetime(df_processed[date_col])
        df_processed[quantity_col] = pd.to_numeric(df_processed[quantity_col], errors='coerce')
        df_processed.dropna(subset=[quantity_col, customer_col, date_col], inplace=True)
        return df_processed
    except KeyError as e:
        st.error(f"A specified column is not found: {e}. Please check column mappings.")
        return None
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return None

# --- Analytical Functions ---
# ### NEW AND IMPROVED RFM FUNCTION ###
@st.cache_data
def calculate_rfm(df, date_col, customer_col, quantity_col, order_col):
    """
    Calculates RFM metrics and segments customers using a robust and clear method.
    """
    # 1. Calculate R, F, M values
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm_df = df.groupby(customer_col).agg({
        date_col: lambda date: (snapshot_date - date.max()).days,
        order_col: 'nunique',
        quantity_col: 'sum'
    }).rename(columns={date_col: 'Recency', order_col: 'Frequency', quantity_col: 'Monetary'})

    # 2. Create Scores - Handles cases with non-unique bin edges
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop').astype(int)
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)

    # 3. Combine scores for segmentation
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

    # 4. Define segments based on clear, combined score logic
    # This mapping is ordered from highest priority (Champions) to lowest.
    seg_map = {
        r'[3-4][3-4][3-4]': 'Champions',
        r'[3-4]4[1-2]': 'Champions (Low Spenders)',
        r'4[1-3][1-4]': 'New Customers',
        r'3[1-3][1-4]': 'Recent Customers',
        r'[1-2][3-4][3-4]': 'At Risk (High Value)',
        r'[1-2]2[1-4]': 'At Risk (Slipping)',
        r'[1-2]1[1-4]': 'Hibernating',
        r'2[3-4][1-4]': 'Potential Loyalists',
        r'[3-4][1-2][1-4]': 'Promising'
    }

    # Default to 'Regular' if no specific segment is matched
    rfm_df['Segment'] = 'Regular'
    # Use regex to map the combined RFM score string to a segment
    for pattern, segment_name in seg_map.items():
        rfm_df.loc[rfm_df['RFM_Score'].str.match(pattern), 'Segment'] = segment_name
        
    return rfm_df


# --- Main App ---
st.title("🚀 Comprehensive Sales Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is None:
        st.info("Please upload an Excel file to begin.")
        st.stop()
    df_raw = load_data(uploaded_file)
    if df_raw is None:
        st.stop()

    st.header("2. Map Your Columns")
    all_columns = df_raw.columns.tolist()
    def find_default(name, options):
        return options.index(name) if name in options else 0
    date_col = st.selectbox("Date Column", all_columns, index=find_default('Ship date', all_columns))
    customer_col = st.selectbox("Customer Name Column", all_columns, index=find_default('Customer name', all_columns))
    quantity_col = st.selectbox("Quantity Column", all_columns, index=find_default('Quantity', all_columns))
    order_col = st.selectbox("Sales Order Column", all_columns, index=find_default('Sales order', all_columns))
    product_col = st.selectbox("Product Name Column", all_columns, index=find_default('Product name', all_columns))
    dom_exp_col = st.selectbox("Domestic/Export Column", all_columns, index=find_default('Domestic/Export', all_columns))
    country_col = st.selectbox("Country Column", all_columns, index=find_default('Country', all_columns))

    df = preprocess_data(df_raw, date_col, quantity_col, customer_col)
    if df is None:
        st.stop()

    st.header("3. Master Filters")
    start_date, end_date = st.date_input("Select Timeframe",
        value=(df[date_col].min().date(), df[date_col].max().date()),
        min_value=df[date_col].min().date(), max_value=df[date_col].max().date()
    )
    selected_customers = st.multiselect("Filter by Customer", options=sorted(df[customer_col].unique()), default=[])
    selected_products = st.multiselect("Filter by Product", options=sorted(df[product_col].unique()), default=[])
    selected_dom_exp = st.multiselect("Filter by Domestic/Export", options=sorted(df[dom_exp_col].unique()), default=[])
# New, corrected line
country_options = sorted([str(country) for country in df[country_col].unique()])
selected_countries = st.multiselect("Filter by Country", options=country_options, default=[])
# --- Filtering Logic ---
mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
if selected_customers: mask &= (df[customer_col].isin(selected_customers))
if selected_products: mask &= (df[product_col].isin(selected_products))
if selected_dom_exp: mask &= (df[dom_exp_col].isin(selected_dom_exp))
if selected_countries: mask &= (df[country_col].isin(selected_countries))
filtered_df = df[mask]
if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- Main Dashboard with Tabs ---
tab_list = ["📊 Overview", "👤 Customer Deep Dive", "📦 Product Analysis", "🔄 Timeframe Comparison", "📈 Sales Forecast"]
overview_tab, customer_tab, product_tab, comparison_tab, forecast_tab = st.tabs(tab_list)

with overview_tab:
    # ... (code for this tab is unchanged) ...
    st.header("Dashboard Overview")
    total_volume = filtered_df[quantity_col].sum()
    unique_customers = filtered_df[customer_col].nunique()
    new_customers_mask = df.groupby(customer_col)[date_col].min().between(pd.to_datetime(start_date), pd.to_datetime(end_date))
    new_customer_count = new_customers_mask.sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales Volume", f"{total_volume:,.0f}")
    col2.metric("Unique Customers", f"{unique_customers:,}")
    col3.metric("New Customers in Period", f"{new_customer_count:,}")
    st.markdown("---")
    st.header("Sales Trend")
    monthly_sales = filtered_df.set_index(date_col)[quantity_col].resample('M').sum()
    fig = px.line(monthly_sales, title="Monthly Sales Volume")
    st.plotly_chart(fig, use_container_width=True)


with customer_tab:
    st.header("Customer Deep Dive Analysis")
    # ... (code for High-Volume and Individual Customer is unchanged) ...
    st.markdown("### High-Volume Customer Analysis")
    if not filtered_df.empty:
        customer_volumes = filtered_df.groupby(customer_col)[quantity_col].sum()
        if not customer_volumes.empty:
            max_volume = int(customer_volumes.max())
            volume_threshold = st.slider("Minimum Total Quantity per Customer:", min_value=0, max_value=max_volume, value=int(max_volume/4))
            high_volume_customers = customer_volumes[customer_volumes > volume_threshold]
            st.metric(f"Customers with > {volume_threshold:,} units", len(high_volume_customers))
            with st.expander("View High-Volume Customer List"):
                st.dataframe(high_volume_customers.sort_values(ascending=False))
    
    st.markdown("---")
    st.markdown("### Individual Customer Consumption")
    if not filtered_df[customer_col].empty:
        selected_customer = st.selectbox("Select a customer to analyze:", options=sorted(filtered_df[customer_col].unique()))
        if selected_customer:
            # ... (code for individual plot unchanged)
            customer_data = filtered_df[filtered_df[customer_col] == selected_customer]
            customer_monthly_consumption = customer_data.set_index(date_col)[quantity_col].resample('M').sum()
            avg_consumption = customer_monthly_consumption.mean()
            st.metric(f"Avg. Monthly Consumption for {selected_customer}", f"{avg_consumption:,.2f} units")
            fig = px.bar(customer_monthly_consumption, title=f"Monthly Purchases for {selected_customer}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### RFM Segmentation")
    if not filtered_df.empty:
        rfm_results = calculate_rfm(filtered_df, date_col, customer_col, quantity_col, order_col)
        fig = px.bar(rfm_results['Segment'].value_counts(), title="Customer Count by RFM Segment")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- NEW: Add expander for detailed RFM data ---
        with st.expander("View Detailed RFM Data and Scores"):
            st.dataframe(rfm_results[['Segment', 'Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score']].sort_values(by='RFM_Score', ascending=False))


# ... (The rest of the script for other tabs is unchanged) ...
with product_tab:
    st.header("Product Performance (Pareto 80/20 Analysis)")
    if not filtered_df.empty:
        product_sales = filtered_df.groupby(product_col)[quantity_col].sum().sort_values(ascending=False)
        product_sales_df = product_sales.reset_index()
        product_sales_df['Cumulative_Percentage'] = (product_sales_df[quantity_col].cumsum() / product_sales_df[quantity_col].sum()) * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=product_sales_df[product_col], y=product_sales_df[quantity_col], name='Volume'), secondary_y=False)
        fig.add_trace(go.Scatter(x=product_sales_df[product_col], y=product_sales_df['Cumulative_Percentage'], name='Cumulative %'), secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

with comparison_tab:
    st.header("Timeframe Comparison")
    col1, col2 = st.columns(2)
    with col1:
        p1_start, p1_end = st.date_input("Period 1", value=(df[date_col].min().date(), df[date_col].min().date() + pd.Timedelta(days=365)), key="p1")
    with col2:
        p2_start, p2_end = st.date_input("Period 2", value=(df[date_col].max().date() - pd.Timedelta(days=365), df[date_col].max().date()), key="p2")

    period1_df = df[(df[date_col].dt.date >= p1_start) & (df[date_col].dt.date <= p1_end)]
    period2_df = df[(df[date_col].dt.date >= p2_start) & (df[date_col].dt.date <= p2_end)]
    p1_sales = period1_df[quantity_col].sum()
    p2_sales = period2_df[quantity_col].sum()
    p1_customers = period1_df[customer_col].nunique()
    p2_customers = period2_df[customer_col].nunique()

    st.subheader("Comparison Results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Period 1 Sales", f"{p1_sales:,.0f}")
        st.metric("Period 1 Customers", f"{p1_customers:,}")
    with c2:
        st.metric("Period 2 Sales", f"{p2_sales:,.0f}", delta=f"{p2_sales - p1_sales:,.0f}")
        st.metric("Period 2 Customers", f"{p2_customers:,}", delta=f"{p2_customers - p1_customers:,}")

with forecast_tab:
    st.header("Sales Forecasting")
    if st.button("Generate 90-Day Forecast"):
        with st.spinner("Training model..."):
            forecast_df = df.set_index(date_col)[quantity_col].resample('D').sum().reset_index().rename(columns={date_col: 'ds', quantity_col: 'y'})
            model = Prophet()
            model.fit(forecast_df)
            future = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)
            fig = model.plot(forecast)
            st.pyplot(fig)
