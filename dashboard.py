import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import openpyxl
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

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
@st.cache_data
def calculate_rfm(df, date_col, customer_col, quantity_col, order_col):
    """
    Calculates RFM metrics and segments customers. Handles cases with few customers.
    """
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm_df = df.groupby(customer_col).agg({
        date_col: lambda date: (snapshot_date - date.max()).days,
        order_col: 'nunique',
        quantity_col: 'sum'
    }).rename(columns={date_col: 'Recency', order_col: 'Frequency', quantity_col: 'Monetary'})

    if rfm_df.shape[0] < 4:
        rfm_df['R_Score'] = 4
        rfm_df['F_Score'] = 4
        rfm_df['M_Score'] = 4
        rfm_df['RFM_Score'] = '444'
        rfm_df['Segment'] = 'Individual Analysis'
        return rfm_df

    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop').astype(int)
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

    def segment_customer(row):
        if row['R_Score'] == 4 and row['F_Score'] == 4 and row['M_Score'] == 4: return 'Champions'
        if row['R_Score'] >= 3 and row['F_Score'] >= 3: return 'Loyal Customers'
        if row['R_Score'] >= 3 and row['Frequency'] > 1: return 'Potential Loyalists'
        if row['R_Score'] == 4 and row['F_Score'] == 1: return 'New Customers'
        if row['R_Score'] == 3 and row['F_Score'] == 1: return 'Promising'
        if row['R_Score'] <= 2 and row['F_Score'] >= 3: return 'At Risk'
        if row['R_Score'] <= 2 and row['F_Score'] <= 2: return 'Hibernating'
        return 'Regular'

    rfm_df['Segment'] = rfm_df.apply(segment_customer, axis=1)
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
    country_col = st.selectbox("Country Column", all_columns, index=find_default('Country', all_columns))
    quantity_col = st.selectbox("Quantity Column", all_columns, index=find_default('Quantity', all_columns))
    order_col = st.selectbox("Sales Order Column", all_columns, index=find_default('Sales order', all_columns))
    product_col = st.selectbox("Product Name Column", all_columns, index=find_default('Product name', all_columns))
    dom_exp_col = st.selectbox("Domestic/Export Column", all_columns, index=find_default('Domestic/Export', all_columns))
    df = preprocess_data(df_raw, date_col, quantity_col, customer_col)
    if df is None:
        st.stop()
    st.header("3. Master Filters")
    start_date, end_date = st.date_input("Select Timeframe",
        value=(df[date_col].min().date(), df[date_col].max().date()),
        min_value=df[date_col].min().date(), max_value=df[date_col].max().date()
    )
    selected_customers = st.multiselect("Filter by Customer", options=sorted([str(c) for c in df[customer_col].unique()]), default=[])
    selected_products = st.multiselect("Filter by Product", options=sorted([str(p) for p in df[product_col].unique()]), default=[])
    selected_dom_exp = st.multiselect("Filter by Domestic/Export", options=sorted([str(de) for de in df[dom_exp_col].unique()]), default=[])
    selected_countries = st.multiselect("Filter by Country", options=sorted([str(c) for c in df[country_col].unique()]), default=[])

# --- Filtering Logic ---
mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
if selected_customers: mask &= df[customer_col].isin(selected_customers)
if selected_products: mask &= df[product_col].isin(selected_products)
if selected_dom_exp: mask &= df[dom_exp_col].isin(selected_dom_exp)
if selected_countries: mask &= df[country_col].isin(selected_countries)
filtered_df = df[mask]
if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- Main Dashboard with Tabs ---
tab_list = ["📊 Overview", "👤 Customer Deep Dive", "📦 Product Analysis", "🌍 Geospatial View", "🔄 Timeframe Comparison", "📈 Sales Forecast", "💎 CLV Prediction"] # <-- NEW TAB
overview_tab, customer_tab, product_tab, geo_tab, comparison_tab, forecast_tab, clv_tab = st.tabs(tab_list) # <-- NEW TAB

with overview_tab:
    # ... (code is unchanged) ...
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
    if len(selected_customers) == 1 or len(selected_countries) == 1 or len(selected_products) > 0:
        st.markdown("---")
        st.header("Filtered Order History")
        if len(selected_customers) == 1: st.subheader(f"Showing all orders for customer: {selected_customers[0]}")
        elif len(selected_countries) == 1: st.subheader(f"Showing all orders for country: {selected_countries[0]}")
        elif len(selected_products) > 0: st.subheader(f"Showing all orders for product(s): {', '.join(selected_products)}")
        history_columns = [date_col, customer_col, country_col, product_col, quantity_col, order_col]
        display_cols = [col for col in history_columns if col in filtered_df.columns]
        history_df = filtered_df[display_cols].sort_values(by=date_col, ascending=False)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

with customer_tab:
    # ... (code is unchanged) ...
    st.header("Customer Deep Dive Analysis")
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
        customer_options = sorted(filtered_df[customer_col].unique())
        selected_customer = st.selectbox("Select a customer to analyze:", options=customer_options)
        if selected_customer:
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
        with st.expander("View Detailed RFM Data and Scores"):
            st.dataframe(rfm_results)

with product_tab:
    # ... (code is unchanged) ...
    st.header("Product Performance (Pareto 80/20 Analysis)")
    if not filtered_df.empty:
        product_sales = filtered_df.groupby(product_col)[quantity_col].sum().sort_values(ascending=False)
        product_sales_df = product_sales.reset_index()
        product_sales_df['Cumulative_Percentage'] = (product_sales_df[quantity_col].cumsum() / product_sales_df[quantity_col].sum()) * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=product_sales_df[product_col], y=product_sales_df[quantity_col], name='Volume'), secondary_y=False)
        fig.add_trace(go.Scatter(x=product_sales_df[product_col], y=product_sales_df['Cumulative_Percentage'], name='Cumulative %'), secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# --- NEW GEOSPATIAL TAB ---
with geo_tab:
    st.header("🌍 Geospatial Sales View")
    st.markdown("This map shows the total sales volume by country for the selected filters. Darker colors indicate higher sales volume.")
    
    if not filtered_df.empty:
        # Group data by country and sum the quantity
        country_sales = filtered_df.groupby(country_col)[quantity_col].sum().reset_index()

        # Create the choropleth map
        fig = px.choropleth(
            country_sales,
            locations=country_col,
            locationmode='country names', # This tells plotly how to interpret the locations
            color=quantity_col,
            hover_name=country_col,
            color_continuous_scale=px.colors.sequential.Plasma, # Color scale for the heatmap
            title="Total Sales Volume by Country"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data to display on the map for the current filter selection.")


with comparison_tab:
    # ... (code is unchanged) ...
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
    # ... (code is unchanged) ...
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

with clv_tab:
    # ... (code is unchanged) ...
    st.header("💎 Customer Lifetime Value (CLV) Prediction")
    st.markdown("This analysis forecasts the future value of your customers based on their past transaction history.")
    prediction_days = st.slider("Select prediction timeframe (days):", 30, 365, 90)
    if st.button("Calculate CLV"):
        if filtered_df.empty or filtered_df[customer_col].nunique() < 10:
            st.warning("Not enough unique customer data to build a reliable CLV model. Please select a larger date range or more customers.")
        else:
            with st.spinner("Preparing data and training CLV models..."):
                clv_df = summary_data_from_transaction_data(filtered_df, customer_id_col=customer_col, datetime_col=date_col, monetary_value_col=quantity_col, observation_period_end=pd.to_datetime(end_date))
                clv_df = clv_df[clv_df['monetary_value'] > 0]
                if clv_df.empty:
                    st.error("No customers with repeat purchases and positive monetary value found. Cannot calculate CLV.")
                else:
                    bgf = BetaGeoFitter(penalizer_coef=0.0)
                    bgf.fit(clv_df['frequency'], clv_df['recency'], clv_df['T'])
                    ggf = GammaGammaFitter(penalizer_coef=0.0)
                    ggf.fit(clv_df['frequency'], clv_df['monetary_value'])
                    clv_df['predicted_clv'] = ggf.customer_lifetime_value(bgf, clv_df['frequency'], clv_df['recency'], clv_df['T'], clv_df['monetary_value'], time=prediction_days, discount_rate=0.01)
                    st.success("CLV Calculation Complete!")
                    st.subheader(f"Top Customers by Predicted CLV over the next {prediction_days} days")
                    clv_results = clv_df[['predicted_clv']].sort_values(by='predicted_clv', ascending=False).reset_index()
                    st.dataframe(clv_results, use_container_width=True, hide_index=True, column_config={"predicted_clv": st.column_config.NumberColumn(format="%.2f")})
