import streamlit as st
import pandas as pd
import numpy as np
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
    (CUSTOMER RFM) Calculates RFM metrics and segments customers. 
    This function is NOT affected by the new country logic.
    """
    if df.empty or df[customer_col].nunique() == 0:
        return pd.DataFrame()

    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm_df = df.groupby(customer_col).agg({
        date_col: lambda date: (snapshot_date - date.max()).days,
        order_col: 'nunique',
        quantity_col: 'sum'
    }).rename(columns={date_col: 'Recency', order_col: 'Frequency', quantity_col: 'Monetary'})

    if rfm_df.shape[0] < 4:
        rfm_df['R_Score'] = 4; rfm_df['F_Score'] = 4; rfm_df['M_Score'] = 4
        rfm_df['RFM_Score'] = '444'; rfm_df['Segment'] = 'Individual Analysis'
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

# --- MODIFIED: Country "FM" Function (Idea FM) ---
@st.cache_data
def calculate_country_fm(df, date_col, country_col, quantity_col, customer_col, active_period_months):
    """
    (COUNTRY RFM - IDEA FM) Calculates metrics based on an "active period".
    R = Recency of last order from the total period
    F = Frequency (breadth) defined by number of unique *active* customers
    M = Monetary defined by total quantity from the total period
    """
    if df.empty or df[country_col].nunique() == 0:
        return pd.DataFrame()

    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    
    # Define the "active" cutoff date
    active_cutoff_date = snapshot_date - pd.DateOffset(months=active_period_months)
    
    # Create the "active" dataframe
    active_df = df[df[date_col] >= active_cutoff_date]

    # Calculate base metrics
    rfm_df = df.groupby(country_col).agg(
        Recency=(date_col, lambda date: (snapshot_date - date.max()).days),
        Monetary_Volume=(quantity_col, 'sum')
    )
    
    # Calculate Active Customer Breadth
    active_customers_per_country = active_df.groupby(country_col)[customer_col].nunique()
    # Calculate Total Customer Breadth
    total_customers_per_country = df.groupby(country_col)[customer_col].nunique()

    # Join both to the main rfm_df
    rfm_df = rfm_df.join(active_customers_per_country.rename('Frequency_Active_Breadth')).fillna(0)
    rfm_df = rfm_df.join(total_customers_per_country.rename('Frequency_Total_Breadth')).fillna(0)

    # Convert counts to integer
    rfm_df['Frequency_Active_Breadth'] = rfm_df['Frequency_Active_Breadth'].astype(int)
    rfm_df['Frequency_Total_Breadth'] = rfm_df['Frequency_Total_Breadth'].astype(int)

    # --- NEW: Create the new index format ---
    rfm_df.index = rfm_df.index.astype(str) + ' (' + rfm_df['Frequency_Total_Breadth'].astype(str) + ')'
    # --- END NEW ---

    if rfm_df.shape[0] < 4:
        rfm_df['R_Score'] = 4; rfm_df['F_Score'] = 4; rfm_df['M_Score'] = 4
        rfm_df['RFM_Score'] = '444'; rfm_df['Segment'] = 'Single Market'
        return rfm_df

    # Scoring remains based on the *active* customer count
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop').astype(int)
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency_Active_Breadth'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary_Volume'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

    def segment_market(row):
        if row['R_Score'] >= 3 and row['F_Score'] >= 3: return 'Healthy Markets'
        if row['R_Score'] == 4 and row['F_Score'] == 1: return 'Single Customer Market'
        if row['R_Score'] >= 3 and row['F_Score'] == 1: return 'Promising Markets'
        if row['R_Score'] <= 2 and row['F_Score'] >= 3: return 'At-Risk (High Breadth)'
        if row['R_Score'] <= 2 and row['F_Score'] <= 2: return 'Hibernating Markets'
        return 'Regular Markets'

    rfm_df['Segment'] = rfm_df.apply(segment_market, axis=1)
    return rfm_df
# --- END MODIFIED FUNCTION ---

# --- Main App ---
st.title("🚀 Comprehensive Sales Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Your Data"); uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is None: st.info("Please upload an Excel file to begin."); st.stop()
    df_raw = load_data(uploaded_file)
    if df_raw is None: st.stop()
    st.header("2. Map Your Columns")
    all_columns = df_raw.columns.tolist()
    def find_default(name, options): return options.index(name) if name in options else 0
    date_col = st.selectbox("Date Column", all_columns, index=find_default('Ship date', all_columns))
    customer_col = st.selectbox("Customer Name Column", all_columns, index=find_default('Customer name', all_columns))
    country_col = st.selectbox("Country Column", all_columns, index=find_default('Country', all_columns))
    quantity_col = st.selectbox("Quantity Column", all_columns, index=find_default('Quantity', all_columns))
    order_col = st.selectbox("Sales Order Column", all_columns, index=find_default('Sales order', all_columns))
    product_group_col = st.selectbox("Product Group Column", all_columns, index=find_default('Product', all_columns))
    product_col = st.selectbox("Product Name Column", all_columns, index=find_default('Product name', all_columns))
    salesman_col = st.selectbox("Salesman Column", all_columns, index=find_default('Salesman', all_columns))
    dom_exp_col = st.selectbox("Domestic/Export Column", all_columns, index=find_default('Domestic/Export', all_columns))
    df = preprocess_data(df_raw, date_col, quantity_col, customer_col)
    if df is None: st.stop()
    st.header("3. Master Filters")
    start_date, end_date = st.date_input("Select Timeframe", value=(df[date_col].min().date(), df[date_col].max().date()), min_value=df[date_col].min().date(), max_value=df[date_col].max().date())
    selected_customers = st.multiselect("Filter by Customer", options=sorted([str(c) for c in df[customer_col].unique()]), default=[])
    selected_product_groups = st.multiselect("Filter by Product Group", options=sorted([str(pg) for pg in df[product_group_col].unique()]), default=[])
    selected_products = st.multiselect("Filter by Product Name", options=sorted([str(p) for p in df[product_col].unique()]), default=[])
    selected_salesmen = st.multiselect("Filter by Salesman", options=sorted([str(s) for s in df[salesman_col].unique()]), default=[])
    selected_dom_exp = st.multiselect("Filter by Domestic/Export", options=sorted([str(de) for de in df[dom_exp_col].unique()]), default=[])
    selected_countries = st.multiselect("Filter by Country", options=sorted([str(c) for c in df[country_col].unique()]), default=[])

# --- Filtering Logic ---
mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
if selected_customers: mask &= df[customer_col].isin(selected_customers)
if selected_product_groups: mask &= df[product_group_col].isin(selected_product_groups)
if selected_products: mask &= df[product_col].isin(selected_products)
if selected_salesmen: mask &= df[salesman_col].isin(selected_salesmen)
if selected_dom_exp: mask &= df[dom_exp_col].isin(selected_dom_exp)
if selected_countries: mask &= df[country_col].isin(selected_countries)
filtered_df = df[mask]
if filtered_df.empty: st.warning("No data matches the selected filters."); st.stop()

# --- Main Dashboard with Tabs ---
tab_list = ["📊 Overview", "👤 Customer Deep Dive", "🌎 Country Analysis", "📦 Product Analysis", "🔬 Sample Analysis", "🔄 Timeframe Comparison", "📈 Sales Forecast", "💎 CLV Prediction"]
overview_tab, customer_tab, country_analysis_tab, product_tab, sample_tab, comparison_tab, forecast_tab, clv_tab = st.tabs(tab_list)

with overview_tab:
    st.header("Dashboard Overview"); total_volume = filtered_df[quantity_col].sum(); unique_customers = filtered_df[customer_col].nunique()
    new_customers_mask = df.groupby(customer_col)[date_col].min().between(pd.to_datetime(start_date), pd.to_datetime(end_date)); new_customer_count = new_customers_mask.sum()
    col1, col2, col3 = st.columns(3); col1.metric("Total Sales Volume", f"{total_volume:,.0f}"); col2.metric("Unique Customers", f"{unique_customers:,}"); col3.metric("New Customers in Period", f"{new_customer_count:,}")
    st.markdown("---"); st.header("Sales Trend"); monthly_sales = filtered_df.set_index(date_col)[quantity_col].resample('M').sum()
    fig = px.line(monthly_sales, title="Monthly Sales Volume"); st.plotly_chart(fig, use_container_width=True)
    if len(selected_customers) == 1 or len(selected_countries) == 1 or len(selected_products) > 0 or len(selected_salesmen) == 1:
        st.markdown("---"); st.header("Filtered Order History")
        if len(selected_customers) == 1: st.subheader(f"Showing all orders for customer: {selected_customers[0]}")
        elif len(selected_countries) == 1: st.subheader(f"Showing all orders for country: {selected_countries[0]}")
        elif len(selected_salesmen) == 1: st.subheader(f"Showing all orders for salesman: {selected_salesmen[0]}")
        elif len(selected_products) > 0: st.subheader(f"Showing all orders for product(s): {', '.join(selected_products)}")
        history_columns = [date_col, customer_col, country_col, product_group_col, product_col, salesman_col, quantity_col, order_col]
        display_cols = [col for col in history_columns if col in filtered_df.columns]
        history_df = filtered_df[display_cols].sort_values(by=date_col, ascending=False); st.dataframe(history_df, use_container_width=True, hide_index=True)

with customer_tab:
    st.header("Customer Deep Dive Analysis"); st.markdown("### High-Volume Customer Analysis")
    if not filtered_df.empty:
        customer_volumes = filtered_df.groupby(customer_col)[quantity_col].sum()
        if not customer_volumes.empty:
            max_volume = int(customer_volumes.max())
            volume_threshold = st.slider("Minimum Total Quantity per Customer:", min_value=0, max_value=max_volume, value=int(max_volume/4))
            high_volume_customers = customer_volumes[customer_volumes > volume_threshold]; st.metric(f"Customers with > {volume_threshold:,} units", len(high_volume_customers))
            with st.expander("View High-Volume Customer List"): st.dataframe(high_volume_customers.sort_values(ascending=False))
    st.markdown("---"); st.markdown("### Individual Customer Consumption")
    if not filtered_df[customer_col].empty:
        customer_options = sorted(filtered_df[customer_col].unique())
        selected_customer = st.selectbox("Select a customer to analyze:", options=customer_options)
        if selected_customer:
            customer_data = filtered_df[filtered_df[customer_col] == selected_customer]; customer_monthly_consumption = customer_data.set_index(date_col)[quantity_col].resample('M').sum()
            avg_consumption = customer_monthly_consumption.mean(); st.metric(f"Avg. Monthly Consumption for {selected_customer}", f"{avg_consumption:,.2f} units")
            fig = px.bar(customer_monthly_consumption, title=f"Monthly Purchases for {selected_customer}"); st.plotly_chart(fig, use_container_width=True)
    st.markdown("---"); st.markdown("### RFM Segmentation (by Customer)")
    if not filtered_df.empty:
        rfm_results = calculate_rfm(filtered_df, date_col, customer_col, quantity_col, order_col)
        fig = px.bar(rfm_results['Segment'].value_counts(), title="Customer Segment Counts"); st.plotly_chart(fig, use_container_width=True)
        with st.expander("View Detailed Customer RFM Data"): st.dataframe(rfm_results)

# --- MODIFIED: Country Analysis Tab (Idea FM) ---
with country_analysis_tab:
    st.header("🌎 Country Market Analysis (FM Model)")
    st.markdown("""
    This analysis segments your markets based on their **Active Health**:
    * **Recency (R):** How recently was the last order? (from total period)
    * **Frequency (F):** How many unique customers (breadth) were **active** in the defined period?
    * **Monetary (M):** What is the total sales volume (size)? (from total period)
    """)
    
    active_months = st.slider("Define 'Active Period' (in months):", 1, 24, 6)

    if not filtered_df.empty:
        # Use the *unfiltered* df for a complete historical analysis
        country_rfm_results = calculate_country_fm(df, date_col, country_col, quantity_col, customer_col, active_months)
        
        if country_rfm_results.empty:
            st.warning("Not enough data to perform country analysis.")
        else:
            st.subheader(f"Market Segments based on {active_months}-Month Activity")
            fig = px.bar(country_rfm_results['Segment'].value_counts(), title="Market Segment Counts")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Detailed Market Scores"):
                # Rename columns for a cleaner display
                display_df = country_rfm_results.rename(columns={
                    'Recency': 'Recency (Days)',
                    'Frequency_Active_Breadth': 'Active Customers', # <-- MODIFIED
                    'Monetary_Volume': 'Total Volume'
                })
                
                # Define the columns we want to show
                display_cols = [
                    'Segment', 
                    'Recency (Days)', 
                    'Active Customers', # <-- MODIFIED
                    'Total Volume', 
                    'R_Score', 
                    'F_Score', 
                    'M_Score',
                    'RFM_Score'
                ]
                
                # Ensure all desired columns exist in the dataframe
                final_display_cols = [col for col in display_cols if col in display_df.columns]
                
                # Set the index name to "Country (Total Customers)"
                display_df.index.name = "Country (Total Customers)"
                
                st.dataframe(
                    display_df[final_display_cols].sort_values(by='RFM_Score', ascending=False), 
                    use_container_width=True
                )

with product_tab:
    st.header("📦 Product Performance (Pareto 80/20 Analysis)")
    if not filtered_df.empty:
        product_sales = filtered_df.groupby(product_col)[quantity_col].sum().sort_values(ascending=False)
        product_sales_df = product_sales.reset_index(); product_sales_df['Cumulative_Percentage'] = (product_sales_df[quantity_col].cumsum() / product_sales_df[quantity_col].sum()) * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]]); fig.add_trace(go.Bar(x=product_sales_df[product_col], y=product_sales_df[quantity_col], name='Volume'), secondary_y=False); fig.add_trace(go.Scatter(x=product_sales_df[product_col], y=product_sales_df['Cumulative_Percentage'], name='Cumulative %'), secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

with sample_tab:
    st.header("🔬 Sample Conversion Analysis"); st.markdown("This tool identifies samples based on the combined quantity of related products within a single order.")
    sample_threshold = st.number_input("Set the maximum combined quantity for a 'sample' (in kg):", min_value=1, value=200)
    df_analysis = df.copy(); df_analysis['Product_Base'] = df_analysis[product_col].str.extract(r'(\d+)'); df_analysis.dropna(subset=['Product_Base'], inplace=True)
    df_analysis['Combined_Order_Quantity'] = df_analysis.groupby([order_col, 'Product_Base'])[quantity_col].transform('sum')
    df_analysis['Order_Type'] = np.where(df_analysis['Combined_Order_Quantity'] < sample_threshold, 'Sample', 'Regular Order')
    samples_df = df_analysis[df_analysis['Order_Type'] == 'Sample']; regular_orders_df = df_analysis[df_analysis['Order_Type'] == 'Regular Order']
    if samples_df.empty: st.info("No sample orders found.")
    else:
        customers_in_view = filtered_df[customer_col].unique(); sampled_customers_total = samples_df[samples_df[customer_col].isin(customers_in_view)][customer_col].unique()
        if len(sampled_customers_total) == 0: st.info("No customers who have received samples are present in the current filter.")
        else:
            conversion_data = []
            for customer in sampled_customers_total:
                first_sample_date = samples_df[samples_df[customer_col] == customer][date_col].min()
                converted_orders = regular_orders_df[(regular_orders_df[customer_col] == customer) & (regular_orders_df[date_col] > first_sample_date)]
                status = "Not Converted"; days_to_convert = None; post_conversion_volume = 0
                if not converted_orders.empty:
                    status = "Converted"; first_conversion_date = converted_orders[date_col].min()
                    days_to_convert = (first_conversion_date - first_sample_date).days
                    post_conversion_volume = converted_orders[quantity_col].sum()
                conversion_data.append({'Customer': customer, 'Conversion Status': status, 'First Sample Date': first_sample_date.date(), 'Days to Convert': days_to_convert, 'Post-Conversion Volume': post_conversion_volume})
            conversion_df = pd.DataFrame(conversion_data); st.subheader("Conversion Summary"); col1, col2, col3 = st.columns(3)
            converted_count = conversion_df[conversion_df['Conversion Status'] == 'Converted'].shape[0]; total_sampled_count = len(sampled_customers_total)
            conversion_rate = (converted_count / total_sampled_count * 100) if total_sampled_count > 0 else 0
            col1.metric("Sampled Customers (in view)", f"{total_sampled_count}"); col2.metric("Converted Customers", f"{converted_count}"); col3.metric("Conversion Rate", f"{conversion_rate:.1f}%")
            st.subheader("Customer Conversion Details"); st.dataframe(conversion_df.sort_values(by="Post-Conversion Volume", ascending=False), use_container_width=True, hide_index=True)
            with st.expander("View All Sample Line Items (in current filter)"):
                samples_in_view_df = df_analysis[(df_analysis['Order_Type'] == 'Sample') & (df_analysis[customer_col].isin(customers_in_view))]
                sample_display_cols = [date_col, customer_col, country_col, product_col, quantity_col, 'Product_Base', 'Combined_Order_Quantity', 'Order_Type']
                st.dataframe(samples_in_view_df[sample_display_cols].sort_values(by=date_col, ascending=False), use_container_width=True, hide_index=True)

with comparison_tab:
    st.header("Timeframe Comparison"); col1, col2 = st.columns(2)
    with col1: p1_start, p1_end = st.date_input("Period 1", value=(df[date_col].min().date(), df[date_col].min().date() + pd.Timedelta(days=365)), key="p1")
    with col2: p2_start, p2_end = st.date_input("Period 2", value=(df[date_col].max().date() - pd.Timedelta(days=365), df[date_col].max().date()), key="p2")
    period1_df = df[(df[date_col].dt.date >= p1_start) & (df[date_col].dt.date <= p1_end)]; period2_df = df[(df[date_col].dt.date >= p2_start) & (df[date_col].dt.date <= p2_end)]
    p1_sales = period1_df[quantity_col].sum(); p2_sales = period2_df[quantity_col].sum(); p1_customers = period1_df[customer_col].nunique(); p2_customers = period2_df[customer_col].nunique()
    st.subheader("Comparison Results"); c1, c2 = st.columns(2)
    with c1: st.metric("Period 1 Sales", f"{p1_sales:,.0f}"); st.metric("Period 1 Customers", f"{p1_customers:,}")
    with c2: st.metric("Period 2 Sales", f"{p2_sales:,.0f}", delta=f"{p2_sales - p1_sales:,.0f}"); st.metric("Period 2 Customers", f"{p2_customers:,}", delta=f"{p2_customers - p1_customers:,}")

with forecast_tab:
    st.header("Sales Forecasting");
    if st.button("Generate 90-Day Forecast"):
        with st.spinner("Training model..."):
            forecast_df = df.set_index(date_col)[quantity_col].resample('D').sum().reset_index().rename(columns={date_col: 'ds', quantity_col: 'y'})
            model = Prophet(); model.fit(forecast_df); future = model.make_future_dataframe(periods=90); forecast = model.predict(future)
            fig = model.plot(forecast); st.pyplot(fig)

with clv_tab:
    st.header("💎 Customer Lifetime Value (CLV) Prediction")
    st.markdown("This analysis forecasts the future value of your customers based on their past transaction history.")
    prediction_days = st.slider("Select prediction timeframe (days):", 30, 365, 90)
    if st.button("Calculate CLV"):
        if filtered_df.empty or filtered_df[customer_col].nunique() < 10: st.warning("Not enough unique customer data to build a reliable CLV model.")
        else:
            with st.spinner("Preparing data and training CLV models..."):
                clv_df = summary_data_from_transaction_data(filtered_df, customer_id_col=customer_col, datetime_col=date_col, monetary_value_col=quantity_col, observation_period_end=pd.to_datetime(end_date))
                clv_df = clv_df[clv_df['monetary_value'] > 0]
                if clv_df.empty: st.error("No customers with repeat purchases found. Cannot calculate CLV.")
                else:
                    bgf = BetaGeoFitter(penalizer_coef=0.0); bgf.fit(clv_df['frequency'], clv_df['recency'], clv_df['T'])
                    ggf = GammaGammaFitter(penalizer_coef=0.0); ggf.fit(clv_df['frequency'], clv_df['monetary_value'])
                    clv_df['predicted_clv'] = ggf.customer_lifetime_value(bgf, clv_df['frequency'], clv_df['recency'], clv_df['T'], clv_df['monetary_value'], time=prediction_days, discount_rate=0.01)
                    st.success("CLV Calculation Complete!")
                    st.subheader(f"Top Customers by Predicted CLV over the next {prediction_days} days")
                    clv_results = clv_df[['predicted_clv']].sort_values(by='predicted_clv', ascending=False).reset_index()
                    st.dataframe(clv_results, use_container_width=True, hide_index=True, column_config={"predicted_clv": st.column_config.NumberColumn(format="%.2f")})
