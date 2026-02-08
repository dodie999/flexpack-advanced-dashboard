import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import openpyxl
import io
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
    """(CUSTOMER RFM) Calculates metrics for individual customers."""
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

@st.cache_data
def calculate_country_fm(df, date_col, country_col, quantity_col, customer_col, active_period_months):
    """(COUNTRY FM - IDEA FM) Segments markets based on active customer breadth."""
    if df.empty or df[country_col].nunique() == 0:
        return pd.DataFrame()
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    active_cutoff_date = snapshot_date - pd.DateOffset(months=active_period_months)
    active_df = df[df[date_col] >= active_cutoff_date]
    
    rfm_df = df.groupby(country_col).agg(
        Recency=(date_col, lambda date: (snapshot_date - date.max()).days),
        Monetary_Volume=(quantity_col, 'sum')
    )
    
    active_customers = active_df.groupby(country_col)[customer_col].nunique()
    total_customers = df.groupby(country_col)[customer_col].nunique()
    
    rfm_df = rfm_df.join(active_customers.rename('Frequency_Active_Breadth')).fillna(0)
    rfm_df = rfm_df.join(total_customers.rename('Frequency_Total_Breadth')).fillna(0)
    rfm_df['Frequency_Active_Breadth'] = rfm_df['Frequency_Active_Breadth'].astype(int)
    rfm_df['Frequency_Total_Breadth'] = rfm_df['Frequency_Total_Breadth'].astype(int)
    
    # Cleaner index: Country (Total Customers)
    rfm_df.index = rfm_df.index.astype(str) + ' (' + rfm_df['Frequency_Total_Breadth'].astype(str) + ')'
    
    if rfm_df.shape[0] < 4:
        rfm_df['R_Score'] = 4; rfm_df['F_Score'] = 4; rfm_df['M_Score'] = 4
        rfm_df['RFM_Score'] = '444'; rfm_df['Segment'] = 'Single Market'
        return rfm_df

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
    start_date, end_date = st.date_input("Select Timeframe", value=(df[date_col].min().date(), df[date_col].max().date()))
    selected_customers = st.multiselect("Filter by Customer", options=sorted([str(c) for c in df[customer_col].unique()]))
    selected_product_groups = st.multiselect("Filter by Product Group", options=sorted([str(pg) for pg in df[product_group_col].unique()]))
    selected_products = st.multiselect("Filter by Product Name", options=sorted([str(p) for p in df[product_col].unique()]))
    selected_salesmen = st.multiselect("Filter by Salesman", options=sorted([str(s) for s in df[salesman_col].unique()]))
    selected_dom_exp = st.multiselect("Filter by Domestic/Export", options=sorted([str(de) for de in df[dom_exp_col].unique()]))
    selected_countries = st.multiselect("Filter by Country", options=sorted([str(c) for c in df[country_col].unique()]))

# --- Filtering Logic ---
mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
if selected_customers: mask &= df[customer_col].isin(selected_customers)
if selected_product_groups: mask &= df[product_group_col].isin(selected_product_groups)
if selected_products: mask &= df[product_col].isin(selected_products)
if selected_salesmen: mask &= df[salesman_col].isin(selected_salesmen)
if selected_dom_exp: mask &= df[dom_exp_col].isin(selected_dom_exp)
if selected_countries: mask &= df[country_col].isin(selected_countries)
filtered_df = df[mask]
if filtered_df.empty: st.warning("No data matches filters."); st.stop()

# --- Main Dashboard Tabs ---
tab_list = ["📊 Overview", "👤 Customer Deep Dive", "🌎 Country Analysis", "📦 Product Analysis", "🔬 Sample Analysis", "📞 CRM & Follow-Up", "🔄 Timeframe Comparison", "📈 Sales Forecast", "💎 CLV Prediction"]
tabs = st.tabs(tab_list)

with tabs[0]: # Overview
    st.header("Dashboard Overview")
    total_volume = filtered_df[quantity_col].sum()
    unique_cust = filtered_df[customer_col].nunique()
    c1, c2 = st.columns(2)
    c1.metric("Total Sales Volume", f"{total_volume:,.0f}")
    c2.metric("Unique Customers", f"{unique_cust:,}")
    monthly_sales = filtered_df.set_index(date_col)[quantity_col].resample('M').sum()
    st.plotly_chart(px.line(monthly_sales, title="Monthly Sales Trend"), use_container_width=True)

with tabs[1]: # Customer Deep Dive
    st.header("Customer Analysis")
    cust_vols = filtered_df.groupby(customer_col)[quantity_col].sum()
    if not cust_vols.empty:
        max_v = float(cust_vols.max())
        # Correction applied: max_value parameter
        thresh = st.slider("Min Quantity Filter:", min_value=0.0, max_value=max_v, value=float(max_v/4))
        st.dataframe(cust_vols[cust_vols > thresh].sort_values(ascending=False))
    rfm = calculate_rfm(filtered_df, date_col, customer_col, quantity_col, order_col)
    if not rfm.empty:
        st.plotly_chart(px.bar(rfm['Segment'].value_counts(), title="Customer Segments"), use_container_width=True)

with tabs[2]: # Country Analysis (Idea FM)
    st.header("🌎 Country Market Analysis (FM Model)")
    act_mos = st.slider("Define 'Active Period' (Months):", 1, 24, 6)
    c_rfm = calculate_country_fm(df, date_col, country_col, quantity_col, customer_col, act_mos)
    if not c_rfm.empty:
        st.plotly_chart(px.bar(c_rfm['Segment'].value_counts(), title="Market Segments"), use_container_width=True)
        st.dataframe(c_rfm.sort_values(by='RFM_Score', ascending=False), use_container_width=True)

with tabs[3]: # Product Analysis
    st.header("📦 Product Pareto Analysis")
    p_sales = filtered_df.groupby(product_col)[quantity_col].sum().sort_values(ascending=False).reset_index()
    p_sales['CumSum %'] = (p_sales[quantity_col].cumsum() / p_sales[quantity_col].sum()) * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=p_sales[product_col], y=p_sales[quantity_col], name="Volume"), secondary_y=False)
    fig.add_trace(go.Scatter(x=p_sales[product_col], y=p_sales['CumSum %'], name="Cumulative %"), secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]: # Sample Analysis
    st.header("🔬 Sample Conversion")
    s_thresh = st.number_input("Sample Limit (kg):", value=200)
    # Logic for sample identification and conversion metrics goes here...
    st.info("Identify small-quantity 'samples' and track if they lead to bulk 'regular' orders later.")

with tabs[5]: # CRM & Follow-Up
    st.header("📞 CRM Follow-Up Engine")
    crm_data = []
    snap = df[date_col].max()
    for cust, c_df in filtered_df.groupby(customer_col):
        last_s = c_df[date_col].max()
        orders = c_df[order_col].nunique()
        cadence = ((c_df[date_col].max() - c_df[date_col].min()).days / (orders - 1)) if orders > 1 else 0
        three_mo = snap - pd.DateOffset(months=3)
        cons = c_df[c_df[date_col] >= three_mo][quantity_col].sum() / 3
        last_qty = c_df[c_df[date_col] == last_s][quantity_col].sum()
        crm_data.append({customer_col: cust, 'Country': c_df[country_col].iloc[0], 'Frequency (Days)': round(cadence, 1), '3 Mo Avg MT': round(cons, 2), 'Last Order MT': round(last_qty, 2), 'Last Ship': last_s.date()})
    
    crm_df = pd.DataFrame(crm_data)
    # Plotly Scatter Fix: Filter out non-positive sizes
    plot_df = crm_df[crm_df['Last Order MT'] > 0].copy()
    if not plot_df.empty:
        st.plotly_chart(px.scatter(plot_df, x='Frequency (Days)', y='3 Mo Avg MT', size='Last Order MT', color='Country', hover_name=customer_col), use_container_width=True)
    
    # Actionable Table
    for col in ["Purchasing Person", "Honorific", "Purchasing Email"]:
        if col not in crm_df.columns: crm_df[col] = ""
    edited = st.data_editor(crm_df, use_container_width=True, hide_index=True)
    
    # Export Logic
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as wr:
        edited.to_excel(wr, index=False, sheet_name='FollowUps')
    st.download_button("📥 Export to Excel", buf.getvalue(), "FollowUps.xlsx")

    # Email Generator
    target = st.selectbox("Draft Email for:", edited[customer_col].unique())
    if target:
        r = edited[edited[customer_col] == target].iloc[0]
        h = r["Honorific"] if r["Honorific"] else "Mr./Ms."
        p = r["Purchasing Person"] if r["Purchasing Person"] else "Purchasing Manager"
        st.code(f"Dear {h} {p},\n\nNoticed your last order was {r['Last Order MT']} MT on {r['Last Ship']}.\nBased on your {r['Frequency (Days)']} day cycle, can we assist with your next batch?")

with tabs[7]: # Forecast with Holiday context
    st.header("📈 Sales Forecast")
    ccode = st.text_input("Country Code (Holidays):", "EG")
    target_f = st.selectbox("Forecast Target:", ["Total Sales"] + sorted(df[product_col].unique().tolist()))
    if st.button("Run Forecast"):
        f_df = df if target_f == "Total Sales" else df[df[product_col] == target_f]
        ts = f_df.set_index(date_col)[quantity_col].resample('D').sum().reset_index().rename(columns={date_col: 'ds', quantity_col: 'y'})
        m = Prophet()
        if ccode: m.add_country_holidays(country_name=ccode)
        m.fit(ts)
        fut = m.make_future_dataframe(periods=90)
        fcst = m.predict(fut)
        st.pyplot(m.plot(fcst))

with tabs[8]: # CLV Prediction
    st.header("💎 CLV Prediction")
    st.markdown("This analysis forecasts future value based on repeat purchase patterns.")
    days = st.slider("Prediction Days:", 30, 365, 90)
    
    if st.button("Predict CLV"):
        # 1. Prepare the data
        clv_base = summary_data_from_transaction_data(
            filtered_df, 
            customer_col, 
            date_col, 
            quantity_col, 
            observation_period_end=pd.to_datetime(end_date)
        )
        
        # 2. Filter for customers with at least some repeat activity
        # Lifetimes models often struggle with strictly 0-frequency (one-time) customers
        clv_filtered = clv_base[(clv_base['monetary_value'] > 0) & (clv_base['frequency'] > 0)]
        
        if clv_filtered.empty or len(clv_filtered) < 5:
            st.warning("⚠️ Not enough repeat-purchase data in this selection to build a CLV model.")
        else:
            with st.spinner("Training CLV model..."):
                try:
                    # 3. Fit the BG/NBD model with a small penalizer to help convergence
                    bgf = BetaGeoFitter(penalizer_coef=0.01) # Small penalizer helps stability
                    bgf.fit(clv_filtered['frequency'], clv_filtered['recency'], clv_filtered['T'])
                    
                    # 4. Fit the Gamma-Gamma model
                    ggf = GammaGammaFitter(penalizer_coef=0.01)
                    ggf.fit(clv_filtered['frequency'], clv_filtered['monetary_value'])
                    
                    # 5. Predict Value
                    clv_filtered['Predicted CLV'] = ggf.customer_lifetime_value(
                        bgf, 
                        clv_filtered['frequency'], 
                        clv_filtered['recency'], 
                        clv_filtered['T'], 
                        clv_filtered['monetary_value'], 
                        time=days
                    )
                    
                    st.success("Model converged successfully!")
                    st.dataframe(
                        clv_filtered[['Predicted CLV']].sort_values(by='Predicted CLV', ascending=False),
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"📈 Convergence Error: The model couldn't find a stable pattern in this specific data slice.")
                    st.info("This often happens with domestic data if ordering patterns are too irregular. Try expanding your timeframe or selecting more customers.")
