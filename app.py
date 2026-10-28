"""
Regional Sales Intelligence App
Run with:  streamlit run app.py
Place Denormalize_regional_sales_data.xlsx next to this file (or upload it in the sidebar).
"""
import io
import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_SM = True
except Exception:  # statsmodels not installed
    HAS_SM = False

DEFAULT_FILE = "Denormalize_regional_sales_data.xlsx"

REQUIRED = [
    "Date", "ProductID", "SalesrepID", "CityID", "ScenarioID", "Units Sold", "Sales",
    "Cost of sales", "Sale price", "Cost price", "Product Name", "Category", "Brand",
    "Warranty", "Cities", "SalesPerson Name", "Designation", "Age", "Scenario Name",
]
NUMERIC = [
    "Index", "ProductID", "SalesrepID", "CityID", "ScenarioID", "Units Sold", "Sales",
    "Cost of sales", "Sale price", "Cost price", "Latitude", "Longitude", "Age",
]
TEXT = [
    "Product Name", "Category", "Brand", "Warranty", "Cities",
    "SalesPerson Name", "Designation", "Scenario Name",
]
CATEGORICAL_FEATURES = [
    "Product Name", "Category", "Brand", "Warranty", "Cities",
    "SalesPerson Name", "Designation", "Scenario Name",
]
NUMERIC_FEATURES = [
    "Units Sold", "Sale price", "Cost price", "Age", "Year", "Month", "Quarter", "DayOfWeek",
]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _find_header(raw):
    for i in range(min(20, len(raw))):
        vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        if "date" in vals and "sales" in vals:
            return i
    return None


@st.cache_data(show_spinner="Reading and cleaning the data...")
def load_data(file_bytes, filename):
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        raw = pd.read_csv(buf, header=None, low_memory=False)
    else:
        raw = pd.read_excel(buf, header=None)
    h = _find_header(raw)
    if h is None:
        raise ValueError("Could not find a header row that contains both 'Date' and 'Sales'.")
    df = raw.iloc[h + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[h].tolist()]
    df = df.dropna(how="all").reset_index(drop=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "DOJ" in df.columns:
        df["DOJ"] = pd.to_datetime(df["DOJ"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"]).reset_index(drop=True)
    for c in TEXT:
        df[c] = df[c].astype(str).str.strip()

    df["Profit"] = df["Sales"] - df["Cost of sales"]
    df["Margin %"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"] * 100, np.nan)
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def fmt(v):
    for div, s in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(v) >= div:
            return f"{v / div:,.2f}{s}"
    return f"{v:,.0f}"


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else float("nan")


def reg_metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "MAPE %": mape(y, p),
    }


def sidebar_filters(df):
    st.sidebar.header("Filters")
    st.sidebar.caption("Applied to the Analysis and Visualization tabs.")
    scen_all = sorted(df["Scenario Name"].unique())
    scen = st.sidebar.multiselect(
        "Scenario", scen_all, default=["Actual"] if "Actual" in scen_all else scen_all
    )
    cities = st.sidebar.multiselect("City", sorted(df["Cities"].unique()), default=sorted(df["Cities"].unique()))
    cats = st.sidebar.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    start, end = (dr[0], dr[1]) if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax)

    base = df[
        df["Cities"].isin(cities)
        & df["Category"].isin(cats)
        & (df["Date"].dt.date >= start)
        & (df["Date"].dt.date <= end)
    ]
    filtered = base[base["Scenario Name"].isin(scen)]
    if len(scen) > 1:
        st.sidebar.warning("Both scenarios selected: totals add Actual and Budget together.")
    return filtered, base


# ----------------------------------------------------------------------------
# TAB 1: Data analysis
# ----------------------------------------------------------------------------
def tab_analysis(df, base, full):
    st.subheader("Data analysis")
    if df.empty:
        st.warning("No rows match the current filters.")
        return

    c = st.columns(5)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Total sales", fmt(df["Sales"].sum()))
    c[2].metric("Total profit", fmt(df["Profit"].sum()))
    c[3].metric("Profit margin", f"{df['Profit'].sum() / df['Sales'].sum() * 100:.1f}%")
    c[4].metric("Units sold", f"{int(df['Units Sold'].sum()):,}")

    # Automatic insights
    yearly = df.groupby("Year")["Sales"].sum()
    bullets = []
    for dim, label in [("Category", "category"), ("Cities", "city"), ("Brand", "brand"), ("SalesPerson Name", "salesperson")]:
        s = df.groupby(dim)["Sales"].sum().sort_values(ascending=False)
        bullets.append(f"Top {label}: **{s.index[0]}** with {fmt(s.iloc[0])} ({s.iloc[0] / s.sum() * 100:.1f}% of sales).")
    if len(yearly) > 1 and yearly.iloc[0] > 0:
        cagr = (yearly.iloc[-1] / yearly.iloc[0]) ** (1 / (len(yearly) - 1)) - 1
        bullets.append(f"Sales grew from {fmt(yearly.iloc[0])} in {yearly.index[0]} to {fmt(yearly.iloc[-1])} in {yearly.index[-1]} ({cagr * 100:.1f}% per year).")
    st.markdown("**Quick insights**\n\n" + "\n".join(f"- {b}" for b in bullets))

    st.markdown("#### Data preview")
    st.dataframe(df.head(200))

    with st.expander("Schema, missing values and data quality checks"):
        info = pd.DataFrame({
            "dtype": full.dtypes.astype(str),
            "missing": full.isna().sum(),
            "unique values": full.nunique(),
        })
        st.dataframe(info)
        mism = int((full["Sales"] != full["Units Sold"] * full["Sale price"]).sum())
        cmism = int((full["Cost of sales"] != full["Units Sold"] * full["Cost price"]).sum())
        st.write(f"Duplicate rows: **{int(full.duplicated().sum()):,}**")
        st.write(f"Rows where Sales is not Units Sold x Sale price: **{mism:,}**")
        st.write(f"Rows where Cost of sales is not Units Sold x Cost price: **{cmism:,}**")

    with st.expander("Descriptive statistics"):
        num = ["Units Sold", "Sales", "Cost of sales", "Profit", "Sale price", "Cost price", "Margin %", "Age"]
        st.dataframe(df[num].describe().T)

    st.markdown("#### Group and compare")
    dims = ["Category", "Brand", "Product Name", "Cities", "SalesPerson Name", "Designation",
            "Warranty", "Scenario Name", "Year", "Quarter", "Month"]
    dim = st.selectbox("Group by", dims, key="an_dim")
    g = df.groupby(dim).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                            Units=("Units Sold", "sum"), Orders=("Sales", "size"))
    g["Margin %"] = (g["Profit"] / g["Sales"] * 100).round(2)
    g["Avg order value"] = (g["Sales"] / g["Orders"]).round(0)
    g["Share of sales %"] = (g["Sales"] / g["Sales"].sum() * 100).round(2)
    st.dataframe(g.sort_values("Sales", ascending=False))

    st.markdown("#### Actual vs Budget by year")
    if {"Actual", "Budget"} <= set(base["Scenario Name"].unique()):
        p = base.pivot_table(index="Year", columns="Scenario Name", values="Sales", aggfunc="sum")
        p["Variance"] = p["Actual"] - p["Budget"]
        p["Variance %"] = (p["Variance"] / p["Budget"] * 100).round(2)
        st.dataframe(p)
    else:
        st.info("Both Actual and Budget rows are needed for this table.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Correlation")
        cols = ["Units Sold", "Sales", "Cost of sales", "Profit", "Sale price", "Cost price", "Age", "Year", "Month"]
        fig = px.imshow(df[cols].corr(), text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig)
    with right:
        st.markdown("#### Outliers in Sales (IQR rule, per category)")
        q1 = df.groupby("Category")["Sales"].transform(lambda s: s.quantile(0.25))
        q3 = df.groupby("Category")["Sales"].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        is_out = (df["Sales"] > q3 + 1.5 * iqr) | (df["Sales"] < q1 - 1.5 * iqr)
        o = df.assign(Outlier=is_out).groupby("Category")["Outlier"].agg(["sum", "mean"])
        o.columns = ["Outlier rows", "Share"]
        o["Share"] = (o["Share"] * 100).round(2).astype(str) + "%"
        st.dataframe(o)
        st.caption("Large orders in expensive categories are legitimate sales, so review before removing anything.")


# ----------------------------------------------------------------------------
# TAB 2: Visualization
# ----------------------------------------------------------------------------
def tab_viz(df, base):
    st.subheader("Data visualization")
    if df.empty:
        st.warning("No rows match the current filters.")
        return

    metric = st.radio("Metric", ["Sales", "Profit", "Units Sold"], horizontal=True, key="viz_metric")
    gran = st.radio("Time granularity", ["Month", "Quarter", "Year"], horizontal=True, key="viz_gran")

    d = base.copy()
    if gran == "Month":
        d["Period"] = d["YearMonth"]
    elif gran == "Quarter":
        d["Period"] = d["Date"].dt.to_period("Q").dt.to_timestamp()
    else:
        d["Period"] = d["Date"].dt.to_period("Y").dt.to_timestamp()
    t = d.groupby(["Period", "Scenario Name"])[metric].sum().reset_index()
    st.plotly_chart(px.line(t, x="Period", y=metric, color="Scenario Name", markers=(gran != "Month"),
                            title=f"{metric} over time"))

    c1, c2 = st.columns(2)
    with c1:
        s = df.groupby("Category")[metric].sum().sort_values().reset_index()
        st.plotly_chart(px.bar(s, x=metric, y="Category", orientation="h", title=f"{metric} by category"))
    with c2:
        s = df.groupby("Cities")[metric].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(s, x="Cities", y=metric, title=f"{metric} by city"))

    c3, c4 = st.columns(2)
    with c3:
        top_n = st.slider("Top N products", 5, 30, 10, key="viz_topn")
        s = df.groupby("Product Name")[metric].sum().sort_values(ascending=False).head(top_n).sort_values().reset_index()
        st.plotly_chart(px.bar(s, x=metric, y="Product Name", orientation="h", title=f"Top {top_n} products"))
    with c4:
        s = df.groupby(["Year", "Category"])[metric].sum().reset_index()
        st.plotly_chart(px.bar(s, x="Year", y=metric, color="Category", title=f"{metric} by year and category"))

    c5, c6 = st.columns(2)
    with c5:
        h = df.pivot_table(index="Year", columns="Month", values=metric, aggfunc="sum")
        st.plotly_chart(px.imshow(h, aspect="auto", color_continuous_scale="Viridis",
                                  title=f"{metric}: year by month heatmap"))
    with c6:
        s = df.groupby("Brand")[metric].sum().reset_index()
        st.plotly_chart(px.pie(s, names="Brand", values=metric, hole=0.45, title=f"{metric} share by brand"))

    st.markdown("#### Category, brand and product hierarchy")
    st.plotly_chart(px.treemap(df.groupby(["Category", "Brand", "Product Name"])[metric].sum().reset_index(),
                               path=["Category", "Brand", "Product Name"], values=metric))

    c7, c8 = st.columns(2)
    with c7:
        s = df.groupby("SalesPerson Name").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
        st.plotly_chart(px.bar(s.sort_values("Sales"), x="Sales", y="SalesPerson Name", orientation="h",
                               color="Profit", title="Salesperson leaderboard"))
    with c8:
        st.plotly_chart(px.box(df.sample(min(len(df), 20000), random_state=1), x="Category", y="Sales", log_y=True,
                               title="Sales distribution by category (log scale, sampled)"))

    c9, c10 = st.columns(2)
    with c9:
        s = df.groupby("Category").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
        s["Margin %"] = s["Profit"] / s["Sales"] * 100
        st.plotly_chart(px.bar(s.sort_values("Margin %"), x="Margin %", y="Category", orientation="h",
                               title="Profit margin by category"))
    with c10:
        s = df.groupby("Warranty")[metric].sum().reset_index()
        st.plotly_chart(px.bar(s, x="Warranty", y=metric, title=f"{metric} by warranty type"))

    st.markdown("#### Sales map")
    m = df.groupby(["Cities", "Latitude", "Longitude"])["Sales"].sum().reset_index()
    m["size"] = m["Sales"] / m["Sales"].max() * 40000 + 5000
    st.map(m.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}), size="size")


# ----------------------------------------------------------------------------
# TAB 3: Predictive analysis
# ----------------------------------------------------------------------------
def build_pipeline(model_name, cat_cols, num_cols):
    parts = []
    if cat_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        parts.append(("num", StandardScaler() if model_name == "Ridge Regression" else "passthrough", num_cols))
    pre = ColumnTransformer(parts)
    if model_name == "Random Forest":
        est = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5,
                                    max_samples=0.5, n_jobs=-1, random_state=42)
    elif model_name == "Gradient Boosting":
        est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=8, random_state=42)
    else:
        est = Ridge(alpha=1.0)
    return Pipeline([("pre", pre), ("model", est)])


def train_and_evaluate(df, target, feats, model_name, test_months):
    d = df.dropna(subset=feats + [target]).copy()
    cat_cols = [c for c in feats if c in CATEGORICAL_FEATURES]
    num_cols = [c for c in feats if c not in CATEGORICAL_FEATURES]

    test_start = d["Date"].max().to_period("M").to_timestamp() - pd.DateOffset(months=test_months - 1)
    train, test = d[d["Date"] < test_start], d[d["Date"] >= test_start]
    if len(train) < 500 or len(test) < 50:
        raise ValueError("Not enough rows in the train or test split. Reduce the hold out period.")

    pipe = build_pipeline(model_name, cat_cols, num_cols).fit(train[feats], train[target])
    pred = np.clip(pipe.predict(test[feats]), 0, None)
    metrics = reg_metrics(test[target], pred)

    base_pred = DummyRegressor(strategy="mean").fit(train[feats], train[target]).predict(test[feats])
    baseline = reg_metrics(test[target], base_pred)

    sample = test.sample(min(2000, len(test)), random_state=0)
    imp = permutation_importance(pipe, sample[feats], sample[target], n_repeats=2, random_state=0)
    importance = (pd.DataFrame({"Feature": feats, "Importance": imp.importances_mean})
                  .sort_values("Importance", ascending=True))

    final = build_pipeline(model_name, cat_cols, num_cols).fit(d[feats], d[target])
    out = test[["Date", "Product Name", "Cities", "Scenario Name", target]].copy()
    out["Predicted"] = pred
    return {
        "final": final, "feats": feats, "target": target, "model_name": model_name,
        "metrics": metrics, "baseline": baseline, "importance": importance, "test": out,
        "train_rows": len(train), "test_rows": len(test), "test_start": test_start,
    }


def build_price_table(df):
    return df.groupby(["Product Name", "Scenario Name", "Year"])[["Sale price", "Cost price"]].mean().reset_index()


def estimate_prices(tbl, product, scenario, year):
    t = tbl[(tbl["Product Name"] == product) & (tbl["Scenario Name"] == scenario)].sort_values("Year")
    out = {}
    for col in ["Sale price", "Cost price"]:
        if year in t["Year"].values:
            val = float(t.loc[t["Year"] == year, col].iloc[0])
        elif len(t) >= 2:
            k, b = np.polyfit(t["Year"].astype(float), t[col].astype(float), 1)
            val = float(k * year + b)
        else:
            val = float(t[col].iloc[-1])
        out[col] = max(val, 1.0)
    return out


def predict_one(res, full, prod_attr, rep_attr, price_tbl, product, city, rep, scenario, when, units, override=0.0):
    """Predict one sale. For dates after the data, tree models cannot extrapolate prices,
    so the model is evaluated at last year's price level and scaled by the price ratio."""
    when = pd.Timestamp(when)
    last_year = int(full["Year"].max())
    future = when.year > last_year
    est = estimate_prices(price_tbl, product, scenario, when.year)
    sale_price = float(override) if override and override > 0 else est["Sale price"]
    adjust = future and res["target"] in ("Sales", "Profit") and "Sale price" in res["feats"]
    ref = estimate_prices(price_tbl, product, scenario, last_year) if adjust else est

    def row_for(u):
        use_price = ref["Sale price"] if adjust else sale_price
        year = last_year if adjust else when.year
        row = {
            "Product Name": product, "Category": prod_attr.loc[product, "Category"],
            "Brand": prod_attr.loc[product, "Brand"], "Warranty": prod_attr.loc[product, "Warranty"],
            "Cities": city, "SalesPerson Name": rep,
            "Designation": rep_attr.loc[rep, "Designation"], "Age": rep_attr.loc[rep, "Age"],
            "Scenario Name": scenario, "Units Sold": u, "Sale price": use_price,
            "Cost price": ref["Cost price"] if adjust else est["Cost price"], "Year": year,
            "Month": when.month, "Quarter": when.quarter, "DayOfWeek": when.dayofweek,
        }
        return pd.DataFrame([{k: row[k] for k in res["feats"]}])

    def predict(u):
        p = float(res["final"].predict(row_for(u))[0])
        if adjust:
            p *= sale_price / ref["Sale price"]
        return max(p, 0.0)

    return {"predict": predict, "sale_price": sale_price, "future": future, "adjusted": adjust}


def tab_predict(full):
    st.subheader("Predictive analysis")
    st.caption("Uses all data (both scenarios). Trains on the past and tests on the most recent months.")
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("What do you want to predict?", ["Sales", "Profit", "Units Sold"])
    model_name = c2.selectbox("Model", ["Gradient Boosting", "Random Forest", "Ridge Regression"])
    test_months = c3.slider("Hold out last N months for testing", 3, 24, 12)

    banned = {"Sales", "Profit", "Cost of sales", "Margin %"} | ({"Units Sold"} if target == "Units Sold" else set())
    options = [f for f in CATEGORICAL_FEATURES + NUMERIC_FEATURES if f not in banned]
    defaults = ["Product Name", "Cities", "SalesPerson Name", "Scenario Name", "Year", "Month"]
    if target in ("Sales", "Profit"):
        defaults += ["Units Sold", "Sale price"]
    if target == "Profit":
        defaults += ["Cost price"]
    feats = st.multiselect("Input features", options, default=[f for f in defaults if f in options],
                           key=f"feats_{target}")
    st.caption("Cost of sales is excluded automatically for these targets because it is part of the answer.")
    if "Sale price" in feats and target == "Sales":
        st.info("Sale price and Units Sold are included. Sales is literally Units x Price, so expect a very high score. "
                "Remove them to see how well product, city, rep and season alone explain sales.")

    if st.button("Train and evaluate model", type="primary"):
        if not feats:
            st.error("Pick at least one feature.")
        else:
            with st.spinner("Training..."):
                try:
                    st.session_state["pred"] = train_and_evaluate(full, target, feats, model_name, test_months)
                except Exception as e:
                    st.error(str(e))

    res = st.session_state.get("pred")
    if res is None:
        st.info("Choose settings and click the button to train a model.")
        return

    tgt = res["target"]
    st.markdown(f"### Results: {res['model_name']} predicting {tgt}")
    st.caption(f"Trained on {res['train_rows']:,} rows before {res['test_start'].date()}; "
               f"tested on {res['test_rows']:,} later rows.")
    m, b = res["metrics"], res["baseline"]
    k = st.columns(4)
    k[0].metric("R2 (test)", f"{m['R2']:.3f}")
    k[1].metric("MAE", f"{m['MAE']:,.1f}", f"{(m['MAE'] / b['MAE'] - 1) * 100:.0f}% vs always guessing the average",
                delta_color="inverse")
    k[2].metric("RMSE", f"{m['RMSE']:,.1f}")
    k[3].metric("MAPE", f"{m['MAPE %']:.1f}%")

    t = res["test"]
    g1, g2 = st.columns(2)
    with g1:
        s = t.sample(min(3000, len(t)), random_state=1)
        fig = px.scatter(s, x=tgt, y="Predicted", opacity=0.4, title="Actual vs predicted (test sample)")
        lim = [0, float(max(s[tgt].max(), s["Predicted"].max()))]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect", line=dict(dash="dash", color="red")))
        st.plotly_chart(fig)
    with g2:
        st.plotly_chart(px.bar(res["importance"], x="Importance", y="Feature", orientation="h",
                               title="Permutation importance (drop in score when shuffled)"))

    g3, g4 = st.columns(2)
    with g3:
        mo = t.groupby(t["Date"].dt.to_period("M").dt.to_timestamp())[[tgt, "Predicted"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mo["Date"], y=mo[tgt], name="Actual", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=mo["Date"], y=mo["Predicted"], name="Predicted", mode="lines+markers"))
        fig.update_layout(title="Monthly total in the test period")
        st.plotly_chart(fig)
    with g4:
        st.plotly_chart(px.histogram(t.assign(Error=t["Predicted"] - t[tgt]), x="Error", nbins=60,
                                     title="Prediction error distribution"))
    st.download_button("Download test predictions (CSV)", t.to_csv(index=False).encode("utf-8"),
                       file_name="test_predictions.csv", mime="text/csv")

    # ---- single prediction form
    st.markdown("### Predict a new sale")
    prod_attr = full.drop_duplicates("Product Name").set_index("Product Name")[["Category", "Brand", "Warranty"]]
    rep_attr = full.drop_duplicates("SalesPerson Name").set_index("SalesPerson Name")[["Designation", "Age"]]
    price_tbl = build_price_table(full)

    max_day = pd.Timestamp(f"{full['Date'].max().year + 10}-12-31").date()
    with st.form("whatif"):
        a, b_, c = st.columns(3)
        product = a.selectbox("Product", sorted(prod_attr.index))
        city = b_.selectbox("City", sorted(full["Cities"].unique()))
        rep = c.selectbox("Salesperson", sorted(rep_attr.index))
        d_, e_, f_ = st.columns(3)
        scenario = d_.selectbox("Scenario", sorted(full["Scenario Name"].unique()))
        when = e_.date_input("Sale date (up to 2030)", value=(full["Date"].max() + timedelta(days=30)).date(),
                             min_value=full["Date"].min().date(), max_value=max_day)
        units = f_.number_input("Units sold", min_value=1, max_value=50, value=3)
        override = st.number_input("Sale price override (0 means estimate automatically)", min_value=0, value=0, step=100)
        go_ = st.form_submit_button("Predict")

    if go_:
        out = predict_one(res, full, prod_attr, rep_attr, price_tbl, product, city, rep, scenario, when, units, override)
        pred = out["predict"](units)
        r = st.columns(3)
        r[0].metric(f"Predicted {tgt}", f"{pred:,.0f}")
        r[1].metric("Estimated sale price", f"{out['sale_price']:,.0f}")
        if tgt == "Sales":
            r[2].metric("Units x price check", f"{units * out['sale_price']:,.0f}")
        if out["future"]:
            if out["adjusted"]:
                st.info("This date is after the data. Prices are extended along each product's yearly trend, and the model "
                        "result is scaled by the price ratio because tree models cannot extrapolate. Treat it as an estimate.")
            else:
                st.warning("This date is after the data and Sale price is not a model feature, so tree models will "
                           "under-predict growth. Add Sale price to the features for future dates.")
        if "Units Sold" in res["feats"]:
            sens = pd.DataFrame({"Units": range(1, 6)})
            sens["Predicted"] = [out["predict"](u) for u in sens["Units"]]
            st.plotly_chart(px.bar(sens, x="Units", y="Predicted", title=f"Predicted {tgt} by units sold"))


# ----------------------------------------------------------------------------
# TAB 4: Forecasting
# ----------------------------------------------------------------------------
def monthly_series(df, metric):
    s = df.groupby("YearMonth")[metric].sum().sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(idx, fill_value=0.0).astype(float)


def _future_index(y, h):
    return pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def _trend_growth(n, h):
    """Prediction interval growth factor for extrapolating a linear trend h steps ahead."""
    t = np.arange(n, n + h)
    tbar = (n - 1) / 2
    sxx = np.sum((np.arange(n) - tbar) ** 2)
    return np.sqrt(1 + 1 / n + (t - tbar) ** 2 / sxx)


def fc_seasonal_naive(y, h):
    last = y.values[-12:]
    mean = np.array([last[i % 12] for i in range(h)])
    sd = float(np.std(y.values[12:] - y.values[:-12]))
    band = 1.96 * sd * np.sqrt(np.ceil((np.arange(h) + 1) / 12))
    return mean, mean - band, mean + band


def fc_trend_season_ridge(y, h):
    scale = float(y.mean())
    n = len(y)

    def design(t, months):
        return np.column_stack([t] + [(months == m).astype(float) for m in range(2, 13)])

    X = design(np.arange(n), y.index.month.values)
    mdl = Ridge(alpha=0.1).fit(X, y.values / scale)
    sd = float(np.std(y.values / scale - mdl.predict(X))) * scale
    fi = _future_index(y, h)
    mean = mdl.predict(design(np.arange(n, n + h), fi.month.values)) * scale
    band = 1.96 * sd * _trend_growth(n, h)
    return mean, mean - band, mean + band


def fc_gbr_lags(y, h):
    scale = float(y.mean())
    n = len(y)
    yy = y.values / scale
    t = np.arange(n)
    k, b = np.polyfit(t, yy, 1)
    resid = yy - (k * t + b)
    lags = [1, 2, 3, 6, 12]
    X = np.array([[resid[i - l] for l in lags] + [y.index[i].month] for i in range(12, n)])
    Y = resid[12:]
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, random_state=42).fit(X, Y)
    sd = max(float(np.std(Y - gbr.predict(X))) * 1.5, float(np.std(resid[12:]))) * scale
    hist, fi, out = list(resid), _future_index(y, h), []
    for j in range(h):
        row = [hist[-l] for l in lags] + [fi[j].month]
        r = float(gbr.predict(np.array([row]))[0])
        hist.append(r)
        out.append(((k * (n + j) + b) + r) * scale)
    mean = np.array(out)
    band = 1.96 * sd * _trend_growth(n, h)
    return mean, mean - band, mean + band


def fc_holt_winters(y, h):
    scale = float(y.mean())
    m = ExponentialSmoothing(y / scale, trend="add", damped_trend=False, seasonal="add",
                             seasonal_periods=12, initialization_method="estimated").fit()
    mean = np.asarray(m.forecast(h)) * scale
    sd = float(np.std(m.resid)) * scale * 1.2
    band = 1.96 * sd * np.sqrt(1 + np.arange(h) / 12.0)
    return mean, mean - band, mean + band


def fc_sarima(y, h):
    scale = float(y.mean())
    m = SARIMAX(y / scale, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pr = m.get_forecast(h)
    ci = pr.conf_int(alpha=0.05)
    return (np.asarray(pr.predicted_mean) * scale,
            np.asarray(ci.iloc[:, 0]) * scale, np.asarray(ci.iloc[:, 1]) * scale)


def get_models():
    models = {
        "Seasonal Naive (baseline)": fc_seasonal_naive,
        "Trend + Seasonality (Ridge)": fc_trend_season_ridge,
        "Gradient Boosting (detrended lags)": fc_gbr_lags,
    }
    if HAS_SM:
        models["Holt-Winters"] = fc_holt_winters
        models["SARIMA"] = fc_sarima
    return models


def run_forecast(y, model_names, horizon, backtest_months):
    models = get_models()
    train, test = y.iloc[:-backtest_months], y.iloc[-backtest_months:]
    rows, back, final, errors = [], {}, {}, {}
    for name in model_names:
        fn = models[name]
        try:
            bm, _, _ = fn(train, backtest_months)
            m = reg_metrics(test.values, bm)
            rows.append({"Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE %": m["MAPE %"]})
            back[name] = bm
            mean, lo, hi = fn(y, horizon)
            final[name] = {"index": _future_index(y, horizon), "mean": mean, "lo": lo, "hi": hi}
        except Exception as e:
            errors[name] = str(e)
    table = pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True) if rows else pd.DataFrame()
    return {"table": table, "back": back, "final": final, "errors": errors,
            "y": y, "test": test, "horizon": horizon}


def tab_forecast(full):
    st.subheader("Forecasting")
    st.caption("Aggregates the data to monthly totals, backtests every model on the latest months, then forecasts up to 2030 (change the end year below).")
    c1, c2, c3 = st.columns(3)
    metric = c1.selectbox("Forecast", ["Sales", "Profit", "Units Sold"], key="fc_metric")
    scen = c2.radio("Scenario", sorted(full["Scenario Name"].unique()), horizontal=True, key="fc_scen",
                    index=sorted(full["Scenario Name"].unique()).index("Actual") if "Actual" in set(full["Scenario Name"]) else 0)
    last_date = full["Date"].max()
    years = list(range(last_date.year + 1, last_date.year + 11))
    end_year = c3.selectbox("Forecast until end of", years, index=len(years) - 1, key="fc_end")
    horizon = (end_year - last_date.year) * 12 + (12 - last_date.month)
    c4, c5 = st.columns(2)
    cities = c4.multiselect("Cities", sorted(full["Cities"].unique()), default=sorted(full["Cities"].unique()), key="fc_city")
    cats = c5.multiselect("Categories", sorted(full["Category"].unique()), default=sorted(full["Category"].unique()), key="fc_cat")
    c6, c7 = st.columns(2)
    backtest_months = c6.slider("Backtest on the last N months", 3, 18, 12, key="fc_bt")
    models = get_models()
    chosen = c7.multiselect("Models", list(models), default=list(models), key="fc_models")
    st.caption("Seasonal Naive repeats the last 12 months, so it stays flat. It is only a benchmark, not a 2030 forecast.")
    if not HAS_SM:
        st.info("Install statsmodels (pip install statsmodels) to also get Holt-Winters and SARIMA.")

    if st.button("Run forecast", type="primary"):
        d = full[(full["Scenario Name"] == scen) & full["Cities"].isin(cities) & full["Category"].isin(cats)]
        if d.empty or not chosen:
            st.error("Nothing to forecast. Check the filters and models.")
        else:
            y = monthly_series(d, metric)
            if len(y) - backtest_months < 24:
                st.error(f"Only {len(y)} months available. Need at least 24 months before the backtest window.")
            else:
                with st.spinner("Fitting models..."):
                    res = run_forecast(y, chosen, horizon, backtest_months)
                res["metric"], res["scen"] = metric, scen
                st.session_state["fc"] = res

    res = st.session_state.get("fc")
    if res is None:
        st.info("Pick your scope and click Run forecast.")
        return
    if res["errors"]:
        for n, e in res["errors"].items():
            st.warning(f"{n} failed: {e}")
    if res["table"].empty:
        return

    y, metric = res["y"], res["metric"]
    st.markdown("#### Backtest accuracy (lower is better)")
    st.dataframe(res["table"].round(2))
    best = res["table"].iloc[0]["Model"]
    st.success(f"Best model on the backtest: **{best}** "
               f"(MAPE {res['table'].iloc[0]['MAPE %']:.1f}%). Always compare against the seasonal naive baseline.")

    show = st.selectbox("Show confidence band for", list(res["final"]), index=list(res["final"]).index(best) if best in res["final"] else 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y.values, name="History", mode="lines", line=dict(color="black")))
    for n, r in res["final"].items():
        fig.add_trace(go.Scatter(x=r["index"], y=r["mean"], name=n, mode="lines", line=dict(width=3 if n == show else 1.5)))
    r = res["final"][show]
    fig.add_trace(go.Scatter(x=list(r["index"]) + list(r["index"])[::-1], y=list(r["hi"]) + list(r["lo"])[::-1],
                             fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(width=0),
                             name=f"95% band ({show})", hoverinfo="skip"))
    fig.update_layout(title=f"{res['scen']} {metric}: monthly history and forecast", yaxis_title=metric)
    st.plotly_chart(fig)

    with st.expander("Backtest chart: how each model did on months it had not seen"):
        fb = go.Figure()
        fb.add_trace(go.Scatter(x=res["test"].index, y=res["test"].values, name="Actual", mode="lines+markers",
                                line=dict(color="black")))
        for n, bm in res["back"].items():
            fb.add_trace(go.Scatter(x=res["test"].index, y=bm, name=n, mode="lines"))
        st.plotly_chart(fb)

    tbl = pd.DataFrame({n: pd.Series(r["mean"], index=r["index"]) for n, r in res["final"].items()})
    tbl.index.name = "Month"
    st.markdown("#### Forecast table")
    st.dataframe(tbl.round(0))
    yr = tbl.groupby(tbl.index.year).sum()
    st.markdown("#### Forecast totals by calendar year (partial years only cover forecasted months)")
    st.dataframe(yr.round(0))
    st.download_button("Download forecast (CSV)", tbl.round(2).to_csv().encode("utf-8"),
                       file_name="forecast.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Regional Sales Intelligence", page_icon="📊", layout="wide")
    st.title("Regional Sales Intelligence")
    st.caption("Analysis, visualization, prediction and forecasting for the regional sales dataset.")

    up = st.sidebar.file_uploader("Upload sales file (.xlsx or .csv)", type=["xlsx", "csv"])
    try:
        if up is not None:
            full = load_data(up.getvalue(), up.name)
        elif os.path.exists(DEFAULT_FILE):
            with open(DEFAULT_FILE, "rb") as f:
                full = load_data(f.read(), DEFAULT_FILE)
        else:
            st.info(f"Upload your file in the sidebar, or place {DEFAULT_FILE} next to app.py.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    filtered, base = sidebar_filters(full)
    t1, t2, t3, t4 = st.tabs(["Data analysis", "Visualization", "Predict sales", "Forecast"])
    with t1:
        tab_analysis(filtered, base, full)
    with t2:
        tab_viz(filtered, base)
    with t3:
        tab_predict(full)
    with t4:
        tab_forecast(full)


if __name__ == "__main__":
    main()
