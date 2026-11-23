from hashlib import sha256
from html import escape
from pathlib import Path
from inspect import signature

import altair as alt
import pandas as pd
import streamlit as st

from chat_ui import render_chat_panel
from financial_copilot import FinancialCopilot

from data_processing import (
    REQUIRED_COLUMNS,
    anomaly_trend,
    calculate_metrics,
    calculate_var,
    detect_anomalies,
    load_data,
    monthly_growth,
    monthly_revenue,
    portfolio_performance,
    revenue_forecast,
    rolling_volatility,
    top_revenue,
)


st.set_page_config(page_title="Financial Decision Support Dashboard", layout="wide")

TEAL = "#087f72"
BLUE = "#3875a0"
AMBER = "#a65b16"
RED = "#b74748"
INK = "#173247"
MUTED = "#536979"
PALETTE = [TEAL, BLUE, "#6b63a6", "#b97135", "#2a8f9e", "#647d4c", "#a35d79", "#8798a4"]
FULL_WIDTH = (
    {"width": "stretch"}
    if "width" in signature(st.altair_chart).parameters
    else {"use_container_width": True}
)


def apply_style():
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stHeader"] { background: #f4f7fa; color: #173247; }
        [data-testid="stMainBlockContainer"], .main .block-container {
            max-width: 1410px;
            padding-top: 3.3rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: #eef4f6;
            border-right: 1px solid #dce6eb;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
            color: #173247;
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #dce6ec;
        }
        div[class*="st-key-panel_"] {
            background: #fff;
            border-color: #dce6ec;
            border-radius: 14px;
        }
        [data-testid="stMain"] h2, [data-testid="stMain"] h3 {
            color: #173247;
            letter-spacing: -.023em;
        }
        [data-testid="stMain"] h2 { font-size: 1.55rem; }
        [data-testid="stMain"] h3 { font-size: 1.12rem; }
        [data-testid="stCaptionContainer"] p {
            color: #536979 !important;
        }
        [data-testid="stTabs"] [role="tablist"] {
            gap: .28rem;
            flex-wrap: wrap;
            padding: .28rem;
            border: 1px solid #dce6ec;
            border-radius: 11px;
            background: #fff;
        }
        [data-testid="stTabs"] [role="tab"] {
            border-radius: 7px;
            padding: .38rem .75rem;
            color: #425d70;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: #dff2ee;
            color: #075b51;
            font-weight: 650;
        }
        [data-testid="stTabs"] .react-aria-SelectionIndicator,
        [data-testid="stTabs"] [data-baseweb="tab-highlight"],
        [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
        .fd-hero {
            position: relative;
            isolation: isolate;
            overflow: hidden;
            padding: clamp(1.25rem, 3vw, 2rem) clamp(1.15rem, 3.5vw, 2.5rem);
            border-radius: 18px;
            background: linear-gradient(112deg, #102a43 0%, #143d50 60%, #086358 100%);
            color: #fff;
            box-shadow: 0 8px 22px rgba(16, 42, 67, .10);
        }
        .fd-hero h1 {
            font-size: clamp(1.7rem, 3vw, 2.7rem);
            line-height: 1.14;
            letter-spacing: -.035em;
            color: #fff;
            margin: 0;
            padding: 0;
            font-weight: 730;
        }
        .fd-hero p {
            margin: .6rem 0 0;
            max-width: 36rem;
            color: #d3e8e9;
            font-size: .97rem;
        }
        .fd-eyebrow {
            color: #a1e6d7;
            font-size: .70rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: .58rem;
        }
        .fd-motif {
            position: absolute;
            z-index: -1;
            right: 1rem;
            bottom: .4rem;
            width: 30%;
            min-width: 13rem;
            opacity: .36;
        }
        .fd-brand {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin: .2rem 0 1.25rem;
            color: #173247;
            font-size: 1rem;
            font-weight: 690;
        }
        .fd-brand-icon {
            display: grid;
            place-items: center;
            width: 2.2rem;
            height: 2.2rem;
            background: #086b60;
            border-radius: 9px;
            color: #fff;
        }
        .fd-period {
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: .25rem .7rem;
            color: #536979;
            font-size: .83rem;
            padding: .5rem 0 .38rem;
        }
        .fd-period strong { color: #173247; }
        .fd-kpis {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
            gap: .7rem;
            margin: .15rem 0 .25rem;
        }
        .fd-kpi {
            padding: .85rem 1rem .9rem;
            background: #fff;
            border: 1px solid #dce6ec;
            border-top: 3px solid #0d8174;
            border-radius: 12px;
        }
        .fd-kpi-label { color: #536979; font-size: .79rem; font-weight: 650; }
        .fd-kpi-value {
            color: #102a43;
            font-size: clamp(1.3rem, 2.3vw, 1.85rem);
            font-weight: 730;
            letter-spacing: -.035em;
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
            margin-top: .28rem;
            overflow-wrap: anywhere;
        }
        div.st-key-dashboard_home {
            max-width: 73rem;
            margin: clamp(.25rem, 2vw, 1.5rem) auto 0;
        }
        div.st-key-home_intro, div.st-key-home_preview {
            min-height: 26rem;
            padding: clamp(1.4rem, 2.7vw, 2.4rem);
            border-radius: 22px;
        }
        div.st-key-home_intro {
            justify-content: center;
            background: radial-gradient(ellipse at 100% 0%, #235953 0%, transparent 70%), #123433;
            color: #fff;
            box-shadow: 0 16px 36px rgba(18, 52, 51, .14);
        }
        .fd-home-title {
            margin: 0;
            padding: 0;
            font-size: clamp(2.5rem, 4.65vw, 4.3rem);
            font-weight: 730;
            line-height: .99;
            letter-spacing: -.05em;
            color: #fff;
        }
        .fd-home-copy {
            max-width: 23rem;
            margin: 1.15rem 0 .65rem;
            color: #d0e6e1;
            font-size: 1.04rem;
            line-height: 1.6;
        }
        div.st-key-data_source_actions button { min-height: 2.75rem; border-radius: 9px; }
        div.st-key-home_intro button[data-testid="stBaseButton-primary"] {
            color: #113731;
            background: #a2e8d3;
            border-color: #a2e8d3;
            font-weight: 650;
        }
        div.st-key-home_intro button[data-testid="stBaseButton-primary"]:hover {
            background: #c0f4e4;
            border-color: #c0f4e4;
        }
        div.st-key-home_intro button[data-testid="stBaseButton-secondary"] {
            color: #f0faf7;
            background: transparent;
            border-color: #759891;
        }
        div.st-key-home_intro button[data-testid="stBaseButton-secondary"]:hover {
            background: #28534d;
            border-color: #a8d0c7;
        }
        div.st-key-home_preview {
            background: #fff;
            border: 1px solid #d9e7e5;
            box-shadow: 0 12px 32px rgba(20, 54, 52, .07);
        }
        .fd-home-preview h2 { margin: 0; padding: 0; }
        .fd-home-dates { margin: .25rem 0 0; color: #536979; font-size: .87rem; }
        .fd-home-metrics {
            display: grid;
            grid-template-columns: 1.45fr 1fr .85fr;
            gap: .85rem;
            margin: 1rem 0 0;
            padding: .85rem 0;
            border-block: 1px solid #e5edeb;
        }
        .fd-home-metrics dt { color: #536979; font-size: .8rem; }
        .fd-home-metrics dd {
            margin: .15rem 0 0;
            color: #143b36;
            font-size: clamp(1.12rem, 2vw, 1.65rem);
            font-variant-numeric: tabular-nums;
            font-weight: 710;
            letter-spacing: -.04em;
        }
        [data-testid="stMainBlockContainer"]:has(.st-key-copilot_page) { max-width: 64rem; }
        .st-key-copilot_page h1 { letter-spacing: -.035em; }
        .st-key-copilot_page [data-testid="stChatMessage"] {
            border-radius: 14px;
        }
        [data-testid="stMain"]:has(.st-key-copilot_page) [data-testid="stBottomBlockContainer"] {
            max-width: 64rem;
        }
        .st-key-copilot_chat_list [data-testid="stButton"] button {
            justify-content: flex-start;
            text-align: left;
        }
        .st-key-copilot_chat_list button[data-testid="stBaseButton-primary"] {
            color: #075b51;
            background: #dff2ee;
            border-color: #b9dcd5;
        }
        @media (max-width: 820px) {
            div.st-key-dashboard_home [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
            div.st-key-dashboard_home [data-testid="stColumn"] { flex: 1 1 100%; min-width: 0; }
            div.st-key-home_intro, div.st-key-home_preview { min-height: 0; }
        }
        @media (max-width: 650px) {
            .fd-hero { border-radius: 13px; }
            .fd-motif { opacity: .14; }
            [data-testid="stTabs"] [role="tab"] { padding: .3rem .45rem; }
        }
        @media (max-width: 450px) {
            .fd-home-metrics { grid-template-columns: 1fr 1fr; }
            .fd-home-metrics div:first-child { grid-column: 1 / -1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <div class="fd-hero">
            <svg class="fd-motif" viewBox="0 0 340 130" fill="none"
                 aria-hidden="true" focusable="false">
                <path d="M0 110H340M0 80H340M0 50H340M0 20H340
                         M30 0V130M90 0V130M150 0V130M210 0V130M270 0V130M330 0V130"
                      stroke="#bcf2e5" stroke-opacity=".30"/>
                <path d="M8 113L57 89L105 98L153 62L201 75L253 39L300 45L332 11"
                      stroke="#a6f0dc" stroke-width="2.5" stroke-linecap="round"
                      stroke-linejoin="round"/>
                <circle cx="332" cy="11" r="4" fill="#caf9ed"/>
            </svg>
            <div class="fd-eyebrow">Finance Analytics</div>
            <h1>Financial Decision Support Dashboard</h1>
            <p>Explore sales performance, revenue risk, forecasts, and unusual transactions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand():
    st.markdown(
        """
        <div class="fd-brand">
            <span class="fd-brand-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" width="19" height="19" fill="none">
                    <path d="M4 18V13M10 18V8M16 18V4M22 18H2"
                          stroke="currentColor" stroke-width="2"
                          stroke-linecap="round"/>
                </svg>
            </span>
            <span>Financial Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_cards(items):
    cards = "".join(
        '<div class="fd-kpi">'
        f'<div class="fd-kpi-label">{escape(label)}</div>'
        f'<div class="fd-kpi-value">{escape(value)}</div>'
        '</div>'
        for label, value in items
    )
    st.markdown(f'<div class="fd-kpis">{cards}</div>', unsafe_allow_html=True)


def panel(name):
    return st.container(border=True, key=f"panel_{name}")


def show_chart(chart, height=255):
    chart = (
        chart.properties(height=height)
        .configure(background="#ffffff", font="sans-serif")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#e7eef2",
            domain=False,
            tickColor="#dce6ec",
            labelColor=MUTED,
            labelFontSize=11,
            titleColor=INK,
            titlePadding=12,
            labelPadding=7,
        )
        .configure_legend(
            orient="top", labelColor=INK, titleColor=INK, labelFontSize=11
        )
    )
    st.altair_chart(chart, theme=None, **FULL_WIDTH)


def month_axis():
    return alt.X("sale_date:T", title="Month", axis=alt.Axis(format="%b %y", tickCount=7))


def month_bar_axis():
    return alt.X(
        "yearmonth(sale_date):O",
        title="Month",
        axis=alt.Axis(format="%b %y", labelAngle=0, labelOverlap="greedy"),
    )


def month_tooltip():
    return alt.Tooltip("sale_date:T", title="Month", format="%b %Y")


def monthly_line(data, field, label, color=TEAL):
    return (
        alt.Chart(data)
        .mark_line(
            color=color, strokeWidth=2.5,
            point=alt.OverlayMarkDef(filled=True, color=color, size=28),
        )
        .encode(
            x=month_axis(),
            y=alt.Y(f"{field}:Q", title=label, scale=alt.Scale(zero=True)),
            tooltip=[month_tooltip(), alt.Tooltip(f"{field}:Q", title=label, format=",.2f")],
        )
    )


def select_data_mode(mode):
    st.session_state["data_mode"] = mode


def source_actions():
    with st.container(key="data_source_actions", horizontal=True, gap="small"):
        if st.button(
            "Explore demo", type="primary", icon=":material/arrow_forward:", icon_position="right",
        ):
            select_data_mode("demo")
            st.switch_page(COPILOT_PAGE)
        st.button("Use my data", icon=":material/upload_file:", on_click=select_data_mode, args=("upload",))


def show_home(demo_csv):
    with st.container(key="dashboard_home"):
        introduction, preview = st.columns([1, 1.2], gap="medium")
        with introduction, st.container(key="home_intro"):
            st.markdown(
                '<h1 class="fd-home-title">Financial<br>Dashboard</h1>'
                '<p class="fd-home-copy">Explore sales and forecasts, or ask the AI Financial Copilot '
                'about invoices and unusual transactions.</p>',
                unsafe_allow_html=True,
            )
            source_actions()

        with preview, st.container(key="home_preview"):
            try:
                sales = load_data(demo_csv)
            except (OSError, ValueError):
                st.info("The demo sales preview is unavailable. You can still upload your own CSV.")
                return
            if sales.empty:
                st.info("The demo CSV has no valid sales to preview.")
                return

            start = sales["sale_date"].min().date()
            end = sales["sale_date"].max().date()
            metrics = calculate_metrics(sales)
            st.markdown(
                '<div class="fd-home-preview">'
                '<h2>Demo sales overview</h2>'
                f'<p class="fd-home-dates">{start:%b %Y} – {end:%b %Y}</p>'
                '<dl class="fd-home-metrics">'
                f'<div><dt>Total revenue</dt><dd>{metrics["total_revenue"]:,.2f}</dd></div>'
                f'<div><dt>Transactions</dt><dd>{metrics["number_of_transactions"]:,}</dd></div>'
                f'<div><dt>Customers</dt><dd>{metrics["number_of_customers"]:,}</dd></div>'
                '</dl></div>',
                unsafe_allow_html=True,
            )
            monthly = monthly_revenue(sales, start, end)
            line = monthly_line(monthly, "revenue", "Revenue").encode(
                x=alt.X("sale_date:T", title=None, axis=alt.Axis(format="%b %y", tickCount=4)),
                y=alt.Y("revenue:Q", title="Monthly revenue", axis=alt.Axis(format="~s", tickCount=4)),
            )
            area = line.mark_area(color=TEAL, opacity=.09)
            show_chart(area + line, height=174)


def show_ranking(data, field, label, title, color):
    with panel(field):
        st.subheader(title)
        ranking = top_revenue(data, field)
        if ranking.empty:
            st.info(f"No {label.lower()} values are available for this period.")
            return
        chart = (
            alt.Chart(ranking)
            .mark_bar(color=color, cornerRadiusEnd=4)
            .encode(
                x=alt.X("revenue:Q", title="Revenue", axis=alt.Axis(tickCount=5, format="~s")),
                y=alt.Y(f"{field}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=170)),
                tooltip=[
                    alt.Tooltip(f"{field}:N", title=label),
                    alt.Tooltip("revenue:Q", title="Revenue", format=",.2f"),
                ],
            )
        )
        show_chart(chart, height=max(140, 24 * len(ranking)))
        st.caption(f"Ranked by revenue. Sales with missing {label.lower()} are excluded.")


def show_business(data, monthly, growth):
    st.header("Business Performance")
    metrics = calculate_metrics(data)
    kpi_cards([
        ("Total Revenue", f"{metrics['total_revenue']:,.2f}"),
        ("Customers", f"{metrics['number_of_customers']:,}"),
        ("Transactions", f"{metrics['number_of_transactions']:,}"),
        ("Average Transaction Value", f"{metrics['average_transaction_value']:,.2f}"),
    ])

    with panel("monthly_revenue"):
        st.subheader("Revenue Over Time")
        show_chart(monthly_line(monthly, "revenue", "Monthly revenue"))
        st.caption("Revenue in the selected period. The first and last months can be partial.")

    with panel("revenue_growth"):
        st.subheader("Revenue Growth")
        valid_growth = growth.dropna(subset=["growth_pct"])
        if valid_growth.empty:
            st.info("Select at least two full months with positive revenue in the earlier month.")
        else:
            chart = (
                alt.Chart(valid_growth)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=month_bar_axis(),
                    y=alt.Y("growth_pct:Q", title="Growth (%)"),
                    color=alt.condition(alt.datum.growth_pct < 0, alt.value(RED), alt.value(TEAL)),
                    tooltip=[
                        month_tooltip(),
                        alt.Tooltip("growth_pct:Q", title="Growth (%)", format="+,.1f"),
                    ],
                )
            )
            show_chart(chart, height=210)
            st.caption("Percentage change versus the previous full month. Red bars show a decline.")

    product, customer = st.columns(2)
    with product:
        show_ranking(data, "product_name", "Product name", "Top 10 Products by Revenue", TEAL)
    with customer:
        show_ranking(data, "customer_id", "Customer ID", "Top 10 Customers by Revenue", BLUE)

    with st.expander("Data Preview"):
        columns = ["sale_id", "sale_date", "customer_id", "product_name", "category", "revenue"]
        st.dataframe(
            data[columns].head(10),
            hide_index=True,
            **FULL_WIDTH,
            column_config={
                "sale_date": st.column_config.DateColumn("Sale date", format="YYYY-MM-DD"),
                "revenue": st.column_config.NumberColumn("Revenue", format="%.2f"),
            },
        )


def show_risk(data, growth, start_date, end_date):
    st.header("Risk Analytics")

    with panel("volatility"):
        st.subheader("Revenue Growth Volatility")
        risk = rolling_volatility(growth)
        if risk["volatility"].notna().any():
            show_chart(monthly_line(risk, "volatility", "Volatility (percentage points)", BLUE))
            st.caption(
                "Six-month rolling standard deviation of monthly revenue growth. "
                "Higher values mean less stable growth."
            )
        else:
            st.info("Select at least seven full months with consecutive valid growth observations.")

    with panel("portfolio"):
        st.subheader("Portfolio Performance")
        portfolio, summary = portfolio_performance(data, start_date, end_date)
        category_order = summary["category"].tolist()
        chart = (
            alt.Chart(portfolio)
            .mark_bar()
            .encode(
                x=month_bar_axis(),
                y=alt.Y("revenue:Q", title="Monthly revenue", stack="zero"),
                color=alt.Color(
                    "category:N",
                    title="Product category",
                    scale=alt.Scale(domain=category_order, range=PALETTE[:len(category_order)]),
                ),
                tooltip=[
                    month_tooltip(),
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("revenue:Q", title="Revenue", format=",.2f"),
                ],
            )
        )
        show_chart(chart)
        st.caption(
            "Product categories make up the business portfolio here. "
            "The chart shows each category's contribution to monthly sales revenue."
        )
        st.dataframe(
            summary,
            hide_index=True,
            **FULL_WIDTH,
            column_config={
                "category": "Category",
                "revenue": st.column_config.NumberColumn("Total revenue", format="%.2f"),
                "share_pct": st.column_config.NumberColumn("Revenue share (%)", format="%.1f"),
            },
        )

    with panel("value_at_risk"):
        st.subheader("Value at Risk (95%)")
        var = calculate_var(growth)
        if var is None:
            st.info(
                "VaR needs 12 valid monthly growth observations. "
                "Select a longer period covering at least 13 full months."
            )
        else:
            kpi_cards([
                ("Monthly Revenue VaR", f"{var['var_pct']:.2f}%"),
                ("Estimated Revenue Shortfall", f"{var['shortfall']:,.2f}"),
            ])
            baseline_month = var["baseline_month"].strftime("%b %Y")
            st.caption(
                f"Based on the 5th percentile of {var['observations']} historical monthly "
                f"growth observations. The amount is relative to {baseline_month} revenue "
                f"of {var['baseline_revenue']:,.2f}. This is an estimate; worse outcomes are possible."
            )


def show_predictions(growth):
    st.header("Predictive Analytics")
    results = revenue_forecast(growth)
    if results is None:
        st.info("Select at least 12 full calendar months to train and test the forecast.")
        return

    with panel("forecast"):
        st.subheader("Revenue Forecast")
        actual = growth[["sale_date", "revenue"]].assign(series="Observed")
        future = results["future"].assign(series="Forecast")
        forecast_data = pd.concat([actual, future], ignore_index=True)
        series_order = ["Observed", "Forecast"]
        chart = (
            alt.Chart(forecast_data)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=month_axis(),
                y=alt.Y("revenue:Q", title="Monthly revenue", scale=alt.Scale(zero=True)),
                color=alt.Color(
                    "series:N", title=None,
                    scale=alt.Scale(domain=series_order, range=[TEAL, AMBER]),
                ),
                strokeDash=alt.StrokeDash(
                    "series:N", title=None,
                    scale=alt.Scale(domain=series_order, range=[[1, 0], [6, 4]]),
                ),
                tooltip=[
                    month_tooltip(),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("revenue:Q", title="Revenue", format=",.2f"),
                ],
            )
        )
        show_chart(chart, height=275)
        st.caption(
            "The model predicts three months after the last complete month in the selected "
            "period. It uses the previous month's revenue and the average of the past three months."
        )

    with panel("predictions"):
        st.subheader("ML Predictions")
        st.caption(
            f"Ridge regression trains on the first {results['train_count']} months and is "
            f"tested on the last {results['test_count']} months. Each test prediction uses only "
            "values observed before the month being predicted."
        )
        st.dataframe(
            results["test"],
            hide_index=True,
            **FULL_WIDTH,
            column_config={
                "sale_date": st.column_config.DateColumn("Test month", format="MMM YYYY"),
                "actual": st.column_config.NumberColumn("Actual revenue", format="%.2f"),
                "prediction": st.column_config.NumberColumn("ML prediction", format="%.2f"),
                "baseline": st.column_config.NumberColumn("Baseline prediction", format="%.2f"),
                "error": st.column_config.NumberColumn("Prediction minus actual", format="%.2f"),
            },
        )

    with panel("model_performance"):
        st.subheader("Model Performance")
        errors = pd.DataFrame({
            "Model": ["Ridge regression", "Last month baseline"],
            "MAE": [results["mae"], results["baseline_mae"]],
            "RMSE": [results["rmse"], results["baseline_rmse"]],
        })
        st.dataframe(
            errors,
            hide_index=True,
            **FULL_WIDTH,
            column_config={
                "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
            },
        )
        st.caption(
            "Lower errors are better. MAE is the average absolute error; RMSE gives more "
            "weight to large errors. The baseline predicts the previous month's revenue."
        )
        if results["mae"] > results["baseline_mae"]:
            st.caption("The baseline had a lower MAE on this test period.")
        elif results["mae"] < results["baseline_mae"]:
            st.caption("Ridge regression had a lower MAE on this test period.")


def show_anomalies(data, start_date, end_date):
    st.header("Anomaly Detection")
    flagged = detect_anomalies(data)
    if flagged is None:
        st.info("Select a period with at least 20 valid transactions to detect anomalies.")
        return

    kpi_cards([("Number of Detected Anomalies", f"{len(flagged):,}")])
    st.caption(
        f"Flagged {len(flagged):,} of {len(data):,} selected transactions. The IQR rule "
        "flags amounts outside the middle 50% range extended by 1.5 times its width on "
        "each side. Categories with fewer than 20 rows use the selected transactions as "
        "the comparison group. Flags identify transactions to review."
    )

    with panel("flagged_transactions"):
        st.subheader("Suspicious Transactions")
        if flagged.empty:
            st.info("No transactions were flagged in this period.")
        else:
            columns = [
                "sale_date", "sale_id", "customer_id", "product_name",
                "category", "revenue", "threshold", "reason", "comparison",
            ]
            st.dataframe(
                flagged[columns],
                hide_index=True,
                **FULL_WIDTH,
                column_config={
                    "sale_date": st.column_config.DateColumn("Sale date", format="YYYY-MM-DD"),
                    "sale_id": "Sale ID",
                    "customer_id": "Customer ID",
                    "product_name": "Product",
                    "category": "Category",
                    "revenue": st.column_config.NumberColumn("Revenue", format="%.2f"),
                    "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                    "reason": "Flag reason",
                    "comparison": "Comparison group",
                },
            )

    with panel("anomaly_trend"):
        st.subheader("Anomaly Trends")
        trend = anomaly_trend(flagged, start_date, end_date)
        chart = (
            alt.Chart(trend)
            .mark_bar(color=AMBER, cornerRadiusEnd=3)
            .encode(
                x=month_bar_axis(),
                y=alt.Y("anomalies:Q", title="Flagged transactions", axis=alt.Axis(tickMinStep=1)),
                tooltip=[month_tooltip(), alt.Tooltip("anomalies:Q", title="Flagged transactions")],
            )
        )
        show_chart(chart, height=210)
        st.caption("Number of flagged transactions by month within the selected dates.")


def demo_csv_path():
    folder = Path(__file__).resolve().parent
    path = folder / "clean_financial_data.csv"
    return path if path.is_file() else folder.parent / path.name


def save_upload():
    uploaded_file = st.session_state.get("sales_file")
    st.session_state.pop("uploaded_sales", None)
    st.session_state.pop("sales_file_error", None)
    if uploaded_file is None:
        return
    try:
        data = load_data(uploaded_file)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.session_state["sales_file_error"] = f"Could not read the CSV: {error}"
        return
    if data.empty:
        st.session_state["sales_file_error"] = "No valid transactions were found in the CSV."
        return
    source = f"csv:{sha256(uploaded_file.getbuffer()).hexdigest()}"
    st.session_state["uploaded_sales"] = (data, source)


def active_sales(is_demo):
    if not is_demo:
        return st.session_state.get("uploaded_sales")
    try:
        data = load_data(demo_csv_path())
    except (OSError, ValueError):
        st.error("The demo sales CSV is unavailable. You can upload your own sales CSV.")
        return None
    if data.empty:
        st.warning("The demo CSV has no valid sales.")
        return None
    return data, "demo"


def reporting_period(data, source, *, expanded=False):
    first_date = data["sale_date"].min().date()
    last_date = data["sale_date"].max().date()
    saved_periods = st.session_state.setdefault("reporting_periods", {})
    start, end = saved_periods.get(source, (first_date, last_date))
    date_key = f"{source}:{first_date}:{last_date}"
    with st.sidebar.expander("Reporting period", expanded=expanded):
        start = st.date_input(
            "Start date", max(first_date, min(start, last_date)),
            min_value=first_date, max_value=last_date, key=f"start_{date_key}",
        )
        end = st.date_input(
            "End date", max(first_date, min(end, last_date)),
            min_value=first_date, max_value=last_date, key=f"end_{date_key}",
        )
    saved_periods[source] = start, end
    if start > end:
        st.warning("Start date must be on or before the end date.")
        return None
    return start, end


def dashboard():
    mode = st.session_state.get("data_mode")
    if mode is None:
        show_home(demo_csv_path())
        return

    hero()
    source_actions()
    with st.sidebar:
        sidebar_brand()
    is_demo = mode == "demo"
    if not is_demo:
        st.file_uploader("Upload your sales CSV", type=["csv"], key="sales_file", on_change=save_upload)
        st.download_button(
            "Download CSV template",
            data=",".join(REQUIRED_COLUMNS) + "\nS001,2025-01-01,120,C001,P001,Desk,Office\n",
            file_name="sales_template.csv",
            mime="text/csv",
            on_click="ignore",
        )
        if error := st.session_state.get("sales_file_error"):
            st.error(error)

    current = active_sales(is_demo)
    if current is None:
        return
    all_data, source = current
    period = reporting_period(all_data, source, expanded=True)
    if period is None:
        return
    start_date, end_date = period
    data = all_data[all_data["sale_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    if data.empty:
        st.info("No transactions in this period. Try a wider date range.")
        return

    source_label = "Demo ERP data" if is_demo else "Uploaded sales CSV"
    st.markdown(
        '<div class="fd-period"><strong>'
        f'{start_date:%d %b %Y} – {end_date:%d %b %Y}'
        f'</strong><span>{source_label}</span><span>{len(data):,} transactions in view</span>'
        '<span>Amounts shown in the CSV currency</span></div>',
        unsafe_allow_html=True,
    )

    monthly = monthly_revenue(data, start_date, end_date)
    growth = monthly_growth(data, start_date, end_date)
    business, risk, predictions, anomalies = st.tabs([
        "Business Performance", "Risk Analytics", "Predictive Analytics", "Anomaly Detection",
    ])
    with business:
        show_business(data, monthly, growth)
    with risk:
        show_risk(data, growth, start_date, end_date)
    with predictions:
        show_predictions(growth)
    with anomalies:
        show_anomalies(data, start_date, end_date)


def copilot_workspace():
    is_demo = st.session_state.setdefault("data_mode", "demo") == "demo"
    current = active_sales(is_demo)
    if current is None:
        st.title("AI Financial Copilot", anchor=False)
        if not is_demo:
            st.info("Upload a sales CSV on the Dashboard to start a chat with your data.")
        return
    data, source = current
    period = reporting_period(data, source)
    if period is None:
        return
    start, end = period
    with st.container(key="copilot_page"):
        try:
            with FinancialCopilot(sales_data=data, use_demo_erp=is_demo) as copilot:
                render_chat_panel(copilot, start_date=start, end_date=end, source=source)
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Financial Copilot could not load data: {error}")


COPILOT_PAGE = st.Page(copilot_workspace, title="AI Financial Copilot", url_path="copilot", icon=":material/insights:")


def main():
    apply_style()
    st.navigation([
        st.Page(dashboard, title="Dashboard", icon=":material/dashboard:", default=True),
        COPILOT_PAGE,
    ]).run()


if __name__ == "__main__":
    main()
