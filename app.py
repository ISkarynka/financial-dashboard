from dataclasses import dataclass
from datetime import date
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

TEAL = "#238779"
BLUE = "#4b85ba"
AMBER = "#b77246"
RED = "#bf6371"
FULL_WIDTH = (
    {"width": "stretch"}
    if "width" in signature(st.altair_chart).parameters
    else {"use_container_width": True}
)


def page_heading(title, report):
    start, end = report.start, report.end
    source = "Demo data" if report.is_demo else "My data"
    st.html(
        '<header class="fd-page-heading">'
        f'<div><h1>{escape(title)}</h1><p class="fd-period">{start:%d %b %Y} – {end:%d %b %Y}</p></div>'
        f'<span class="fd-source">{source}</span></header>'
    )


def kpi_cards(items):
    cards = "".join(
        '<div class="stat">'
        f'<div class="stat-title">{escape(label)}</div>'
        f'<div class="stat-value">{escape(value)}</div>'
        '</div>'
        for label, value in items
    )
    st.html(f'<div class="stats fd-metrics" data-count="{len(items)}">{cards}</div>')


def panel(name):
    return st.container(key=f"panel_{name}")


def show_chart(chart, height=255):
    chart = (
        chart.properties(height=height)
        .configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(domain=False, gridOpacity=.58, labelFontSize=11, titlePadding=9, labelPadding=6)
        .configure_axisX(grid=False)
        .configure_legend(orient="top", labelFontSize=11)
    )
    st.altair_chart(chart, theme="streamlit", **FULL_WIDTH)


def month_axis():
    return alt.X("sale_date:T", title=None, axis=alt.Axis(format="%b %y", tickCount=7))


def month_bar_axis():
    return alt.X(
        "yearmonth(sale_date):O",
        title=None,
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


def show_overview(data, monthly, growth):
    metrics = calculate_metrics(data)
    kpi_cards([
        ("Total revenue", f"{metrics['total_revenue']:,.2f}"),
        ("Transactions", f"{metrics['number_of_transactions']:,}"),
        ("Customers", f"{metrics['number_of_customers']:,}"),
        ("Average sale", f"{metrics['average_transaction_value']:,.2f}"),
    ])

    with st.container(key="report_charts"):
        revenue, change = st.columns([1.65, 1], gap="medium")
        with revenue, panel("monthly_revenue"):
            st.subheader("Revenue over time")
            line = monthly_line(monthly, "revenue", "Monthly revenue").encode(
                y=alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format="~s", tickCount=5)),
            )
            show_chart(line.mark_area(color=TEAL, opacity=.12) + line, height=245)
            st.caption("First and last months may be partial.")
        with change, panel("revenue_growth"):
            st.subheader("Monthly change")
            valid_growth = growth.dropna(subset=["growth_pct"])
            if valid_growth.empty:
                st.info("Select at least two complete months to see the change.")
            else:
                chart = (
                    alt.Chart(valid_growth)
                    .mark_bar(cornerRadiusEnd=3)
                    .encode(
                        x=month_bar_axis(),
                        y=alt.Y("growth_pct:Q", title="Change (%)", axis=alt.Axis(tickCount=5)),
                        color=alt.condition(alt.datum.growth_pct < 0, alt.value(RED), alt.value(TEAL)),
                        tooltip=[month_tooltip(), alt.Tooltip("growth_pct:Q", title="Change (%)", format="+,.1f")],
                    )
                )
                show_chart(chart, height=245)
                st.caption("Compared with the previous complete month.")

    with st.container(key="ranking_charts"):
        product, customer = st.columns(2, gap="medium")
        with product:
            show_ranking(data, "product_name", "Product name", "Top products", TEAL)
        with customer:
            show_ranking(data, "customer_id", "Customer ID", "Top customers", BLUE)

    with st.expander("Transaction preview"):
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
    var = calculate_var(growth)
    flagged = detect_anomalies(data)
    metrics = []
    if var is not None:
        metrics.extend([
            ("95% revenue VaR", f"{var['var_pct']:.1f}%"),
            ("Estimated shortfall", f"{var['shortfall']:,.2f}"),
        ])
    if flagged is not None:
        metrics.append(("Transactions to review", f"{len(flagged):,}"))
    if metrics:
        kpi_cards(metrics)

    with st.container(key="report_charts"):
        volatility_panel, anomalies_panel = st.columns(2, gap="medium")
        with volatility_panel, panel("volatility"):
            st.subheader("Revenue volatility")
            risk = rolling_volatility(growth)
            if risk["volatility"].notna().any():
                show_chart(monthly_line(risk, "volatility", "Volatility (pp)", BLUE), height=218)
                st.caption("Rolling six-month standard deviation of monthly revenue growth.")
            else:
                st.info("Select seven complete months with consecutive growth observations.")
        with anomalies_panel, panel("anomaly_trend"):
            st.subheader("Flags by month")
            if flagged is None:
                st.info("Select a period with at least 20 valid transactions.")
            else:
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
                show_chart(chart, height=218)
                st.caption(f"{len(flagged):,} of {len(data):,} transactions flagged by the IQR rule.")

    if var is not None:
        baseline_month = var["baseline_month"].strftime("%b %Y")
        with st.expander("About the risk estimate"):
            st.write(
                f"The historical estimate uses the 5th percentile of {var['observations']} monthly growth "
                f"observations and {baseline_month} revenue of {var['baseline_revenue']:,.2f}. Future losses can exceed it."
            )
    else:
        st.caption("Revenue VaR needs 12 valid monthly growth observations.")

    if flagged is not None:
        with st.expander("Transactions to review"):
            st.caption(
                "The IQR rule flags amounts beyond 1.5 times the interquartile range. "
                "Categories with fewer than 20 rows use the selected transactions as the comparison group."
            )
            if flagged.empty:
                st.info("No transactions were flagged in this period.")
            else:
                columns = [
                    "sale_date", "sale_id", "customer_id", "product_name",
                    "category", "revenue", "threshold", "reason", "comparison",
                ]
                st.dataframe(
                    flagged[columns], hide_index=True, **FULL_WIDTH,
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

    with panel("portfolio"):
        st.subheader("Revenue mix")
        portfolio, summary = portfolio_performance(data, start_date, end_date)
        chart = (
            alt.Chart(portfolio)
            .mark_bar()
            .encode(
                x=month_bar_axis(),
                y=alt.Y("revenue:Q", title="Monthly revenue", stack="zero"),
                color=alt.Color("category:N", title="Product category", sort=summary["category"].tolist()),
                tooltip=[
                    month_tooltip(), alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("revenue:Q", title="Revenue", format=",.2f"),
                ],
            )
        )
        show_chart(chart, height=235)
        with st.expander("Category breakdown"):
            st.dataframe(
                summary, hide_index=True, **FULL_WIDTH,
                column_config={
                    "category": "Category",
                    "revenue": st.column_config.NumberColumn("Total revenue", format="%.2f"),
                    "share_pct": st.column_config.NumberColumn("Revenue share (%)", format="%.1f"),
                },
            )


def show_predictions(growth):
    results = revenue_forecast(growth)
    if results is None:
        st.info("Select at least 12 full calendar months to train and test the forecast.")
        return

    kpi_cards([
        (row.sale_date.strftime("%b %Y"), f"{row.revenue:,.2f}")
        for row in results["future"].itertuples()
    ])
    with panel("forecast"):
        st.subheader("Revenue forecast")
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
                    scale=alt.Scale(domain=series_order),
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

    with st.expander("Backtest by month"):
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
        st.subheader("Forecast accuracy")
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
    st.session_state["uploaded_sales"] = (data, source, uploaded_file.name)


def active_sales(is_demo):
    if not is_demo:
        record = st.session_state.get("uploaded_sales")
        return record[:2] if record else None
    try:
        data = load_data(demo_csv_path())
    except (OSError, ValueError):
        st.error("The demo sales CSV is unavailable. You can upload your own sales CSV.")
        return None
    if data.empty:
        st.warning("The demo CSV has no valid sales.")
        return None
    return data, "demo"


def reporting_period(data, source):
    first_date = data["sale_date"].min().date()
    last_date = data["sale_date"].max().date()
    saved_periods = st.session_state.setdefault("reporting_periods", {})
    start, end = saved_periods.get(source, (first_date, last_date))
    date_key = f"{source}:{first_date}:{last_date}"
    with st.expander("Reporting period"):
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


@dataclass(frozen=True)
class Report:
    sales: pd.DataFrame
    source: str
    start: date
    end: date

    @property
    def is_demo(self):
        return self.source == "demo"

    def transactions(self):
        return self.sales[self.sales.sale_date.between(pd.Timestamp(self.start), pd.Timestamp(self.end))]


def sidebar_report():
    mode = st.segmented_control(
        "Data source", ["demo", "upload"],
        format_func={"demo": "Demo data", "upload": "My data"}.get,
        default="demo", required=True, width="stretch", key="data_mode",
    )
    if mode == "upload":
        selected = st.file_uploader("Sales CSV", type=["csv"], key="sales_file", on_change=save_upload)
        if selected is None and (previous := st.session_state.get("uploaded_sales")):
            st.text(previous[2])
        st.download_button(
            "Download CSV template",
            data=",".join(REQUIRED_COLUMNS) + "\nS001,2025-01-01,120,C001,P001,Desk,Office\n",
            file_name="sales_template.csv", mime="text/csv", on_click="ignore",
            icon=":material/download:",
        )
        if error := st.session_state.get("sales_file_error"):
            st.error(error)

    current = active_sales(mode == "demo")
    if current is None:
        return None
    sales, source = current
    period = reporting_period(sales, source)
    return Report(sales, source, *period) if period else None


def report_transactions(title, report):
    if report is None:
        st.title(title, anchor=False)
        if st.session_state.get("data_mode") == "upload" and "uploaded_sales" not in st.session_state:
            st.info("Choose a sales CSV in the sidebar to view your report.")
        return None
    page_heading(title, report)
    data = report.transactions()
    if data.empty:
        st.info("No transactions match these dates. Select a wider period in the sidebar.")
        return None
    return data


def overview(report):
    data = report_transactions("Overview", report)
    if data is not None:
        monthly = monthly_revenue(data, report.start, report.end)
        growth = monthly_growth(data, report.start, report.end)
        show_overview(data, monthly, growth)


def forecast(report):
    data = report_transactions("Forecast", report)
    if data is not None:
        show_predictions(monthly_growth(data, report.start, report.end))


def risk_and_anomalies(report):
    data = report_transactions("Risk & anomalies", report)
    if data is not None:
        show_risk(data, monthly_growth(data, report.start, report.end), report.start, report.end)


def copilot_workspace(report):
    if report is None:
        st.title("AI Financial Copilot", anchor=False)
        if st.session_state.get("data_mode") == "upload" and "uploaded_sales" not in st.session_state:
            st.info("Choose a sales CSV in the sidebar to start a chat with your data.")
        return
    with st.container(key="copilot_page"):
        try:
            with FinancialCopilot(sales_data=report.sales, use_demo_erp=report.is_demo) as copilot:
                render_chat_panel(copilot, start_date=report.start, end_date=report.end, source=report.source)
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Financial Copilot could not load data: {error}")


def main():
    st.html(Path(__file__).with_name("dashboard.css"))
    with st.sidebar:
        st.html('<div class="fd-brand">Financial Dashboard</div>')
        navigation = st.container(key="workspace_nav", gap=None)
        st.divider()
        report = sidebar_report()
    pages = [
        st.Page(lambda: overview(report), title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page(lambda: forecast(report), title="Forecast", icon=":material/timeline:", url_path="forecast"),
        st.Page(lambda: risk_and_anomalies(report), title="Risk & anomalies", icon=":material/monitoring:", url_path="risk"),
        st.Page(lambda: copilot_workspace(report), title="AI Financial Copilot", icon=":material/insights:", url_path="copilot"),
    ]
    selected = st.navigation(pages, position="hidden")
    with navigation:
        for page in pages:
            st.page_link(page, width="stretch")
    selected.run()


if __name__ == "__main__":
    main()
