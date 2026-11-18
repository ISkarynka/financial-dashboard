import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "sale_id", "sale_date", "revenue", "customer_id",
    "product_id", "product_name", "category",
)


def load_data(file):
    """Load the cleaned sales CSV and keep valid transactions."""
    text_columns = {
        column: "string"
        for column in REQUIRED_COLUMNS
        if column not in ("sale_date", "revenue")
    }
    data = pd.read_csv(file, dtype=text_columns)
    data.columns = data.columns.str.strip()

    missing = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    for column in text_columns:
        data[column] = data[column].astype("string").str.strip().replace("", pd.NA)

    data["sale_date"] = (
        pd.to_datetime(data["sale_date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
    data = data.dropna(subset=["sale_id", "sale_date", "revenue"])
    data = data[np.isfinite(data["revenue"]) & (data["revenue"] >= 0)]

    for column in ("invalid_discount", "revenue_mismatch"):
        if column in data:
            flag = data[column].astype("string").str.strip().str.lower()
            data = data[flag.isin(["false", "0", "0.0"])]

    data = data.drop_duplicates(subset="sale_id").sort_values("sale_date").copy()
    data["category"] = data["category"].fillna("Unknown category")
    return data.reset_index(drop=True)


def calculate_metrics(data):
    revenue = data["revenue"].sum()
    transactions = data["sale_id"].nunique()
    return {
        "total_revenue": revenue,
        "number_of_customers": data["customer_id"].nunique(),
        "number_of_transactions": transactions,
        "average_transaction_value": revenue / transactions if transactions else 0,
    }


def month_index(start_date, end_date, full_months_only=False):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    first_month = start.replace(day=1)
    last_month = end.replace(day=1)
    if full_months_only:
        if start.day != 1:
            first_month += pd.offsets.MonthBegin(1)
        if not end.is_month_end:
            last_month -= pd.offsets.MonthBegin(1)
    return pd.date_range(first_month, last_month, freq="MS", name="sale_date")


def monthly_revenue(data, start_date, end_date, full_months_only=False):
    months = month_index(start_date, end_date, full_months_only)
    monthly = data.groupby(pd.Grouper(key="sale_date", freq="MS"))["revenue"].sum()
    return monthly.reindex(months, fill_value=0).reset_index()


def top_revenue(data, column):
    return (
        data.groupby(column)["revenue"]
        .sum()
        .nlargest(10)
        .reset_index()
    )


def monthly_growth(data, start_date, end_date):
    monthly = monthly_revenue(data, start_date, end_date, full_months_only=True)
    previous_revenue = monthly["revenue"].shift(1)
    monthly["growth_pct"] = (
        monthly["revenue"].div(previous_revenue.where(previous_revenue > 0)) - 1
    ) * 100
    return monthly


def rolling_volatility(monthly):
    risk = monthly.copy()
    risk["volatility"] = risk["growth_pct"].rolling(6).std()
    return risk


def calculate_var(monthly):
    growth = monthly["growth_pct"].dropna()
    if len(growth) < 12:
        return None

    var_pct = max(0.0, -float(growth.quantile(0.05)))
    latest_revenue = float(monthly["revenue"].iloc[-1])
    return {
        "var_pct": var_pct,
        "shortfall": latest_revenue * var_pct / 100,
        "baseline_revenue": latest_revenue,
        "baseline_month": monthly["sale_date"].iloc[-1],
        "observations": len(growth),
    }


def portfolio_performance(data, start_date, end_date):
    """Treat the sales mix across product categories as a business portfolio."""
    categories = data.groupby("category")["revenue"].sum().sort_values(ascending=False)
    if len(categories) > 8:
        other_label = "Other categories"
        while other_label in categories.index:
            other_label += " (grouped)"
        data = data.assign(
            category=data["category"].where(
                data["category"].isin(categories.head(7).index), other_label
            )
        )
        categories = data.groupby("category")["revenue"].sum().sort_values(ascending=False)

    monthly = (
        data.groupby([pd.Grouper(key="sale_date", freq="MS"), "category"])["revenue"]
        .sum()
        .unstack(fill_value=0)
        .reindex(month_index(start_date, end_date), fill_value=0)
        .reset_index()
        .melt(id_vars="sale_date", var_name="category", value_name="revenue")
    )
    summary = categories.rename("revenue").reset_index()
    total = summary["revenue"].sum()
    summary["share_pct"] = summary["revenue"] / total * 100 if total > 0 else 0.0
    return monthly, summary


def _features(history):
    return np.array([history[-1], np.mean(history[-3:])], dtype=float)


def _fit_revenue_model(revenue):
    """Fit ridge regression on previous-month revenue and the last 3-month mean."""
    x = np.array([_features(revenue[i - 3:i]) for i in range(3, len(revenue))])
    y = np.asarray(revenue[3:], dtype=float)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    x_scaled = (x - center) / scale
    intercept = y.mean()
    weights = np.linalg.solve(
        x_scaled.T @ x_scaled + np.eye(x.shape[1]),
        x_scaled.T @ (y - intercept),
    )
    return center, scale, intercept, weights


def _predict_revenue(model, history):
    center, scale, intercept, weights = model
    prediction = intercept + ((_features(history) - center) / scale) @ weights
    return max(0.0, float(prediction))


def revenue_forecast(monthly, horizon=3):
    """Test one-step forecasts on later months, then forecast future months."""
    revenue = monthly["revenue"].to_numpy(dtype=float)
    if len(revenue) < 12:
        return None

    test_count = max(3, int(np.ceil(len(revenue) * 0.2)))
    train_count = len(revenue) - test_count
    test_model = _fit_revenue_model(revenue[:train_count])

    # Each test prediction only uses revenue observed before that month.
    test = monthly.iloc[train_count:][["sale_date", "revenue"]].copy()
    test = test.rename(columns={"revenue": "actual"})
    test["prediction"] = [
        _predict_revenue(test_model, revenue[i - 3:i])
        for i in range(train_count, len(revenue))
    ]
    test["baseline"] = revenue[train_count - 1:-1]
    test["error"] = test["prediction"] - test["actual"]
    baseline_error = test["baseline"] - test["actual"]

    # Refit on all available months for the final forecast.
    final_model = _fit_revenue_model(revenue)
    history = revenue.tolist()
    predictions = []
    for _ in range(horizon):
        prediction = _predict_revenue(final_model, history)
        predictions.append(prediction)
        history.append(prediction)

    future_months = pd.date_range(
        monthly["sale_date"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    future = pd.DataFrame({"sale_date": future_months, "revenue": predictions})

    return {
        "future": future,
        "test": test,
        "train_count": train_count,
        "test_count": test_count,
        "mae": test["error"].abs().mean(),
        "rmse": np.sqrt(test["error"].pow(2).mean()),
        "baseline_mae": baseline_error.abs().mean(),
        "baseline_rmse": np.sqrt(baseline_error.pow(2).mean()),
    }


def _iqr_bounds(values):
    q1, q3 = values.quantile([0.25, 0.75])
    margin = 1.5 * (q3 - q1)
    return max(0.0, q1 - margin), q3 + margin


def detect_anomalies(data):
    """Flag revenue outside the IQR range for the product category."""
    if len(data) < 20:
        return None

    overall_low, overall_high = _iqr_bounds(data["revenue"])
    bounds = []
    for category, group in data.groupby("category"):
        if len(group) >= 20:
            low, high = _iqr_bounds(group["revenue"])
            comparison = "Product category"
        else:
            low, high = overall_low, overall_high
            comparison = "All selected transactions"
        bounds.append((category, low, high, comparison))

    limits = pd.DataFrame(
        bounds, columns=["category", "lower_limit", "upper_limit", "comparison"]
    )
    candidates = data.merge(limits, on="category", how="left", validate="many_to_one")
    high = candidates["revenue"] > candidates["upper_limit"]
    low = candidates["revenue"] < candidates["lower_limit"]
    flagged = candidates[high | low].copy()
    flagged["reason"] = np.where(
        flagged["revenue"] > flagged["upper_limit"],
        "Unusually high revenue",
        "Unusually low revenue",
    )
    flagged["threshold"] = np.where(
        flagged["revenue"] > flagged["upper_limit"],
        flagged["upper_limit"],
        flagged["lower_limit"],
    )
    return flagged.sort_values(["sale_date", "sale_id"], ascending=False)


def anomaly_trend(flagged, start_date, end_date):
    months = month_index(start_date, end_date)
    counts = flagged.groupby(pd.Grouper(key="sale_date", freq="MS")).size()
    return counts.reindex(months, fill_value=0).rename("anomalies").reset_index()
