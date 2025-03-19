import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.ar_model import AutoReg

def compare_flavors(
    df, predictor1, predictor2, target, color_by="flavor_category",
    category_list=None, year_range=None, country_list=None, flavor_list=None
):
    """
    Improved 3D scatter plot with correctly ordered AR trendline.

    Fixes:
    - Ensures correct time sorting before applying the AR model.
    - Uses proper datetime representation for ordering and display.
    - Aligns AR model predictions with actual time points.
    - Ensures AR trend line extends across all months.
    """

    # Apply filters
    if category_list:
        df = df[df["product_category"].isin(category_list)]
    if year_range:
        df = df[df["year"].between(year_range[0], year_range[-1])]
    if country_list:
        df = df[df["country"].isin(country_list)]
    if flavor_list:
        df = df[df["flavor_category"].isin(flavor_list)]

    # Ensure the month column is correctly formatted as a string
    month_map = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    
    # Convert numeric months to string representation
    if df["month"].dtype in [np.int64, np.float64]:
        df["month"] = df["month"].map(month_map)
    else:
        df["month"] = df["month"].astype(str).str.strip().str.capitalize()

    # Create a proper datetime format for correct ordering and labeling
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"], format="%Y-%b")

    # Aggregate data
    agg_df = df.groupby([predictor1, "date", color_by], as_index=False).mean(numeric_only=True)

    # Generate full date range from first to last available month
    full_date_range = pd.date_range(start=agg_df["date"].min(), end=agg_df["date"].max(), freq='MS')

    # Ensure every flavor category has a full time series
    all_flavors = agg_df[color_by].unique()
    filled_dfs = []

    for flavor in all_flavors:
        subset = agg_df[agg_df[color_by] == flavor].set_index("date")

        # Reindex with full date range, forward-fill missing values
        subset = subset.reindex(full_date_range).interpolate(method='linear')
        subset[color_by] = flavor  # Reintroduce category column
        subset["date"] = subset.index  # Reset index to get back the date column
        filled_dfs.append(subset)

    # Merge all filled data
    final_df = pd.concat(filled_dfs).reset_index(drop=True)

    # Assign unique colors to each flavor category
    unique_flavors = final_df[color_by].unique()
    color_map = px.colors.qualitative.Set1[:len(unique_flavors)]
    flavor_color_dict = {flavor: color_map[i] for i, flavor in enumerate(unique_flavors)}

    # Create a 3D plot
    fig = go.Figure()

    # Process each unique (product category, flavor category) pair
    for flavor in unique_flavors:
        subset = final_df[final_df[color_by] == flavor].sort_values(by=["date"])

        # Scatter plot of actual data points
        fig.add_trace(go.Scatter3d(
            x=subset[predictor1],
            y=subset["date"].astype(str),  # Use formatted date string for labels
            z=subset[target],
            mode='markers',
            name=f"{flavor}",
            marker=dict(size=5, color=flavor_color_dict[flavor], opacity=0.8),
            hovertemplate='<b>Time:</b> %{y}<br>Prevalence: %{z:.2f}'
        ))

        # Apply AutoRegressive (AR) model for trend line
        try:
            ar_model = AutoReg(subset[target], lags=3).fit()
            z_pred = ar_model.predict(start=0, end=len(subset) - 1)
        except:
            z_pred = subset[target].rolling(window=3, min_periods=1).mean()

        # Add smoothed trend line over the full range
        fig.add_trace(go.Scatter3d(
            x=[subset[predictor1].iloc[0]] * len(subset),
            y=subset["date"].astype(str),
            z=z_pred,
            mode='lines',
            line=dict(width=4, color=flavor_color_dict[flavor], dash='solid'),
            showlegend=False
        ))

    # Update layout for better time axis representation
    fig.update_layout(
        title="Extended 3D Comparison of Flavor Prevalence over Time",
        scene=dict(
            xaxis_title="Product Category",
            yaxis_title="Time (Year-Month)",
            zaxis_title="Flavor Prevalence",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        legend=dict(itemclick="toggle", bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="lightgray", borderwidth=1)
    )

    fig.show()


# # Load the dataset
# df = pd.read_csv("data/flavor_prev.csv")

# # Display the first few rows to verify
# print(df.head())

# compare_flavors(
#     df,
#     predictor1="product_category",
#     predictor2="year",
#     target="prevalence_flavor",
#     color_by="flavor_category",
#     category_list=["Alcoholic Beverages"],  # Focus on Alcoholic Beverages
#     year_range=(2015, 2018),  # Select the date range
#     country_list=["AU"],  # Filter for Australia
#     flavor_list=["Fruit", "Berry Fruit"]  # Select specific flavors
# )
