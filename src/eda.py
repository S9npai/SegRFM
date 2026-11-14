import calendar
import pathlib

import pandas as pd
import plotly.express as px

from src.transform import Transform


class RetailEDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df


    def monthlyRevenue(self):
        monthly_rev = self.df.groupby(['Year', 'Month'], as_index=False)['TotalLineRevenue'].sum()
        monthly_rev['YearMonth'] = monthly_rev['Year'].astype(str) + '-' + monthly_rev['Month'].astype(str).str.zfill(2)

        fig = px.line(
            monthly_rev,
            x='YearMonth',
            y='TotalLineRevenue',
            title='Monthly Revenue & Q4 Surges',
            markers=True,
        )
        fig.update_layout(
            xaxis_title='Year-Month',
            yaxis_title='Total Revenue ($)',
            template='plotly_white',
        )
        fig.update_traces(
            line=dict
            (color='#27c8e8', width=3)
        )
        return fig


    def paretoRevenue(self):
        stock_rev = self.df.groupby('StockCode', as_index=False)['TotalLineRevenue'].sum()
        stock_rev = stock_rev.sort_values(by='TotalLineRevenue', ascending=False).reset_index(drop=True)

        stock_rev['Cumulative_Revenue'] = stock_rev['TotalLineRevenue'].cumsum()
        stock_rev['Cumulative_Percentage'] = stock_rev['Cumulative_Revenue'] / stock_rev['TotalLineRevenue'].sum() * 100
        stock_rev['Product_Percentage'] = (stock_rev.index + 1) / len(stock_rev) * 100

        pareto_subset = stock_rev[stock_rev['Cumulative_Percentage'] <= 80]
        if not pareto_subset.empty:
            pareto_sku_count = len(pareto_subset)
            pareto_sku_pct = pareto_subset['Product_Percentage'].max()
        else:
            pareto_sku_count = 0
            pareto_sku_pct = 0.0

        paretoFig = px.line(
            stock_rev,
            x='Product_Percentage',
            y='Cumulative_Percentage',
            title='Cumulative Revenue Curve (Pareto Analysis)',
            labels={'Product_Percentage': 'Percentage of Products (%)',
                    'Cumulative_Percentage': 'Cumulative Revenue (%)'}
        )
        paretoFig.update_traces(
            line=dict
            (color='#1c1c99', width=3)
        )
        paretoFig.update_layout(template='plotly_white')

        paretoFig.add_hline(
            y=80, line_dash="dash", line_color="#C0392B",
            annotation_text="80% Revenue Threshold", annotation_position="bottom right"
        )
        paretoFig.add_vline(
            x=20, line_dash="dot", line_color="#E67E22",
            annotation_text=f"{pareto_sku_pct:.1f}% of SKUs ({pareto_sku_count:,} items)",
            annotation_position="top left"
        )

        return paretoFig


    def analyze_hourly_traffic(self):
        hourly_orders = self.df.groupby('Hour', as_index=False)['InvoiceNo'].nunique()

        hrfig = px.bar(
            hourly_orders,
            x='Hour',
            y='InvoiceNo',
            color='InvoiceNo',
            color_continuous_scale='Teal',
            title='Hourly Unique Invoices Traffic',
            labels={'Hour': 'Hour of Day', 'InvoiceNo': 'Unique Invoices'}
        )

        cadenceGrid = (
            self.df.groupby(["DayOfWeek", "Hour"])["InvoiceNo"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(list(calendar.day_name))
        )

        contmap = px.imshow(
            cadenceGrid,
            labels=dict(x="Hour of Day (24h)", y="Day of Week", color="Unique Orders"),
            x=cadenceGrid.columns,
            y=cadenceGrid.index,
            color_continuous_scale="Blues",
            aspect="auto",
            title="<b>Unique Orders by Day of Week & Hour</b>"
        )
        contmap.update_traces(
            hovertemplate="<b>Day:</b> %{y}<br><b>Hour:</b> %{x}:00<br><b>Unique Orders:</b> %{z:,}<extra></extra>"
        )
        contmap.update_layout(height=420, xaxis=dict(dtick=1))

        return hrfig, contmap


    def average_order_values(self):
        order_values = self.df.groupby('InvoiceNo', as_index=False).agg({
            'TotalLineRevenue': 'sum',
            'CustomerID': 'first'
        })

        positive_orders = order_values[order_values['TotalLineRevenue'] > 0]
        mean_ov = positive_orders['TotalLineRevenue'].mean()
        median_ov = positive_orders['TotalLineRevenue'].median()
        print(f"Dataset Overall Mean AOV: ${mean_ov:.2f} | Median AOV: ${median_ov:.2f}")

        fig = px.box(
            positive_orders,
            y='TotalLineRevenue',
            title='Average Order Value (AOV) Distribution per Invoice',
            labels={'TotalLineRevenue': 'Invoice Revenue ($)'}
        )
        fig.update_traces(marker_color='#4fbd77')
        fig.update_layout(yaxis_title='Invoice Order Value ($)')
        return fig


    def basket_sizes(self):
        basket_df = self.df.groupby('InvoiceNo', as_index=False).agg({
            'Quantity': 'sum',
            'CustomerID': 'first',
            'Country': 'first'
        })

        positive_baskets = basket_df[basket_df['Quantity'] > 0]

        fig = px.histogram(
            positive_baskets,
            x='Quantity',
            nbins=2000,
            title='Basket Size Distribution',
            labels={'Quantity': 'Per Invoice Total Items', 'count': 'Invoice Count'}
        )
        fig.update_traces(marker_color='#e0892b')
        fig.update_layout(bargap=0.1, template='plotly_white')
        fig.update_xaxes(range=[0, 2000])

        return fig


    def dead_stock(self, days_threshold=60):
        max_date = self.df['InvoiceDate'].max()
        cutoff = max_date - pd.Timedelta(days=days_threshold)

        historical_sales = self.df[self.df['InvoiceDate'] < cutoff]
        hist_volume = (
            historical_sales.groupby(
            ['StockCode', 'Description'], as_index=False)['Quantity']
            .sum().rename(columns={'Quantity': 'Historical_Volume'})
        )

        recent_sales = self.df[(self.df['InvoiceDate'] >= cutoff) & (self.df['Quantity'] > 0)]
        recent_volume = (
            recent_sales.groupby(['StockCode', 'Description'], as_index=False)['Quantity']
            .sum().rename(columns={'Quantity': 'Recent_Volume'})
        )

        dead_stock_df = pd.merge(hist_volume, recent_volume, on='StockCode', how='left')
        dead_stock_df['Recent_Volume'] = dead_stock_df['Recent_Volume'].fillna(0)

        median_vol = dead_stock_df['Historical_Volume'].median()
        dead_stock = dead_stock_df[
            (dead_stock_df['Historical_Volume'] > median_vol) & (dead_stock_df['Recent_Volume'] == 0)]

        dead_stock = dead_stock.sort_values(by='Historical_Volume', ascending=False)
        print(f"Identified {len(dead_stock)} dead stock products matching criteria.")

        return dead_stock


    def cancellations(self):
        cancellations = self.df[self.df['Cancelled'] == True]

        top_cancelled_products = cancellations.groupby(['StockCode', 'Description'], as_index=False)['Quantity'].sum()
        top_cancelled_products['Absolute_Quantity'] = top_cancelled_products['Quantity'].abs()
        top_cancelled_products = top_cancelled_products.sort_values(by='Absolute_Quantity', ascending=False).head(50)

        fig = px.bar(
            top_cancelled_products,
            x='StockCode',
            y='Absolute_Quantity',
            color='Absolute_Quantity',
            color_continuous_scale='Blues',
            hover_data=['Description'],
            title='Top Products Returned or Cancelled',
            labels={'Absolute_Quantity': 'Total Units Returned', 'Product Label': 'Stock Code'}
        )

        fig.update_layout(
            xaxis_type='category',
            xaxis_tickangle=-45,
            template='plotly_white',
            coloraxis_showscale=False,
            margin=dict(b=150)
        )

        stats = {
            "cancelled": len(cancellations),
            "revenue_lost": cancellations['TotalLineRevenue'].sum(),
        }
        return fig, stats




if __name__ == "__main__":
    df = pd.read_csv(pathlib.Path("../data/OnlineRetail.csv"))

    transformer = Transform(df)
    transformer.clean()
    dxf = transformer.transform_all()

    red = RetailEDA(dxf["df_reg"])

    pareto = red.paretoRevenue()
    pareto.show()

    aov = red.average_order_values()
    aov.show()

    monRev = red.monthlyRevenue()
    monRev.show()

    bskv = red.basket_sizes()
    bskv.show()

    redc = RetailEDA(dxf["df_cl"])
    cncl, stats = redc.cancellations()
    cncl.show()
