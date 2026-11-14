import pathlib
import pandas as pd



class Transform:
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df_reg = None
        self.df_cl = None
        self.df_v = None


    def clean(self):
        dff = self.raw_df.drop_duplicates(keep="first").copy()

        cancellations = (
                dff["InvoiceNo"].str.startswith("C", na=False)
                & (dff["Quantity"] < 0)
                & (dff["UnitPrice"] > 0)
        )
        self.df_cl = dff[cancellations].copy()

        valid_sales = (
                (~dff["InvoiceNo"].str.startswith(("C", "A"), na=False))
                & (dff["Quantity"] > 0)
                & (dff["UnitPrice"] > 0)
        )
        dfv = dff[valid_sales].copy()

        dfv["nonProduct"] = (
                ~dfv["StockCode"].str.match(r"^\d{5}", na=False) |
                dfv["StockCode"].str.lower().str.startswith("gift_")
        )
        dfv["Description"] = dfv["Description"].fillna("Unknown Product").str.strip()

        self.df_v = dfv
        self.df_reg = dfv[dfv["CustomerID"].notna()].copy()

        print(f"Cleaned Summary:")
        print(f" - Valid Regular Sales (df_reg): {len(self.df_reg)} rows")
        print(f" - Cancellations (df_cl): {len(self.df_cl)} rows")
        print(f" - Broad Valid Sales (df_v): {len(self.df_v)} rows")

        return self


    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        df = df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df["YearMonth"] = df['InvoiceDate'].dt.to_period("M").astype(str)
        df["Quarter"] = "Q" + df['InvoiceDate'].dt.quarter.astype(str)
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        df['Hour'] = df['InvoiceDate'].dt.hour

        df['Cancelled'] = df['InvoiceNo'].astype(str).str.startswith('C')
        df['TotalLineRevenue'] = df['Quantity'] * df['UnitPrice']
        df['Is_UK'] = (df['Country'] == 'United Kingdom').astype(int)

        return df


    def transform_all(self):
        self.df_reg = self._apply_transformations(self.df_reg)
        self.df_cl = self._apply_transformations(self.df_cl)
        self.df_v = self._apply_transformations(self.df_v)

        return {
            "df_reg": self.df_reg,
            "df_cl": self.df_cl,
            "df_v": self.df_v
        }


    @staticmethod
    def compute_RawRFM(df_registered: pd.DataFrame) -> pd.DataFrame:
        snapshot_date = df_registered["InvoiceDate"].max() + pd.Timedelta(days=1)

        if "TotalLineRevenue" not in df_registered.columns:
            df_registered["TotalLineRevenue"] = df_registered["Quantity"] * df_registered["UnitPrice"]

        rfm_raw = df_registered.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalLineRevenue", "sum")
        ).reset_index()

        return rfm_raw[rfm_raw['Monetary'] > 0].copy()



def testTF():
    df = pd.read_csv(pathlib.Path("../data/OnlineRetail.csv"))

    pipeline = Transform(df)
    pipeline.clean()
    transformed_sets = pipeline.transform_all()

    print("\nRegistered Sales (df_reg)", transformed_sets.get("df_reg"))
    print(transformed_sets["df_reg"][['InvoiceNo', 'InvoiceDate', 'TotalLineRevenue', 'Is_UK']].head(10))

    print("\nCancellations (df_cl)", transformed_sets.get("df_cl"))
    print(transformed_sets["df_cl"][['InvoiceNo', 'InvoiceDate', 'TotalLineRevenue']].head(10))


"""
if __name__ == "__main__":
    testTF()
"""

