import pandas as pd
import numpy as np

class DataProcessor:
    @staticmethod
    def load_data(file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)

    @staticmethod
    def get_summary_stats(df):
        """Extracts metadata and stats to send to the LLM."""
        stats = {
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numerical_summary": df.describe().to_dict(),
            "shape": df.shape,
            "correlations": df.select_dtypes(include=[np.number]).corr().to_dict()
        }
        return stats

    @staticmethod
    def identify_anomalies(df):
        """Simple IQR-based outlier detection."""
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}
        for col in numeric_df.columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            count = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
            outliers[col] = int(count)
        return outliers