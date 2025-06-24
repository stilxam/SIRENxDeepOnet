from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_df(data_dir, num_files: int):
    files = os.listdir(data_dir)
    path_csvs = [filename for filename in files if filename[-4:] == ".csv"]

    df_list = [pd.read_csv(data_dir.joinpath(f_name)) for i, f_name in enumerate(path_csvs) if i < num_files]

    df = pd.concat(df_list, ignore_index=True)
    df = convert_side_to_num(df)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df.set_index('timestamp', inplace=True)
    return df


def convert_side_to_num(df):
    map_dct= {"sell": -1, "buy": 1}

    df["side"] = df["side"].apply(lambda x: map_dct[x])

    return df



def engineer_features(df, interval="1min", rolling_window=5):
    # Resample to interval
    df_resampled = df.copy()
    df_resampled["interval"] = df_resampled.index.floor(interval)

    # Aggregate volume stats
    volume_stats = df_resampled.groupby("interval")["amount"].sum().rename("volume_total")
    volume_by_side = df_resampled.groupby(["interval", "side"])["amount"].sum().unstack(fill_value=0)

    # VWAP
    vwap = (df_resampled["price"] * df_resampled["amount"]).groupby(df_resampled["interval"]).sum() / volume_stats
    vwap = vwap.rename("price_vwap")

    # OHLC
    ohlc = df_resampled.groupby("interval")["price"].agg(["first", "max", "min", "last"])
    ohlc.columns = ["open", "high", "low", "close"]
    price_range = (ohlc["high"] - ohlc["low"]).rename("price_range")

    # Rolling features
    features = pd.DataFrame(index=volume_stats.index)
    features = features.join([volume_stats, vwap, price_range])
    features = features.join(volume_by_side.rename(columns={1: "volume_buy", -1: "volume_sell"}))
    features = features.join(ohlc)

    features["price_rolling_mean"] = features["price_vwap"].rolling(window=rolling_window, min_periods=1).mean()
    features["price_rolling_std"] = features["price_vwap"].rolling(window=rolling_window, min_periods=1).std()
    features["price_momentum"] = features["price_vwap"] - features["price_rolling_mean"]

    # OFI and rolling OFI
    features["ofi"] = features["volume_buy"].fillna(0) - features["volume_sell"].fillna(0)
    features["ofi_rolling_cumulative"] = features["ofi"].rolling(window=rolling_window, min_periods=1).sum()

    # Rolling volatility by side
    def calc_vol(side):
        return df_resampled[df_resampled["side"] == side].groupby("interval")["price"].std().rolling(window=rolling_window, min_periods=1).mean()
    features["volatility_rolling_buy"] = calc_vol(1)
    features["volatility_rolling_sell"] = calc_vol(-1)
    features["volatility_skew"] = features["volatility_rolling_buy"] - features["volatility_rolling_sell"]

    # Price impact proxy
    features["price_impact_proxy"] = (features["close"] - features["open"]) / features["volume_total"].replace(0, np.nan)
    features["price_impact_proxy"] = features["price_impact_proxy"].replace([np.inf, -np.inf], 0).fillna(0)

    # Time features
    features["time_hour"] = features.index.hour

    return features


def main(num_files: int = 1, time_group: str = "min", rolling_window: int = 5):
    cwd = Path.cwd()
    datadir = cwd.joinpath("data")
    df = get_df(datadir, num_files)
    features = engineer_features(df, interval=time_group, rolling_window=rolling_window)
    print(features.head())
    df["t_group"] = df.index.floor(time_group)
    return df, features







if __name__ == "__main__":
    main()