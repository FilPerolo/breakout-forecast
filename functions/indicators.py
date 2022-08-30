import ta
import pandas as pd
import numpy as np


# https://medium.com/codex/detecting-ranging-and-trending-markets-with-choppiness-index-in-python-1942e6450b58#:~:text=Choppiness%20Index%20is%20a%20volatility,are%20almost%20similar%20to%20ATR.
def get_ci(high, low, close, lookback):
    tr1 = pd.DataFrame(high - low).rename(columns={0: 'tr1'})
    tr2 = pd.DataFrame(abs(high - close.shift(1))).rename(columns={0: 'tr2'})
    tr3 = pd.DataFrame(abs(low - close.shift(1))).rename(columns={0: 'tr3'})
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').dropna().max(axis=1)
    atr = tr.rolling(1).mean()
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    ci = 100 * np.log10((atr.rolling(lookback).sum()) /
                        (highh - lowl)) / np.log10(lookback)
    return ci


def get_indicators(df):
    # df['ci_14'] = get_ci(df['high'], df['low'], df['close'], 14)
    # IF CHOPPINESS INDEX >= 61.8 --> MARKET IS CONSOLIDATING
    # IF CHOPPINESS INDEX <= 38.2 --> MARKET IS TRENDING

    # df['pct_change'] = df.close.pct_change()
    # df['log_return'] = 1 + df['pct_change']
    #dfTest['ATR']=ta.volatility.AverageTrueRange(dfTest['high'], dfTest['low'], dfTest['close'], window = 14, fillna=False).average_true_range()

    #df['SMA50'] = ta.trend.SMAIndicator(
    #    df['close'], window=50, fillna=False).sma_indicator()
    #df['SMA200'] = ta.trend.SMAIndicator(
    #    df['close'], window=200, fillna=False).sma_indicator()

    # MOMENTUM
    df['awsOsc'] = ta.momentum.AwesomeOscillatorIndicator(
        df['high'], df['low'], window1=5, window2=34, fillna=False).awesome_oscillator()
    df['kama'] = ta.momentum.KAMAIndicator(
        df['close'], window=10, pow1=2, pow2=30, fillna=False).kama()
    df['PVO'] = ta.momentum.PercentageVolumeOscillator(
        df['volume'], window_slow=26, window_fast=12, window_sign=9, fillna=False).pvo()
    df['ROC'] = ta.momentum.ROCIndicator(
        df['close'], window=12, fillna=False).roc()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['stochRSI'] = ta.momentum.StochRSIIndicator(
        df['close'], window=14, smooth1=3, smooth2=3, fillna=False).stochrsi()
    df['stochOsc'] = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'], window=14, smooth_window=3, fillna=False).stoch()
    df['TSI'] = ta.momentum.TSIIndicator(
        df['close'], window_slow=25, window_fast=13, fillna=False).tsi()
    df['ultimateOsc'] = ta.momentum.UltimateOscillator(
        df['high'], df['low'], df['close'], window1=7, window2=14, window3=28, weight1=4.0, weight2=2.0, weight3=1.0, fillna=False).ultimate_oscillator()
    df['willR'] = ta.momentum.WilliamsRIndicator(
        df['high'], df['low'], df['close'], lbp=14, fillna=False).williams_r()

    # VOLUME
    df['ADI'] = ta.volume.AccDistIndexIndicator(
        df['high'], df['low'], df['close'], df['volume'], fillna=False).acc_dist_index()
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
        df['high'], df['low'], df['close'], df['volume'], window=20, fillna=False).chaikin_money_flow()
    df['EoM'] = ta.volume.EaseOfMovementIndicator(
        df['high'], df['low'], df['volume'], window=14, fillna=False).ease_of_movement()
    df['FI'] = ta.volume.ForceIndexIndicator(
        df['close'], df['volume'], window=13, fillna=False).force_index()
    df['MFI'] = ta.volume.MFIIndicator(
        df['high'], df['low'], df['close'], df['volume'], window=14, fillna=False).money_flow_index()
    df['NVI'] = ta.volume.NegativeVolumeIndexIndicator(
        df['close'], df['volume'], fillna=False).negative_volume_index()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        df['close'], df['volume'], fillna=False).on_balance_volume()
    df['VPT'] = ta.volume.VolumePriceTrendIndicator(
        df['close'], df['volume'], fillna=False).volume_price_trend()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        df['high'], df['low'], df['close'], df['volume'], window=14, fillna=False).volume_weighted_average_price()

    # VOLATILITY
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14).average_true_range()
    BgBANDS = ta.volatility.BollingerBands(
        df['close'], window=20, window_dev=2, fillna=False)
    df['bbPer'] = BgBANDS.bollinger_pband()
    df['bbWidth'] = BgBANDS.bollinger_wband()
    DonChann = ta.volatility.DonchianChannel(
        df['high'], df['low'], df['close'], window=20, fillna=False)
    df['DONC'] = ta.volatility.DonchianChannel(
        df['high'], df['low'], df['close'], window=20, offset=0).donchian_channel_wband()
    df['dcPer'] = DonChann.donchian_channel_pband()
    Keltner = ta.volatility.KeltnerChannel(
        df['high'], df['low'], df['close'], window=20, window_atr=10, fillna=False, original_version=True, multiplier=2)
    df['kelPer'] = Keltner.keltner_channel_pband()
    df['kelWidth'] = Keltner.keltner_channel_wband()
    df['UI'] = ta.volatility.UlcerIndex(
        df['close'], window=14, fillna=False).ulcer_index()

    # TREND
    df['ADX'] = ta.trend.ADXIndicator(
        df['high'], df['low'], df['close'], window=14).adx()
    df['Aroon'] = ta.trend.AroonIndicator(
        df['close'], window=25, fillna=False).aroon_indicator()
    df['CCI'] = ta.trend.CCIIndicator(
        df['high'], df['low'], df['close'], window=20, constant=0.015, fillna=False).cci()
    df['DPO'] = ta.trend.DPOIndicator(
        df['close'], window=20, fillna=False).dpo()
    df['KST'] = ta.trend.KSTIndicator(df['close'], roc1=10, roc2=15, roc3=20, roc4=30,
                                      window1=10, window2=10, window3=10, window4=15, nsig=9, fillna=False).kst()
    df['MACD'] = ta.trend.MACD(df['close'], window_slow=26,
                               window_fast=12, window_sign=9, fillna=False).macd_diff()
    df['MI'] = ta.trend.MassIndex(
        df['high'], df['low'], window_fast=9, window_slow=25, fillna=False).mass_index()
    df['PSAR'] = ta.trend.PSARIndicator(
        df['high'], df['low'], df['close'], step=0.02, max_step=0.2, fillna=False).psar()
    df['STC'] = ta.trend.STCIndicator(
        df['close'], window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3, fillna=False).stc()
    df['TRIX'] = ta.trend.TRIXIndicator(
        df['close'], window=15, fillna=False).trix()
    df['VI'] = ta.trend.VortexIndicator(
        df['high'], df['low'], df['close'], window=14, fillna=False).vortex_indicator_diff()
