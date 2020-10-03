from lib.log import logger
from lib.dataset import builder, load_dataset, save_symbol_dataset
from lib.dataset.utils import convergence_between_series
import pandas as pd


def build(source_index, dest_index, W=10):
    _dataset = load_dataset(source_index, return_index=True)

    for _sym, entry in _dataset.items():
        _df = pd.read_csv(entry['csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        _target = pd.read_csv(entry['target_csv'], sep=',', encoding='utf-8', index_col='Date', parse_dates=True)
        ohlcv = _df[entry['features']['ohlcv']]

        ta = _df[entry['features']['ta']]
        ta_7 = _df[entry['features']['ta_7d']]
        cm = _df[entry['features']['cm']]

        ohlcv_stats = pd.DataFrame(index=ohlcv.index)
        #ohlcv_stats['volume'] = ohlcv.volume
        #ohlcv_stats['volume_pct'] = ohlcv.volume.pct_change()
        #ohlcv_stats['close_pct'] = ohlcv.close.pct_change()
        ohlcv_stats['day_range_pct'] = (ohlcv.high - ohlcv.low).pct_change() # Showld always be > 0, price oscillation range for current day
        ohlcv_stats['direction'] = ohlcv.close - ohlcv.open # Price direction for the day green > 0, red < 0. Modulus is range.

        cm_picked = pd.DataFrame(index=ohlcv.index)
        if'adractcnt' in cm.columns:
            cm_picked['adractcnt_pct'] = cm.adractcnt.pct_change()
            # cm_picked['adractcnt_mean3_pct'] = cm.adractcnt.rolling(3).mean().pct_change()
            # cm_picked['adractcnt_mean7_pct'] = cm.adractcnt.rolling(7).mean().pct_change()
        # if 'splycur' in cm.columns: ## Correlated with volume and close
        #     cm_picked['vol_supply'] = ohlcv.volume / cm.splycur # Ratio between transacted volume and total supply (mined)
        if 'txtfrvaladjntv' in cm.columns and 'isstotntv' in cm.columns and 'feetotntv' in cm.columns:
            # I want to represent miners earnings (fees + issued coins) vs amount transacted in that interval
            cm_picked['earned_vs_transacted'] = (cm.isstotntv + cm.feetotntv) / cm.txtfrvaladjntv
        if 'isstotntv' in cm.columns:
            # isstotntv is total number of coins mined in the time interval
            # splycur is total number of coins mined (all time)
            total_mined = cm.isstotntv.rolling(365, min_periods=7).sum() # total mined in a year
            cm_picked['isstot365_isstot1_pct'] = (total_mined / cm.isstotntv).pct_change()
        if 'splycur' in cm.columns and 'isstotntv' in cm.columns:
            cm_picked['splycur_isstot1_pct'] = (cm.splycur / cm.isstotntv).pct_change()
        if 'hashrate' in cm.columns:
            #cm_picked['hashrate_mean3_pct'] = cm.hashrate.rolling(3).mean().pct_change()
            #cm_picked['hashrate_mean7_pct'] = cm.hashrate.rolling(7).mean().pct_change()
            cm_picked['hashrate_pct'] = cm.hashrate.pct_change()
        if 'roi30d' in cm.columns:
            cm_picked['roi30d'] = cm.roi30d
        if 'isstotntv' in cm.columns:
            cm_picked['isstotntv_pct'] = cm.isstotntv.pct_change()
        if 'feetotntv' in cm.columns:
            cm_picked['feetotntv_pct'] = cm.feetotntv.pct_change()
        if 'txtfrcount' in cm.columns:
            cm_picked['txtfrcount_pct'] = cm.txtfrcount.pct_change()
            cm_picked['txtfrcount_volume'] = cm.txtfrcount.pct_change()
        if 'vtydayret30d' in cm.columns:
            cm_picked['vtydayret30d'] = cm.vtydayret30d
        if 'isscontpctann' in cm.columns:
            cm_picked['isscontpctann'] = cm.isscontpctann

        ta_picked = pd.DataFrame(index=ta.index)
        # REMA / RSMA are already used and well-estabilished in ATSA,
        # I'm taking the pct change since i want to encode the relative movement of the ema's not their positions
        # ta_picked['rema_5_20_pct'] = ta.rema_5_20.pct_change()
        ta_picked['rema_8_15_pct'] = ta.rema_8_15.pct_change()
        # ta_picked['rema_20_50_pct'] = ta.rema_20_50.pct_change()
        # ta_picked['rsma_5_20_pct'] = ta.rema_5_20.pct_change()
        ta_picked['rsma_8_15_pct'] = ta.rema_8_15.pct_change()
        # ta_picked['rsma_20_50_pct'] = ta.rema_20_50.pct_change()

        # Stoch is a momentum indicator comparing a particular closing price of a security to a range of its prices
        # over a certain period of time.
        # The sensitivity of the oscillator to market movements is reducible by adjusting that time period or
        # by taking a moving average of the result.
        # It is used to generate overbought and oversold trading signals, utilizing a 0-100 bounded range of values.
        # IDEA => decrease sensitivity by 3-mean and divide by 100 to get fp values
        ta_picked['stoch_14_mean3_div100'] = ta.stoch_14.rolling(3).mean() / 100

        #Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows
        # the relationship between two moving averages of a security’s price.
        # The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.
        #  A nine-day EMA of the MACD called the "signal line," is then plotted on top of the MACD line,
        #  which can function as a trigger for buy and sell signals.
        #  Traders may buy the security when the MACD crosses above its signal line and sell - or short - the security
        #  when the MACD crosses below the signal line.
        #  Moving Average Convergence Divergence (MACD) indicators can be interpreted in several ways,
        #  but the more common methods are crossovers, divergences, and rapid rises/falls.
        signal_line = builder.exponential_moving_average(ta.macd_12_26, 9)
        ta_picked['macd_12_26_signal'] = (ta.macd_12_26 - signal_line).pct_change() # Relationship with signal line
        ta_picked['macd_12_26_pct'] = ta.macd_12_26.pct_change() # Information about slope

        # PPO is identical to the moving average convergence divergence (MACD) indicator,
        # except the PPO measures percentage difference between two EMAs, while the MACD measures absolute (dollar) difference.
        signal_line = builder.exponential_moving_average(ta.ppo_12_26, 9)
        ta_picked['ppo_12_26_signal'] = (ta.ppo_12_26 - signal_line).pct_change()  # Relationship with signal line
        ta_picked['ppo_12_26_pct'] = ta.ppo_12_26.pct_change()  # Information about slope

        # ADI Accumulation/distribution is a cumulative indicator that uses volume and price to assess whether
        # a stock is being accumulated or distributed.
        # The accumulation/distribution measure seeks to identify divergences between the stock price and volume flow.
        # This provides insight into how strong a trend is. If the price is rising but the indicator is falling
        # this indicates that buying or accumulation volume may not be enough to support
        # the price rise and a price decline could be forthcoming.
        # ==> IDEA: if we can fit a line to the price y1 = m1X+q1 and a line to ADI y2=m2X+q2 then we can identify
        #           divergences by simply looking at the sign of M.
        #           Another insight would be given by the slope (ie pct_change)
        ta_picked['adi_pct'] = ta.adi.pct_change()
        ta_picked['adi_close_convergence'] = convergence_between_series(ta.adi, ohlcv.close, 3)

        # RSI goes from 0 to 100, values <= 20 mean BUY, while values >= 80 mean SELL.
        # Dividing it by 100 to get a floating point feature, makes no sense to pct_change it
        ta_picked['rsi_14_div100'] = ta.rsi_14 / 100

        # The Money Flow Index (MFI) is a technical indicator that generates overbought or oversold
        #   signals using both prices and volume data. The oscillator moves between 0 and 100.
        # An MFI reading above 80 is considered overbought and an MFI reading below 20 is considered oversold,
        #   although levels of 90 and 10 are also used as thresholds.
        # A divergence between the indicator and price is noteworthy. For example, if the indicator is rising while
        #   the price is falling or flat, the price could start rising.
        ta_picked['mfi_14_div100'] = ta.mfi_14 / 100

        # The Chande momentum oscillator is a technical momentum indicator similar to other momentum indicators
        #   such as Wilder’s Relative Strength Index (Wilder’s RSI) and the Stochastic Oscillator.
        #   It measures momentum on both up and down days and does not smooth results, triggering more frequent
        #   oversold and overbought penetrations. The indicator oscillates between +100 and -100.
        # Many technical traders add a 10-period moving average to this oscillator to act as a signal line.
        #   The oscillator generates a bullish signal when it crosses above the moving average and a
        #   bearish signal when it drops below the moving average.
        ta_picked['cmo_14_div100'] = ta.cmo_14 / 100
        signal_line = builder.simple_moving_average(ta.cmo_14, 10)
        ta_picked['cmo_14_signal'] = (ta.cmo_14 - signal_line) / 100

        # On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict changes in stock price.
        # Eventually, volume drives the price upward. At that point, larger investors begin to sell, and smaller investors begin buying.
        # Despite being plotted on a price chart and measured numerically,
        # the actual individual quantitative value of OBV is not relevant.
        # The indicator itself is cumulative, while the time interval remains fixed by a dedicated starting point,
        # meaning the real number value of OBV arbitrarily depends on the start date.
        # Instead, traders and analysts look to the nature of OBV movements over time;
        # the slope of the OBV line carries all of the weight of analysis. => We want percent change
        ta_picked['obv_pct'] = ta.obv.pct_change()
        ta_picked['obv_mean3_pct'] = ta.obv.rolling(3).mean().pct_change()

        # Strong rallies in price should see the force index rise.
        # During pullbacks and sideways movements, the force index will often fall because the volume
        # and/or the size of the price moves gets smaller.
        # => Encoding the percent variation could be a good idea
        ta_picked['fi_13_pct'] = ta.fi_13.pct_change()
        ta_picked['fi_50_pct'] = ta.fi_50.pct_change()

        # The Aroon Oscillator is a trend-following indicator that uses aspects of the
        # Aroon Indicator (Aroon Up and Aroon Down) to gauge the strength of a current trend
        # and the likelihood that it will continue.
        # It moves between -100 and 100. A high oscillator value is an indication of an uptrend
        # while a low oscillator value is an indication of a downtrend.
        ta_picked['ao_14'] = ta.ao_14 / 100

        # The average true range (ATR) is a technical analysis indicator that measures market volatility
        #   by decomposing the entire range of an asset price for that period.
        # ATRP is pct_change of volatility
        ta_picked['atrp_14'] = ta.atrp_14

        # Percentage Volume Oscillator (PVO) is momentum volume oscillator used in technical analysis
        #   to evaluate and measure volume surges and to compare trading volume to the average longer-term volume.
        # PVO does not analyze price and it is based solely on volume.
        #  It compares fast and slow volume moving averages by showing how short-term volume differs from
        #  the average volume over longer-term.
        #  Since it does not care a trend's factor in its calculation (only volume data are used)
        #  this technical indicator cannot be used alone to predict changes in a trend.
        ta_picked['pvo_12_26'] = ta.pvo_12_26

        # IGNORED: tsi, wd, adx,

        #lagged_stats = pd.concat([ohlcv_stats] + [builder.make_lagged(ohlcv_stats, i) for i in range(1,10+1)], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Build the dataframe with base features
        # lagged_close = pd.concat([ohlcv.close.pct_change()] + [builder.make_lagged(ohlcv.close.pct_change(), i) for i in range(1,10+1)], axis='columns', verify_integrity=True, sort=True, join='inner')
        # lagged_close.columns = ['close_pct'] + ['close_pct_lag-{}'.format(i) for i in range(1, W +1)]

        ohlc = ohlcv[['close', 'volume']].pct_change()
        lagged_ohlc = pd.concat([ohlc] + [builder.make_lagged(ohlc, i) for i in range(1, W + 1)], axis='columns',
                                verify_integrity=True, sort=True, join='inner')

        # Add lagged features to the dataframe
        improved_df = pd.concat([ohlcv_stats, lagged_ohlc, cm_picked, ta_picked], axis='columns', verify_integrity=True, sort=True, join='inner')

        # Drop the first 30 rows
        improved_df = improved_df[30:]
        # Drop columns whose values are all nan or inf
        with pd.option_context('mode.use_inf_as_na', True): # Set option temporarily
            improved_df = improved_df.dropna(axis='columns', how='all')
        logger.info('Saving {}'.format(_sym))
        save_symbol_dataset(dest_index, _sym, improved_df, target=_target)
        logger.info('Saved {}'.format(_sym))