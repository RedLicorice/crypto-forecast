import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from lib.log import logger
from lib.trading.models import Asset, Order, Equity, Signal, SignalType, OrderType, OrderStatus
from lib.dataset import load_symbol
from matplotlib import pyplot as plt
from adjustText import adjust_text

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

class Backtest:
    def __init__(self, name, exchange, strategies, sessionFactory, orderSize=0.1):
        self.createSession = sessionFactory
        self.name = name
        self.strategies = strategies
        self.exchange = exchange
        self.order_size = orderSize

    def __repr__(self):
        return 'Simulator(name={})'.format(self.name)

    def trading_period(self, begin, end):
        begin = datetime.strptime(begin, "%Y-%m-%d")
        end = datetime.strptime(end, "%Y-%m-%d")
        for day in daterange(begin, end):
            self.trading_day(day)#ToDo: maybe parallelize day operations, though it'd make no sense IRL

    def trading_day(self, day):
        session = self.createSession()
        for symbol, strat in self.strategies.items():
            try:
                signal = strat.get_signal(day) #ToDo: Handle exceptions
            except Exception as e:
                logger.debug("Exception for {} at day {}: {}".format(symbol, day, e))
                continue
            session.add(signal) # Save signals to db for history
            # Get asset
            asset = self.exchange.get_or_create_asset(session, symbol)
            # Manage open positions
            longs = self.exchange.get_open_long(session, asset)
            for o in longs:
                if o.should_stop(signal.close):
                    logger.info('[Day: {}] Closing long position {} on {} due to stop loss'.format(day, o.id, o.symbol))
                    _, log = self.exchange.close_order(day, asset, o, o.stop_loss)
                    session.add(log)
                    continue

                # If signal is SELL or position has a 1% profit
                if signal.signal == SignalType.SELL:
                    logger.info(
                        '[Day: {}] Closing long position {} on {} due to SELL signal'.format(day, o.id, o.symbol))
                    _, log = self.exchange.close_order(day, asset, o, signal.close)
                    session.add(log)
                    continue
            shorts = self.exchange.get_open_short(session, asset)
            for o in shorts:
                # If close meets stop loss, close position
                if o.should_stop(signal.close):
                    logger.info(
                        '[Day: {}] Closing short position {} on {} due to stop loss'.format(day, o.id, o.symbol))
                    _, log = self.exchange.close_order(day, asset, o, o.stop_loss)
                    session.add(log)
                    continue
                # If signal is BUY we're going to lose money, so we close position
                if signal.signal == SignalType.BUY:
                    logger.info(
                        '[Day: {}] Closing short position {} on {} due to BUY signal'.format(day, o.id, o.symbol))
                    _, log = self.exchange.close_order(day, asset, o, signal.close)
                    session.add(log)
                    continue
                # If signal is HOLD and position is old
                if o.get_age_in_days(day) > 2 and signal == SignalType.HOLD:
                    logger.info('[Day: {}] Closing short position {} on {} due to age'.format(day, o.id, o.symbol))
                    _, log = self.exchange.close_order(day, asset, o, signal.close)
                    session.add(log)
                    continue
            ## Open new positions
            # Determine position sizing
            position_coins = asset.position_size(signal.close, self.order_size)
            # Open the order
            if signal.signal == SignalType.BUY:
                logger.info(
                    '[Day: {}] Opening long position on {} due to BUY signal [Close {}, Price {}, Coins {}]'.format(
                        day, symbol, signal.close, position_coins * signal.close, position_coins))
                o, log = self.exchange.open_order(day, OrderType.LONG, asset, position_coins, signal.close,
                                        stop_loss=-0.01)  # Stop loss is -1%
                if not o:
                    logger.error("LONG FAILED")
                else:
                    session.add(o)
                    session.add(log)
            elif signal.signal == SignalType.SELL:
                logger.info(
                    '[Day: {}] Opening short position on {} due to BUY signal [Close {}, Price {}, Coins {}]'.format(
                        day, symbol, signal.close, position_coins * signal.close, position_coins))
                o, log = self.exchange.open_order(day, OrderType.SHORT, asset, position_coins, signal.close,
                                        stop_loss=0.01)  # Stop loss is +1%
                if not o:
                    logger.error("SHORT FAILED")
                else:
                    session.add(o)
                    session.add(log)
            session.add(Equity(
                day=day,
                symbol=symbol,
                equity=asset.equity(signal.close),
                longs=asset.long_orders,
                shorts=asset.short_orders
            ))
            session.commit()

    def plot_period(self, symbol, begin, end, save=None):
        begin = datetime.strptime(begin, "%Y-%m-%d")
        end = datetime.strptime(end, "%Y-%m-%d")
        session = self.createSession()
        asset = self.exchange.get_asset(session, symbol)
        long = self.exchange.get_orders_between(session, asset, begin, end, OrderType.LONG)
        short = self.exchange.get_orders_between(session, asset, begin, end, OrderType.SHORT)
        equities = session.query(Equity).filter(Equity.symbol == asset.symbol, Equity.day >= begin, Equity.day <= end).all()
        equity = pd.Series({e.day: e.equity for e in equities})
        features, target, close = load_symbol('atsa', symbol)
        if not close.loc[begin:end].shape[0]:
            print("Skip plot of {}".format(symbol))
            return

        daysToIndex = {d:i for i,d in enumerate(daterange(begin, end))}
        days = [i for i in daysToIndex.values()]
        labels = [d.strftime("%Y-%m-%d") for d in daterange(begin, end)]
        fig, (axl, axs, axe) = plt.subplots(3, 1, figsize=(12,18), sharex=True)
        plt.xticks(ticks=days, labels=labels, rotation=90)
        # Plot Longs
        axl.set_title("{} Long Orders on Close from {} to {}".format(asset.symbol, begin.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        axl.grid(axis='x', color='0.5', linestyle='--')
        axl.plot(days, close.loc[begin:end].values)

        textsl = []
        for o in long:
            x = daysToIndex[o.open_at]
            y = o.open_price
            axl.scatter(x, y, color='green', marker='o')
            #ax.annotate("O:{}".format(o.id), xy=(x, y), xytext=(5,2), textcoords='offset points', ha='left', va='bottom')
            textsl.append(axl.text(x, y, "O:{}".format(o.id)))
            if o.closed_at:
                x = daysToIndex[o.closed_at]
                y = o.close_price
                axl.scatter(x, y, color='red', marker='o')
                #ax.annotate("C:{}".format(o.id), xy=(x, y), xytext=(5,2), textcoords='offset points', ha='left', va='bottom')
                textsl.append(axl.text(x, y, "C:{}".format(o.id)))
        adjust_text(textsl, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), ax=axl)

        # Plot Shorts
        axs.set_title("{} Short Orders on Close from {} to {}".format(asset.symbol, begin.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        axs.grid(axis='x', color='0.5', linestyle='--')
        axs.plot(days, close.loc[begin:end].values)

        textss = []
        for o in short:
            x = daysToIndex[o.open_at]
            y = o.open_price
            axs.scatter(x, y, color='green', marker='D')
            #ax.annotate("O:{}".format(o.id), xy=(x, y), xytext=(5,2), textcoords='offset points', ha='left', va='bottom')
            textss.append(axs.text(x, y, "O:{}".format(o.id)))
            if o.closed_at:
                x = daysToIndex[o.closed_at]
                y = o.close_price
                axs.scatter(x, y, color='red', marker='D')
                #ax.annotate("C:{}".format(o.id), xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='left', va='bottom')
                textss.append(axs.text(x, y, "C:{}".format(o.id)))
        adjust_text(textss, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), ax=axs)

        # Plot Equity
        if equity.shape[0]:
            axe.set_title("{} Equity from {} to {}".format(asset.symbol, begin.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
            axe.grid(axis='x', color='0.5', linestyle='--')
            axe.plot(days, equity.values, color='red')
            axe.scatter(days, equity.values, color='red', marker='o')

            texte = []
            for e in equities:
                x = daysToIndex[e.day]
                y = e.equity
                texte.append(axe.text(x, y, "[L:{} S:{}]".format(e.longs, e.shorts)))
            adjust_text(texte, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), ax=axe)

        plt.tight_layout()
        if save:
            plt.savefig(save)
        plt.close()