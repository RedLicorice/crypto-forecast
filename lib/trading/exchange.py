#
#   This class handles Exchange-related operations.
#   For methods where DB access is needed, a 'session' parameter is required.
#   Please note that created orders need to be added to the session manually
#
#   Margin wallets hold the lent positions, they don't count for the sake of equities
from lib.trading.models import Asset, Order, OrderLog, OrderType, OrderStatus
from lib.log import logger
from sqlalchemy import and_, or_

class Exchange:
    MARGIN_SHORT_FIXED_FEE = 0.0001 # 0.01% fee at position open
    MARGIN_SHORT_DAILY_FEE = 0.0002 * 5 # Kraken applies 0.02% fee every 4 hours period after the first so 24/4 - 1
    MARGIN_LONG_FIXED_FEE = 0.0001 # 0.01% fee at position open
    MARGIN_LONG_DAILY_FEE = 0.0001 * 5 # Kraken applies 0.02% fee every 4 hours period after the first so 24/4 - 1
    SPOT_FIXED_FEE = 0.0026 # 0.16-0.26% fee at every spot transaction (maker-taker)

    def __init__(self, sessionFactory):
        self.createSession = sessionFactory

    def get_or_create_asset(self, symbol, **kwargs):
        session = self.createSession()
        asset = session.query(Asset).filter(Asset.symbol == symbol).first()
        if not asset:
            kwargs.update({'symbol': symbol})
            asset = Asset(**kwargs)
            session.add(asset)
        session.commit()
        return asset

    def get_asset(self, symbol):
        session = self.createSession()
        asset = session.query(Asset).filter(Asset.symbol == symbol).first()
        return asset

    def can_open_order(self, asset: Asset , o: Order):
        # Margin trades usually require a FIAT balance greater or equal than the collateral value,
        # Exchanges impose an hard cap on margin orders
        # In margin long orders, there's a fixed + rolling fee which is paid in the base currency
        # In margin short orders, there's a fixed + rolling fee which is paid in cryptocurrency,
        # Only requirement would be 1.5x the order's value, but we simplify it to 1.0x
        if o.type == OrderType.LONG:
            # Each user has an allowance limit to the fiat he can borrow from the pool,
            # this varies with trading volume, we suppose it is fixed
            if asset.long_allowance < o.open_price * o.coins:
                return False
            # Our <s>margin</s> account should have 1.5x the FIAT value we're asking to the broker,
            # this requirement has been relapsed to 1.0x. Opening fee is not counted,
            # since the amount is not deducted (it is just a warranty)
            if asset.fiat < o.open_price * o.coins:
                return False
        elif type == OrderType.SHORT:
            # Each user has an allowance limit to the coins he can borrow from the pool,
            # this varies with trading volume, we suppose it is fixed
            if asset.short_allowance < o.coins:
                return False
            # Our margin account should have 1.5x the value we're asking to the broker,
            # this requirement has been relapsed to 1.0x. Opening fee is not counted,
            # since the amount is not deducted (it is just a warranty)
            if asset.fiat < o.coins*o.open_price:
                return False
        elif type == OrderType.SPOT:
            # In spot orders, fee is paid upfront over position's opening transaction,
            # so we must own it all.
            if asset.fiat < o.open_price * o.coins + self.get_open_fee(o):
                return False
        return True

    def can_close_order(self, asset: Asset , o: Order):
        if o.type == OrderType.LONG:
            # To close a margin long, we need to return the amount of FIAT spent when opening the position
            debt_fiat = o.coins*o.open_price
            # To the debt, we must add the fixed fee (opening fee) and the rolling (closing) fee..
            debt_fiat += self.get_open_fee(o) + self.get_close_fee(o)
            # The debt should be paid by selling the coins we purchased, at close price..
            sell_fiat = o.coins*o.close_price
            # if sell_fiat < debt_fiat, we're at loss: we should pay the difference with our funds!
            if sell_fiat < debt_fiat:
                loss = debt_fiat - sell_fiat
                if asset.fiat < loss:
                    return False

        elif o.type == OrderType.SHORT:
            # To close a margin short, we need to return the amount of COLLATERAL spent when opening the position
            # To the debt, we must add the fixed fee (opening fee) and the rolling (closing) fee..
            debt_collateral = o.coins + self.get_open_fee(o) + self.get_close_fee(o)
            # We want to buy back our debt at CLOSE PRICE
            debt_fiat = debt_collateral * o.close_price # How much we spend from buying back the debt
            # We initially sold collateral at OPEN PRICE
            sell_fiat = o.coins*o.open_price
            # If our initial sell is lower than our debt, we're at loss: we pay difference with our funds!
            if sell_fiat < debt_fiat:
                loss = debt_fiat - sell_fiat
                if asset.fiat < loss:
                    return False

        elif o.type == OrderType.SPOT:
            if asset.coins < o.coins:
                return False
            if asset.fiat < self.get_close_fee(o):
                return False

        return True

    def get_open_fee(self, o: Order):
        if o.type == OrderType.LONG:
            # Open fee is calculated on the FIAT debt
            fiat_debt = o.coins*o.open_price
            return fiat_debt * self.MARGIN_LONG_FIXED_FEE  # Fixed fee
        elif o.type == OrderType.SHORT:
            # Open fee is calculated on the COLLATERAL debt
            return o.coins * self.MARGIN_SHORT_FIXED_FEE
        elif o.type == OrderType.SPOT:
            # In spot orders, fee is paid for every transaction's value
           return self.SPOT_FIXED_FEE * (o.coins * o.open_price)

    def get_close_fee(self, o: Order, days=None):
        position_days = o.get_age_in_days(o.closed_at) if not days else days
        if o.type == OrderType.LONG:
            # Daily fee is calculated on the FIAT debt
            fiat_debt = o.coins * o.open_price
            # Daily fee by fiat debt gives us daily interest
            daily_interest = fiat_debt * self.MARGIN_LONG_DAILY_FEE
            # Finally, closing fee is daily interest by number of days
            return daily_interest * position_days
        elif o.type == OrderType.SHORT:
            # Daily fee is calculated on the COLLATERAL debt
            daily_interest = o.coins * self.MARGIN_SHORT_DAILY_FEE
            # Closing fee is daily interest by number of days
            return  daily_interest * position_days
        elif o.type == OrderType.SPOT:
            # In spot orders, fee is paid for every transaction's value
            return self.SPOT_FIXED_FEE * (o.coins * o.close_price)

    def get_orders_between(self, asset, begin, end, type):
        session = self.createSession()
        return session.query(Order).filter(
            and_(
                Order.open_at >= begin,
                Order.symbol == asset.symbol,
                Order.type == type,
                or_(Order.closed_at <= end, Order.status == OrderStatus.OPEN)
            )
        ).all()

    def get_operations_between(self, asset, begin, end, type):
        session = self.createSession()
        return session.query(OrderLog).filter(
            and_(
                OrderLog.timestamp >= begin,
                OrderLog.timestamp <= end,
                OrderLog.type == type,
                OrderLog.symbol == asset.symbol
            )
        ).all()

    def get_open_short(self, asset):
        session = self.createSession()
        return session.query(Order).filter(and_(Order.symbol == asset.symbol, Order.type == OrderType.SHORT, Order.status == OrderStatus.OPEN)).all()

    def get_open_long(self, asset):
        session = self.createSession()
        return session.query(Order).filter(and_(Order.symbol == asset.symbol, Order.type == OrderType.LONG, Order.status == OrderStatus.OPEN)).all()

    def get_open_spot(self, asset):
        session = self.createSession()
        return session.query(Order).filter(and_(Order.symbol == asset.symbol, Order.type == OrderType.SPOT, Order.status == OrderStatus.OPEN)).all()

    def open_order(self, day, type: str, asset: Asset, coins, price, stop_loss=0.01):
        session = self.createSession()
        # Create an order instance
        o = Order(
            symbol=asset.symbol,
            type=type,
            status=OrderStatus.OPEN,
            coins=coins,
            open_price=price,
            #close_price=None,
            last_price=price,
            stop_loss=price + (price * stop_loss) if stop_loss else None,
            open_at=day,
            #closed_at=None,
        )
        log = OrderLog(
            symbol=asset.symbol,
            type=o.type,
            status=o.status,
            price=price,
            timestamp=day
        )
        # Fail if order can't be placed
        if not self.can_open_order(asset, o):
            logger.debug("[Day {}] Cannot open order for {}".format(day, asset.symbol))
            return None
        if o.type == OrderType.LONG:
            # Deduct order from allowance, which is in fiat for margin longs
            asset.long_allowance -= o.open_price * o.coins
            # In margin long orders we purchase coins using FIAT lent from our broker (so subject to allowance)
            asset.margin_coins += o.coins
            # Increase long orders count
            asset.long_orders += 1
        elif o.type == OrderType.SHORT:
            # Deduct order from allowance, which is in coin for margin short
            asset.short_allowance -= o.coins
            # In margin short orders we sell coins lent from our broker (subject to allowance) at open price
            asset.margin_fiat += o.coins*o.open_price
            # Increase short orders count
            asset.short_orders += 1
        elif o.type == OrderType.SPOT:
            # Deduct order buy price + fee from FIAT wallet
            asset.fiat -= o.open_price*o.coins + self.get_open_fee(o)
            # Add purchased coins to balance
            asset.coins += o.coins
            # Increase spot orders count
            asset.spot_orders += 1
        session.add(o)
        session.add(log)
        session.commit()
        return o, log

    def close_order(self, day, asset: Asset, o: Order, price):
        session = self.createSession()
        if o.status == OrderStatus.CLOSED:
            logger.error('Order {} is already closed'.format(o.id))
            return
        o.close_price = price
        o.closed_at = day

        o.status=OrderStatus.CLOSED
        log = OrderLog(
            symbol=asset.symbol,
            type=o.type,
            status=o.status,
            price=price,
            timestamp=day
        )

        if o.type == OrderType.LONG:
            # Sell coins we purchased when opening position, at close_price
            asset.margin_coins -= o.coins
            sell_fiat = o.coins * o.close_price
            # Debt is calculated on opening FIAT price
            debt_fiat = o.coins * o.open_price + self.get_open_fee(o) + self.get_close_fee(o)
            # Profit is what's left after we pay back the debt
            # If negative, will be deducted from fiat
            profit = sell_fiat - debt_fiat
            asset.fiat += profit
            # Restore allowance, which is in fiat for margin longs
            asset.long_allowance += o.open_price * o.coins
            # Increase long orders count
            asset.long_orders -= 1

            o.profit = profit
            o.open_fee = self.get_open_fee(o)
            o.close_fee = self.get_close_fee(o)
        elif o.type == OrderType.SHORT:
            # Debt is given by coins + open fee + interest (close fee)
            debt_collateral = o.coins + self.get_open_fee(o) + self.get_close_fee(o)
            # Buy coins to fill our debt, at close_price
            debt_fiat = debt_collateral * o.close_price
            # Profit is what's left of sell_fiat  after buying back the coins
            sell_fiat = o.coins * o.open_price
            asset.margin_fiat -= sell_fiat
            profit = sell_fiat - debt_fiat
            asset.fiat += profit

            o.profit = profit
            o.open_fee = self.get_open_fee(o)
            o.close_fee = self.get_close_fee(o)
        elif o.type == OrderType.SPOT:
            # Deduct sold coins from balance
            asset.coins -= o.coins
            # Trading fee is deducted from the sale profit
            asset.fiat += o.close_price * o.coins - self.get_close_fee(o)
            # Increase spot orders count
            asset.spot_orders -= 1

            o.profit = o.coins * (o.close_price - o.open_price) - (self.get_open_fee(o)+ self.get_close_fee(o))
            o.open_fee = self.get_open_fee(o)
            o.close_fee = self.get_close_fee(o)
        session.add(o)
        session.add(log)
        session.commit()
        return o, log
