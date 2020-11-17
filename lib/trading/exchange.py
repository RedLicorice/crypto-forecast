from lib.trading.models import Asset, Order, OrderType, OrderStatus
from sqlalchemy import and_

class Exchange:
    MARGIN_SHORT_FIXED_FEE = 0.0001 # 0.01% fee at position open
    MARGIN_SHORT_ROLLING_FEE = 0.0002 * 5 # Kraken applies 0.02% fee every 4 hours period after the first so 24/4 - 1
    MARGIN_LONG_FIXED_FEE = 0.0001 # 0.01% fee at position open
    MARGIN_LONG_ROLLING_FEE = 0.0001 * 5 # Kraken applies 0.02% fee every 4 hours period after the first so 24/4 - 1
    SPOT_FIXED_FEE = 0.0026 # 0.16-0.26% fee at every spot transaction (maker-taker)
    USE_MARGIN_COINS = False

    def __init__(self, session):
        self.db = session

    def get_or_create_asset(self, symbol, **kwargs):
        asset = self.db.query(Asset).filter(Asset.symbol == symbol).first()
        if not asset:
            kwargs.update({'symbol': symbol})
            asset = Asset(**kwargs)
            self.db.add(asset)
        return asset

    def fund_margin_fiat(self, asset, amount):
        if asset.fiat < amount:
            #raise Exception("Not enough FIAT to fund margin account!")
            return False
        asset.fiat -= amount
        asset.margin_fiat += amount
        return True

    def fund_margin_coins(self, asset, amount, price=None):
        if not self.USE_MARGIN_COINS:
            if not price:
                raise Exception("Margin coins balance is disabled, funding requires close price.")
            return self.fund_margin_fiat(asset, amount*price)
        if asset.coins < amount:
            #raise Exception("Not enough COINS to fund margin account!")
            return False
        asset.coins -= amount
        asset.margin_coins += amount
        return True

    def can_open_order(self, asset: Asset , o: Order):
        # Margin trades usually require a FIAT balance greater or equal than the collateral value,
        # Exchanges impose an hard cap on margin orders
        # In margin long orders, there's a fixed + rolling fee which is paid in the base currency
        # In margin short orders, there's a fixed + rolling fee which is paid in cryptocurrency,
        #   but we suppose it is converted at market price.
        # Only requirement would be 1.5x the order's value, but we simplify it to 1.0x
        if o.type == OrderType.LONG:
            # Each user has an allowance limit to the fiat he can borrow from the pool,
            # this varies with trading volume, we suppose it is fixed
            if asset.long_allowance < o.open_price * o.coins:
                return False
            # Our margin account should have 1.5x the FIAT value we're asking to the broker,
            # this requirement has been relapsed to 1.0x. Opening fee is not counted,
            # since the amount is not deducted (it is just a warranty)
            if asset.margin_fiat < o.open_price * o.coins:
                return False
        elif type == OrderType.SHORT:
            # Each user has an allowance limit to the coins he can borrow from the pool,
            # this varies with trading volume, we suppose it is fixed
            if asset.short_allowance < o.coins:
                return False
            # Our margin account should have 1.5x the COINS value we're asking to the broker,
            # this requirement has been relapsed to 1.0x. Opening fee is not counted,
            # since the amount is not deducted (it is just a warranty)
            if self.USE_MARGIN_COINS:
                if asset.margin_coins < o.coins:
                    return False
            else:
                if asset.margin_fiat < o.coins*o.open_price:
                    return False
        elif type == OrderType.SPOT:
            # In spot orders, fee is paid upfront over position's opening transaction,
            # so we must own it all.
            if asset.fiat < o.open_price * o.coins + self.get_open_fee(o):
                return False
        return True

    def can_close_order(self, asset: Asset , o: Order):
        if o.type == OrderType.LONG:
            #In order to close a margin long, we want to sell the coins we purchased
            # - therefore, we must still own them.
            if asset.margin_fiat < o.coins*o.open_price:
                return False
            # If our profit is < 0 something went wrong (maybe fees exceed our profit?! This should never happen!)
            #if o.coins * (o.close_price - o.open_price) - self.get_close_fee(o) < 0:
            #    raise Exception("Negative profit for Margin LONG ID: {}".format(o.id))
        elif o.type == OrderType.SHORT:
            if self.USE_MARGIN_COINS:
                if asset.margin_coins < o.coins + self.get_close_fee(o):
                    return False
            else:
                if asset.margin_fiat < (o.coins + self.get_close_fee(o))*o.close_price:
                    return False
        elif o.type == OrderType.SPOT:
            if asset.coins < o.coins:
                return False
            if asset.fiat < self.get_close_fee(o):
                return False
        return True

    def get_open_fee(self, o: Order):
        if o.type == OrderType.SHORT:
            # When shorting, we need to pay back the debt we have with the broker, as well as fees.
            # Fees are paid in Native tokens
            # Fixed fee for kraken BTCUSD is 0.01%
            # Variable fee for kraken BTCUSD is 0.01% every 4 hours period, excluding the first so 0.08% every 24 hours
            open_fee = o.coins * self.MARGIN_SHORT_FIXED_FEE # Fixed fee
            return open_fee # Short order fees are paid in COINS
        elif o.type == OrderType.LONG:
            open_fee = o.coins * self.MARGIN_LONG_FIXED_FEE  # Fixed fee
            return open_fee * o.open_price # Long order fees are paid in FIAT
        elif o.type == OrderType.SPOT:
            # In spot orders, fee is paid for every transaction
           return self.SPOT_FIXED_FEE * (o.coins * o.open_price)

    def get_close_fee(self, o: Order):
        position_days = o.get_age_in_days(o.closed_at)
        if o.type == OrderType.SHORT:
            close_fee =  o.coins * (self.MARGIN_SHORT_ROLLING_FEE * position_days)  # variable fee
            return close_fee
        elif o.type == OrderType.LONG:
            close_fee = o.coins * (self.MARGIN_LONG_ROLLING_FEE * position_days)
            return close_fee * o.close_price
        elif o.type == OrderType.SPOT:
            # In spot orders, fee is paid for every transaction
            return self.SPOT_FIXED_FEE * (o.coins * o.close_price)

    def get_open_short(self, asset):
        return self.db.query(Order).filter(and_(Order.symbol == asset.symbol, Order.type == OrderType.SHORT, Order.status == OrderStatus.OPEN)).all()

    def get_open_long(self, asset):
        return self.db.query(Order).filter(and_(Order.symbol == asset.symbol, Order.type == OrderType.LONG, Order.status == OrderStatus.OPEN)).all()

    def get_open_spot(self, asset):
        return self.db.query(Order).filter(and_(Order.symbol == asset.symbol, Order.type == OrderType.SPOT, Order.status == OrderStatus.OPEN)).all()

    def open_order(self, day, type: OrderType, asset: Asset, coins, price, stop_loss=0.01):
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
        # Fail if order can't be placed
        if not self.can_open_order(asset, o):
            return None
        if o.type == OrderType.LONG:
            # Deduct order from allowance, which is in fiat for margin longs
            asset.long_allowance -= o.open_price * o.coins
            # In margin long orders we purchase coins using FIAT lent from our broker (so subject to allowance)
            # Fixed opening fee is paid in FIAT
            if self.USE_MARGIN_COINS:
                asset.margin_coins += o.coins
            asset.margin_fiat -= self.get_open_fee(o)
            # Increase long orders count
            asset.long_orders += 1
        elif o.type == OrderType.SHORT:
            # Deduct order from allowance, which is in coin for margin short
            asset.short_allowance -= o.coins
            # In margin short orders we sell coins lent from our broker (subject to allowance)
            # Fixed opening fee is paid in coins, we deduct it from lent coins
            asset.margin_fiat += o.open_price* ( o.coins - self.get_open_fee(o))# We sell lent coins minus open fee at open price
            # Increase short orders count
            asset.short_orders += 1
        elif o.type == OrderType.SPOT:
            # Deduct order buy price + fee from FIAT wallet
            asset.fiat -= o.open_price*o.coins + self.get_open_fee(o)
            # Add purchased coins to balance
            asset.coins += o.coins
            # Increase spot orders count
            asset.spot_orders += 1
        self.db.add(o)
        return o

    def close_order(self, day, asset: Asset, o: Order, price):
        o.close_price = price
        o.closed_at = day

        if not self.can_close_order(asset, o):
            raise Exception("Cannot close order {}".format(o.id))
            #return None
        o.status=OrderStatus.CLOSED

        if o.type == OrderType.LONG:
            # Sell coins we purchased when opening position, at close_price
            # and give back the borrowed fiat (value of coins at open price)
            # so profit P would be: P = coins*close - coins*open - rolling_fee = coins*(close - open) - rolling_fee
            if self.USE_MARGIN_COINS:
                asset.margin_coins -= o.coins
            asset.margin_fiat += o.coins * (o.close_price - o.open_price) - self.get_close_fee(o)
            # Restore allowance, which is in fiat for margin longs
            asset.long_allowance += o.open_price * o.coins
            # Increase long orders count
            asset.long_orders -= 1

            o.profit = o.coins * (o.close_price - o.open_price) - self.get_close_fee(o)
            o.open_fee = self.get_open_fee(o)
            o.close_fee = self.get_close_fee(o)
        elif o.type == OrderType.SHORT:
            # Buy back the coins we borrowed when opening position + close_fee, at close_price
            give_back_amount = o.coins + self.get_close_fee(o)
            asset.margin_fiat -= o.close_price * give_back_amount
            # Restore allowance, which is in COINS for margin short
            asset.short_allowance += o.coins
            # Increase short orders count
            asset.short_orders -= 1

            o.profit = (o.coins- self.get_open_fee(o) - self.get_close_fee(o))* (o.open_price - o.close_price)
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
        return o