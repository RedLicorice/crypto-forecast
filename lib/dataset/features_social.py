import pandas as pd


def blockchain_symbol_diff(blockchain, symbol, **kwargs):
	periods = kwargs.get('periods', 1)
	# Prendi solo le features il cui nome inizia per il simbolo scelto
	df = blockchain[[c for c in blockchain.columns if c.startswith(symbol)]]
	df.columns = ['{}_d{}'.format(c.lower(), periods) for c in df.columns]
	# calcola la differenza su 1 periodo di default
	df = df.diff(axis=0, periods=kwargs.get('periods', 1))
	# Prendi solo le righe valide
	df = df.loc[df.first_valid_index():df.last_valid_index()]
	# Riempi eventuali null interpolando i dati
	df = df.interpolate(axis=1, method='linear')
	return df

def blockchain_symbol_pct_change(blockchain, symbol, **kwargs):
	periods = kwargs.get('periods', 1)
	# Prendi solo le features il cui nome inizia per il simbolo scelto
	df = blockchain[[c for c in blockchain.columns if c.startswith(symbol)]]
	df.columns = ['{}_p{}'.format(c.lower(), periods) for c in df.columns]
	# calcola la variazione percentuale su 1 periodo di default
	df = df.pct_change(periods=periods)
	# Prendi solo le righe valide
	df = df.loc[df.first_valid_index():df.last_valid_index()]
	# Riempi eventuali null interpolando i dati
	df = df.interpolate(axis=1, method='linear')
	return df