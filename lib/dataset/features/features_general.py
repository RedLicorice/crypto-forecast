import pandas as pd

def pct_change(df, **kwargs):
	periods = kwargs.get('periods', 1)
	# calcola la variazione percentuale su 1 periodo di default
	df = df.pct_change(periods=periods)
	df.columns = [c+'_p{}'.format(periods) for c in df.columns]
	# Prendi solo le righe valide
	df = df.loc[df.first_valid_index():df.last_valid_index()]
	# Riempi eventuali null interpolando i dati
	df = df.interpolate(axis=1, method='linear')
	return df

def difference(df, **kwargs):
	periods = kwargs.get('periods', 1)
	# calcola la differenza su 1 periodo di default
	df = df.diff(periods=periods)
	df.columns = [c + '_d{}'.format(periods) for c in df.columns]
	# Prendi solo le righe valide
	df = df.loc[df.first_valid_index():df.last_valid_index()]
	# Riempi eventuali null interpolando i dati
	df = df.interpolate(axis=1, method='linear')
	return df

def make_lagged(df, periods=1):
	shift = df.shift(periods=periods)
	if hasattr(shift, 'columns'):
		shift.columns = ['{}_lag_-{}'.format(c, periods) for c in shift.columns]
	elif hasattr(shift, 'name'):
		shift.rename('{}_lag_-{}'.format(shift.name, periods))
	return shift