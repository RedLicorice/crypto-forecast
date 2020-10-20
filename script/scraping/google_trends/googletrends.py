from datetime import datetime
import os
import json
from pytrends.request import TrendReq
import pandas as pd
from functools import reduce

TRENDS_MIN_DATE="2015-01-08 00:00:00"

# Split time interval in periods of specifdied length
def get_intervals(b, e, duration=3600 * 24 * 7):
	beg = b.timestamp()
	end = e.timestamp()
	results = []
	while beg < end:
		next = beg + duration - 1
		if next > end:
			next = end
		results.append((datetime.fromtimestamp(beg), datetime.fromtimestamp(next)))
		beg = next + 1
	return reversed(results)

# Split a list l in successive n-sized chunks
def chunks(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]

class GoogleTrendsScraper:
	basename = None
	filename = None
	result = None

	def __init__(self, basename='trends'):
		self.basename = basename
		self.pytrend = TrendReq(hl='it-IT', backoff_factor=0.2)

	# Build a state with operations to be done
	# A state is made of chunks, a chunk consists of a set of search terms and
	# a list of queries (one for each time interval)
	def build_state(self, begin, end, symbols):
		# Data is absent
		begin = datetime.strptime(begin, "%Y-%m-%d")
		end = datetime.strptime(end, "%Y-%m-%d")
		f = datetime.strptime(TRENDS_MIN_DATE, "%Y-%m-%d %H:%M:%S")
		if begin.timestamp() < f.timestamp():
			begin = f
		#
		state = {
			"begin": begin.strftime("%Y-%m-%d %H:%M:%S"),
			"end": end.strftime("%Y-%m-%d %H:%M:%S"),
		}

		for c,terms in enumerate(chunks(symbols, 5)):
			if not c in state:
				state[c] = {
					"terms": terms,
					"queries":[]
				}

			for f, (b, e) in enumerate(get_intervals(begin, end)):
				# b, e = t
				query = {
					"begin": b.strftime("%Y-%m-%d %H:%M:%S"),
					"end": e.strftime("%Y-%m-%d %H:%M:%S"),
					"done":False,
					"filename":None
				}
				state[c]["queries"].append(query)
		return state

	# Save a state
	def save_state(self, state, filename = None):
		if not self.filename:
			if not filename:
				raise RuntimeError("Need a filename!")
			self.filename = filename
		with open(self.filename, "w") as fp:
			json.dump(state, fp, indent=4)

	# Load a state
	def load_state(self, filename):
		self.filename = filename
		with open(filename, "r") as fp:
			return json.load(fp)

	def is_state_complete(self, state):
		for c, chunk in state.items():
			if not type(chunk) is dict:
				continue
			for query in chunk["queries"]:
				if not query["done"]:
					return False
		return True

	def run(self, state):
		# When a query succeeds, save results and update the state
		complete = True
		for c, chunk in state.items():
			if not type(chunk) is dict:
				continue
			for q, query in enumerate(chunk["queries"]):
				if query["done"]:
					continue
				df = self.do_query(chunk["terms"], query)
				if not df.empty:
					query["done"] = True
					query["filename"] = self.save_query_result(c, q, df)
					self.save_state(state)
				else:
					complete = False
					print("Failed to fetch chunk {} query {}".format(c, q))
					print("Head:")
					df.head()
		if self.is_state_complete(state):
			self.save(state)
			print("State exported!")
		else:
			print("State is incomplete. Please rerun!")

	def do_query(self, terms, query):
		b = datetime.strptime(query["begin"], "%Y-%m-%d %H:%M:%S")
		e = datetime.strptime(query["end"], "%Y-%m-%d %H:%M:%S")
		print("{}|{}|Query: Terms: [{}] Begin: \"{}\" End: \"{}\"".format(
			datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.basename, ','.join(terms), query["begin"], query["end"])
		)
		df = self.pytrend.get_historical_interest(
			terms,
			year_start=b.timetuple().tm_year, month_start=b.timetuple().tm_mon, day_start=b.timetuple().tm_mday, hour_start=0,
			year_end=e.timetuple().tm_year, month_end=e.timetuple().tm_mon, day_end=e.timetuple().tm_mday, hour_end=0,
			cat=0, geo='', gprop='', sleep=30
		)
		return df

	def save_query_result(self, c, f, df, dir='temp'):
		filename = "{}_{}-{}.csv".format(self.basename, c, f)
		if dir and not os.path.isdir(dir):
			os.makedirs(dir, True)
		filename = os.path.join(dir, filename)
		df.to_csv(filename, sep=',', encoding='utf-8')
		return filename

	def save(self, state, dir = 'output'):
		b = datetime.strptime(state["begin"], "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S")
		e = datetime.strptime(state["end"], "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S")
		filenames = []
		for c, chunk in state.items():
			if not type(chunk) is dict:
				continue
			# Rebuild full csv
			files = [q["filename"] for q in chunk["queries"] if q["done"] is True]
			merged = pd.concat([pd.read_csv(f, index_col=None, header=0, sep=',', encoding='utf-8') for f in files], axis=0)
			# Sort by date
			merged['date'] = pd.to_datetime(merged.date)
			merged.sort_values(by='date', ascending=True, inplace=True)
			# Save result
			filename = "{}{}_{}_{}_{}.csv".format(self.basename, c, ','.join(chunk["terms"]), b, e)
			if dir and not os.path.isdir(dir):
				os.makedirs(dir, True)
			filename = os.path.join(dir, filename)
			merged.drop("isPartial", axis=1, inplace=True) # drop isPartial column
			merged.to_csv(filename, sep=',', encoding='utf-8', index=False)
			filenames.append(filename)
		# Merge all datasets in a big one
		filename = "{}_all_{}_{}.csv".format(self.basename, b, e)
		filename = os.path.join(dir, filename)
		dfs = [pd.read_csv(f, header=0, sep=',', encoding='utf-8') for f in filenames]
		main = reduce(lambda left, right: pd.merge(left, right, on='date'), dfs)
		main.to_csv(filename, sep=',', encoding='utf-8', index=False)