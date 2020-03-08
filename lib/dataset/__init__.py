import importlib
import pkgutil
from lib.log import logger

class DatasetFactory:
	registered_features = {}
	registered_targets = {}
	registered_modules = None

	@classmethod
	def discover(cls):
		if cls.registered_modules:
			return
		cls.registered_modules = {
			name: importlib.import_module(name)
			for finder, name, ispkg
			in pkgutil.iter_modules(__path__, __name__ + ".")
		}

	@classmethod
	def register_features(cls, name, model):
		#logger.debug("Registered features {} from {}".format(name, str(model)))
		cls.registered_features[name] = model

	@classmethod
	def register_target(cls, name, model):
		#logger.debug("Registered target {} from {}".format(name, str(model)))
		cls.registered_targets[name] = model

	@classmethod
	def get_features(cls, features, input, **kwargs):
		if features not in cls.registered_features:
			raise ValueError("Feature {} not registered!".format(features))
		return cls.registered_features[features](input, **kwargs)

	@classmethod
	def get_target(cls, target, input, **kwargs):
		if target not in cls.registered_targets:
			raise ValueError("Target {} not registered!".format(target))
		return cls.registered_targets[target](input, **kwargs)