import logging
import yaml


class YParams:
    """
    Small YAML parameter helper used by the starter examples.

    Parameters are available both as attributes and through ``params``.
    """

    def __init__(self, yaml_filename, config_name, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as yaml_file:
            configs = yaml.safe_load(yaml_file)

        if config_name not in configs:
            available = ", ".join(configs.keys())
            raise KeyError(
                f"Config {config_name!r} not found in {yaml_filename}. "
                f"Available configs: {available}"
            )

        for key, val in configs[config_name].items():
            if print_params:
                print(key, val)
            if val == "None":
                val = None
            self.params[key] = val
            self.__setattr__(key, val)

        if print_params:
            print("---------------------------------------------------")

    def __setattr__(self, key, val):
        if key not in {"_yaml_filename", "_config_name", "params"} and "params" in self.__dict__:
            self.params[key] = val
        super().__setattr__(key, val)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def get(self, key, default=None):
        return self.params.get(key, default)

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + " " + str(val))
        logging.info("---------------------------------------------------")
