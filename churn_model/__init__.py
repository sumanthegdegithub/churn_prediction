from churn_model.config.core import parent, config
with open(parent / "VERSION") as version_file:
    __version__ = version_file.read().strip()