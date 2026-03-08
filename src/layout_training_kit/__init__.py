"""layout_training_kit package."""

__version__ = "0.1.0"

try:
	from .celery_app import layout_celery
except Exception:
	layout_celery = None

__all__ = ["__version__", "layout_celery"]
