"""Model implementations are imported only when selected."""

from importlib import import_module


def load_backend(spec, options):
    module = import_module(f"pretrained_ocr.models.{spec.module}")
    return module.Backend(spec, options)
