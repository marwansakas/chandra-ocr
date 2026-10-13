"""TrOCR processor compatibility for legacy RoBERTa vocab/merges checkpoints."""

import json
import warnings


def load_processor(model_name, **options):
    """Keep normal/custom processors intact; repair only the known legacy failure.

    Transformers 5 can route TrOCR's legacy RoBERTa files to the generic tokenizer
    backend, which requires tokenizer.json even when vocab.json/merges.txt exist.
    An explicit RobertaTokenizer loads the checkpoint's original vocabulary and
    special tokens on both supported Transformers major versions.
    """
    from transformers import AutoImageProcessor, RobertaTokenizer, TrOCRProcessor
    from transformers.utils.hub import cached_file

    try:
        return TrOCRProcessor.from_pretrained(model_name, **options)
    except ValueError as error:
        if "Couldn't instantiate the backend tokenizer" not in str(error):
            raise
        metadata_path = cached_file(model_name, "tokenizer_config.json",
                                    revision=options.get("revision"),
                                    local_files_only=options.get("local_files_only", False))
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("tokenizer_class") not in ("RobertaTokenizer", "RobertaTokenizerFast"):
            # Never guess a tokenizer for an Arabic/custom checkpoint.
            raise
        tokenizer = RobertaTokenizer.from_pretrained(model_name, **options)
        image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False, **options)
        warnings.warn("Loaded legacy TrOCR tokenizer explicitly from its original RoBERTa files.",
                      UserWarning, stacklevel=2)
        return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
