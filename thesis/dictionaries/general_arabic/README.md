# General Arabic dictionary

The general-Arabic TrOCR decoder uses the Arabic Hunspell dictionary maintained
by LibreOffice. The source files are not committed because `ar.dic` is several
megabytes and should retain its upstream licensing files.

Download the pinned dictionary release from the repository root:

```bash
python thesis/scripts/download_general_arabic_dictionary.py
```

This creates `ar.aff`, `ar.dic`, `AUTHORS.txt`, `COPYING.txt`, and
`README_ar.txt` in this directory. The downloader pins LibreOffice commit
`32b006a2c22a4ac7e8ed3f03346f7b3d85a970a4` so experiments remain
reproducible.

Source: https://github.com/LibreOffice/dictionaries/tree/master/ar

The dictionary is used only during inference. It is not ground truth and is not
used to train the visual TrOCR weights.
