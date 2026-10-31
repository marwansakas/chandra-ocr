# OpenOCR attribution

OpenOCR, https://github.com/Topdu/OpenOCR, Apache License 2.0 (LICENSE).
Pinned source: 1ccfc6ee6161f7133b192e16af3a9265273997ba.
svtrv2.py originates from openrec/modeling/encoders/svtrv2_lnconv_two33.py;
common_upstream.py originates from openrec/modeling/common.py.

Changes: relative common import; mathematically equivalent Torch SDPA attention
to avoid materializing large full-line attention matrices. No encoder parameter
names or shapes changed. BatchNorm running statistics are frozen by the wrapper;
encoder operates separately on each unpadded line. No pretrained positional
embeddings exist in the selected use_pos_embed=false configuration.

Official weights/config: topdu/MDiff4STR, mdiff4str_small,
Hugging Face revision 70c320bb8ab539ecc08f820ef94ba7679b6e40f2.
Only the encoder is transferred. English decoder/output weights are discarded.
This full-line Arabic decoder is an independent adaptation of MDiff4STR ideas,
not a faithful reproduction or an official Arabic checkpoint.
