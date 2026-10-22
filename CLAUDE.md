# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CalliReader (ICCV 2025): a plug-and-play add-on to InternVL2 for Chinese calligraphy OCR/understanding. Research code — no test suite, linter, or build step. Needs CUDA (`inference.py` calls `.cuda()`); upstream targets Linux + `flash-attn`, but Windows works via the 4-bit path below.

## Commands

```
pip install -r requirements.txt && pip install flash-attn
python inference.py --tgt=<image or folder> [--save_name=x]   # folder results -> ./results/
python evaluate.py --type=<full_page|region_wise|choice|bilingual|intent> --data=./CalliBench --save_name=exp
python get_single_embeddings.py   # synthesize pseudo-text embeddings (.pt) from a jpg + json annotation
```
Bilingual/intent scoring is separate: `eval/eval_bilingual.py` (SentenceTransformer) and `eval/eval_intent.py` (DeepSeek API, needs own key); edit the `src` path inside the script.

Training (Xtuner, separate env, see `train/README.md`):
```
cd train && export PYTHONPATH="../InternVL"
NPROC_PER_NODE=2 xtuner train ./xtuner/configs/internvl/v2/e-IT.py --work-dir ./logs/e_it --deepspeed deepspeed_zero1
python ./xtuner/configs/internvl/v1_5/convert_to_official.py --config=./xtuner/configs/internvl/v2/e-IT.py <iter.pth> <out_dir>   # merge LoRA
```
Paths in `e-IT.py` and the training JSON are absolute and must be edited by hand.

## Running on Windows / 8 GB GPU (verified on RTX 5060)

```
uv venv --python 3.12 ../callireader-venv   # keep venv outside the repo; 3.13+ lacks `imghdr`, 3.14 lacks many wheels
python inference.py --tgt=examples/0.jpg --load_4bit
```
- `--load_4bit` quantizes only the LLM (bitsandbytes nf4) and forces eager attention, so `flash-attn` is not needed. Without it the 18 GB bf16 model will not fit.
- Pins that matter: `transformers==4.45.2`, `sentencepiece==0.2.0` (newer raises "piece must not include null character" and `AutoTokenizer` then silently returns `bool`), torch cu128 for Blackwell GPUs, `protobuf`, `bitsandbytes`.
- Set `PYTHONIOENCODING=utf-8` when printing Chinese output on a GBK/cp1252 console.
- `InternVL/modeling_internvl_chat.py` is copied to `~/.cache/huggingface/modules/transformers_modules/` on load; edit the repo copy.

## Architecture

Pipeline lives in `InternVL/modeling_internvl_chat.py` (a modified copy of the InternVL remote-code model, loaded with `trust_remote_code=True`). `chat_ocr` → `calli_align`:
1. YOLO (`params/best.pt`) detects character boxes.
2. `OrderFormer` (`models/model.py`, transformer over box coords, `params/orderformer.pth`) plus KMeans column grouping (`char2col_with_kmeans`) recovers reading order.
3. Characters are sliced and encoded by the ViT + `mlp1`, then a Perceiver Resampler (`models/perceiver_resampler.py`, `params/callialign.pth`) aligns them to LLM *text-token embedding space* ("pseudo-text embeddings"). `hard_vq` snaps to nearest token embedding via cosine similarity (`models/similarity.py`); `drop_zero` removes padding tokens.
4. Those embeddings are fed to the fine-tuned LLM alongside the normal image tokens.

Training is two-stage: the resampler (alignment, settings in `config/configu.py`) and the LLM e-IT LoRA finetune in `train/` (a vendored Xtuner fork). `models/get_embeds.py` and `get_single_embeddings.py` produce the pseudo-text embeddings that e-IT training data references (`"embedding": "xxx.pt"`).

## Gotchas

- `config/configu.py` is star-imported everywhere; it holds all checkpoint paths (`./params/...`, `InternVL/` for weights, `SEED`) and resampler training hyperparameters. Weights are downloaded separately from HuggingFace (`gtang666/CalliReader`) into `params/` and `InternVL/`; `params/` is not in the repo. Note `MLP1_PATH` is `./params/params/mlp1.pth`.
- The argparse flags `--use_p/--hard_vq/--drop_zero/--verbose` use `type=bool`, so any non-empty string (even `False`) is truthy.
- `inference.py` and `evaluate.py` duplicate helpers (`set_seed`, `setup_logger`); shared utilities are in `utils/utils.py`.
- `.pyc` files are committed in the repo.
