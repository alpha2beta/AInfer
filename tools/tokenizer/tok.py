"""AInfer tokenizer (T4.3): exact-match wrapper around HF `tokenizers`.

Loads models/Qwen3.8-27B/tokenizer.json, registers the 33 added special tokens
from tokenizer_config.json (the T1.4 trap: tokenizer.json alone drops chat
markup like <|im_start|>), and renders chat prompts with the pinned
chat_template.jinja via Jinja2.

Reference for validation: transformers AutoTokenizer for the same directory.
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes")
if not os.path.exists(DEFAULT_MODEL_DIR):
    DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "models", "Qwen3.8-27B")
MODEL_DIR = os.environ.get("AINFER_MODEL_DIR", DEFAULT_MODEL_DIR)

BOS_ID = 248044  # <|endoftext|> doubles as BOS-pad in this config
EOS_IDS = (248046, 248044)  # <|im_end|>, <|endoftext|> per generation_config


def load_tokenizer_config():
    with open(os.path.join(MODEL_DIR, "tokenizer_config.json")) as f:
        return json.load(f)


def added_special_tokens(cfg=None):
    """{id: content} for the added special tokens, sorted by id."""
    cfg = cfg or load_tokenizer_config()
    return {int(k): v["content"] for k, v in cfg["added_tokens_decoder"].items()}


def load():
    from tokenizers import Tokenizer, AddedToken
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    cfg = load_tokenizer_config()
    specials = [AddedToken(content, single_word=False, lstrip=False,
                           rstrip=False, normalized=False, special=True)
                for _, content in sorted(added_special_tokens(cfg).items())]
    tok.add_special_tokens(specials)
    return tok


def encode(tok, text):
    return tok.encode(text).ids


def decode(tok, ids, skip_special_tokens=False):
    return tok.decode(ids, skip_special_tokens=skip_special_tokens)


def render_chat(messages, add_generation_prompt=True, enable_thinking=False, tools=None):
    """Render with the pinned chat_template.jinja.

    Defaults mirror the T4.4 CLI path: thinking disabled gives a minimal,
    deterministic prompt. Callers needing parity with HF defaults must pass
    the same kwargs to both renderers (see validation report).

    tools: OpenAI-style list of {"type": "function", "function": {...}} dicts
    (or None). The pinned Hermes template renders them into a <tools> system
    block natively; None preserves the exact legacy prompt.
    """
    from jinja2 import Environment, BaseLoader
    with open(os.path.join(MODEL_DIR, "chat_template.jinja")) as f:
        src = f.read()
    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    def _raise(msg):
        raise ValueError(msg)
    env.globals["raise_exception"] = _raise
    tmpl = env.from_string(src)
    return tmpl.render(messages=messages,
                       add_generation_prompt=add_generation_prompt,
                       enable_thinking=enable_thinking,
                       tools=tools,
                       bos_token=None, eos_token="<|im_end|>",
                       add_vision_id=True)


def prompt_ids(tok, messages, **kw):
    return encode(tok, render_chat(messages, **kw))
