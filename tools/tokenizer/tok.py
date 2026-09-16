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
MODEL_DIR = os.path.join(REPO_ROOT, "models", "Qwen3.8-27B")

BOS_ID = 248044  # <|endoftext|> doubles as BOS-pad in this config
EOS_IDS = (248046, 248044)  # <|im_end|>, <|endoftext|> per generation_config


def load_tokenizer_config():
    with open(os.path.join(MODEL_DIR, "tokenizer_config.json")) as f:
        return json.load(f)


def added_special_tokens(cfg=None):
    """{id: content} for the 33 added special tokens, sorted by id."""
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


def render_chat(messages, add_generation_prompt=True, enable_thinking=False):
    """Render with the pinned chat_template.jinja.

    Defaults mirror the T4.4 CLI path: thinking disabled gives a minimal,
    deterministic prompt. Callers needing parity with HF defaults must pass
    the same kwargs to both renderers (see validation report).
    """
    from jinja2 import Environment, BaseLoader
    with open(os.path.join(MODEL_DIR, "chat_template.jinja")) as f:
        src = f.read()
    # Default (lenient) Undefined, matching HF apply_chat_template: assistant
    # messages without tool_calls render instead of raising (T8.4).
    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    # The template calls raise_exception(); provide it like HF does.
    def _raise(msg):
        raise ValueError(msg)
    tmpl = env.from_string(src)
    return tmpl.render(messages=messages,
                       add_generation_prompt=add_generation_prompt,
                       enable_thinking=enable_thinking,
                       tools=None,
                       bos_token=None, eos_token="<|im_end|>",
                       add_vision_id=True)


def prompt_ids(tok, messages, **kw):
    return encode(tok, render_chat(messages, **kw))
