from conftest import LLAMA_033_ARGV, LLAMA_SERVER_ARGV, LLAMA_SHORT_ARGV, OLLAMA_ENGINE_ARGV

import mtop


def test_non_runner_argv_is_ignored():
    assert mtop.parse_runner_argv([]) is None
    assert mtop.parse_runner_argv(["/usr/local/bin/ollama", "serve"]) is None
    assert mtop.parse_runner_argv(["ollama", "run", "llama3"]) is None
    assert mtop.parse_runner_argv(["/usr/bin/python3", "--model", "x"]) is None


def test_llama_server_long_flags():
    info = mtop.parse_runner_argv(LLAMA_SERVER_ARGV)
    assert info["ctx"] == "8192"
    assert info["batch"] == "512"
    assert info["ngl"] == "33"
    assert info["threads"] == "8"
    assert info["flash_attn"] == "on"      # bare flag == on in older builds
    assert info["parallel"] == "1"
    assert info["port"] == "40561"
    assert info["engine"] == "llama"
    assert info["digest"].startswith("6a0746a1ec1a")


def test_llama_server_short_flags():
    info = mtop.parse_runner_argv(LLAMA_SHORT_ARGV)
    assert info["ctx"] == "4096"
    assert info["batch"] == "256"
    assert info["ubatch"] == "64"
    assert info["ngl"] == "99"
    assert info["kv_k"] == "q8_0"
    assert info["kv_v"] == "q4_0"
    assert info["flash_attn"] == "on"
    assert info["parallel"] == "2"
    assert info["tensor_split"] == "1,1"
    assert info["main_gpu"] == "0"
    assert info["no_mmap"] is True
    assert info["digest"] == "deadbeef"


def test_ollama_engine_flags():
    info = mtop.parse_runner_argv(OLLAMA_ENGINE_ARGV)
    assert info["engine"] == "ollama"
    assert info["kv_k"] == info["kv_v"] == "q8_0"   # --kv-cache-type folds into K/V
    assert info["threads"] == "12"
    assert info["multiuser_cache"] is True
    assert info["ctx"] == "32768"
    assert "ollama_engine" not in info                # folded into engine


def test_ollama_033_llama_server():
    info = mtop.parse_runner_argv(LLAMA_033_ARGV)
    assert info["ctx"] == "65536" and info["parallel"] == "1"
    assert info["batch"] == "2048" and info["ubatch"] == "2048"
    assert info["kv_k"] == info["kv_v"] == "f16"
    assert info["flash_attn"] == "on"
    assert info["load_mode"] == "dio"
    assert info["context_shift"] is True
    assert info["engine"] == "llama"
    assert info["digest"] == "819dce06"
    # unknown flags with values (--host, --chat-template, --keep) are skipped
    assert "ngl" not in info


def test_flash_attn_value_forms():
    base = ["llama-server", "--model", "/b/sha256-x"]
    assert mtop.parse_runner_argv(base + ["--flash-attn", "auto"])["flash_attn"] == "auto"
    assert mtop.parse_runner_argv(base + ["--flash-attn", "off"])["flash_attn"] == "off"
    # bare flag followed by another flag must not eat that flag
    info = mtop.parse_runner_argv(base + ["--flash-attn", "--ctx-size", "2048"])
    assert info["flash_attn"] == "on" and info["ctx"] == "2048"


def test_trailing_value_flag_does_not_crash():
    info = mtop.parse_runner_argv(["llama-server", "--ctx-size"])
    assert info is not None and "ctx" not in info


def _runner(ctx, **kw):
    return {"ctx": str(ctx), "digest": "abc", **kw}


def _model(name, ctx):
    return {"name": name, "context_length": ctx, "size_vram": 10, "size": 12}


def test_match_by_unique_context_length():
    runners = [_runner(8192), _runner(32768)]
    models = [_model("a:7b", 32768), _model("b:13b", 8192)]
    mtop.match_runners_to_models(runners, models)
    assert runners[0]["model_name"] == "b:13b"
    assert runners[1]["model_name"] == "a:7b"
    assert runners[0]["vram"] == 10 and runners[0]["model_size"] == 12


def test_match_single_leftover():
    runners = [_runner(8192), _runner(8192)]
    models = [_model("a", 8192), _model("b", 4096)]
    mtop.match_runners_to_models(runners, models)
    # ctx 8192 is ambiguous between the two runners (2 hits? no: models with
    # ctx 8192 -> exactly one, so first runner claims it), second gets leftover.
    names = {r.get("model_name") for r in runners}
    assert names == {"a", "b"}


def test_ambiguous_stays_unmatched():
    runners = [_runner(8192), _runner(8192)]
    models = [_model("a", 8192), _model("b", 8192)]
    mtop.match_runners_to_models(runners, models)
    # two runners face two models with the same ctx: no honest join exists
    assert all("model_name" not in r for r in runners)


def test_runner_without_ctx_can_still_match_as_leftover():
    runners = [{"digest": "x"}]
    models = [_model("only", 2048)]
    mtop.match_runners_to_models(runners, models)
    assert runners[0]["model_name"] == "only"
