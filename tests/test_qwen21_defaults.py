import importlib.util
from pathlib import Path

import pytest
from PIL import Image

module_spec = importlib.util.spec_from_file_location("paint_server", Path(__file__).parents[1] / "server.py")
server = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(server)

TURBO_MODELS = [name for name in server.MODELS if name.startswith("qwen21-turbo-")]


@pytest.mark.parametrize("name", TURBO_MODELS)
@pytest.mark.parametrize("steps", [6, 8])
def test_official_turbo_recipe_reaches_cli(name, steps):
    spec = server.MODELS[name]
    cmd = server.build_cmd(spec, "teapot", steps, 1.0, [42], None, None, 1024, 1024, "out.png", "in.png")
    assert cmd[cmd.index("--scheduler") + 1] == "viggle_turbo"
    assert cmd[cmd.index("--lora") + 2] == "1.0"
    assert "v0.2.1-6step-lora-" in cmd[cmd.index("--lora") + 1]
    assert cmd[cmd.index("--steps") + 1] == str(steps)
    assert cmd[cmd.index("--output-resolution") + 1] == "1024"
    assert cmd[cmd.index("--reference-resolution") + 1] == "1024"
    assert "--use-kv-cache" in cmd and "--no-bake-lora" in cmd
    assert "--quantize" not in cmd and "--low-ram" not in cmd
    assert ("--image-paths" in cmd) == name.endswith("-edit")


@pytest.mark.parametrize("steps,guidance,negative", [(4, 1, None), (7, 1, None), (6, 2, None), (6, 1, "blurry")])
def test_invalid_turbo_recipe_rejected(steps, guidance, negative):
    with pytest.raises(ValueError):
        server.build_cmd(server.MODELS[TURBO_MODELS[0]], "teapot", steps, guidance, [42], negative,
                         None, 512, 512, "out.png", "in.png")


def test_small_edit_keeps_reference_detail_and_area_size(monkeypatch):
    source = Image.new("RGB", (1600, 1200))
    captured = {}

    def run(cmd, steps, jid):
        with Image.open(cmd[cmd.index("--image-paths") + 1]) as ref:
            captured["reference"] = ref.size
        captured["output"] = (int(cmd[cmd.index("--width") + 1]), int(cmd[cmd.index("--height") + 1]))
        return None

    monkeypatch.setattr(server, "_popen_stream", run)
    monkeypatch.setattr(server, "_collect_outputs", lambda _: [])
    server.run_edit_or_fill(server.MODELS["qwen21-turbo-r128-edit"], source, None, "teapot", 6,
                            1.0, [42], 512, "test", None, None)
    assert captured == {"reference": (1600, 1200), "output": (576, 448)}


def test_registry_exposes_step_limits_and_no_retired_adapter():
    entries = [m for m in server.available_models() if m["id"].startswith("qwen21")]
    assert len(entries) == 12
    assert len(TURBO_MODELS) == 4
    for model in entries:
        assert model["snap_multiple"] == 32
        assert model["fixed_guidance"] == 1.0
        if "turbo" in model["id"]:
            assert model["step_choices"] == [6, 8] and model["fixed_guidance"] == 1.0


def test_custom_lora_strength_reaches_cli_without_changing_default():
    original = server.MODELS["qwen21-turbo-r128-edit"]
    spec = {**original, "lora_scale": 0.85}
    cmd = server.build_cmd(spec, "red dress", 6, 1.0, [42], None, None, 1600, 960, "out.png", "in.png")
    assert cmd[cmd.index("--lora") + 2] == "0.85"
    assert original["lora_scale"] == 1.0


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), 3])
def test_invalid_lora_strength_rejected(scale):
    spec = {**server.MODELS["qwen21-turbo-r128-edit"], "lora_scale": scale}
    with pytest.raises(ValueError, match="LoRA scale"):
        server.build_cmd(spec, "red dress", 6, 1, [42], None, None, 1600, 960, "out.png", "in.png")


@pytest.mark.parametrize("requested,expected", [(0, (1600, 960)), (1024, (1312, 800)), (1536, (1984, 1184))])
def test_edit_output_keeps_generated_size(monkeypatch, requested, expected):
    source = Image.new("RGBA", (1600, 960))

    def run(cmd, steps, jid):
        size = (int(cmd[cmd.index("--width") + 1]), int(cmd[cmd.index("--height") + 1]))
        with Image.open(cmd[cmd.index("--image-paths") + 1]) as ref:
            assert ref.size == source.size
        Image.new("RGBA", size, (12, 34, 56, 128)).save(cmd[cmd.index("--output") + 1])

    monkeypatch.setattr(server, "_popen_stream", run)
    results = server.run_edit_or_fill(server.MODELS["qwen21-turbo-r128-edit"], source, None,
                                     "red dress", 6, 1, [42], requested, "test", None, None)
    assert results[0].size == expected
    assert results[0].getpixel((0, 0)) == (12, 34, 56, 128)


def test_auto_edit_area_limit_and_other_model_sizing():
    qwen = server.MODELS["qwen21-turbo-r128-edit"]
    assert server.edit_size(qwen, 4000, 2400, 0) == (1984, 1184)
    other = {"gen_max": 512, "snap_multiple": 16}
    assert server.edit_size(other, 1600, 960, 0) == server.gen_size(1600, 960, 512)


def test_masked_edit_keeps_canvas_size(monkeypatch):
    source = Image.new("RGBA", (1600, 960))

    def run(cmd, *_):
        Image.new("RGBA", (1312, 800), "blue").save(cmd[cmd.index("--output") + 1])

    monkeypatch.setattr(server, "_popen_stream", run)
    results = server.run_edit_or_fill(server.MODELS["qwen21-turbo-r128-edit"], source,
                                     Image.new("L", source.size, 255), "blue", 6, 1, [42],
                                     1024, "test", None, None)
    assert results[0].size == source.size


def test_save_preserves_rgba_pixels(tmp_path, monkeypatch):
    source = Image.new("RGBA", (32, 32), (117, 83, 29, 128))
    monkeypatch.setattr(server, "SAVE_DIR", str(tmp_path))
    result = server.handle_save({"image": server.img_to_b64(source), "name": "alpha.png"})
    with Image.open(result["path"]) as saved:
        assert saved.mode == "RGBA"
        assert saved.tobytes() == source.tobytes()


def test_input_exif_orientation_preserved_without_canvas():
    source = Image.new("RGB", (32, 64), "red")
    exif = Image.Exif()
    exif[274] = 6
    buffer = server.io.BytesIO()
    source.save(buffer, "JPEG", exif=exif)
    data = "data:image/jpeg;base64," + server.base64.b64encode(buffer.getvalue()).decode()
    result = server.b64_to_img(data)
    assert result.size == (64, 32)
    assert result.getexif().get(274) is None
