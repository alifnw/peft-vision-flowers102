"""Protect the pretrained backbone from overly broad trainable-parameter masks."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from peft_methods import method_settings, is_method_parameter, expected_method_parameters


class MethodBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.config = {"methods": ["linear", "lora", "ssf", "convpass"], "lora_rank": 8,
                       "convpass": {"bottleneck": 8, "scale": 0.1, "xavier_init": False}}

    def test_native_counts(self):
        for method, count in (("linear", 0), ("lora", 294912), ("ssf", 205824), ("convpass", 327552)):
            with self.subTest(method=method):
                self.assertEqual(expected_method_parameters(self.config, method), count)

    def test_backbone_and_layernorm_are_never_selected(self):
        for method in self.config["methods"]:
            for name in ("blocks.0.attn.qkv.weight", "blocks.0.norm1.weight", "patch_embed.proj.weight", "norm.bias"):
                self.assertFalse(is_method_parameter(name, method), (method, name))

    def test_methods_do_not_select_each_other(self):
        examples = {"lora": "blocks.0.attn.lora.lora_a.weight",
                    "ssf": "blocks.0.attn.ssf_scale_qkv",
                    "convpass": "blocks.0.ft_mlp_module.adapter_conv.weight"}
        for owner, name in examples.items():
            for method in self.config["methods"]:
                self.assertEqual(is_method_parameter(name, method), method == owner)

    def test_switching_method_clears_other_modules(self):
        for method in self.config["methods"]:
            setting = method_settings(self.config, method)
            self.assertEqual(setting["ssf"], method == "ssf")
            self.assertEqual(setting["lora_bottleneck"], 8 if method == "lora" else 0)
            self.assertEqual(setting["ft_attn_module"], "convpass" if method == "convpass" else None)

    def test_unlisted_method_is_rejected(self):
        self.config["methods"] = ["ssf"]
        with self.assertRaises(ValueError):
            method_settings(self.config, "convpass")


if __name__ == "__main__":
    unittest.main()
