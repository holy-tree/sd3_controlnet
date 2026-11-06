"""CLI/report checks only; no real or substituted model is executed here."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.diagnose_dynamic_texture import parse_args, report
from dpo.texture_diagnostics_runtime import summarize_runtime


class DiagnosticEntryPointTest(unittest.TestCase):
    def test_inference_only_selects_full_comparison_without_gradient_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=root/"training"/"dpo_config.yaml"
            args=["diagnose","--config",str(config),"--checkpoint",str(config.parent/"checkpoint-200"),
                  "--output",str(root/"report"),"--inference-only"]
            with patch("sys.argv",args):
                parsed=parse_args()
            self.assertTrue(parsed.inference_only)
            self.assertTrue(parsed.full_inference)
            self.assertFalse(parsed.per_sample_gradients)
            runtime={"gradient_diagnostics_skipped":True}
            summarize_runtime(runtime,[])
            self.assertEqual(runtime["gradient_summary"],{})
            self.assertEqual(runtime["per_sample_gradient_summary"],{})
            self.assertEqual(runtime["measured_accumulation_windows"],0)
            self.assertFalse(any("GT/DPO" in finding for finding in runtime["findings"]))

    def test_empty_failed_measurement_report_does_not_infer_weighting_advice(self):
        runtime={}
        summarize_runtime(runtime,[])
        self.assertEqual(len(runtime["priorities"]),1)
        self.assertIn("没有完成",runtime["priorities"][0])
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            state={"runtime":runtime,"status":"failed_or_partial","error":"scratch disk full", "snapshot":None,
                   "arguments":{},"code_audit":{"findings":[]},"environment":{}}
            report(root,state,[])
            text=(root/"diagnostic_report.md").read_text(encoding="utf-8")
            self.assertIn("未测得梯度",text)
            self.assertNotIn("|micro_gradient/",text)
            self.assertNotIn("只有GT相对弱、冲突证据少",text)

    def test_inference_only_rejects_keep_gradient_buffers(self):
        with patch("sys.argv",["diagnose","--inference-only","--keep-gradient-buffers","--output","unused"]),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
