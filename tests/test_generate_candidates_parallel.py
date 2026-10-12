"""Coordinator tests use fake child processes, not GPU inference."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from scripts import generate_dpo_candidates as cli


class FakeProcess:
    def __init__(self, code=0):
        self.code = code
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def kill(self):
        self.killed = True
        self.code = -9

    def wait(self, timeout=None):
        self.waited = True
        return self.code


class FakeProgressBar:
    def __init__(self, iterable=(), **kwargs):
        self.iterable = iterable
        self.n = kwargs.get("initial", 0)
        self.postfix = {}
        self.closed = False

    def __iter__(self):
        return iter(self.iterable)

    def update(self, count):
        self.n += count

    def set_postfix(self, values):
        self.postfix = values

    def close(self):
        self.closed = True


class CandidateCoordinatorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.output = self.base / "output"
        self.plan = {"num_shards": 3, "fingerprint": "test-plan",
                     "sample_records": [{"global_index": i} for i in range(5)]}
        self.summary = {"parallel_generation": {"num_shards": 3, "n_candidates": 60}}
        progress_patch = patch.object(cli, "tqdm", FakeProgressBar)
        progress_patch.start()
        self.addCleanup(progress_patch.stop)

    def journal(self, shard, index, fingerprint="test-plan", corrupt=False):
        directory = cli.shard_output_dir(self.output, shard) / "group_cache"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": cli.SCHEMA_VERSION, "plan_fingerprint": fingerprint,
                   "global_index": index, "result": {"status": "retained"}}
        payload["fingerprint"] = "corrupt" if corrupt else cli._fingerprint(payload)
        path = directory / f"{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def run_coordinator(self, processes=None, resume=False, completed=()):
        processes = processes if processes is not None else [FakeProcess() for _ in range(3)]
        calls = []

        def prepare(command, **kwargs):
            self.output.mkdir(exist_ok=True)
            return Mock(returncode=0)

        def spawn(command, **kwargs):
            calls.append((command, kwargs))
            return processes[len(calls) - 1]

        with patch.object(cli.subprocess, "run", side_effect=prepare) as prepare_plan, \
                patch.object(cli, "load_generation_plan", return_value=self.plan), \
                patch.object(cli, "validate_shard_completion", side_effect=lambda root, plan, i: i in completed), \
                patch.object(cli.subprocess, "Popen", side_effect=spawn), \
                patch.object(cli, "merge_candidate_shards", return_value=self.summary) as merge, \
                contextlib.redirect_stdout(io.StringIO()):
            result = cli.generate_shards([sys.executable, "-m", "utils.randomness_check"],
                                         self.output, ["0", "1", "2"], resume)
        return result, calls, prepare_plan, merge

    def test_all_three_independent_children_are_gpu_pinned_and_merged(self):
        processes = [FakeProcess() for _ in range(3)]
        with patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "3", "MASTER_PORT": "1234"}):
            result, calls, prepare, merge = self.run_coordinator(processes)
        self.assertEqual(result, self.summary)
        self.assertEqual(len(calls), 3)
        self.assertIn("--plan_only", prepare.call_args.args[0])
        for index, (command, kwargs) in enumerate(calls):
            self.assertEqual(command[command.index("--shard_index") + 1], str(index))
            self.assertEqual(command[command.index("--num_shards") + 1], "3")
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(index))
            self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
            self.assertNotIn("WORLD_SIZE", kwargs["env"])
            self.assertNotIn("LOCAL_RANK", kwargs["env"])
            self.assertTrue(kwargs["stdout"].closed)
        self.assertTrue(all(p.waited and not p.terminated for p in processes))
        merge.assert_called_once_with(self.output, self.plan)
        self.assertFalse((self.base / ".output.generation.lock").exists())

    def test_resume_skips_verified_shard_and_passes_resume_to_remaining(self):
        processes = [FakeProcess(), FakeProcess()]
        _, calls, prepare, _ = self.run_coordinator(processes, resume=True, completed=(1,))
        self.assertIn("--resume", prepare.call_args.args[0])
        self.assertEqual([kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in calls], ["0", "2"])
        self.assertTrue(all("--resume" in command for command, _ in calls))

    def test_all_completed_shards_merge_without_loading_children(self):
        _, calls, _, merge = self.run_coordinator([], resume=True, completed=(0, 1, 2))
        self.assertEqual(calls, [])
        merge.assert_called_once()

    def test_failure_terminates_only_live_owned_children_and_never_merges(self):
        failed, live, done = FakeProcess(1), FakeProcess(None), FakeProcess(0)
        self.output.mkdir()
        (self.output / "COMPLETE.json").write_text("old marker")
        with patch.object(cli.subprocess, "run"), patch.object(cli, "load_generation_plan", return_value=self.plan), \
                patch.object(cli.subprocess, "Popen", side_effect=[failed, live, done]), \
                patch.object(cli, "merge_candidate_shards") as merge, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "shard 0 failed"):
            cli.generate_shards(["python"], self.output, ["0", "1", "2"], False)
        self.assertTrue(live.terminated)
        self.assertFalse(done.terminated)
        self.assertTrue(all(p.waited for p in (failed, live, done)))
        merge.assert_not_called()
        self.assertFalse((self.output / "COMPLETE.json").exists())
        self.assertFalse((self.base / ".output.generation.lock").exists())

    def test_existing_lock_prevents_duplicate_coordinators(self):
        lock = self.base / ".output.generation.lock"
        lock.write_text(json.dumps({"pid": 123}))
        with patch.object(cli.subprocess, "run") as prepare, self.assertRaisesRegex(RuntimeError, "lock exists"):
            cli.generate_shards(["python"], self.output, ["0", "1", "2"], False)
        prepare.assert_not_called()
        self.assertTrue(lock.is_file())

    def test_cli_dry_run_supports_inherited_config_and_writes_nothing(self):
        base = self.base / "base.yaml"
        base.write_text(yaml.safe_dump({
            "candidate_generation": {"output_dir": str(self.output), "num_candidates_per_image": 12,
                                     "candidate_guidance_scales": [1.0] * 12},
            "model": {"pretrained_model_name_or_path": "fake-sd3", "controlnet_model_path": "fake-controlnet",
                      "ra_fusion_path": "fake-ra"},
        }))
        derived = self.base / "derived.yaml"
        derived.write_text("base_config: base.yaml\n")
        other = self.base / "new-output"
        args = ["generate", "--config", str(derived), "--gpu_ids", "0", "1", "2", "--output_dir", str(other), "--dry_run"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"WORLD_SIZE": "1"}), \
                patch.object(cli.subprocess, "run") as run, patch.object(cli.subprocess, "Popen") as spawn, \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            cli.main()
        run.assert_not_called()
        spawn.assert_not_called()
        self.assertIn("3 disjoint source shards", printed.getvalue())
        self.assertIn(str(other), printed.getvalue())
        self.assertFalse(other.exists())
        self.assertFalse(self.output.exists())

    def test_cli_rejects_duplicate_gpu_ids_and_ddp_launch(self):
        for args, world in ((["--gpu_ids", "0", "0"], "1"), (["--gpu_ids", "0", "1", "2"], "3")):
            with patch.object(sys, "argv", ["generate", *args]), patch.dict(os.environ, {"WORLD_SIZE": world}), \
                    self.assertRaises(ValueError):
                cli.main()

    def test_legacy_command_still_uses_one_subprocess(self):
        root = Path(__file__).resolve().parents[1]
        args = ["generate", "--config", str(root / "config" / "dpo_sd3.yaml")]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"WORLD_SIZE": "1"}), \
                patch.object(cli.subprocess, "run") as run, patch.object(cli.subprocess, "Popen") as spawn, \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        run.assert_called_once()
        spawn.assert_not_called()
        self.assertNotIn("--plan_only", run.call_args.args[0])

    def test_progress_counts_only_matching_atomic_owned_journals(self):
        first = self.journal(0, 0)
        self.journal(0, 1)  # Not owned by shard zero.
        self.journal(1, 1, fingerprint="different-plan")
        self.journal(2, 2, corrupt=True)
        progress = cli.CandidateProgress(self.output, self.plan, ["0", "1", "2"])
        self.assertEqual(progress.bar.n, 1)
        self.assertEqual(progress.bar.postfix["GPU0"], "1/2")
        self.assertEqual(progress.bar.postfix["GPU1"], "0/2")
        self.journal(1, 1)
        self.journal(2, 2)
        progress.refresh()
        self.assertEqual(progress.bar.n, 3)
        first.unlink()
        progress.refresh()
        self.assertEqual(progress.bar.n, 2)
        progress.close()
        self.assertTrue(progress.bar.closed)

    def test_resume_progress_excludes_damaged_existing_cache(self):
        self.journal(0, 0)
        self.journal(1, 1)
        with patch.object(cli, "load_group_cache", side_effect=lambda root, row, plan:
                          {} if row["global_index"] == 0 else None) as validate:
            progress = cli.CandidateProgress(self.output, self.plan, ["0", "1", "2"], verify_existing=True)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(progress.bar.n, 1)
        progress.refresh()  # Unchanged damaged cache must not become valid by recounting.
        self.assertEqual(progress.bar.n, 1)
        self.journal(1, 1)
        progress.refresh()
        self.assertEqual(progress.bar.n, 2)
        progress.close()

    def test_monitor_is_read_only_and_does_not_start_generation(self):
        self.output.mkdir()
        complete = self.output / "COMPLETE.json"
        complete.write_text(json.dumps({"plan_fingerprint": self.plan["fingerprint"]}))
        before = complete.read_bytes()
        with patch.object(cli, "load_generation_plan", return_value=self.plan), \
                patch.object(cli.subprocess, "run") as run, patch.object(cli.subprocess, "Popen") as spawn, \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            cli.monitor_candidates(self.output, ["0", "1", "2"])
        run.assert_not_called()
        spawn.assert_not_called()
        self.assertIn("merged generation complete", printed.getvalue())
        self.assertEqual(complete.read_bytes(), before)

    def test_cli_monitor_dispatches_without_plan_or_worker_launch(self):
        root = Path(__file__).resolve().parents[1]
        args = ["generate", "--config", str(root / "config" / "dpo_sd3.yaml"),
                "--gpu_ids", "0", "1", "2", "--output_dir", str(self.output), "--monitor"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"WORLD_SIZE": "1"}), \
                patch.object(cli, "monitor_candidates") as monitor, \
                patch.object(cli, "generate_shards") as generate, patch.object(cli.subprocess, "run") as run:
            cli.main()
        monitor.assert_called_once_with(self.output, ["0", "1", "2"])
        generate.assert_not_called()
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_stopping_monitor_does_not_stop_workers_or_remove_lock(self):
        self.output.mkdir()
        lock = self.base / ".output.generation.lock"
        lock.write_text('{"pid": 123}')
        with patch.object(cli, "load_generation_plan", return_value=self.plan), \
                patch.object(cli.time, "sleep", side_effect=KeyboardInterrupt), \
                patch.object(cli.subprocess, "Popen") as spawn, contextlib.redirect_stdout(io.StringIO()) as printed:
            cli.monitor_candidates(self.output)
        spawn.assert_not_called()
        self.assertTrue(lock.is_file())
        self.assertIn("workers are not affected", printed.getvalue())

    def test_monitor_stops_when_coordinator_has_failed(self):
        self.output.mkdir()
        with patch.object(cli, "load_generation_plan", return_value=self.plan), \
                patch.object(cli.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()) as printed:
            cli.monitor_candidates(self.output)
        sleep.assert_not_called()
        self.assertIn("coordinator is not active", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
