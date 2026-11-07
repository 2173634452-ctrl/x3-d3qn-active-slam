"""Exercise orchestration without Gazebo: ordering, isolation and failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pipeline


class PipelineTests(unittest.TestCase):
    def test_collection_precedes_training_and_domains_are_unique(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            domains = []
            collected = []
            calls = []

            def fake_run(module, *args, env=None):
                options = dict(zip(args[::2], args[1::2]))
                if module == 'collect_long.py':
                    domains.append(env['X3_ROS_DOMAIN_BASE'])
                    target = Path(options['--output'])
                    target.mkdir()
                    (target / 'transitions.jsonl').write_text('{}\n' * 8)
                    collected.append(target)
                else:
                    self.assertEqual(len(collected), 2)
                    calls.append(module)

            with patch('sys.argv', ['pipeline.py', '--output', str(output), '--actors', '2']), patch.object(pipeline, 'run', fake_run):
                pipeline.main()
            self.assertEqual(len(set(domains)), 2)
            self.assertEqual(calls, ['train_offline.py', 'evaluate_checkpoint.py'])
            state = json.loads((output / 'pipeline.json').read_text())
            self.assertTrue(state['completed'])
            self.assertEqual(state['samples'], 16)

    def test_collection_failure_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            def fail(module, *args, **kwargs):
                self.assertEqual(module, 'collect_long.py')
                raise RuntimeError('collector failed')
            with patch('sys.argv', ['pipeline.py', '--output', str(output), '--actors', '1']), patch.object(pipeline, 'run', fail):
                with self.assertRaisesRegex(RuntimeError, 'collector failed'):
                    pipeline.main()
            self.assertEqual(json.loads((output / 'pipeline.json').read_text())['phase'], 'failed')


if __name__ == '__main__':
    unittest.main()
