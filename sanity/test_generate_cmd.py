#!/usr/bin/env python3
"""Output selection regression tests; no GPU or encoding required."""
import os
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'generate_cmd.sh').read_text().rsplit('\nmain\n', 1)[0]


class OutputPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aji-output-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.inputs = [str(self.directory / f'episode [{i:02d}].mkv') for i in range(25)]

    def select(self, answers, inputs=None, settings='', expected=0):
        inputs = self.inputs if inputs is None else inputs
        result_file = self.directory / 'outputs'
        body = SOURCE + '\nINPUT_VIDEOS=(' + shlex.join(inputs) + ')\n'
        body += settings + '\nselect_output_path || exit $?\n'
        body += 'printf "%s\\0" "${OUTPUT_VIDEOS[@]}" "$OVERWRITE_FLAG" > ' + shlex.quote(str(result_file))
        # Verify selection consumed exactly its own answers, leaving the next step intact.
        body += '\nread -r next_answer\n[[ "$next_answer" == NEXT_STEP ]]\n'
        result = subprocess.run(['bash', '-c', body], input=answers + 'NEXT_STEP\n',
                                capture_output=True, text=True, timeout=10,
                                env={**os.environ, 'USE_RUNTIME_CONFIG': '0'})
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        if expected:
            return result.stdout + result.stderr
        values = result_file.read_bytes().decode().split('\0')[:-1]
        return values[:-1], values[-1]

    def test_selected_engine_overrides_config_in_generated_command(self):
        engine = self.directory / '2x_APISR_RRDB_GAN_fp16_1080p.engine'
        engine.write_bytes(b'fixture')
        argv_file = self.directory / 'argv.json'
        encoder = self.directory / 'fake_encode'
        encoder.write_text('#!/usr/bin/env python3\nimport json, sys\n'
                           + f'open({str(argv_file)!r}, "w").write(json.dumps(sys.argv[1:]))\n')
        encoder.chmod(0o755)
        q = shlex.quote
        body = SOURCE + f"\nPROJECT_ROOT={q(str(self.directory))}\n"
        body += f"MODELS_DIR={q(str(self.directory))}\n"
        # Isolate engine discovery from the machine's real model directories.
        body += f"find() {{ printf '%s\\n' {q(str(engine))}; }}\n"
        body += "USE_RUNTIME_CONFIG=1\nselect_engine <<< 1 || exit $?\n"
        body += f"INPUT_VIDEOS=({q(self.inputs[0])})\n"
        body += f"OUTPUT_VIDEOS=({q(str(self.directory / 'out.mkv'))})\n"
        body += "SOURCE_WIDTHS=(1920)\nSOURCE_HEIGHTS=(1080)\nIS_CLIP=0\nRIFE_ENABLED=0\n"
        body += f"AJI_ENCODE_BIN={q(str(encoder))}\n"
        body += "generate_final_command_and_script || exit $?\n"
        body += f"bash {q(str(self.directory / 'run_encode.sh'))}\n"
        result = subprocess.run(['bash', '-c', body], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = json.loads(argv_file.read_text())
        self.assertEqual(args[args.index('--engine') + 1], str(engine))
        self.assertNotIn('--conf', args)
        self.assertNotIn('--trtexec', args)
        self.assertNotIn('--slot', args)

    def test_batch_default_needs_one_path_answer(self):
        outputs, overwrite = self.select('\n\n')
        self.assertEqual(outputs, [str(Path(p).with_suffix('')) + '_upscaled.mkv' for p in self.inputs])
        self.assertEqual(overwrite, '--overwrite')

    def test_batch_custom_directory_is_created_for_all_videos(self):
        target = self.directory / "输出 [A&B] 'quoted' $literal" / 'new'
        outputs, overwrite = self.select(f'"{target}"\nn\n')
        self.assertTrue(target.is_dir())
        self.assertEqual(outputs, [str(target / (Path(p).stem + '_upscaled.mkv')) for p in self.inputs])
        self.assertEqual(overwrite, '')

    def test_default_preserves_each_input_directory(self):
        inputs = [str(self.directory / folder / 'episode.mkv') for folder in ('a', 'b')]
        outputs, _ = self.select('\n\n', inputs=inputs)
        self.assertEqual(outputs, [str(Path(p).with_suffix('')) + '_upscaled.mkv' for p in inputs])

    def test_processing_suffixes(self):
        for clip in (0, 1):
            for rife, upscale in ((0, 1), (1, 0), (1, 1)):
                with self.subTest(clip=clip, rife=rife, upscale=upscale):
                    settings = f'IS_CLIP={clip}\nCLIP_DURATION=120\nRIFE_ENABLED={rife}\nUPSCALE_ENABLED={upscale}\nRIFE_FACTOR=4'
                    suffix = ('_clip_120s' if clip else '') + ('_rife4x' if rife else '')
                    suffix += ('_upscaled' if upscale else '') + '.mkv'
                    outputs, _ = self.select('\n\n', settings=settings)
                    self.assertEqual(outputs, [str(Path(p).with_suffix('')) + suffix for p in self.inputs])

    def test_single_video_can_still_choose_a_filename(self):
        target = self.directory / 'custom name.mkv'
        outputs, _ = self.select(f'{target}\n\n', inputs=self.inputs[:1])
        self.assertEqual(outputs, [str(target)])

    def test_batch_rejects_duplicate_output_names(self):
        inputs = [str(self.directory / folder / 'episode.mkv') for folder in ('a', 'b')]
        target = self.directory / 'combined'
        message = self.select(f'{target}\n', inputs=inputs, expected=1)
        self.assertIn('输出冲突', message)
        self.assertFalse(target.exists())

    def test_rejects_output_overwriting_an_input(self):
        inputs = [str(self.directory / name) for name in ('episode.mkv', 'episode_upscaled.mkv')]
        self.assertIn('输出冲突', self.select('\n', inputs=inputs, expected=1))

    def test_rejects_file_as_batch_directory(self):
        target = self.directory / 'existing.mkv'
        target.touch()
        self.assertIn('不是文件夹', self.select(f'{target}\n', expected=1))


if __name__ == '__main__':
    unittest.main()
