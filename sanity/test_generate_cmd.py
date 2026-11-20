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
                                env=os.environ)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        if expected:
            return result.stdout + result.stderr
        values = result_file.read_bytes().decode().split('\0')[:-1]
        return values[:-1], values[-1]

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


class ProfileTests(unittest.TestCase):
    """Every profile runs through a conf slot; customizations go to a task conf."""
    MODELS = ('2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16',
              '2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo',
              'RealESRGAN_x4plus_anime_6B', '4x_Custom')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aji-profile-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.models = self.directory / 'my models'
        (self.models / 'rife').mkdir(parents=True)
        for name in self.MODELS:
            (self.models / f'{name}.onnx').write_text('fake')
        (self.models / 'rife/rife_v4.26.onnx').write_text('fake')
        self.argv_file = self.directory / 'argv.json'
        self.encoder = self.directory / 'fake_encode'
        self.encoder.write_text('#!/usr/bin/env python3\nimport json, sys\n'
                                + f'open({str(self.argv_file)!r}, "w").write(json.dumps(sys.argv[1:]))\n')
        self.encoder.chmod(0o755)

    def generate(self, answers, settings='', expected=0):
        q = shlex.quote
        body = SOURCE + f'''
CONFIG_MODEL_DIR={q(str(self.models))}; RIFE_MODEL_DIR={q(str(self.models / 'rife'))}
CONFIG_TRTEXEC=/opt/trt/trtexec
GEN_SCRIPT_PATH={q(str(self.directory / 'run_encode.sh'))}
TASK_CONF_PATH={q(str(self.directory / 'run_encode.conf'))}
AJI_ENCODE_BIN={q(str(self.encoder))}
INPUT_VIDEOS=({q(str(self.directory / 'in.mkv'))}); SOURCE_WIDTHS=(1920); SOURCE_HEIGHTS=(1080)
OUTPUT_VIDEOS=({q(str(self.directory / 'out.mkv'))})
IS_CLIP=0
{settings}
select_processing_profile || exit $?
select_rife_settings || exit $?
check_profile_models || exit $?
generate_final_command_and_script >/dev/null || exit $?
bash {q(str(self.directory / 'run_encode.sh'))}
'''
        result = subprocess.run(['bash', '-c', body], input=answers, capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        if expected:
            return result.stdout + result.stderr, None
        args = json.loads(self.argv_file.read_text())
        conf = self.directory / 'run_encode.conf'
        return args, conf.read_text() if conf.exists() else None

    @staticmethod
    def value(args, option):
        return args[args.index(option) + 1]

    def test_builtin_slot_uses_the_base_conf(self):
        args, task_conf = self.generate('3\nn\n')
        self.assertEqual(self.value(args, '--slot'), '1003')
        self.assertTrue(self.value(args, '--conf').endswith('packaging/ubuntu24/runtime/animejanai.conf'))
        self.assertEqual(self.value(args, '--model-dir'), str(self.models))
        self.assertEqual(self.value(args, '--rife-model-dir'), str(self.models / 'rife'))
        self.assertEqual(self.value(args, '--trtexec'), '/opt/trt/trtexec')
        for removed in ('--engine', '--max-width', '--rife-model', '--rife-factor', '--rife-order'):
            self.assertNotIn(removed, args)
        self.assertIsNone(task_conf)

    def test_custom_rife_on_a_builtin_slot_writes_a_task_conf(self):
        args, task_conf = self.generate('3\ny\n1\n2\n2\n0.2\n')
        self.assertEqual(self.value(args, '--slot'), '9001')
        self.assertEqual(self.value(args, '--conf'), str(self.directory / 'run_encode.conf'))
        self.assertIn('[slot_9001]', task_conf)
        self.assertIn('config_version=2', task_conf)
        # each built-in chain keeps its own model and gets the same RIFE keys
        for chain in (1, 2):
            for line in (f'chain_{chain}_rife=yes', f'chain_{chain}_rife_model=426',
                         f'chain_{chain}_rife_factor_numerator=5',
                         f'chain_{chain}_rife_factor_denominator=2',
                         f'chain_{chain}_rife_before_upscale=no',
                         f'chain_{chain}_rife_scene_detect_threshold=0.2'):
                self.assertIn(line + '\n', task_conf)
        self.assertIn('chain_1_model_1_name=' + self.MODELS[0] + '\n', task_conf)
        self.assertIn('chain_2_model_1_name=' + self.MODELS[1] + '\n', task_conf)

    def test_custom_4x_model_adds_pipeline_depth(self):
        args, task_conf = self.generate('14\n1\nn\n')
        self.assertEqual(self.value(args, '--slot'), '9001')
        self.assertEqual(self.value(args, '--pipeline-depth'), '2')
        self.assertIn('chain_1_model_1_name=4x_Custom\n', task_conf)

    def test_slot_2006_runs_from_the_packaged_conf(self):
        args, task_conf = self.generate('9\nn\n')
        self.assertEqual(self.value(args, '--slot'), '2006')
        self.assertEqual(self.value(args, '--pipeline-depth'), '2')
        self.assertIsNone(task_conf)

    def test_missing_model_stops_generation(self):
        output, _ = self.generate('10\n', expected=1)
        self.assertIn('rife_v4.25.onnx', output)
        self.assertFalse((self.directory / 'run_encode.sh').exists())

    def test_tensorrt_comparison_writes_native_off(self):
        args, task_conf = self.generate('3\nn\n', settings='NATIVE_OFF=1')
        self.assertEqual(self.value(args, '--slot'), '1003')
        self.assertIn('native=off\n', task_conf)
        self.assertNotIn('[slot_', task_conf)

    def test_rife_model_codes_round_trip(self):
        body = SOURCE + '''
for name in rife_v4.26 rife_v4.6 rife_v4.22_lite rife_v4.25_ensemble rife_v4.6_lite rife_vx; do
    printf '%s=%s\\n' "$name" "$(rife_name_to_code "$name" || echo invalid)"
done'''
        result = subprocess.run(['bash', '-c', body], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stdout.splitlines(), ['rife_v4.26=426 no', 'rife_v4.6=46 no',
                                                 'rife_v4.22_lite=4221 no',
                                                 'rife_v4.25_ensemble=425 yes',
                                                 'rife_v4.6_lite=invalid', 'rife_vx=invalid'])


if __name__ == '__main__':
    unittest.main()
