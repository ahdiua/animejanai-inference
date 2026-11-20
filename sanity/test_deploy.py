#!/usr/bin/env python3
"""Isolated deploy regression tests. No apt, system writes, network or GPU calls."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('download_model', ROOT / 'scripts/download_model.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DeployTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aji-deploy-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.environment = {**os.environ, 'AUDIT_DIR': str(self.directory), 'DEPLOY_SCRIPT': str(ROOT / 'deploy.sh')}
        # Never inherit a user's project/root selection, preload or linker paths.
        for key in ('PROJECT_ROOT', 'CUDA_ROOT', 'TRT_ROOT', 'TRTEXEC_BIN', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'ASSUME_DEFAULTS'):
            self.environment.pop(key, None)

    def run_shell(self, body, expected=0, stdin=''):
        preamble = '''source "$DEPLOY_SCRIPT"
print_header() { :; }
check_install_platform() { CUDA_KEYRING_SHA256=unused; return 0; }
run_as_root() { printf 'UNEXPECTED_ROOT_CALL: %s\\n' "$*" >&2; return 99; }
'''
        result = subprocess.run(['bash', '--noprofile', '--norc', '-c', preamble + body],
                                input=stdin, env=self.environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def executable(self, relative, text):
        file = self.directory / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('#!/bin/bash\n' + text)
        file.chmod(0o755)
        return file

    def test_source_and_help_have_no_initialization(self):
        result = self.run_shell('init_project_root() { echo BAD; return 1; }; main --help')
        self.assertNotIn('BAD', result.stdout)

    def test_script_root_wins_over_working_directory(self):
        (self.directory / 'CMakeLists.txt').write_text('project(animejanai-fake)')
        result = self.run_shell('cd "$AUDIT_DIR"; init_project_root; printf "%s" "$PROJECT_ROOT"')
        self.assertEqual(result.stdout, str(ROOT))

    def test_invalid_project_fails_without_clone(self):
        result = self.run_shell('PROJECT_ROOT="$AUDIT_DIR"; git() { echo CLONED; }; init_project_root', 1)
        self.assertNotIn('CLONED', result.stdout)

    def test_empty_library_paths_removed_and_deduplicated(self):
        result = self.run_shell('LD_LIBRARY_PATH=":/a::/b:/a:"; prepend_path LD_LIBRARY_PATH /b; printf "%s" "$LD_LIBRARY_PATH"')
        self.assertEqual(result.stdout, '/b:/a')

    def test_input_rejects_arithmetic_commands(self):
        self.run_shell('''value='a[$(touch "$AUDIT_DIR/injected")0]+1'; valid_index "$value" 1 3''', 1)
        self.assertFalse((self.directory / 'injected').exists())

    def test_indices_are_decimal_and_bounded(self):
        self.run_shell('valid_index 08 1 9 && ! valid_index 0 1 9 && ! valid_index 10 1 9 && ! valid_index 1234567 1 999999')

    def test_eof_is_not_a_default_answer(self):
        self.run_shell("prompt_value choice 'choice: ' 1", 1)

    def test_explicit_default_mode(self):
        result = self.run_shell("ASSUME_DEFAULTS=1; prompt_value choice 'choice: ' 1; printf '%s' \"$choice\"")
        self.assertEqual(result.stdout, '1')

    def test_all_without_stdin_fails_before_changes(self):
        self.run_shell('one_click_setup', 1)

    def test_menu_eof_exits(self):
        self.run_shell('main_menu', 1)

    def test_venv_apt_failure_stops_before_python(self):
        result = self.run_shell('python3() { echo UNEXPECTED_PYTHON; }; setup_python_venv', 1)
        self.assertNotIn('UNEXPECTED_PYTHON', result.stdout)

    def test_venv_creation_failure_propagates(self):
        result = self.run_shell('''init_project_root() { VENV_DIR="$AUDIT_DIR/venv"; }
run_as_root() { :; }
python3() { return 7; }
setup_python_venv''', 1)
        self.assertNotIn('配置完成', result.stdout)

    def test_one_click_stops_on_dependency_failure(self):
        result = self.run_shell("""ASSUME_DEFAULTS=1
check_nvidia_driver() { :; }
setup_nvidia_network_repo() { :; }
install_cuda_and_tensorrt() { return 7; }
install_build_tools_and_ffmpeg() { echo MUST_NOT_RUN; }
one_click_setup""", 1)
        self.assertNotIn('MUST_NOT_RUN', result.stdout)

    def test_keyring_install_failure_propagates(self):
        self.run_shell('''check_nvidia_driver() { :; }
fetch_verified() { printf fake > "$2"; }
run_as_root() { if [ "$1 $2" = 'dpkg -i' ]; then return 7; fi; return 0; }
setup_nvidia_network_repo''', 1)

    def test_corrupt_ffmpeg_is_checked_before_old_install_removed(self):
        result = self.run_shell('''FFMPEG_VARIANT=n8.1
run_as_root() { printf '%s\\n' "$*"; }
fetch_verified() { printf corrupt > "$2"; }
install_build_tools_and_ffmpeg''', 1)
        self.assertNotIn('rm -rf', result.stdout)
        self.assertNotIn('ln -sf', result.stdout)

    def test_nvenc_failed_download_never_compiles_stale_source(self):
        result = self.run_shell('''fetch_verified() { printf stale > "$2"; return 7; }
gcc() { echo COMPILED; }
install_nvenc_fix''', 1)
        self.assertNotIn('COMPILED', result.stdout)

    def test_checksum_mismatch_is_rejected(self):
        self.run_shell('''curl() { local arg previous=''; for arg; do
 if [ "$previous" = --output ]; then printf incorrect > "$arg"; fi; previous=$arg; done; }
fetch_verified https://example.test/file "$AUDIT_DIR/download" "$(printf '%064d' 0)"''', 1)

    def test_cuda_selection_is_read_only(self):
        self.executable('cuda/bin/nvcc', "printf 'release 13.2, V13.2.0\\n'\n")
        result = self.run_shell('CUDA_ROOT="$AUDIT_DIR/cuda"; check_cuda_toolkit')
        self.assertNotIn('UNEXPECTED_ROOT_CALL', result.stderr)
        self.assertIn('13.2', result.stdout)

    def test_old_cuda_packages_include_shared_config_and_version_suffixes(self):
        result = self.run_shell("""dpkg-query() { cat <<'EOF'
cuda-nvcc-12-6 12.6.77-1 ii
cuda-toolkit-12-config-common 12.6.77-1 ii
libcudnn9-cuda-12:amd64 9.6.0-1 ii
libcudnn9-dev-cuda-12 9.6.0-1 ii
libnccl2 2.23.4-1+cuda12.6 ii
libnvinfer10 10.7.0-1+cuda12.6 ii
libnvonnxparsers10 10.7.0-1+cuda12.6 ii
cuda-nvcc-11-8 11.8-1 ii
cuda-nvcc-10-2 10.2-1 ii
cuda-nvcc-13-2 13.2.86-1 ii
cuda-toolkit-13-config-common 13.4.92-1 ii
cuda-toolkit-config-common 13.4.92-1 ii
libnvinfer11 11.2.1-1+cuda13.3 ii
libnvinfer10 10.16.1-1+cuda13.2 ii
libnccl2 2.31.2-1+cuda13.4 ii
libcuda1 580.0-1 ii
nvidia-driver-580 580.0-1 ii
libcurl4 8.5.0-1 ii
libcutensor2 2.0.2.5-1 ii
nsight-compute-2024.3.2 2024.3.2-1 ii
cuda-nvcc-12-9 12.9-1 rc
EOF
}
get_old_cuda_packages""")
        self.assertEqual(result.stdout.splitlines(), [
            'cuda-nvcc-12-6', 'cuda-toolkit-12-config-common',
            'libcudnn9-cuda-12:amd64', 'libcudnn9-dev-cuda-12',
            'libnccl2', 'libnvinfer10', 'libnvonnxparsers10',
            'cuda-nvcc-11-8', 'cuda-nvcc-10-2'])

    def test_old_cuda_scan_failure_stops_cleanup(self):
        result = self.run_shell('dpkg-query() { return 7; }; purge_old_cuda_packages', 1)
        self.assertNotIn('UNEXPECTED_ROOT_CALL', result.stderr)

    def test_old_cuda_scan_can_be_empty(self):
        result = self.run_shell('dpkg-query() { :; }; get_old_cuda_packages')
        self.assertEqual(result.stdout, '')

    def test_cuda_wrong_major_rejected(self):
        self.executable('cuda/bin/nvcc', "printf 'release 14.0, V14.0.0\\n'\n")
        self.run_shell('CUDA_ROOT="$AUDIT_DIR/cuda"; check_cuda_toolkit', 1)

    def make_trt(self, version='110201'):
        include = self.directory / 'trt/include'
        include.mkdir(parents=True)
        (include / 'NvInferVersion.h').write_text('''#define TRT_MAJOR_ENTERPRISE 11
#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE
#define NV_TENSORRT_MINOR 2
#define NV_TENSORRT_PATCH 1
''')
        lib = self.directory / 'trt/lib'
        lib.mkdir()
        (lib / 'libnvinfer.so').write_text('mock library')
        self.executable('trt/bin/trtexec', f"printf 'TensorRT v{version}\\n'\n")

    def test_trt_header_alias_and_non_path_tool(self):
        self.make_trt()
        result = self.run_shell('TRT_ROOT="$AUDIT_DIR/trt"; resolve_tensorrt && printf "%s %s" "$TRT_VER" "$TRTEXEC_BIN"')
        self.assertIn('11.2.1', result.stdout)
        self.assertIn('/trt/bin/trtexec', result.stdout)

    def test_trt_mismatch_rejected(self):
        self.make_trt('110300')
        self.run_shell('TRT_ROOT="$AUDIT_DIR/trt"; resolve_tensorrt', 1)

    def test_diagnostics_aggregate_failures(self):
        result = self.run_shell('''check_nvidia_driver() { return 1; }
check_os_info() { :; }
check_cuda_toolkit() { return 1; }
check_tensorrt() { return 1; }
check_ffmpeg() { return 1; }
check_nvenc_and_patch() { return 1; }
check_build_tools() { return 1; }
check_python_venv() { return 1; }
check_build_and_models() { return 1; }
diagnose_all''', 1)
        self.assertIn('失败项目: 8', result.stdout)

    def test_native_nvenc_never_loads_patch(self):
        (self.directory / 'patch.so').write_text('not loaded')
        result = self.run_shell('''NVENC_FIX_SO="$AUDIT_DIR/patch.so"
ffmpeg() { [ -z "${LD_PRELOAD:-}" ]; }
check_nvenc_and_patch && [ -z "$NVENC_PRELOAD" ]''')
        self.assertIn('原生编码正常', result.stdout)

    def test_patch_only_for_enumeration_failure(self):
        (self.directory / 'patch.so').write_text('not loaded')
        self.run_shell('''NVENC_FIX_SO="$AUDIT_DIR/patch.so"
ffmpeg() { if [ -z "${LD_PRELOAD:-}" ]; then echo 'unsupported device' >&2; return 1; fi; return 0; }
check_nvenc_and_patch && [ "$NVENC_PRELOAD" = "$NVENC_FIX_SO" ] && [ -z "${LD_PRELOAD:-}" ]''')
        self.run_shell('''NVENC_FIX_SO="$AUDIT_DIR/patch.so"
ffmpeg() { echo 'cannot load libnvidia-encode' >&2; return 1; }
check_nvenc_and_patch; [ "$NVENC_PATCH_NEEDED" = 0 ] && [ -z "$NVENC_PRELOAD" ]''')

    def test_failed_model_download_is_reported(self):
        self.run_shell('''init_project_root() { PROJECT_ROOT="$PWD"; VENV_DIR="$AUDIT_DIR/venv"; }
MODELS_DIR="$AUDIT_DIR/models"
install_model_row() { return 7; }
download_models''', 1, '2\n')

    def test_model_groups_follow_the_manifest(self):
        result = self.run_shell('''init_project_root() { PROJECT_ROOT="$PWD"; VENV_DIR="$AUDIT_DIR/venv"; }
MODELS_DIR="$AUDIT_DIR/models"
install_model_row() { printf 'ROW=%s %s\\n' "$1" "$3"; }
download_models''', 0, '7\n')
        rows = [x for x in result.stdout.splitlines() if x.startswith('ROW=')]
        self.assertEqual(rows, ['ROW=rife rife/rife_v4.26.onnx', 'ROW=rife rife/rife_v4.25.onnx'])

    def test_custom_models_keep_names_and_never_overwrite(self):
        models = self.directory / 'models'
        models.mkdir()
        (models / 'taken.onnx').write_bytes(b'existing')
        body = '''init_project_root() { PROJECT_ROOT="$PWD"; VENV_DIR="$AUDIT_DIR/venv"; }
init_project_root
MODELS_DIR="$AUDIT_DIR/models"
download_model_file() { printf 'DEST=%s\\n' "$1"; }
'''
        result = self.run_shell(body + 'import_custom_model\nimport_custom_model\n', 0,
                                'https://example.test/x/4x_Custom.onnx?download=1\n'
                                'https://example.test/get?id=7\n')
        destinations = [x[5:] for x in result.stdout.splitlines() if x.startswith('DEST=')]
        self.assertEqual(destinations[0], str(models / '4x_Custom.onnx'))
        self.assertRegex(destinations[1], r'/custom_[0-9a-f]{16}\.onnx$')
        for name in ('taken.onnx', 'RealESRGAN_x4plus_anime_6B.onnx'):
            with self.subTest(name=name):
                self.run_shell(body + 'import_custom_model\n', 1, f'https://example.test/{name}\n')
        self.assertEqual((models / 'taken.onnx').read_bytes(), b'existing')

    def test_test_clip_runs_a_conf_slot_and_propagates_failure(self):
        self.executable('build/aji_encode', '''printf '%s\\n' "$@" > "$AUDIT_DIR/encode-args"
exit 7
''')
        (self.directory / 'example.mkv').write_text('fake')
        models = self.directory / 'my models'
        models.mkdir()
        for name in ('2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx',
                     '2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo.onnx'):
            (models / name).write_text('fake')
        self.run_shell(f'''init_project_root() {{ PROJECT_ROOT="$AUDIT_DIR"; }}
load_model_assets() {{ source {str(ROOT / 'scripts/model-assets.sh')!r}; }}
MODELS_DIR="$AUDIT_DIR/my models"
resolve_tensorrt() {{ TRTEXEC_BIN=/opt/trt/trtexec; }}
ffprobe() {{ printf '1920\\n'; }}
ffmpeg() {{ printf fake > "${{@: -1}}"; }}
check_nvenc_and_patch() {{ NVENC_PRELOAD=''; return 0; }}
run_test_clip''', 1)
        args = (self.directory / 'encode-args').read_text().splitlines()
        self.assertEqual(args[args.index('--slot') + 1], '1003')
        self.assertEqual(args[args.index('--model-dir') + 1], str(models))
        self.assertEqual(args[args.index('--rife-model-dir') + 1], str(models / 'rife'))
        self.assertEqual(args[args.index('--trtexec') + 1], '/opt/trt/trtexec')
        self.assertTrue(args[args.index('--conf') + 1].endswith('packaging/ubuntu24/runtime/animejanai.conf'))
        self.assertNotIn('--engine', args)

    def legacy_install(self):
        import hashlib
        models = self.directory / 'models'
        models.mkdir()
        manifest = [line.split('\t') for line in (ROOT / 'scripts/runtime-models.tsv').read_text().splitlines()
                    if line and not line.startswith('#')]
        self.assertTrue(manifest)
        # A fake model under its legacy short name, with its digest pinned in a
        # copy of the manifest so the migration can recognize it.
        payload = b'performance model'
        manifest = [row if row[2] != '2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx'
                    else row[:3] + [hashlib.sha256(payload).hexdigest()] + row[4:] for row in manifest]
        (self.directory / 'manifest.tsv').write_text('\n'.join('\t'.join(r) for r in manifest) + '\n')
        (models / 'performance.onnx').write_bytes(payload)
        (models / 'sd_compact.onnx').write_bytes(b'unknown')
        for name in ('perf_1080p.engine', 'perf_1080p.engine.label', 'perf_1080p.engine.build.log',
                     'perf.timing.cache', 'x.trt-11.4.0.gpu-rtx-sm89.engine',
                     'x.trt-11.4.0.gpu-rtx-sm89.engine.onnxinfo'):
            (models / name).write_text('x')
        (models / '.aji-engine.tmp').mkdir()
        return models, f'''init_project_root() {{ PROJECT_ROOT="$PWD"; }}
load_model_assets() {{ source {str(ROOT / 'scripts/model-assets.sh')!r}; AJI_MODEL_MANIFEST="$AUDIT_DIR/manifest.tsv"; }}
MODELS_DIR="$AUDIT_DIR/models"
'''

    def test_migration_renames_by_checksum_and_deletes_only_when_confirmed(self):
        models, setup = self.legacy_install()
        self.run_shell(setup + 'migrate_legacy_models', 0, 'y\nn\nn\n')
        names = {p.name for p in models.iterdir()}
        self.assertIn('2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx', names)
        self.assertNotIn('performance.onnx', names)
        self.assertTrue({'sd_compact.onnx', 'perf_1080p.engine', 'perf.timing.cache', '.aji-engine.tmp'} <= names)
        self.run_shell(setup + 'migrate_legacy_models', 0, 'y\ny\n')
        names = {p.name for p in models.iterdir()}
        self.assertEqual(names, {'2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx',
                                 'x.trt-11.4.0.gpu-rtx-sm89.engine',
                                 'x.trt-11.4.0.gpu-rtx-sm89.engine.onnxinfo'})

    def test_migration_declined_rename_and_assume_yes_keep_files(self):
        models, setup = self.legacy_install()
        before = {p.name for p in models.iterdir()}
        self.run_shell(setup + 'migrate_legacy_models', 0, 'n\ny\nn\n')
        names = {p.name for p in models.iterdir()}
        self.assertIn('performance.onnx', names)
        self.assertNotIn('sd_compact.onnx', names)
        self.assertIn('perf_1080p.engine', names)
        self.run_shell(setup + 'ASSUME_DEFAULTS=1\nmigrate_legacy_models')
        self.assertTrue({'perf_1080p.engine', 'perf.timing.cache', '.aji-engine.tmp'}
                        <= {p.name for p in models.iterdir()})
        self.assertTrue(before)

    def test_legacy_profile_migration_preserves_user_lines(self):
        home = self.directory / 'home'
        home.mkdir()
        bashrc = home / '.bashrc'
        bashrc.write_text('export CUSTOM=value\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n')
        self.run_shell('''CUDA_ROOT='/opt/cuda with space'
run_as_root() { cp "$4" "$AUDIT_DIR/generated-profile"; }
install_environment_profile cuda "$AUDIT_DIR/home/.bashrc"
unset LD_LIBRARY_PATH
source "$AUDIT_DIR/generated-profile"
[ "$LD_LIBRARY_PATH" = '/opt/cuda with space/lib64' ]''')
        text = bashrc.read_text()
        self.assertIn('export CUSTOM=value', text)
        self.assertNotIn('lib64:$LD_LIBRARY_PATH', text)

    def test_cuda_invalid_manual_version_never_installs(self):
        result = self.run_shell("""check_nvidia_driver() { DRIVER_VER=580.1; }
get_available_cuda_packages() { printf '13.2\\n'; }
CUDA_VERSION=12.8
install_cuda_and_tensorrt""", 1)
        self.assertNotIn('UNEXPECTED_ROOT_CALL', result.stderr)

    def test_cuda_install_uses_exact_selected_package_versions(self):
        result = self.run_shell("""check_nvidia_driver() { DRIVER_VER=580.1; }
get_available_cuda_packages() { printf '13.2\\n'; }
apt-cache() { printf 'tensorrt | 12.0.0.1-1 | repository\\ntensorrt | 11.3.0.99-1 | repository\\n'; }
run_as_root() { printf 'ROOT: %s\\n' "$*"; }
activate_build_environment() { NVCC_VER=13.2; }
check_tensorrt() { TRT_VER=11.3.0; }
install_environment_profile() { :; }
CUDA_VERSION=13.2
ASSUME_DEFAULTS=1
install_cuda_and_tensorrt""")
        self.assertIn('tensorrt=11.3.0.99-1 tensorrt-dev=11.3.0.99-1 cuda-nvcc-13-2', result.stdout)
        self.assertNotIn('ROOT: ln', result.stdout)

    def test_ffmpeg_failed_version_check_is_failure(self):
        self.run_shell('ffmpeg() { return 7; }; check_ffmpeg', 1)

    def test_environment_profile_cleans_inherited_empty_paths(self):
        (self.directory / 'home').mkdir()
        self.run_shell("""CUDA_ROOT=/opt/cuda
run_as_root() { cp "$4" "$AUDIT_DIR/generated-profile"; }
install_environment_profile cuda "$AUDIT_DIR/home/.bashrc"
LD_LIBRARY_PATH=:/old::/other:
source "$AUDIT_DIR/generated-profile"
[ "$LD_LIBRARY_PATH" = /opt/cuda/lib64:/old:/other ]""")


class ManifestTests(unittest.TestCase):
    """deploy.sh and the runtime packager share scripts/runtime-models.tsv;
    every model a packaged or built-in slot names must be in it."""

    def test_every_slot_model_is_in_the_manifest(self):
        import re
        rows = [line.split('\t') for line in (ROOT / 'scripts/runtime-models.tsv').read_text().splitlines()
                if line and not line.startswith('#')]
        self.assertTrue(all(len(r) == 6 for r in rows), rows)
        files = {r[2] for r in rows}
        conf = (ROOT / 'packaging/ubuntu24/runtime/animejanai.conf').read_text()
        names = set(re.findall(r'^chain_\d+_model_\d+_name=(.+)$', conf, re.M))
        builtin = (ROOT / 'src/aji_conf.cpp').read_text()
        names |= set(re.findall(r'^const char \*(?:HD_BAL|HD_PERF|SD) = "([^"]+)";', builtin, re.M))
        self.assertEqual(len(names), 8, names)
        self.assertEqual({n + '.onnx' for n in names} - files, set())
        for code in re.findall(r'^chain_\d+_rife_model=(\d+)$', conf, re.M):
            self.assertIn(f'rife/rife_v{code[0]}.{code[1:]}.onnx', files)
        generator = (ROOT / 'generate_cmd.sh').read_text()
        for constant in ('HD_BAL_MODEL', 'HD_PERF_MODEL', 'SD_MODEL'):
            value = re.search(rf'^{constant}="([^"]+)"', generator, re.M).group(1)
            self.assertIn(value, names)


class Response(io.BytesIO):
    def __init__(self, content, length=None):
        super().__init__(content)
        self.url = 'https://example.test/file'
        self.headers = {'Content-Length': str(len(content) if length is None else length)}


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name) / 'model.onnx'
        self.url = 'https://example.test/a.onnx'
        self.validator = patch.object(downloader, 'validate_onnx')
        self.validator.start()
        self.addCleanup(self.validator.stop)
        aria2_patch = patch.object(downloader.shutil, 'which', return_value=None)
        self.aria2 = aria2_patch.start()
        self.addCleanup(aria2_patch.stop)

    def request(self, content, url=None, length=None, expected=''):
        with patch.object(downloader.urllib.request.OpenerDirector, 'open', return_value=Response(content, length)) as mock:
            downloader.download(url or self.url, self.destination, expected=expected)
            return mock

    def test_partial_download_preserves_existing_model(self):
        self.destination.write_bytes(b'old')
        with self.assertRaises(RuntimeError):
            self.request(b'partial', length=100)
        self.assertEqual(self.destination.read_bytes(), b'old')
        self.assertEqual(list(self.destination.parent.glob('.aji-download-*')), [])

    def test_reuses_only_verified_matching_url(self):
        self.request(b'model-a')
        mock = self.request(b'model-b')
        mock.assert_not_called()
        self.request(b'model-b', url='https://example.test/b.onnx')
        self.assertEqual(self.destination.read_bytes(), b'model-b')

    def test_python_download_sends_cdn_compatible_user_agent(self):
        mock = self.request(b'model')
        request = mock.call_args.args[0]
        self.assertEqual(request.full_url, self.url)
        self.assertEqual(request.get_header('User-agent'), 'Mozilla/5.0')

    def fake_aria2(self, args, **kwargs):
        directory = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--dir='))
        name = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--out='))
        (Path(directory) / name).write_bytes(b'aria2-model')

    def test_aria2_preferred_and_validated_before_publish(self):
        self.aria2.return_value = '/usr/bin/aria2c'
        with patch.object(downloader.subprocess, 'run', side_effect=self.fake_aria2) as run:
            mock = self.request(b'python-model')
        mock.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b'aria2-model')
        self.assertEqual(run.call_args.args[0][-1], self.url)
        downloader.validate_onnx.assert_called_once()

    def test_aria2_failure_falls_back_to_python(self):
        self.aria2.return_value = '/usr/bin/aria2c'
        def fail(args, **kwargs):
            self.fake_aria2(args)
            raise subprocess.CalledProcessError(1, args)
        with patch.object(downloader.subprocess, 'run', side_effect=fail):
            mock = self.request(b'python-model')
        mock.assert_called_once()
        self.assertEqual(self.destination.read_bytes(), b'python-model')
        self.assertEqual(list(self.destination.parent.glob('.aji-download-*')), [])

    def test_invalid_aria2_download_preserves_existing_model(self):
        self.aria2.return_value = '/usr/bin/aria2c'
        self.destination.write_bytes(b'old-model')
        with patch.object(downloader.subprocess, 'run', side_effect=self.fake_aria2):
            with patch.object(downloader, 'validate_onnx', side_effect=ValueError('invalid ONNX')):
                with self.assertRaises(RuntimeError):
                    downloader.download(self.url, self.destination)
        self.assertEqual(self.destination.read_bytes(), b'old-model')

    def test_primary_failure_uses_mirror(self):
        mirror = 'https://mirror.test/model.onnx'
        with patch.object(downloader.urllib.request.OpenerDirector, 'open', side_effect=[
                downloader.urllib.error.HTTPError(self.url, 403, 'Forbidden', {}, None),
                Response(b'mirror-model')]) as mock:
            downloader.download(self.url, self.destination, mirror=mirror)
        self.assertEqual([call.args[0].full_url for call in mock.call_args_list], [self.url, mirror])
        self.assertEqual(self.destination.read_bytes(), b'mirror-model')

    def test_local_corruption_redownloads(self):
        self.request(b'model-a')
        self.destination.write_bytes(b'corrupt')
        mock = self.request(b'model-a')
        mock.assert_called_once()
        self.assertEqual(self.destination.read_bytes(), b'model-a')

    def test_hash_mismatch_cannot_publish(self):
        with self.assertRaises(RuntimeError):
            self.request(b'model-a', expected='0' * 64)
        self.assertFalse(self.destination.exists())

    def test_malformed_receipt_redownloads(self):
        self.destination.write_bytes(b'old')
        self.destination.with_name('model.onnx.download.json').write_text('[]')
        self.request(b'new')
        self.assertEqual(self.destination.read_bytes(), b'new')

    def test_rejects_http_and_https_downgrade(self):
        with self.assertRaises(ValueError):
            downloader.download('http://example.test/model', self.destination)
        with self.assertRaises(ValueError):
            downloader.HTTPSRedirect().redirect_request(None, None, 302, '', {}, 'http://example.test/model')

    def test_validation_failure_preserves_destination(self):
        self.destination.write_bytes(b'old')
        with patch.object(downloader, 'validate_onnx', side_effect=ValueError('invalid ONNX')):
            with self.assertRaises(RuntimeError):
                self.request(b'html response')
        self.assertEqual(self.destination.read_bytes(), b'old')


try:
    import onnx
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, 'install onnx for format validation tests')
class RealOnnxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'model.onnx'

    def model(self):
        return onnx.helper.make_model(onnx.helper.make_graph(
            [onnx.helper.make_node('Identity', ['input'], ['output'])], 'test',
            [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1])]))

    def test_valid_embedded_model(self):
        onnx.save(self.model(), self.path)
        downloader.validate_onnx(self.path)

    def test_html_response_rejected(self):
        self.path.write_text('<html>error</html>')
        with self.assertRaises(ValueError):
            downloader.validate_onnx(self.path)

    def test_external_tensor_rejected_before_checker(self):
        model = self.model()
        tensor = model.graph.initializer.add()
        tensor.name = 'external'
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.append(1)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        tensor.external_data.add(key='location', value='/etc/passwd')
        self.path.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(ValueError, 'embed all tensor data'):
            downloader.validate_onnx(self.path)


if __name__ == '__main__':
    unittest.main()
