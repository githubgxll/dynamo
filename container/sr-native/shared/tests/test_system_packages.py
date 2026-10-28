"""Execute the bootstrap's Bash control flow with fake APT; no Linux/network claim."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / 'container/sr-native/shared/install_system_packages.sh'
BASH = r'D:\workingtools\Git\bin\bash.exe' if os.name == 'nt' else '/bin/bash'

FAKE_APT = r'''
update_count=0
download_count=0
install_count=0
apt-get() {
    printf '%s\0' "$@" >> "$FAKE_APT_CALLS"
    printf '\0' >> "$FAKE_APT_CALLS"
    local phase=install arg
    for arg in "$@"; do
        case "$arg" in
            update) phase=update ;;
            --download-only) phase=download ;;
        esac
    done
    case "$phase" in
        update)
            update_count=$((update_count + 1))
            [[ "$update_count" -gt "${FAIL_UPDATE:-0}" ]] || return 100 ;;
        download)
            download_count=$((download_count + 1))
            [[ "$download_count" -gt "${FAIL_DOWNLOAD:-0}" ]] || return 101 ;;
        install)
            install_count=$((install_count + 1))
            [[ "$install_count" -gt "${FAIL_INSTALL:-0}" ]] || return 102 ;;
    esac
    return 0
}
sleep() { printf '%s\n' "$@" >> "$FAKE_APT_SLEEPS"; }
install_system_packages "$@"
'''


class SystemPackages(unittest.TestCase):
    def setUp(self):
        # check_local supplies a per-run D: evidence directory on Windows.
        parent = Path(os.environ.get('TEMP', ROOT / '.h3-experimental-evidence/sr-native/local'))
        if os.name == 'nt' and parent.drive.lower() != ROOT.drive.lower():
            parent = ROOT / '.h3-experimental-evidence/sr-native/local'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='apt-bootstrap-', dir=parent)
        self.directory = Path(self.temp.name)
        self.apt = self.directory / 'apt with spaces'
        (self.apt / 'sources.list.d').mkdir(parents=True)
        self.ca = self.directory / 'ca.crt'
        self.ca.write_text('fixture CA bundle\n', encoding='utf-8')
        self.calls = self.directory / 'calls.bin'
        self.sleeps = self.directory / 'sleeps.txt'

    def tearDown(self):
        self.temp.cleanup()

    def bash(self, body, *args, **env_overrides):
        harness = self.directory / 'harness.sh'
        harness.write_text('source "$1"\nshift\n' + body + '\n', encoding='utf-8', newline='\n')
        env = dict(os.environ, FAKE_APT_CALLS=self.calls.as_posix(),
                   FAKE_APT_SLEEPS=self.sleeps.as_posix(), **env_overrides)
        result = subprocess.run(
            [BASH, '--noprofile', '--norc', harness.as_posix(), HELPER.as_posix(),
             *(arg.as_posix() if isinstance(arg, Path) else str(arg) for arg in args)],
            cwd=ROOT, env=env, capture_output=True, encoding='utf-8', errors='replace', timeout=30)
        return result

    def configure(self):
        return self.bash('configure_ubuntu_https "$@"', self.apt, self.ca)

    def recorded_calls(self):
        if not self.calls.exists():
            return []
        return [row.decode('utf-8').split('\0') for row in self.calls.read_bytes().split(b'\0\0') if row]

    def delays(self):
        return self.sleeps.read_text().splitlines() if self.sleeps.exists() else []

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_sources_preserve_components_signatures_and_other_hosts(self):
        original = (
            'deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu.gpg] '
            'http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse\n'
            'deb http://security.ubuntu.com/ubuntu jammy-security main universe\n'
            'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /\n'
            'deb http://mirror.example/ubuntu jammy main\n'
            'deb http://archive.ubuntu.com/ubuntu-other jammy main\n')
        source = self.apt / 'sources.list'
        source.write_text(original, encoding='utf-8', newline='\n')
        self.assert_success(self.configure())
        expected = original.replace('http://archive.ubuntu.com/ubuntu/ ', 'https://archive.ubuntu.com/ubuntu/ ')
        expected = expected.replace('http://security.ubuntu.com/ubuntu ', 'https://security.ubuntu.com/ubuntu ')
        self.assertEqual(source.read_text(), expected)

    def test_deb822_and_list_fragments_preserve_fields_and_ignore_backup(self):
        original = ('Types: deb deb-src\nURIs: http://archive.ubuntu.com/ubuntu '
                    'http://security.ubuntu.com/ubuntu\nSuites: jammy jammy-updates jammy-security\n'
                    'Components: main restricted universe multiverse\n'
                    'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n')
        source = self.apt / 'sources.list.d/ubuntu.sources'
        source.write_text(original, encoding='utf-8', newline='\n')
        fragment = self.apt / 'sources.list.d/security.list'
        fragment.write_text('deb http://security.ubuntu.com/ubuntu jammy-security universe\n')
        backup = self.apt / 'sources.list.d/ubuntu.list.save'
        backup.write_text(original)
        self.assert_success(self.configure())
        self.assertEqual(source.read_text(), original.replace('http://', 'https://'))
        self.assertEqual(fragment.read_text(), 'deb https://security.ubuntu.com/ubuntu jammy-security universe\n')
        self.assertEqual(backup.read_text(), original)

    def test_https_configuration_is_idempotent(self):
        source = self.apt / 'sources.list'
        source.write_text('deb http://security.ubuntu.com/ubuntu jammy-security universe\n')
        self.assert_success(self.configure())
        expected = source.read_bytes()
        self.assert_success(self.configure())
        self.assertEqual(source.read_bytes(), expected)

    def test_missing_ca_fails_before_changing_sources(self):
        source = self.apt / 'sources.list'
        source.write_text('deb http://security.ubuntu.com/ubuntu jammy-security universe\n')
        expected = source.read_bytes()
        self.ca.unlink()
        result = self.configure()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no CA bundle', result.stderr)
        self.assertEqual(source.read_bytes(), expected)

    def test_transient_indexes_retry_then_reach_download_and_install(self):
        result = self.bash(FAKE_APT, 'python3.10', 'ffmpeg', FAIL_UPDATE='2')
        self.assert_success(result)
        calls = self.recorded_calls()
        self.assertEqual(len(calls), 5)
        for call in calls[:3]:
            self.assertIn('APT::Update::Error-Mode=any', call)
            self.assertEqual(call[-1], 'update')
        self.assertIn('--download-only', calls[3])
        self.assertIn('--no-download', calls[4])
        self.assertEqual(self.delays(), ['5', '10'])

    def test_incomplete_indexes_exhaust_retries_and_never_install(self):
        result = self.bash(FAKE_APT, 'python3.10', FAIL_UPDATE='99')
        self.assertEqual(result.returncode, 100)
        calls = self.recorded_calls()
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call[-1] == 'update' for call in calls))
        self.assertIn('indexes failed after 3 attempts', result.stderr)
        self.assertEqual(self.delays(), ['5', '10'])

    def test_transient_package_download_retries_without_repeating_indexes(self):
        result = self.bash(FAKE_APT, 'ffmpeg', FAIL_DOWNLOAD='1')
        self.assert_success(result)
        calls = self.recorded_calls()
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][-1], 'update')
        self.assertTrue(all('--download-only' in call for call in calls[1:3]))
        self.assertIn('--no-download', calls[3])
        self.assertEqual(self.delays(), ['5'])

    def test_failed_package_download_never_runs_dpkg_install(self):
        result = self.bash(FAKE_APT, 'ffmpeg', FAIL_DOWNLOAD='99')
        self.assertEqual(result.returncode, 101)
        calls = self.recorded_calls()
        self.assertEqual(len(calls), 4)
        self.assertTrue(all('--download-only' in call for call in calls[1:]))
        self.assertFalse(any('--no-download' in call for call in calls))
        self.assertIn('packages failed after 3 attempts', result.stderr)

    def test_install_failure_is_not_retried(self):
        result = self.bash(FAKE_APT, 'ffmpeg', FAIL_INSTALL='99')
        self.assertEqual(result.returncode, 102)
        calls = self.recorded_calls()
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum('--no-download' in call for call in calls), 1)
        self.assertEqual(self.delays(), [])

    def test_download_policy_and_exact_package_arguments(self):
        packages = ['python3.10=3.10.12-1', 'libexample:amd64', 'literal value']
        self.assert_success(self.bash(FAKE_APT, *packages))
        calls = self.recorded_calls()
        self.assertEqual(len(calls), 3)
        for call in calls[:2]:
            for option in ('Acquire::Retries=3', 'Acquire::http::Timeout=60', 'Acquire::https::Timeout=60'):
                self.assertIn(option, call)
        self.assertEqual(calls[0][-1], 'update')
        for call in calls[1:]:
            self.assertIn('--no-install-recommends', call)
            self.assertEqual(call[-len(packages):], packages)

    def test_empty_package_list_fails_without_running_apt(self):
        result = self.bash(FAKE_APT)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.recorded_calls(), [])

    def test_both_dockerfiles_use_locked_defaults_and_shared_bootstrap(self):
        lock = json.loads((ROOT / 'delivery/sr-native/source-lock.json').read_text(encoding='utf-8'))
        for backend, model in lock['models'].items():
            with self.subTest(backend=backend):
                source = (ROOT / model['dockerfile']).read_text(encoding='utf-8')
                self.assertIn('ARG BASE_IMAGE=' + model['base_image'] + '\nFROM ${BASE_IMAGE}', source)
                self.assertIn('COPY container/sr-native/shared/install_system_packages.sh /opt/sr-native/shared/', source)
                self.assertIn('RUN bash /opt/sr-native/shared/install_system_packages.sh', source)
                self.assertNotIn('apt-get ', source)


if __name__ == '__main__':
    unittest.main()
