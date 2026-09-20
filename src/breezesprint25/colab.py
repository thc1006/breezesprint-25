"""Small notebook controller; never imports JAX into the Colab kernel.

Only a child interpreter from the isolated venv owns the TPU. Rerunning the
transcription cell reuses its weights and JIT executables. No keep-alive tricks.
"""
from __future__ import annotations
import html
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import traceback
from .text_output import download_name


def cpu_budget():
    """Bound threads by both process affinity and cgroup v2/v1 quotas."""
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            cores = min(cores, max(1, math.ceil(int(quota)/int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        try:
            base = Path('/sys/fs/cgroup/cpu')
            quota = int((base/'cpu.cfs_quota_us').read_text())
            period = int((base/'cpu.cfs_period_us').read_text())
            if quota > 0:
                cores = min(cores, max(1, math.ceil(quota/period)))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return max(1, cores)


def tail(path, lines=20, limit=7000):
    path = Path(path)
    if not path.exists():
        return '(No log was created.)'
    # Do not read an unbounded log into the notebook kernel.
    with path.open('rb') as file:
        file.seek(max(0, path.stat().st_size-limit))
        text = file.read().decode('utf-8', errors='replace')
    return '\n'.join(text.splitlines()[-lines:])


class LiveStatus:
    def __init__(self):
        self.handle = None
        self.last = ''

    def show(self, text, error=False):
        if text == self.last:
            return
        self.last = text
        try:
            from IPython.display import HTML, display
            mark = 'ERROR' if error else 'BreezeSprint25'
            view = HTML('<div style="padding:10px 14px;border-left:4px solid '+
                        ('#b91c1c' if error else '#2563eb')+';font:14px sans-serif">'
                        '<b>'+mark+'</b> · '+html.escape(text)+'</div>')
            if self.handle is None:
                self.handle = display(view, display_id=True)
            else:
                self.handle.update(view)
        except ImportError:
            print(text, flush=True)


def _run_logged(command, log):
    with Path(log).open('a', encoding='utf-8') as file:
        file.write('\n$ '+' '.join(map(str, command))+'\n')
        file.flush()
        result = subprocess.run(list(map(str, command)), stdout=file, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed with exit {result.returncode}. See {log}.\n'+tail(log))


def environment_matches(python, lock):
    if not Path(python).is_file():
        return False
    code = '''import importlib.metadata as m, pathlib, sys
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
 line=line.strip()
 if line and not line.startswith('#'):
  name,expected=line.split('==')
  if m.version(name)!=expected: raise SystemExit(1)
'''
    return subprocess.run([str(python), '-c', code, str(lock)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def setup(bundle: Path, root=Path('/content/breezesprint25')):
    start = time.perf_counter()
    root, bundle = Path(root), Path(bundle)
    root.mkdir(exist_ok=True, parents=True)
    ui = LiveStatus()
    log = root/'install.log'
    try:
        if not (3, 11) <= sys.version_info[:2] <= (3, 13):
            raise RuntimeError('This pinned release requires Python 3.11–3.13. '
                               'Use a supported Colab runtime; do not silently upgrade the stack.')
        if sys.platform != 'linux':
            raise RuntimeError('The notebook targets Linux Colab TPU runtimes.')
        python = root/'venv/bin/python'
        lock = bundle/'requirements-tpu.lock'
        ui.show('Checking the isolated environment…')
        if not environment_matches(python, lock):
            if not python.is_file():
                _run_logged([sys.executable, '-m', 'venv', '--without-pip', root/'venv'], log)
            ui.show('Installing the pinned TPU stack; full details are in install.log…')
            _run_logged([sys.executable, '-m', 'pip', '--python', python, 'install',
                         '--disable-pip-version-check', '--no-input', '--prefer-binary',
                         '-r', lock], log)
            _run_logged([sys.executable, '-m', 'pip', '--python', python, 'check'], log)
            if not environment_matches(python, lock):
                raise RuntimeError('Installed versions differ from the pinned runtime lock.')
        if not shutil.which('ffmpeg'):
            ui.show('Installing FFmpeg…')
            _run_logged(['apt-get', 'update', '-qq'], log)
            _run_logged(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], log)
        session = Session(root, bundle, python, start)
        ui.show('TPU worker started. Run cell 2 to upload recorded audio while the model loads.')
        return session
    except BaseException:
        ui.show('Setup failed. The detailed error below was not suppressed.', error=True)
        print(f'Full setup log: {log}\n'+tail(log), file=sys.stderr)
        raise


class Session:
    def __init__(self, root, bundle, python, started=None):
        self.root, self.bundle = Path(root), Path(bundle)
        self.started = time.perf_counter() if started is None else started
        self.directory = self.root/'sessions'/uuid.uuid4().hex
        self.directory.mkdir(parents=True, mode=0o700)
        self.log = self.directory/'worker.log'
        env = os.environ.copy()
        cores = cpu_budget()
        # A 128x201 log-mel GEMM should not spawn every host thread. Leave cores
        # available for XLA, I/O and the one-file prefetch thread.
        threads = min(4, cores)
        env.update(PYTHONPATH=str(self.bundle/'src'), JAX_PLATFORMS='tpu,cpu',
                   PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                   HF_HUB_DISABLE_PROGRESS_BARS='1',
                   OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
                   WHISPERSPRINT_CPU_BUDGET=str(cores))
        # No sysfs writes, privileged THP changes, giant pinned buffers, unsafe
        # math flags or automatic accelerator/paid-plan selection.
        with self.log.open('w') as logfile:
            self.process = subprocess.Popen([str(python), '-m', 'breezesprint25.worker',
                                             '--root', str(self.root), '--session', str(self.directory)],
                                            stdin=subprocess.PIPE, stdout=logfile,
                                            stderr=subprocess.STDOUT, text=True,
                                            env=env, start_new_session=True)
        self.last_result = None
        self.last_text_files = []

    def status(self):
        path = self.directory/'status.json'
        if path.exists():
            return json.loads(path.read_text())
        return dict(state='starting', stage='starting isolated worker')

    def _ensure_alive(self):
        status = self.status()
        if self.process.poll() is not None or status.get('state') == 'failed':
            raise RuntimeError(f"TPU worker failed at {status.get('stage')}: "
                               f"{status.get('message', 'see worker.log')}\n"+tail(self.log))
        return status

    def _wait_ready(self, ui):
        start = time.monotonic()
        while True:
            status = self._ensure_alive()
            if status.get('state') == 'ready':
                return status
            ui.show(status.get('stage', 'preparing model')+'…')
            if time.monotonic()-start > 3600:
                raise TimeoutError('Setup exceeded the one-hour guard. Inspect worker.log; restart rather than hiding the stall.')
            time.sleep(.3)

    def transcribe(self, paths, output=None):
        paths = list(dict.fromkeys(str(Path(path).expanduser().resolve(strict=True)) for path in paths))
        if not paths or any(not Path(p).is_file() for p in paths):
            raise ValueError('Upload one or more recorded audio/video files in cell 2.')
        output = Path(output) if output else self.root/'results'
        ui = LiveStatus()
        start = time.perf_counter()
        self.last_result, self.last_text_files = None, []
        try:
            status = self._wait_ready(ui)
            request_id = uuid.uuid4().hex
            request = dict(id=request_id, op='transcribe', paths=paths, output=str(output))
            self.process.stdin.write(json.dumps(request)+'\n')
            self.process.stdin.flush()
            response_path = self.directory/(request_id+'.json')
            while not response_path.exists():
                status = self._ensure_alive()
                label = status.get('stage', 'transcribing')
                if status.get('request') == request_id and 'file' in status:
                    label += ' · '+status['file'][:100]
                    if 'total' in status:
                        label += f" · {status.get('processed', 0):.1f}/{status['total']:.1f}s"
                ui.show(label)
                time.sleep(.2)
            response = json.loads(response_path.read_text())
            if not response['ok']:
                raise RuntimeError(f"{response['error_type']} at {response['stage']}: "
                                   f"{response['message']}\n{response['recovery']}\n"+tail(self.log))
            result = response['result']
            result['upload_cell_wait_and_processing_seconds'] = time.perf_counter()-start
            stage_start = time.perf_counter()
            text_files = self.stage_text_downloads(result, request_id, output)
            result['txt_download_staging_seconds'] = time.perf_counter()-stage_start
            result['notebook_elapsed_including_user_gaps_seconds'] = time.perf_counter()-self.started
            result['browser_download_seconds'] = None
            (self.directory/'notebook_timing.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            self.last_result, self.last_text_files = result, text_files
            reviews = sum(item['review_windows'] for item in result['outputs'])
            ui.show(f"Done · {len(result['outputs'])} TXT file(s) · {result['audio_seconds']:.2f}s audio · "
                    f"{result['batch_wall_seconds']:.3f}s processing (model setup excluded).")
            if reviews:
                print(f'{reviews} window(s) carry timing/quality flags; not necessarily errors. Text is preserved. Details remain in local transcript.json.')
            if not any(item.get('segments', 0) for item in result['outputs']):
                print('No text segments were detected. The TXT is empty; check the audio if speech was expected.')
            return text_files
        except KeyboardInterrupt:
            self.close(interrupt=True)
            ui.show('Interrupted. Committed windows and partial exports remain on disk. Rerun cell 1 before resuming.', True)
            raise
        except Exception:
            ui.show('Transcription failed. No result download was requested. Full details follow.', True)
            print(f'Worker log: {self.log}', file=sys.stderr)
            # Do not create or download a diagnostics archive. Leave the primary
            # traceback intact even if reading an optional log also fails.
            try:
                print(tail(self.log), file=sys.stderr)
            except Exception as log_error:
                print(f'Could not read the log: {log_error}', file=sys.stderr)
            raise

    def stage_text_downloads(self, result, ident, output):
        """Stage only completed timestamped .txt files under readable names."""
        if result.get('processing_complete') is not True or not result.get('outputs'):
            raise RuntimeError('The worker did not return a complete transcription.')
        destination = self.root/'downloads'/ident
        destination.mkdir(parents=True, exist_ok=False)
        base = Path(output).resolve()
        paths, used = [], set()
        for item in result['outputs']:
            if item.get('processing_complete') is not True:
                raise RuntimeError('An incomplete transcript cannot be downloaded as final.')
            directory = Path(item['directory']).resolve(strict=True)
            if not directory.is_relative_to(base):
                raise RuntimeError('Output directory escaped the results folder.')
            source = directory/'transcript.txt'
            if source.is_symlink() or not source.is_file():
                raise RuntimeError('The completed timestamped TXT file is missing or unsafe.')
            name = download_name(item['input_name'], used)
            target = destination/name
            temporary = target.with_name(target.name+'.tmp')
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            paths.append(target)
        return paths

    def close(self, interrupt=False):
        if self.process.poll() is not None:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            return
        try:
            if not interrupt:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            else:
                raise subprocess.TimeoutExpired('interrupt', 0)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            # A dedicated process group created by this Session; never a global pkill.
            try:
                os.killpg(self.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass  # the owned worker can exit between poll and signal
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
        finally:
            if self.process.stdin:
                self.process.stdin.close()


def upload_and_transcribe(session):
    """The entire second cell: upload only, then automatically download TXT files.

    Cancellation never reuses an earlier upload. Browser download errors are not
    misreported as ASR failures, and no archive or diagnostic file is downloaded.
    """
    from google.colab import files
    ui = LiveStatus()
    try:
        session._ensure_alive()
        directory = session.root/'uploads'/uuid.uuid4().hex
        directory.mkdir(parents=True, mode=0o700)
        old_cwd = Path.cwd()
        uploaded = None
        try:
            os.chdir(directory)
            uploaded = files.upload()
            paths = []
            for name in uploaded:
                # Colab writes the upload into the current folder. Do not accept
                # directory traversal, symlinks or names from outside that folder.
                if Path(name).name != name or '/' in name or '\\' in name:
                    raise ValueError('Unsafe upload filename.')
                path = directory/name
                if path.is_symlink() or not path.is_file():
                    raise ValueError('The upload did not create a regular local file.')
                path = path.resolve(strict=True)
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError('Upload path escaped its private folder.')
                paths.append(path)
        finally:
            if uploaded is not None:
                uploaded.clear()  # release the widget's duplicate byte buffers
            os.chdir(old_cwd)
        if not paths:
            print('No file uploaded. Run this cell again to choose recorded audio.')
            return []
        text_files = session.transcribe(paths)
    except KeyboardInterrupt:
        raise  # Session handles an active TPU job; no extra download on interruption.
    except Exception:
        ui.show('Upload or transcription failed. No automatic download. See the traceback and worker log.', True)
        print(f'Worker log: {session.log}', file=sys.stderr)
        raise
    for path in text_files:
        try:
            files.download(str(path))
        except Exception as error:
            ui.show(f'Transcription completed, but the browser download could not start: {error}', True)
            # Browser controls can prevent automatic saving. Keep a clear fallback
            # instead of disguising a completed transcription as a model error.
            print(f'TXT saved: {path}\nDownload it from the Colab Files panel.', file=sys.stderr)
            try:
                with (session.directory/'download-errors.log').open('a', encoding='utf-8') as stream:
                    stream.write(traceback.format_exc()+'\n')
            except OSError:
                pass  # preserve the already displayed browser error
    return text_files
