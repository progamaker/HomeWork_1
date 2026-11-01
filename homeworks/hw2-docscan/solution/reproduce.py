"""Повторить измерения на примерах курса: python reproduce.py."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    commands = []
    records = []
    def run(args):
        command = [sys.executable, 'docscan.py', *args]
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout.strip().splitlines()[-1])
        commands.append('python docscan.py ' + shlex.join(args))
        records.append({'command_line': commands[-1], 'result': data})
        print(commands[-1], flush=True)
        return data
    for name in ('desk', 'corner', 'tiles', 'plain', 'nosheet'):
        run(['noise', f'sample_data/{name}.jpg'])
    for sigma in ('5', '10', '25'):
        run(['noise', 'sample_data/desk.jpg', '--add-sigma', sigma])
    for name in ('desk', 'corner', 'tiles', 'plain'):
        for label, flags in [('default', []), ('noclose', ['--close', '0']), ('fixed', ['--canny', '50,150'])]:
            run(['edges', f'sample_data/{name}.jpg', *flags, '-o', f'out/{name}_edges_{label}.png'])
    for label, flags in [('default', []), ('fixed', ['--canny', '50,150']), ('noclose', ['--close', '0'])]:
        run(['eval', 'sample_data', '--ann', 'sample_data/corners.json', '--seeds', '20', *flags,
             '--csv', f'out/eval_{label}.csv', '--overlays', f'out/eval_{label}'])
    for name in ('desk', 'corner', 'tiles', 'plain'):
        for method in ('otsu', 'adaptive', 'flat'):
            run(['scan', f'sample_data/{name}.jpg', '--canny', '50,150', '--binarize', method,
                 '-o', f'out/{name}_scan.jpg', '--bw', f'out/{name}_{method}.png', '--gt', 'sample_data/page-gt.png'])
    for method in ('contour', 'hough', 'auto'):
        flags = ['--save', 'out/video.mp4'] if method == 'auto' else []
        run(['video', '--source', 'sample_data/clip.mp4', '--method', method, '--no-window', '--seconds', '0', *flags])
    (out / 'measurements.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    (out / 'commands.sh').write_text('# Запускать из каталога решения\n' + '\n'.join(commands) + '\n')
    subprocess.run([sys.executable, 'make_report.py'], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
