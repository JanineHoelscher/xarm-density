#!/usr/bin/env python3
"""Replot recorded paper results from tracked CSVs, without archived runs."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-csv', action='store_true', help='Compatibility flag; CSV is always the source')
    parser.add_argument('--output', type=Path, help='Output folder; defaults to this figure folder')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output or here
    output.mkdir(parents=True, exist_ok=True)

    def run(script, *arguments):
        subprocess.run([sys.executable, '-B', str(here / script), *map(str, arguments)], check=True)

    with tempfile.TemporaryDirectory(prefix='xarm-paper-') as directory:
        temp = Path(directory)
        run('export_reproducibility_data.py', '--restore-to', temp)
        source = temp / 'results/cdf_qp_validations'
        plots = temp / 'plots'
        run('generate_validation09_figures.py', '--input', source / 'validation_09', '--output', plots)
        run('generate_validation10_figures.py', '--input', source / 'validation_10/performance', '--output', plots)
        for pattern in ('fig*.pdf', 'fig*.png', 'table*.csv', 'table*.tex', '*summary.json'):
            for path in plots.glob(pattern):
                shutil.copy2(path, output / path.name)
    # The selected camera panel is preserved, not rerendered.
    for suffix in ('.pdf', '.png'):
        snapshot = here / ('fig08_robot_snapshots' + suffix)
        if snapshot.resolve() != (output / snapshot.name).resolve():
            shutil.copy2(snapshot, output / snapshot.name)
    print(f'Regenerated numerical figures/tables in {output.resolve()}; preserved figure 8 snapshots.')


if __name__ == '__main__':
    main()
