"""A CLI interface for running the notebooks using nbconvert."""

import click
from . import NOTEBOOKS
import jupyter_client
from pathlib import Path
import subprocess as sbp
import os

k_manager = jupyter_client.kernelspec.KernelSpecManager()
avail_kernels = k_manager.find_kernel_specs()

main = click.Group()

@main.command()
def avail():
    """List available notebooks."""
    for nb in NOTEBOOKS:
        click.echo(nb.stem)


@main.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument("notebook", type=click.Choice([nb.stem for nb in NOTEBOOKS]))
@click.option("-k", "--kernel", type=click.Choice(list(avail_kernels.keys())), default="python3")
@click.option("-t", "--timeout", type=int, default=-1)
@click.option("-f", "--formats", type=str, multiple=True, default=['html'])
@click.option("--ipynb/--no-ipynb", default=True)
@click.option("-o", "--output", type=str, default=None)
@click.option("--output-dir", type=click.Path(exists=True, dir_okay=True, file_okay=False), default='.')
@click.option('--execute-args', type=str, default='')
@click.option('--convert-args', type=str, default='')
@click.option("--params", type=str, default='', help="Extra parameters to pass to the notebook as env vars")
def run(notebook, kernel, timeout, formats, ipynb, output, output_dir, execute_args, convert_args, params):
    """Use nbconvert to run a notebook."""
    nbfile = NOTEBOOKS[[nb.stem for nb in NOTEBOOKS].index(notebook)]

    if output is None:
        output = f'{notebook}.ipynb'
    elif not output.endswith('.ipynb'):
        output = f'{output}.ipynb'

    print("Executing Notebook.")
    print(f"Got notebook params: '{params}'")
    # We might have a params string like "--opt1=this --opt2 that", make this a dict...
    print([
        (p.split('=') if '=' in p else p.split(" ")) for p in (' '+params.strip()).split(' --')[1:]
    ])
    params = {
        k.replace("-", "_").upper(): v for k, v in [
            (p.split('=') if '=' in p else p.split(" ")) for p in (' '+params.strip()).split(' --')[1:]
        ]
    }
    print("Setting the env variables:")
    for k, v in params.items():
        print(f"    {k}={v}")
    
    sbp.run(
        [
            'jupyter', 'nbconvert', 
            "--output", str(output),
            "--output-dir", output_dir, 
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout", str(timeout),
            "--ExecutePreprocessor.kernel_name", kernel,
            execute_args,
            str(nbfile)
        ],
        env={**params, **os.environ},
    )

    for fmt in formats:
        print(f"Converting executed notebook to {fmt}...")
        sbp.run(
            [
                'jupyter', 'nbconvert', 
                "--output", output.replace(".ipynb", f".{fmt}"),
                "--output-dir", output_dir,
                "--to", fmt,
                convert_args,
                f"{output_dir}/{output}"
            ]
        )

    if not ipynb:
        (Path(output_dir) / output).unlink()
