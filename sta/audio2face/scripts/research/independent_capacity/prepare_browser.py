"""Build a research-only width-128 runner; production runtime stays unchanged."""
from prepare import APP, OUT, freeze, sha


def main():
    root = OUT / 'browser'; root.mkdir(exist_ok=True)
    runner = APP / 'src/compact-webgpu.js'
    project = APP / 'src/compact-project.wgsl'
    source = runner.read_text()
    replacements = {
        'const hidden = 96;': 'const hidden = 128;',
        "'./compact-conv.wgsl?raw'": "'/src/compact-conv.wgsl?raw'",
        "'./compact-project.wgsl?raw'": "'./compact-project-128.wgsl?raw'",
        "'./compact-mel.js'": "'/src/compact-mel.js'",
        'pass.dispatchWorkgroups(7, 12);': 'pass.dispatchWorkgroups(7, Math.ceil(hidden / 8));',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('Runner no longer matches: ' + old)
        source = source.replace(old, new)
    freeze(root / 'compact-webgpu-128.js', source.encode())
    freeze(root / 'compact-project-128.wgsl', project.read_text().replace('96u', '128u').encode())
    freeze(root / 'source-lock.json', dict(production_runner_sha256=sha(runner), production_project_sha256=sha(project),
           convolution_sha256=sha(APP / 'src/compact-conv.wgsl'), frontend_sha256=sha(APP / 'src/compact-mel.js'),
           prototype_runner_sha256=sha(root / 'compact-webgpu-128.js'), prototype_project_sha256=sha(root / 'compact-project-128.wgsl')))
    print('Prepared isolated 128-channel prototype; public 96-channel runner unchanged.')


if __name__ == '__main__':
    main()
