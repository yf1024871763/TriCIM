from pathlib import Path


def resolve_config_paths(config, project_root=None):
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()

    paths = dict(config.get("paths", {}))

    workspace_root = paths.get("workspace_root")
    if workspace_root:
        workspace_root = Path(workspace_root).expanduser().resolve()

    arch_name = paths.get("arch_name", "isaac")
    macro_name = paths.get("macro_name", f"{arch_name}_isca_2016")

    resolved = {}
    if workspace_root is not None:
        resolved["workspace_root"] = str(workspace_root)
        resolved["timeloop_scripts"] = str(workspace_root / "scripts")
        resolved["workload_root"] = str(workspace_root / "models" / "workloads")
        resolved["output_root"] = str(workspace_root / "outputs")
        resolved["arch_root"] = str(workspace_root / "models" / "arch" / "3_chip")

    # Backward-compatible overrides.
    for key in (
        "timeloop_scripts",
        "workload_root",
        "output_root",
        "arch_root",
        "booksim_binary",
    ):
        if key in paths:
            resolved[key] = str(Path(paths[key]).expanduser().resolve())

    resolved["project_root"] = str(project_root)
    resolved["plot_root"] = str(
        Path(paths.get("plot_root", project_root / "outputs")).expanduser().resolve()
    )
    resolved["booksim_binary"] = str(
        Path(
            paths.get("booksim_binary", project_root / "booksim2" / "src" / "booksim")
        )
        .expanduser()
        .resolve()
    )
    resolved["arch_name"] = arch_name
    resolved["macro_name"] = macro_name

    config["paths"] = resolved
    return config
