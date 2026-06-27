"""pysamoo's pyclawd config — drives `pyclawd test/lint/typecheck/...` for this repo.
"""

from pyclawd import DescriptionConfig, DoctorConfig, GoldenConfig, Project, QualityConfig, TestConfig

project = Project(
    name='pysamoo',
    conda_env='default',
    root_markers=["pyproject.toml", "setup.py"],
    # The pyclawd this config was built on. `pyclawd doctor` WARNs if the
    # running pyclawd has drifted to a different minor (migration may be needed).
    pyclawd_version='0.1.0',
    # Default directory `pyclawd ls` lists (the code/source root).
    src_dir="src",
    quality=QualityConfig(
        lint_cmd=["ruff", "check"],
        lint_fix_cmd=["ruff", "check", "--fix"],
        format_cmd=["ruff", "format"],
        format_check_cmd=["ruff", "format", "--check", "--quiet"],
        typecheck_cmd=["mypy"],
        check_sequence=["format-check", "lint", "typecheck", "descriptions", "test"],
    ),
    descriptions=DescriptionConfig(
        # vendored third-party code, runnable examples, and WIP experiments are
        # not held to the module-description bar.
        exclude=[r"/vendor/", r"/usage/", r"/experimental/"],
    ),
    test=TestConfig(
        tests_dir='tests',
        classname_prefix="tests.",
        integration_files=[],
        markers={"fast": "not slow and not integration", "default": "not slow", "all": ""},
    ),
    golden=GoldenConfig(
        # Slightly looser than the default 1e-9 to tolerate BLAS/platform float
        # noise in the numerical kernels across machines.
        rtol=1e-7,
        atol=1e-10,
    ),
    doctor=DoctorConfig(
        core_deps=["pymoo", "ezmodel"],
        dev_deps=["pytest", "pytest-xdist", "pytest-cov"],
        tool_files=[],
        binaries=[
            ("ruff", "pip install ruff"),
            ("mypy", "pip install mypy"),
        ],
    ),
)
