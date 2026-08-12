"""Lesson-specific derived code, importable from the notebooks.

Not an installed package: add the repository root to `sys.path` first, the way
`notebooks/lesson0/102_paths.ipynb` does for `src/`.

    import sys; sys.path.append("../..")
    from scripts.pipelines import build_irc_pipeline

What belongs here is code a lesson derived and a later lesson needs — specific
enough that `goad_toolkit` should not ship it, and not part of the `wa_analyzer`
support package.
"""
