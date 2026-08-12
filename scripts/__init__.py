"""Code these lessons derived, kept where later lessons can import it.

Installed with the repo, so `from scripts.pipelines import build_irc_pipeline`
works from any notebook without path juggling.

What belongs here is an example worth keeping: specific enough that
`goad_toolkit` should not ship it, reused often enough that copying it into the
next notebook would leave two versions of one regex. One-off maintenance
scripts live in `tools/` and are not installed.
"""
