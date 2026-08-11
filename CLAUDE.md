# Assistant policy for MADS-DAV

This repo registers two MCP servers as coaches, not oracles:

- **goad** — is my analysis any good?
- **codestyle** — is my code any good?

Each serves a `docs/` folder as its only source of truth, through a staged
conversation it will not short-circuit: it asks questions, waits for your
own words, and only writes code after you've answered them.

## The expectation

The assistant coaches. You decide. Every claim, every plot, every line of
code you submit must be defensible without the assistant present — if you
cannot explain why a choice was made, it was not your choice.

This is graded, not just advised: leerdoel 0.1 is assessed by asking you to
defend a choice, not by policing which tools you used.

## Using the servers

`.mcp.json` (Claude Code) and `.cursor/mcp.json` (Cursor) pin both servers
to a tagged release — never `main`. A breaking change should never land on
you mid-assignment. When a new tag is cut for the cohort, both files get
the same `_REF` bump, together.
