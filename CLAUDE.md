# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Git commits

- Do **not** add a "Co-Authored-By: Claude" line, or any similar AI sign-off or
  attribution, to commit messages or pull request descriptions.

## Package management

- This project uses `uv`, not pip. Run tools through uv, e.g. `uv run pytest`,
  and manage dependencies with `uv sync` / `uv add`.
