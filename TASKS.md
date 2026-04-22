# Handwriting Synthesis - Pen Plotter Tool

## Tasks

- [x] **1. Delete unnecessary files** — Removed lyrics.py, prepare_data.py, data_frame.py, .travis.yml, img/, data/, old TF code
- [x] **2. Extract TF weights to numpy** — Used uv+Python3.11+TF2 to dump checkpoint to weights.npz (40MB)
- [x] **3. Pure numpy inference engine** — Reimplemented 3-layer LSTM+attention+MDN in pure numpy (no TF dependency)
- [x] **4. SVG generation** — Pen-plotter-friendly SVG (no fill, round linecaps, clean paths)
- [x] **5. Text wrapping** — Auto-wrap at configurable character width via textwrap
- [x] **6. SQLite settings** — Persistent settings for style, bias, stroke_width, color, line_width, scale, num_versions
- [x] **7. Streamlit UI** — Text input, style picker, N-version generation, SVG preview + download
- [x] **8. Style previews** — Pre-generated SVG previews for all 13 styles (0-12) in 3-column grid
- [x] **9. E2E testing** — Verified via Playwright: page load, style selection, text input, multi-version generation
- [x] **10. Final cleanup** — Removed old TF code, minimized to 399 lines of Python across 3 files
- [x] **11. Left-align SVG text** — Replaced centering with 20px left margin in strokes_to_svg
- [x] **12. Replace long dashes** — Normalize em/en/figure dashes to hyphen via str.maketrans before generation
- [x] **13. Add pixi** — Configured pyproject.toml with [tool.pixi.workspace], all deps from conda-forge, pixi tasks
- [x] **14. Reverse stroke order** — Pen-up segments within each line reversed for correct plotter draw order
- [x] **15. Remove SVG background rect** — Plotter-safe: no rect element that would trace a border
- [x] **16. Human-like imperfections** — Margin drift (random walk), variable line spacing, per-line rotation, stroke width variation, page tilt
- [x] **17. Preserve blank lines** — Empty lines (single/double/etc) now produce vertical gaps in SVG instead of being stripped
- [x] **18. Dynamic SVG viewBox** — Width computed from actual stroke extents instead of hardcoded 1000px; fixes clipping at wide line_width
- [x] **19. Prompt history** — SQLite prompts table, save on generate, reverse-chronological list at bottom, click-to-load into text area
