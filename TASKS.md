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
