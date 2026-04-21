"""Streamlit UI for handwriting synthesis pen-plotter tool."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import settings
from hand import VALID_CHARS, Hand, strokes_to_svg

PREVIEWS_DIR = Path(__file__).parent / "previews"
NUM_STYLES = 13
COLORS = ["black", "blue", "red", "green", "darkblue", "darkred"]


@st.cache_resource
def load_model() -> Hand:
    return Hand()


def main() -> None:
    st.set_page_config(page_title="Handwriting Synthesis", layout="wide")
    st.title("Handwriting Synthesis")
    st.caption("Generate handwritten text as SVG for pen plotters")

    hand = load_model()
    cfg = settings.get_all()

    with st.sidebar:
        st.header("Settings")

        style = st.selectbox("Style", range(NUM_STYLES), index=int(cfg["style"]),
                             format_func=lambda x: f"Style {x}")
        bias = st.slider("Neatness (bias)", 0.0, 1.5, float(cfg["bias"]), 0.05,
                         help="Higher = neater handwriting")
        line_width = st.slider("Line wrap (chars)", 20, 120, int(cfg["line_width"]),
                               help="Auto-wrap text at this character width")
        scale = st.slider("Scale", 0.5, 3.0, float(cfg["scale"]), 0.1)
        stroke_width = st.slider("Stroke width", 0.5, 5.0, float(cfg["stroke_width"]), 0.25,
                                 help="Pen width for SVG paths")
        stroke_color = st.selectbox("Stroke color", COLORS,
                                    index=COLORS.index(cfg["stroke_color"]) if cfg["stroke_color"] in COLORS else 0)
        num_versions = st.number_input("Versions to generate", 1, 10, int(cfg["num_versions"]))

        if st.button("Save settings"):
            settings.put_all(dict(style=style, bias=bias, line_width=line_width,
                                  scale=scale, stroke_width=stroke_width,
                                  stroke_color=stroke_color, num_versions=num_versions))
            st.success("Saved!")

    with st.expander("Style previews", expanded=False):
        cols = st.columns(3)
        for i in range(NUM_STYLES):
            p = PREVIEWS_DIR / f"style_{i}.svg"
            with cols[i % 3]:
                st.markdown(f"**Style {i}**")
                if p.exists():
                    st.image(p.read_text(), use_container_width=True)

    text = st.text_area("Enter text", height=150,
                        placeholder="Type your letter here...\nLong lines will auto-wrap.")

    invalid = set(text) - VALID_CHARS - {"\n"}
    if invalid:
        st.warning(f"These characters will be stripped: {invalid}")

    if st.button("Generate", type="primary", disabled=not text.strip()):
        for v in range(int(num_versions)):
            with st.spinner(f"Generating version {v + 1}/{int(num_versions)}..."):
                strokes = hand.generate(text, style=style, bias=bias, line_width=line_width)
                svg = strokes_to_svg(strokes, scale=scale, stroke_color=stroke_color,
                                     stroke_width=stroke_width)

            label = f"Version {v + 1}" if num_versions > 1 else "Result"
            st.subheader(label)
            st.image(svg, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(f"Download SVG ({label})", svg,
                                   f"handwriting_v{v + 1}.svg", "image/svg+xml")
            with col2:
                with st.expander("View SVG source"):
                    st.code(svg, language="xml")


if __name__ == "__main__":
    main()
