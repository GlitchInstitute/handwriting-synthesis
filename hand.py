"""Handwriting synthesis: pure numpy inference of Graves' LSTM+attention+MDN model."""
from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ALPHABET = [
    "\x00", " ", "!", '"', "#", "'", "(", ")", ",", "-", ".",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";",
    "?", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z",
]
ALPHA_TO_NUM = defaultdict(int, {ch: i for i, ch in enumerate(ALPHABET)})
VALID_CHARS = set(ALPHABET)
STYLES_DIR = Path(__file__).parent / "styles"

_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
_softplus = lambda x: np.log1p(np.exp(np.clip(x, -20, 20)))
DASH_MAP = str.maketrans("\u2012\u2013\u2014\u2015\u2212", "-----")


def encode(text: str) -> np.ndarray:
    return np.array([ALPHA_TO_NUM[c] for c in text] + [0])



class Hand:
    """Loads pre-extracted weights and generates handwriting strokes."""

    def __init__(self, weights_path: str | Path | None = None):
        path = Path(weights_path) if weights_path else Path(__file__).parent / "weights.npz"
        w = np.load(path)
        pfx = "rnn.LSTMAttentionCell."
        self.lstm1_k, self.lstm1_b = w[pfx + "lstm_cell.kernel"], w[pfx + "lstm_cell.bias"]
        self.lstm2_k, self.lstm2_b = w[pfx + "lstm_cell_1.kernel"], w[pfx + "lstm_cell_1.bias"]
        self.lstm3_k, self.lstm3_b = w[pfx + "lstm_cell_2.kernel"], w[pfx + "lstm_cell_2.bias"]
        self.attn_w, self.attn_b = w[pfx + "attention.weights"], w[pfx + "attention.biases"]
        self.gmm_w, self.gmm_b = w["rnn.gmm.weights"], w["rnn.gmm.biases"]

    @staticmethod
    def _lstm(x: np.ndarray, h: np.ndarray, c: np.ndarray,
              kernel: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gates = np.concatenate([x, h]) @ kernel + bias
        i, j, f, o = np.split(gates, 4)
        i, f, o = _sigmoid(i), _sigmoid(f + 1.0), _sigmoid(o)
        c_new = f * c + i * np.tanh(j)
        return o * np.tanh(c_new), c_new

    def _attention(self, w_prev: np.ndarray, inp: np.ndarray, h1: np.ndarray,
                   kappa_prev: np.ndarray, chars_oh: np.ndarray,
                   chars_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.concatenate([w_prev, inp, h1]) @ self.attn_w + self.attn_b
        alpha, beta, kappa_inc = np.split(_softplus(p), 3)
        kappa = kappa_prev + kappa_inc / 25.0
        beta = np.clip(beta, 0.01, np.inf)
        u = np.arange(chars_oh.shape[0], dtype=np.float32)
        phi = np.sum(
            alpha[:, None] * np.exp(-(kappa[:, None] - u[None, :]) ** 2 / beta[:, None]),
            axis=0,
        )
        mask = np.arange(chars_oh.shape[0]) < chars_len
        w = (phi * mask) @ chars_oh
        return w, phi, kappa

    def _sample_output(self, h3: np.ndarray, bias: float) -> tuple[np.ndarray, float]:
        z = h3 @ self.gmm_w + self.gmm_b
        pis, sigmas, rhos, mus, es = np.split(z, [20, 60, 80, 120])
        pis = pis * (1 + bias)
        sigmas = sigmas - bias
        pis = np.exp(pis - pis.max())
        pis /= pis.sum()
        pis[pis < 0.01] = 0
        if pis.sum() > 0:
            pis /= pis.sum()

        sigmas = np.clip(np.exp(sigmas), 1e-4, np.inf)
        rhos = np.clip(np.tanh(rhos), -0.9999, 0.9999)
        es = np.clip(_sigmoid(es), 1e-8, 1 - 1e-8)
        es[es < 0.01] = 0

        mu1, mu2 = mus[:20], mus[20:]
        s1, s2 = sigmas[:20], sigmas[20:]

        idx = np.random.choice(20, p=pis)
        m = np.array([mu1[idx], mu2[idx]])
        cov = np.array([
            [s1[idx] ** 2, rhos[idx] * s1[idx] * s2[idx]],
            [rhos[idx] * s1[idx] * s2[idx], s2[idx] ** 2],
        ])
        xy = np.random.multivariate_normal(m, cov)
        eos = float(np.random.binomial(1, es[0]))
        return np.array([xy[0], xy[1], eos]), es[0]

    @staticmethod
    def _should_stop(phi: np.ndarray, chars_len: int, eos: float) -> bool:
        idx = int(np.argmax(phi))
        return (idx >= chars_len - 1 and eos >= 0.5) or idx >= chars_len

    @staticmethod
    def _zero_state(window_size: int) -> dict:
        H = 400
        return dict(
            h1=np.zeros(H, dtype=np.float32), c1=np.zeros(H, dtype=np.float32),
            h2=np.zeros(H, dtype=np.float32), c2=np.zeros(H, dtype=np.float32),
            h3=np.zeros(H, dtype=np.float32), c3=np.zeros(H, dtype=np.float32),
            kappa=np.zeros(10, dtype=np.float32),
            w=np.zeros(window_size, dtype=np.float32),
            phi=np.zeros(1, dtype=np.float32),
        )

    def _step(self, inp: np.ndarray, state: dict, chars_oh: np.ndarray,
              chars_len: int) -> dict:
        s1_in = np.concatenate([state["w"], inp])
        state["h1"], state["c1"] = self._lstm(s1_in, state["h1"], state["c1"], self.lstm1_k, self.lstm1_b)
        state["w"], state["phi"], state["kappa"] = self._attention(
            state["w"], inp, state["h1"], state["kappa"], chars_oh, chars_len,
        )
        s2_in = np.concatenate([inp, state["h1"], state["w"]])
        state["h2"], state["c2"] = self._lstm(s2_in, state["h2"], state["c2"], self.lstm2_k, self.lstm2_b)
        s3_in = np.concatenate([inp, state["h2"], state["w"]])
        state["h3"], state["c3"] = self._lstm(s3_in, state["h3"], state["c3"], self.lstm3_k, self.lstm3_b)
        return state

    def _generate_line(self, text: str, style: int | None = None,
                       bias: float = 0.5, max_steps: int = 0) -> np.ndarray:
        if style is not None:
            x_prime = np.load(STYLES_DIR / f"style-{style}-strokes.npy")
            c_prime = np.load(STYLES_DIR / f"style-{style}-chars.npy").tobytes().decode("utf-8")
            full_text = c_prime + " " + text
        else:
            full_text = text

        chars_encoded = encode(full_text)
        chars_oh = np.eye(len(ALPHABET), dtype=np.float32)[chars_encoded]
        chars_len = len(chars_encoded)
        max_steps = max_steps or 40 * len(text)

        state = self._zero_state(len(ALPHABET))

        if style is not None:
            for t in range(len(x_prime)):
                state = self._step(x_prime[t].astype(np.float32), state, chars_oh, chars_len)

        outputs = []
        inp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(max_steps):
            state = self._step(inp, state, chars_oh, chars_len)
            out, es_val = self._sample_output(state["h3"], bias)
            outputs.append(out)
            if self._should_stop(state["phi"], chars_len, es_val):
                break
            inp = out.astype(np.float32)

        return np.array(outputs) if outputs else np.zeros((1, 3))

    def generate(self, text: str, style: int | None = None, bias: float = 0.75,
                 line_width: int = 60) -> list[np.ndarray | None]:
        """Generate strokes for text, auto-wrapping at line_width chars. None = blank line."""
        text = text.translate(DASH_MAP)
        text = "".join(ch for ch in text if ch in VALID_CHARS or ch == "\n")

        lines: list[str] = []
        for paragraph in text.split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue
            lines.extend(textwrap.wrap(paragraph, width=line_width) or [""])

        return [self._generate_line(line, style=style, bias=bias) if line else None
                for line in lines]



def _offsets_to_coords(offsets: np.ndarray) -> np.ndarray:
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def _denoise(coords: np.ndarray) -> np.ndarray:
    splits = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    result = []
    for s in splits:
        if len(s) < 7:
            result.append(s)
            continue
        x = savgol_filter(s[:, 0], 7, 3, mode="nearest")
        y = savgol_filter(s[:, 1], 7, 3, mode="nearest")
        result.append(np.column_stack([x, y, s[:, 2]]))
    return np.vstack(result) if result else coords


def _align(coords: np.ndarray) -> np.ndarray:
    coords = coords.copy()
    X = np.column_stack([np.ones(len(coords)), coords[:, 0]])
    Y = coords[:, 1].reshape(-1, 1)
    try:
        offset, slope = (np.linalg.inv(X.T @ X) @ X.T @ Y).squeeze()
    except np.linalg.LinAlgError:
        return coords
    theta = np.arctan(slope)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords[:, :2] = coords[:, :2] @ R - offset
    return coords


def strokes_to_svg(stroke_list: list[np.ndarray | None], scale: float = 1.5,
                   stroke_color: str = "black", stroke_width: float = 2.0,
                   humanness: float = 1.0) -> str:
    """Convert list of stroke arrays (one per line) to SVG. None entries = blank lines."""
    h = humanness
    line_height = 60
    max_x = 0.0

    paths = []
    y_offset = -(3 * line_height / 4)
    margin_x = 20.0
    page_tilt = np.random.normal(0, 0.001 * h)
    line_idx = 0

    for entry in stroke_list:
        if entry is None:
            y_offset -= line_height
            continue

        offsets = entry.copy()
        offsets[:, :2] *= scale
        coords = _offsets_to_coords(offsets)
        coords = _denoise(coords)
        coords[:, :2] = _align(coords[:, :2])
        coords[:, 1] *= -1
        coords[:, :2] -= coords[:, :2].min(axis=0)

        margin_x += np.random.normal(0, 1.5 * h)
        coords[:, 0] += np.clip(margin_x, 10, 40)
        coords[:, 1] += -y_offset + line_idx * page_tilt * 1000

        theta = np.random.normal(0, 0.004 * h)
        cx = coords[:, 0].min()
        cy = coords[:, 1].mean()
        dx, dy = coords[:, 0] - cx, coords[:, 1] - cy
        coords[:, 0] = cx + dx * np.cos(theta) - dy * np.sin(theta)
        coords[:, 1] = cy + dx * np.sin(theta) + dy * np.cos(theta)

        max_x = max(max_x, float(coords[:, 0].max()))

        segments: list[list[tuple[float, float]]] = [[]]
        for x, y, eos in coords:
            segments[-1].append((x, y))
            if eos == 1.0 and segments[-1]:
                segments.append([])
        segments = [s for s in segments if s]
        segments.reverse()

        d = ""
        for seg in segments:
            d += f"M{seg[0][0]:.1f},{seg[0][1]:.1f} "
            for x, y in seg[1:]:
                d += f"L{x:.1f},{y:.1f} "

        sw = stroke_width * np.clip(np.random.normal(1.0, 0.04 * h), 0.85, 1.15)
        paths.append(
            f'<path d="{d}" stroke="{stroke_color}" stroke-width="{sw:.2f}" '
            f'fill="none" stroke-linecap="round"/>'
        )
        y_offset -= line_height + np.random.normal(0, 2.5 * h)
        line_idx += 1

    view_width = max_x + 40
    view_height = line_height * (len(stroke_list) + 1)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_width:.0f} {view_height}">'
        + "".join(paths)
        + "</svg>"
    )
