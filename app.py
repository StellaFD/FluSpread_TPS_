from time import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="FluSpread SGLC",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Konstanta ────────────────────────────────────────────────
STATE_COLORS = {
    -1: "rgba(0,0,0,0)",
    0:  "#3b82f6",   # Susceptible  — biru
    1:  "#f59e0b",   # Exposed      — kuning
    2:  "#ef4444",   # Infected     — merah
    3:  "#10b981",   # Recovered    — hijau
    4:  "#6b7280",   # Vaccinated   — abu
}
STATE_LABELS = {
    0: "Susceptible",
    1: "Exposed",
    2: "Infected",
    3: "Recovered",
    4: "Vaccinated",
}

# ── Layout SGLC: 8 baris × 14 kolom, 84 kursi aktif ─────────
def build_mask() -> np.ndarray:
    """
    Baris 0    : area depan (Smart Board, Meja Dosen, Papan Tulis) — kosong
    Baris 1–7  : kursi mahasiswa
    Kolom 0–5  : Blok Kiri  (6 × 7 = 42 kursi)
    Kolom 6–7  : Lorong + proyektor — kosong
    Kolom 8–13 : Blok Kanan (6 × 7 = 42 kursi)
    """
    mask = np.zeros((8, 14), dtype=bool)
    for row in range(1, 8):
        for col in list(range(0, 6)) + list(range(8, 14)):
            mask[row, col] = True
    return mask


SEAT_MASK      = build_mask()
SEAT_POS       = np.argwhere(SEAT_MASK)
N_SEATS        = len(SEAT_POS)          # = 84
G_ROWS, G_COLS = SEAT_MASK.shape


# ── Model Spatial SEIR-CA ────────────────────────────────────
class SpatialSEIR_CA:
    """
    Cellular Automaton berbasis denah kelas SGLC (84 kursi).

    State per kursi:
        -1 = bukan kursi (lorong / area depan)
         0 = Susceptible
         1 = Exposed     (inkubasi, belum menular)
         2 = Infected    (bergejala, menular)
         3 = Recovered   (imun sementara)
         4 = Vaccinated  (imun penuh)

    Transisi per hari:
        P(S→E) = 1 − (1−β_eff)^n_I      n_I = jumlah tetangga Infected
        P(E→I) = σ
        P(I→R) = γ
    """

    def __init__(
        self,
        beta: float         = 0.25,
        sigma: float        = 1 / 3,
        gamma: float        = 1 / 7,
        mobility: float     = 0.10,
        vacc_rate: float    = 0.15,
        mask_eff: float     = 0.0,
        contact_radius: int = 1,
        n_initial: int      = 2,
        seed: int           = None,
    ):
        self.G_ROW          = G_ROWS
        self.G_COL          = G_COLS
        self.beta_eff       = beta * (1 - mask_eff)
        self.sigma          = sigma
        self.gamma          = gamma
        self.mobility       = mobility
        self.contact_radius = contact_radius
        self.rng            = np.random.default_rng(seed)
        self.seat_mask      = SEAT_MASK
        self.seat_pos       = SEAT_POS
        self.N              = N_SEATS

        # Inisialisasi grid
        self.grid = np.where(SEAT_MASK, 0, -1).astype(np.int8)

        # Vaksinasi awal
        n_vacc   = min(int(self.N * vacc_rate), self.N)
        vacc_idx = self.rng.choice(self.N, n_vacc, replace=False)
        for idx in vacc_idx:
            r, c = self.seat_pos[idx]
            self.grid[r, c] = 4

        # Kasus awal (pilih dari yang Susceptible)
        susc = [i for i, (r, c) in enumerate(self.seat_pos) if self.grid[r, c] == 0]
        for idx in self.rng.choice(susc, min(n_initial, len(susc)), replace=False):
            r, c = self.seat_pos[idx]
            self.grid[r, c] = 2

        self.history = []
        self._record()

    def _count_infected_neighbors(self) -> np.ndarray:
        """
        Hitung jumlah tetangga Infected (Moore neighborhood, radius r).
        Menggunakan zero-padding (bukan wrap-around) agar sel di tepi grid
        tidak "melihat" sel dari sisi berlawanan ruangan.
        """
        infected = (self.grid == 2).astype(np.float32)
        count    = np.zeros((self.G_ROW, self.G_COL), dtype=np.float32)
        r        = self.contact_radius
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di == 0 and dj == 0:
                    continue
                # Hitung irisan sumber dan tujuan secara eksplisit (tanpa wrap)
                src_row_start = max(0, -di);  src_row_end = min(self.G_ROW, self.G_ROW - di)
                src_col_start = max(0, -dj);  src_col_end = min(self.G_COL, self.G_COL - dj)
                dst_row_start = max(0,  di);  dst_row_end = min(self.G_ROW, self.G_ROW + di)
                dst_col_start = max(0,  dj);  dst_col_end = min(self.G_COL, self.G_COL + dj)
                count[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += \
                    infected[src_row_start:src_row_end, src_col_start:src_col_end]
        count[~self.seat_mask] = 0
        return count

    def step(self):
        # ── Mobilitas: tukar posisi antar mahasiswa secara acak ──────
        # Sejumlah (mobility × N) pasang kursi ditukar, lalu neighbor
        # dihitung ulang setelah pertukaran agar efek mobilitas akurat.
        if self.mobility > 0:
            n_swaps = max(1, int(self.N * self.mobility / 2))
            for _ in range(n_swaps):
                i, j = self.rng.choice(self.N, 2, replace=False)
                r1, c1 = self.seat_pos[i]
                r2, c2 = self.seat_pos[j]
                self.grid[r1, c1], self.grid[r2, c2] = \
                    self.grid[r2, c2], self.grid[r1, c1]

        new_grid      = self.grid.copy()
        inf_neighbors = self._count_infected_neighbors()
        rand_s        = self.rng.random((self.G_ROW, self.G_COL))
        rand_e        = self.rng.random((self.G_ROW, self.G_COL))
        rand_i        = self.rng.random((self.G_ROW, self.G_COL))

        for r, c in self.seat_pos:
            state = self.grid[r, c]
            if state == 0:                                         # S → E
                p = 1 - (1 - self.beta_eff) ** max(float(inf_neighbors[r, c]), 0)
                if rand_s[r, c] < p:
                    new_grid[r, c] = 1
            elif state == 1:                                       # E → I
                if rand_e[r, c] < self.sigma:
                    new_grid[r, c] = 2
            elif state == 2:                                       # I → R
                if rand_i[r, c] < self.gamma:
                    new_grid[r, c] = 3

        self.grid = new_grid
        self._record()

    def _record(self):
        g = self.grid
        S, E, I, R, V = (int(np.sum(g == s)) for s in range(5))
        self.history.append({
            "S": S, "E": E, "I": I, "R": R, "V": V,
            "N": self.N,
            "grid": g.copy(),
            "attack_rate": round((I + R) / self.N * 100, 2),
        })

    def run(self, days: int = 60):
        for _ in range(days):
            h = self.history[-1]
            if h["I"] == 0 and h["E"] == 0:
                break
            self.step()

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Hari": i, **{k: v for k, v in h.items() if k != "grid"}}
            for i, h in enumerate(self.history)
        ])

    def R0(self) -> float:
        """R₀ = β_eff × k̄ / γ,  k̄ = rata-rata tetangga aktual per kursi."""
        r = self.contact_radius
        k_mean = np.mean([
            sum(
                1
                for di in range(-r, r + 1)
                for dj in range(-r, r + 1)
                if not (di == 0 and dj == 0)
                and 0 <= row + di < self.G_ROW
                and 0 <= col + dj < self.G_COL
                and self.seat_mask[row + di, col + dj]
            )
            for row, col in self.seat_pos
        ])
        return round(self.beta_eff * k_mean / self.gamma, 2)


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def grid_to_scatter(grid: np.ndarray):
    """Ubah grid numpy ke list xs, ys, colors, texts untuk Plotly scatter."""
    xs, ys, colors, texts = [], [], [], []
    for row in range(G_ROWS):
        for col in range(G_COLS):
            state = int(grid[row, col])
            xs.append(col)
            ys.append(row)
            colors.append(STATE_COLORS[state])
            label = STATE_LABELS.get(state, "Kosong") if state >= 0 else "Kosong"
            texts.append(f"Baris {row}, Kol {col}: {label}")
    return xs, ys, colors, texts


def add_classroom_shapes(fig: go.Figure):
    """Tambahkan anotasi papan tulis & garis lorong ke figure Plotly."""
    fig.add_shape(type="rect", x0=-0.5, x1=13.5, y0=-0.7, y1=-0.2,
                  fillcolor="#1e3a5f", opacity=0.85, line_width=0)
    for text, xpos in [("SMART BOARD", 2.5), ("MEJA DOSEN | LAYAR", 6.5), ("PAPAN TULIS", 10.5)]:
        fig.add_annotation(x=xpos, y=-0.45, text=text,
                           font=dict(color="white", size=8, family="monospace"),
                           showarrow=False)
    for x in (5.5, 7.5):
        fig.add_shape(type="line", x0=x, x1=x, y0=0.5, y1=G_ROWS - 0.5,
                      line=dict(color="#f59e0b", width=2.5, dash="dash"))
    fig.add_annotation(x=6.5, y=4, text="LORONG",
                       font=dict(color="#f59e0b", size=9),
                       showarrow=False, textangle=-90)


def add_state_legend(fig: go.Figure):
    for state, label in STATE_LABELS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=STATE_COLORS[state], symbol="square"),
            name=label,
        ))


CLASSROOM_LAYOUT = dict(
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
               range=[-0.7, G_COLS - 0.3]),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
               range=[G_ROWS - 0.4, -0.9], autorange=False),
    plot_bgcolor="#f0f4f8",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.05,
                xanchor="center", x=0.5, font=dict(size=10)),
)


# ════════════════════════════════════════════════════════════════
# UI — HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("## 🦠 FluSpread SGLC — Simulasi Penyebaran Flu")
st.markdown("**Kelas SGLC Fakultas Teknik · 84 Mahasiswa · Model Spatial SEIR-CA**")
st.divider()


# ════════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Parameter Simulasi")

    st.subheader("Populasi & Kasus Awal")
    n_initial = st.slider("Kasus awal (I₀)", 1, 10, 2)
    vacc_rate = st.slider("Tingkat vaksinasi (%)", 0, 50, 15, 5,
                          help="% mahasiswa yang sudah divaksin flu sebelum kelas dimulai")
    vacc_rate_f = vacc_rate / 100

    st.subheader("Parameter Penularan")
    beta  = st.slider("β — Prob. penularan/kontak/hari", 0.05, 0.50, 0.25, 0.01)
    sigma = st.slider("σ — Prob. inkubasi→infeksi/hari", 0.10, 0.80, 0.33, 0.01,
                      help="1/σ = durasi inkubasi rata-rata (hari)")
    gamma = st.slider("γ — Prob. sembuh/hari", 0.05, 0.40, 0.14, 0.01,
                      help="1/γ = durasi sakit rata-rata (hari)")

    st.subheader("Intervensi")
    mask_eff = st.slider("Efektivitas masker (%)", 0, 90, 0, 5,
                         help="0% = tidak ada masker | 75% ≈ masker KN95") / 100
    contact_radius = st.radio("Radius kontak", [1, 2, 3], index=0,
                               help="r=1: 8 tetangga | r=2: 24 tetangga | r=3: 48 tetangga")
    mobility = st.slider("Mobilitas (%/hari)", 0, 40, 10, 5,
                         help="% mahasiswa yang pindah kursi setiap hari") / 100

    st.subheader("Durasi & Ensemble")
    sim_days   = st.slider("Hari simulasi", 10, 90, 60, 5)
    n_ensemble = st.slider("Jumlah run ensemble", 5, 30, 10, 5)
    run_btn    = st.button("▶ Jalankan Simulasi", use_container_width=True, type="primary")


# ════════════════════════════════════════════════════════════════
# SIMULASI
# ════════════════════════════════════════════════════════════════
params = dict(
    beta=beta, sigma=sigma, gamma=gamma, mobility=mobility,
    vacc_rate=vacc_rate_f, mask_eff=mask_eff,
    contact_radius=contact_radius, n_initial=n_initial,
)

if "models" not in st.session_state or run_btn:
    with st.spinner("Menjalankan simulasi ensemble..."):
        base_seed = int(time.time())
        models = []
        for i in range(n_ensemble):
            m = SpatialSEIR_CA(**params, seed=i * 137 + base_seed)
            m.run(days=sim_days)
            models.append(m)
    st.session_state.models = models
    st.session_state.params = params

models = st.session_state.models
m0     = models[0]
R0_val = m0.R0()
Reff_val = round(R0_val * (1 - vacc_rate_f), 2)

ar_vals   = [m.df["attack_rate"].iloc[-1] for m in models]
peak_vals = [m.df["I"].max() for m in models]
peak_days = [m.df["I"].idxmax() for m in models]


# ════════════════════════════════════════════════════════════════
# KPI METRICS
# ════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "R₀ CA-Model", f"{R0_val:.2f}",
    help=(
        "R₀ = β_eff × k̄ / γ (definisi per-tetangga CA).\n\n"
        "Nilai ini lebih tinggi dari R₀ populasi WHO (1.2–2.0) karena β "
        "di sini adalah probabilitas penularan per tetangga per hari."
    ),
)
c2.metric(
    "R_eff (dengan vaksinasi)", f"{Reff_val:.2f}",
    "⚠️ Wabah" if Reff_val > 1 else "✅ Terkendali",
    delta_color="inverse" if Reff_val > 1 else "normal",
    help="R_eff = R₀ × (1 - v). Nilai < 1 berarti wabah padam.",
)
c3.metric("Attack Rate (Median)", f"{np.median(ar_vals):.1f}%",
          f"dari {N_SEATS} mahasiswa")
c4.metric("Puncak Infeksi (Median)", f"{np.median(peak_vals):.0f} orang")
c5.metric("Hari Puncak (Median)", f"Hari ke-{np.median(peak_days):.0f}")

if Reff_val < 1:
    st.success(f"R_eff = {Reff_val:.2f} < 1 → Wabah akan padam dengan sendirinya.")
elif Reff_val < 2:
    st.warning(f"R_eff = {Reff_val:.2f} → Wabah menyebar, masih bisa dikendalikan.")
else:
    st.error(
        f"R_eff = {Reff_val:.2f} → Wabah menyebar cepat! "
        f"(R₀ CA = {R0_val:.2f}, dikurangi efek vaksinasi {vacc_rate}%)"
    )

st.divider()


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dinamika SEIR",
    "🎬 Animasi Kelas",
    "🏫 Denah Snapshot",
    "📈 Sensitivitas",
])


# ── Tab 1: Dinamika SEIR ─────────────────────────────────────
with tab1:
    st.subheader("Dinamika Populasi SEIR")

    COMP_COLORS = {"S": "#3b82f6", "E": "#f59e0b", "I": "#ef4444", "R": "#10b981"}
    fig = go.Figure()

    for comp, color in COMP_COLORS.items():
        for m in models:
            d = m.df
            fig.add_trace(go.Scatter(
                x=d["Hari"], y=d[comp], mode="lines",
                line=dict(color=color, width=0.5),
                opacity=0.12, showlegend=False, hoverinfo="skip",
            ))
        max_t = max(len(m.df) for m in models)
        median_vals = [
            np.median([
                m.df[comp].iloc[t] if t < len(m.df) else m.df[comp].iloc[-1]
                for m in models
            ])
            for t in range(max_t)
        ]
        fig.add_trace(go.Scatter(
            x=list(range(max_t)), y=median_vals, mode="lines",
            line=dict(color=color, width=2.8), name=comp,
        ))

    fig.update_layout(
        xaxis_title="Hari",
        yaxis_title="Jumlah Mahasiswa",
        yaxis=dict(range=[0, N_SEATS + 2]),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Garis tebal = median {n_ensemble} run  |  Garis tipis = tiap run individual")


# ── Tab 2: Animasi Kelas ─────────────────────────────────────
with tab2:
    st.subheader("Animasi Penyebaran di Kelas SGLC")
    st.info("Klik ▶ Play. Setiap persegi = satu kursi mahasiswa. Lorong tengah transparan.")

    col_a, col_b = st.columns(2)
    with col_a:
        step_anim = st.radio("Tampilkan setiap", [1, 2, 5], index=1,
                              horizontal=True, format_func=lambda x: f"{x} hari")
    with col_b:
        spd_label = st.select_slider(
            "Kecepatan animasi",
            options=["Lambat (600ms)", "Sedang (350ms)", "Cepat (150ms)"],
            value="Sedang (350ms)",
        )
    frame_ms   = {"Lambat (600ms)": 600, "Sedang (350ms)": 350, "Cepat (150ms)": 150}[spd_label]
    frame_days = list(range(0, len(m0.history), step_anim))

    anim_frames = []
    for day in frame_days:
        h_data = m0.history[day]
        xs, ys, cs, txts = grid_to_scatter(h_data["grid"])
        title_str = (
            f"Hari ke-{day}  |  "
            f"S={h_data['S']}  E={h_data['E']}  "
            f"I={h_data['I']}  R={h_data['R']}  V={h_data['V']}"
        )
        anim_frames.append(go.Frame(
            data=[go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=20, color=cs, symbol="square",
                            line=dict(width=1, color="white")),
                text=txts, hovertemplate="%{text}<extra></extra>",
            )],
            name=str(day),
            layout=go.Layout(title_text=title_str),
        ))

    h0 = m0.history[0]
    xs0, ys0, cs0, txts0 = grid_to_scatter(h0["grid"])

    fig_a = go.Figure(
        data=[go.Scatter(
            x=xs0, y=ys0, mode="markers",
            marker=dict(size=20, color=cs0, symbol="square",
                        line=dict(width=1, color="white")),
            text=txts0, hovertemplate="%{text}<extra></extra>",
        )],
        frames=anim_frames,
    )
    add_classroom_shapes(fig_a)
    add_state_legend(fig_a)
    fig_a.update_layout(
        **CLASSROOM_LAYOUT,
        height=480,
        title=dict(
            text=f"Hari ke-0  |  S={h0['S']}  E={h0['E']}  I={h0['I']}  R={h0['R']}  V={h0['V']}",
            font=dict(size=12),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.18, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 80}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0, yanchor="top", xanchor="left",
            currentvalue=dict(prefix="Hari: ", visible=True, xanchor="right"),
            pad=dict(b=10, t=10), len=0.9, x=0.05,
            steps=[
                dict(
                    args=[[f.name], {"frame": {"duration": frame_ms, "redraw": True},
                                     "mode": "immediate"}],
                    label=str(day), method="animate",
                )
                for day, f in zip(frame_days, anim_frames)
            ],
        )],
    )
    st.plotly_chart(fig_a, use_container_width=True)


# ── Tab 3: Denah Snapshot ────────────────────────────────────
with tab3:
    st.subheader("Denah Kelas SGLC — Snapshot per Hari")

    day_sel = st.slider("Hari ke-", 0, len(m0.history) - 1, 0, key="snap_day")
    h_s     = m0.history[day_sel]

    c1, c2, c3, c4, c5 = st.columns(5)
    for col_ui, key, label in zip(
        [c1, c2, c3, c4, c5],
        ["S", "E", "I", "R", "V"],
        ["Susceptible", "Exposed", "Infected", "Recovered", "Vaccinated"],
    ):
        col_ui.metric(label, h_s[key])

    xs_s, ys_s, cs_s, txts_s = grid_to_scatter(h_s["grid"])
    fig_s = go.Figure(go.Scatter(
        x=xs_s, y=ys_s, mode="markers",
        marker=dict(size=22, color=cs_s, symbol="square",
                    line=dict(width=1.5, color="white")),
        text=txts_s, hovertemplate="%{text}<extra></extra>",
    ))
    add_classroom_shapes(fig_s)
    add_state_legend(fig_s)
    fig_s.update_layout(**CLASSROOM_LAYOUT, height=430)
    st.plotly_chart(fig_s, use_container_width=True)

    # Heatmap kumulatif
    st.subheader("Heatmap Zona Panas Wabah")
    cum = np.zeros((G_ROWS, G_COLS))
    for h in m0.history:
        cum += (h["grid"] == 2).astype(float) + (h["grid"] == 3).astype(float)
    cum[~SEAT_MASK] = np.nan

    hover_text = [
        [
            f"Kursi ({r},{c}): {cum[r, c]:.0f} hari terinfeksi"
            if SEAT_MASK[r, c] else ""
            for c in range(G_COLS)
        ]
        for r in range(G_ROWS)
    ]
    fig_hm = go.Figure(go.Heatmap(
        z=cum, colorscale="YlOrRd",
        text=hover_text, hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title="Akumulasi<br>hari infeksi"),
    ))
    fig_hm.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        plot_bgcolor="#f0f4f8",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Warna merah = kursi yang paling sering menjadi sumber/tujuan infeksi sepanjang simulasi.")


# ── Tab 4: Sensitivitas ──────────────────────────────────────
with tab4:
    st.subheader("Analisis Sensitivitas Parameter")

    col_p, col_m = st.columns(2)
    with col_p:
        param_s = st.selectbox("Parameter sweep",
                               ["β (Penularan)", "γ (Pemulihan)", "Radius Kontak"])
    with col_m:
        metric_s = st.radio("Metrik output",
                            ["Attack Rate (%)", "Puncak Infeksi"], horizontal=True)

    base_p = dict(
        beta=beta, sigma=sigma, gamma=gamma, mobility=mobility,
        vacc_rate=vacc_rate_f, mask_eff=mask_eff,
        contact_radius=contact_radius, n_initial=n_initial,
    )
    if param_s == "β (Penularan)":
        sweep_vals = np.round(np.arange(0.05, 0.50, 0.05), 2)
        pkey       = "beta"
    elif param_s == "γ (Pemulihan)":
        sweep_vals = np.round(np.arange(0.07, 0.40, 0.03), 3)
        pkey       = "gamma"
    else:
        sweep_vals = [1, 2, 3]
        pkey       = "contact_radius"

    N_SWEEP   = 20
    sweep_key = f"sens_{param_s}_{metric_s}_{beta}_{gamma}_{sigma}_{mobility}_{vacc_rate}_{mask_eff}_{contact_radius}"

    if sweep_key not in st.session_state:
        with st.spinner(f"Menghitung sensitivitas ({N_SWEEP} run per nilai)..."):
            medians, q25s, q75s = [], [], []
            for v in sweep_vals:
                p2   = {**base_p, pkey: v}
                runs = [SpatialSEIR_CA(**p2, seed=i * 137 + 42) for i in range(N_SWEEP)]
                for m2 in runs:
                    m2.run(sim_days)
                if "Attack" in metric_s:
                    values = [m2.df["attack_rate"].iloc[-1] for m2 in runs]
                else:
                    values = [m2.df["I"].max() for m2 in runs]
                medians.append(np.median(values))
                q25s.append(np.percentile(values, 25))
                q75s.append(np.percentile(values, 75))
            st.session_state[sweep_key] = (medians, q25s, q75s)
    else:
        medians, q25s, q75s = st.session_state[sweep_key]

    fig_sw = go.Figure()
    fig_sw.add_trace(go.Scatter(
        x=list(sweep_vals) + list(sweep_vals[::-1]),
        y=q75s + q25s[::-1],
        fill="toself", fillcolor="rgba(239,68,68,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="IQR 25–75%", hoverinfo="skip",
    ))
    fig_sw.add_trace(go.Scatter(
        x=sweep_vals, y=medians, mode="lines+markers",
        line=dict(color="#ef4444", width=2.5),
        marker=dict(size=7), name="Median",
    ))
    fig_sw.update_layout(
        xaxis_title=param_s,
        yaxis_title=metric_s,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
        height=340,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_sw, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    "FluSpread SGLC v2.1 · Model Spatial SEIR-CA · "
    "Referensi: Sirakoulis et al. (2000), White et al. (2007), WHO Influenza Factsheet (2023)"
)
