# =====================================================================
# JWCC v2 — Fig. 1a: tadqiqot hududi xaritasi (+ Fig. 1b bilan birlashtirilgan Fig. 1)
# Colab'da BITTA katakka qo'yib ishga tushiring. Boshqa kataklarga bog'liq emas.
#
# Kirish:  CA-discharge GeoPackage (basseynlar va postlar)
#          ixtiyoriy: Fig1b_workflow.png (MATLAB chiqargan) — Drive'ga yuklang
# Chiqish: MyDrive/jwcc_v2/figures/Fig1a_study_area.(png|tif|pdf)
#          MyDrive/jwcc_v2/figures/Fig1_combined.(png|tif)  — agar Fig1b topilsa
# =====================================================================
import os, sys, glob, subprocess, warnings
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run("pip -q install contextily matplotlib-scalebar", shell=True)
ROOT = "/content/drive/MyDrive" if IN_COLAB else os.environ.get("JWCC_ROOT", ".")

# ------------------------------------------------------------------ sozlamalar
GPKG = None            # None -> avtomatik qidiradi; yoki: f"{ROOT}/Scopus/CA-discharge - Copy.gpkg"
FIG1B = None           # None -> avtomatik qidiradi (Fig1b_workflow.png)
OUT = f"{ROOT}/jwcc_v2/figures"; os.makedirs(OUT, exist_ok=True)
W_CM = 17.4            # ikki ustunli kenglik (Acta Geophysica: 174 mm)

# 1-jadval (maqola): gauge -> (nom, muzlik %, tog' tizimi)
BASINS = {16279: ("Chatkal", 0.50), 16290: ("Pskem", 3.21), 16300: ("Ugam", 0.00),
          16936: ("Naryn", 1.96), 17202: ("Karatag", 2.80), 17211: ("Sangardak", 0.04),
          17288: ("Zeravshan", 5.33), 16176: ("Padshaata", 0.69), 16202: ("Chadak", 0.00)}

import numpy as np, pandas as pd, geopandas as gpd, pyogrio
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})

# ------------------------------------------------------------------ GeoPackage
if GPKG is None:
    for pat in [f"{ROOT}/**/CA*discharge*.gpkg", f"{ROOT}/**/*.gpkg"]:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits: GPKG = hits[0]; break
assert GPKG and os.path.exists(GPKG), "GeoPackage topilmadi — GPKG ga yo'lni qo'lda yozing"
layers = pyogrio.list_layers(GPKG)
print("GeoPackage:", GPKG); print("Qatlamlar:", [(l[0], l[1]) for l in layers])

def code_col(df):
    return next(c for c in df.columns if c.lower() in ("code", "gauge", "gid", "station_id", "id"))

poly_layer = next((l[0] for l in layers if "basin" in l[0].lower()), None) or \
             next(l[0] for l in layers if "Polygon" in str(l[1]))
B = gpd.read_file(GPKG, layer=poly_layer)
B["gauge"] = pd.to_numeric(B[code_col(B)].astype(str).str.strip(), errors="coerce")
B = B[B.gauge.isin(BASINS)].dissolve(by="gauge").reset_index()
B["name"] = B.gauge.map(lambda g: BASINS[g][0]); B["glac"] = B.gauge.map(lambda g: BASINS[g][1])
print(f"Basseynlar: {len(B)}/9")

G = None
for lname, gtype in layers:
    if "Point" in str(gtype):
        try:
            g = gpd.read_file(GPKG, layer=lname)
            g["gauge"] = pd.to_numeric(g[code_col(g)].astype(str).str.strip(), errors="coerce")
            g = g[g.gauge.isin(BASINS)].drop_duplicates("gauge")
            if len(g): G = g; print(f"Postlar qatlami: {lname} ({len(g)}/9)"); break
        except StopIteration:
            continue
if G is None:
    print("Postlar qatlami topilmadi — post belgilari chizilmaydi")

B = B.to_crs(3857)
if G is not None: G = G.to_crs(3857)

# davlat chegaralari (Natural Earth); internet bo'lmasa o'tkazib yuboriladi
try:
    import urllib.request
    ne_zip = "/tmp/ne_50m_countries.zip"
    if not os.path.exists(ne_zip):
        urllib.request.urlretrieve("https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip", ne_zip)
    NE = gpd.read_file(f"zip://{ne_zip}").to_crs(3857)
except Exception as e:
    NE = None; print("Davlat chegaralari yuklanmadi:", e)

# ------------------------------------------------------------------ xarita
xmin, ymin, xmax, ymax = B.total_bounds
pad_x, pad_y = (xmax - xmin) * 0.06, (ymax - ymin) * 0.10
ext = (xmin - pad_x, xmax + pad_x, ymin - 2.2 * pad_y, ymax + 1.6 * pad_y)
aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
fig_w = W_CM / 2.54; fig_h = fig_w * aspect * 0.86 + 0.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

try:
    import contextily as cx
    cx.add_basemap(ax, source=cx.providers.Esri.WorldShadedRelief, zoom=7, attribution=False)
except Exception as e:
    ax.set_facecolor("#eeeeee"); print("Relief fon yuklanmadi:", e)

if NE is not None:
    NE = NE.cx[ext[0]:ext[1], ext[2]:ext[3]]
    NE.boundary.plot(ax=ax, color="0.35", linewidth=0.6, linestyle="--", zorder=2)
    for _, r in NE.iterrows():
        p = r.geometry.representative_point()
        if ext[0] < p.x < ext[1] and ext[2] < p.y < ext[3]:
            ax.text(p.x, p.y, r.get("NAME", ""), fontsize=7, color="0.30", style="italic",
                    ha="center", zorder=2, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

bounds = [0, 0.1, 1, 2, 3, 4, 6]
cmap = mpl.colormaps["Blues"].resampled(len(bounds) - 1)
norm = BoundaryNorm(bounds, cmap.N)
B.plot(ax=ax, column="glac", cmap=cmap, norm=norm, edgecolor="k", lw=0.6, alpha=0.85, zorder=3)

# nom siljishlari (gradus): zich joylashgan basseynlar nomi ko'rsatkich chizig'i bilan chetga chiqariladi
LABEL_OFF = {"Pskem": (-0.3, 0.75), "Chatkal": (0.9, 0.75), "Ugam": (-1.2, 0.35), "Padshaata": (0.9, -0.6),
             "Chadak": (-0.9, -0.6), "Karatag": (1.1, -0.25), "Sangardak": (1.2, -0.55),
             "Naryn": (0, 0), "Zeravshan": (0, 0)}
from pyproj import Transformer as _T
_to4326 = _T.from_crs(3857, 4326, always_xy=True); _to3857 = _T.from_crs(4326, 3857, always_xy=True)
for _, r in B.iterrows():
    p = r.geometry.representative_point()
    lon, lat = _to4326.transform(p.x, p.y)
    dx, dy = LABEL_OFF.get(r["name"], (0, 0))
    tx, ty = _to3857.transform(lon + dx, lat + dy)
    if (dx, dy) != (0, 0):
        ax.annotate(r["name"], xy=(p.x, p.y), xytext=(tx, ty), fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=7,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="k", shrinkA=0, shrinkB=2),
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    else:
        ax.text(p.x, p.y, r["name"], fontsize=8, fontweight="bold", ha="center", va="center", zorder=7,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
if G is not None:
    G.plot(ax=ax, marker="^", color="#d62728", edgecolor="k", markersize=28, lw=0.5, zorder=6)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.09, aspect=40, ticks=bounds)
cb.set_label("Glacier fraction (%)", fontsize=8); cb.ax.tick_params(labelsize=8)

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    lat0 = np.deg2rad(40.5)                       # Web Mercator masshtab tuzatishi
    ax.add_artist(ScaleBar(np.cos(lat0), units="m", location="lower right", length_fraction=0.15,
                           font_properties={"size": 7}, box_alpha=0.7))
except Exception as e:
    print("Masshtab chizig'i qo'shilmadi:", e)
ax.annotate("N", xy=(0.96, 0.93), xytext=(0.96, 0.80), xycoords="axes fraction", ha="center",
            fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.2, color="k"))
if G is not None:
    ax.scatter([], [], marker="^", c="#d62728", edgecolors="k", s=28, label="Discharge gauge")
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.8)

# geografik koordinatalar o'q belgilari
from pyproj import Transformer
tf = Transformer.from_crs(3857, 4326, always_xy=True); ti = Transformer.from_crs(4326, 3857, always_xy=True)
lon_min, lat_min = tf.transform(ext[0], ext[2]); lon_max, lat_max = tf.transform(ext[1], ext[3])
lons = np.arange(np.ceil(lon_min), np.floor(lon_max) + 1, 2); lats = np.arange(np.ceil(lat_min), np.floor(lat_max) + 1, 1)
ax.set_xticks([ti.transform(l, lat_min)[0] for l in lons]); ax.set_xticklabels([f"{l:.0f}°E" for l in lons])
ax.set_yticks([ti.transform(lon_min, l)[1] for l in lats]); ax.set_yticklabels([f"{l:.0f}°N" for l in lats])
ax.tick_params(labelsize=7, length=2)
fig.tight_layout()

for ext_ in ("png", "tif"):
    fig.savefig(f"{OUT}/Fig1a_study_area.{ext_}", dpi=600, bbox_inches="tight")
fig.savefig(f"{OUT}/Fig1a_study_area.pdf", bbox_inches="tight")
print("Saqlandi:", f"{OUT}/Fig1a_study_area.(png|tif|pdf)")

# ------------------------------------------------------------------ a + b birlashtirish
if FIG1B is None:
    hits = glob.glob(f"{ROOT}/**/Fig1b_workflow.png", recursive=True); FIG1B = hits[0] if hits else None
if FIG1B and os.path.exists(FIG1B):
    from PIL import Image
    a = Image.open(f"{OUT}/Fig1a_study_area.png").convert("RGB"); b = Image.open(FIG1B).convert("RGB")
    wd = max(a.width, b.width)
    a = a.resize((wd, int(a.height * wd / a.width))); b = b.resize((wd, int(b.height * wd / b.width)))
    gap = int(0.03 * wd); out = Image.new("RGB", (wd, a.height + b.height + gap), "white")
    out.paste(a, (0, 0)); out.paste(b, (0, a.height + gap))
    from PIL import ImageDraw, ImageFont
    dr = ImageDraw.Draw(out)
    try: fnt = ImageFont.truetype("LiberationSans-Bold.ttf", int(0.022 * wd))
    except Exception: fnt = ImageFont.load_default()
    dr.text((int(0.005 * wd), 0), "a", fill="black", font=fnt)
    dr.text((int(0.005 * wd), a.height + gap // 2), "b", fill="black", font=fnt)
    for ext_ in ("png", "tif"):
        out.save(f"{OUT}/Fig1_combined.{ext_}", dpi=(600, 600))
    print("Birlashtirilgan Fig. 1 saqlandi:", f"{OUT}/Fig1_combined.(png|tif)")
else:
    print("Fig1b_workflow.png Drive'da topilmadi — faqat 1a saqlandi. "
          "Uni jwcc_v2/figures ga yuklasangiz, birlashtirilgan Fig. 1 ham chiqadi.")
plt.show() if IN_COLAB else None
