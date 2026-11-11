"""알고핏 졸업심사 발표(5차) 그래프 — 모든 숫자는 capstone/output 결과 파일에서 그대로 읽어 그림
사용법: python make_figures.py <capstone 폴더> <글꼴 폴더(Pretendard otf)> [저장 폴더=figures]
그림 크기(인치)를 슬라이드에 넣는 크기와 같게 맞춰, 글자 크기(pt)가 슬라이드에서도 그대로 보이게 함"""
import sys, os, glob
from pathlib import Path
import matplotlib, matplotlib.ticker, matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd

CAP = Path(sys.argv[1]).resolve(); FD = Path(sys.argv[2]); OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "figures").resolve()
OUT.mkdir(parents=True, exist_ok=True)
for f in glob.glob(str(FD / "Pretendard*.otf")): font_manager.fontManager.addfont(f)
fam = sorted({font_manager.FontProperties(fname=f).get_name() for f in glob.glob(str(FD / "Pretendard*.otf"))})[0]
sys.path.insert(0, str(CAP)); os.chdir(CAP); import engine
TXT, SUB = "#1A1A1A", "#33373D"
NAVY, ORANGE, BLUE, MID = "#22336B", "#E07A4F", "#3D5AFE", "#7C8796"
plt.rcParams.update({"font.family": fam, "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                     "axes.unicode_minus": False, "savefig.dpi": 300, "font.size": 14, "text.color": TXT,
                     "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT, "axes.edgecolor": "#444A52",
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 14, "ytick.labelsize": 13})
o = CAP / "output"
D = pd.read_csv(o / "step3_diagnosis_users.csv")
X = pd.read_csv(o / "step4_behavior_types.csv"); S = engine.load_settings(); names = S["학습유형"]["유형_이름"]
check = {}

# ---------- 그림 1 (01 문제 제기): 정답률 구간별 학습자 수 → 95% 이상 4,081명을 펼쳐 공식 실력 점수 분포
D["정답률"] = D["해결 수"] / D["문제 수"]
labs = ["80% 미만", "80~90%", "90~95%", "95% 이상"]
cnt = pd.cut(D["정답률"], [0, 0.8, 0.9, 0.95, 1.0001], right=False, labels=labs).value_counts().reindex(labs)
top = D[D["정답률"] >= 0.95]["기준일 레이팅"]
bl = ["1200 미만", "1200~1599", "1600~1999", "2000~2399", "2400 이상"]
c2 = pd.cut(top, [0, 1200, 1600, 2000, 2400, 5000], right=False, labels=bl).value_counts().reindex(bl)
assert cnt.sum() == len(D) and c2.sum() == len(top)
check["그림1 왼쪽 막대 합"] = int(cnt.sum()); check["그림1 오른쪽 막대 합"] = int(c2.sum())
share = (D["정답률"] >= 0.9).mean()
fig = plt.figure(figsize=(11.6, 4.1))
ax1 = fig.add_axes([0.085, 0.16, 0.345, 0.65]); ax2 = fig.add_axes([0.665, 0.16, 0.31, 0.65])
ax1.bar(labs, cnt.values, color=[MID, MID, NAVY, ORANGE], width=0.62)
for i, v in enumerate(cnt.values): ax1.text(i, v + 70, f"{v:,}명", ha="center", fontsize=14, color=ORANGE if i == 3 else TXT)
ax1.set_ylim(0, cnt.max() * 1.47); ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")); ax1.set_ylabel("학습자 수 (명)", fontsize=13)
yb = cnt.max() * 1.23
ax1.plot([1.7, 1.7, 3.3, 3.3], [yb - 120, yb, yb, yb - 120], color=TXT, lw=1.6)
ax1.text(2.5, yb + 110, f"정답률 90% 이상 = {share:.1%}", ha="center", fontsize=14, color=TXT)
ax1.set_title(f"정답률 구간별 학습자 수 (진단 대상 {len(D):,}명)", fontsize=16, pad=10, loc="left")
ax2.barh(bl[::-1], c2.values[::-1], color=ORANGE, height=0.6)
for i, v in enumerate(c2.values[::-1]): ax2.text(v + 30, i, f"{v:,}명", va="center", fontsize=14)
ax2.set_xlim(0, c2.max() * 1.3); ax2.set_xlabel("학습자 수 (명)", fontsize=13); ax2.tick_params(axis="y", labelsize=14)
ax2.set_title(f"정답률 95% 이상 {len(top):,}명의 공식 실력 점수", fontsize=16, pad=10, loc="right", color=ORANGE)
ax2.text(0.98, 0.04, f"{top.min():.0f} ~ {top.max():,.0f}점으로 제각각", transform=ax2.transAxes, ha="right", fontsize=14, color=TXT)
con = ConnectionPatch(xyA=(3.36, cnt.values[3] * 0.62), coordsA=ax1.transData, xyB=(-0.40, 0.5), coordsB=ax2.transAxes,
                      arrowstyle="-|>,head_length=0.9,head_width=0.5", color=ORANGE, lw=3.2, connectionstyle="arc3,rad=-0.12")
fig.add_artist(con)
fig.text(0.50, 0.63, "펼쳐 보면", ha="center", fontsize=14, color=ORANGE)
fig.savefig(OUT / "fig1_문제제기_정답률과_공식실력점수.png"); plt.close(fig)

# ---------- 그림 2 (03 분석 결과): 전체 판정 — 공식 실력 점수와 비교 (합이 8,502명인 '전체 중 비율'이라 도넛)
J = D["전체 판정"].value_counts()
order = [("실력 부족", "실력 부족", ORANGE), ("레이팅 수준", "점수만큼", MID), ("레이팅 이상", "점수 이상", BLUE)]
vals = [int(J[k]) for k, _, _ in order]; assert sum(vals) == len(D); check["그림2 합"] = sum(vals)
fig = plt.figure(figsize=(3.75, 3.1))
ax = fig.add_axes([0.0, 0.02, 0.56, 0.80])
ax.pie(vals, colors=[c for *_, c in order], startangle=90, counterclock=False, wedgeprops=dict(width=0.36, edgecolor="white", linewidth=2))
ax.text(0, 0.08, f"{sum(vals):,}명", ha="center", va="center", fontsize=14); ax.text(0, -0.2, "진단 대상", ha="center", va="center", fontsize=10.5, color=SUB)
for i, ((_, n, c), v) in enumerate(zip(order, vals)):
    y = 0.70 - i * 0.24
    fig.patches.append(matplotlib.patches.Rectangle((0.585, y + 0.035), 0.035, 0.05, color=c, transform=fig.transFigure, figure=fig))
    fig.text(0.64, y + 0.06, n, fontsize=13, va="center"); fig.text(0.64, y - 0.035, f"{v:,}명 ({v / sum(vals):.1%})", fontsize=12, va="center", color=SUB)
fig.text(0.03, 0.93, "공식 실력 점수와 비교한 판정", fontsize=15, fontweight="bold", va="center")
fig.savefig(OUT / "fig2_전체판정.png"); plt.close(fig)

# ---------- 그림 3 (03 분석 결과): 고른 약한 분야가 다음 기간에도 약했던 비율
v2 = pd.read_csv(o / "step6_①같은개수.csv", index_col=0)["C에서도 기대 이하(적중률)"] * 100
fig, ax = plt.subplots(figsize=(3.75, 3.1))
ax.bar(["우리 진단", "정답률\n낮은 순", "무작위"], v2.values, color=[NAVY, MID, "#B8BFC9"], width=0.6)
for i, v in enumerate(v2.values): ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=14)
ax.set_ylim(0, 55); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.tick_params(axis="x", labelsize=13)
ax.set_title("약한 분야가 다음 기간에도 약함", fontsize=15, loc="left", pad=26)
ax.text(0.0, 1.035, "고른 약한 분야 중 다음 기간에도 예상보다 덜 맞힌 비율", transform=ax.transAxes, fontsize=10.5, color=SUB, va="bottom")
fig.subplots_adjust(left=0.03, right=0.98, top=0.80, bottom=0.2); fig.savefig(OUT / "fig3_약한분야_적중률.png"); plt.close(fig)
check["그림3"] = [round(x, 1) for x in v2.values]

# ---------- 그림 4 (03 분석 결과): 다음 기간 정답 예측 정확도(AUC) — '이전 → 이후'가 핵심이라 점 두 개를 화살표로 이음
v3 = pd.read_csv(o / "step6_②예측.csv", index_col=0)["AUC"]; v4 = pd.read_csv(o / "step6_③레이팅.csv", index_col=0)["AUC"]
rows = [("공식 점수 대신 평소 실력 점수로", "공식 실력 점수", v4.iloc[1], "평소 실력 점수", v4.iloc[0]),
        ("공식 점수에 진단 결과 추가", "공식 점수만", v3.iloc[0], "+ 진단 결과", v3.iloc[1])]
fig, ax = plt.subplots(figsize=(3.75, 3.1))
for i, (ttl, l0, a0, l1, a1) in enumerate(rows):
    y = (1 - i) * 1.3
    ax.annotate("", xy=(a1, y), xytext=(a0, y), arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.35", color=NAVY, lw=2.4, shrinkA=6, shrinkB=6))
    ax.scatter([a0], [y], s=100, color=MID, zorder=3); ax.scatter([a1], [y], s=120, color=NAVY, zorder=3)
    ax.text(a0 + 0.0006, y + 0.14, l0, ha="right", va="bottom", fontsize=11, color=SUB)          # 이름: 시작 점은 왼쪽으로, 끝 점은 오른쪽으로 뻗게
    ax.text(a1 - 0.0006, y + 0.14, l1, ha="left", va="bottom", fontsize=11, color=NAVY)
    ax.text(a0, y - 0.16, f"{a0:.3f}", ha="center", va="top", fontsize=13, color=SUB)
    ax.text(a1, y - 0.16, f"{a1:.3f}", ha="center", va="top", fontsize=14, color=NAVY)
    ax.text(0.7015, y + 0.50, ttl, fontsize=11.5, va="bottom")
ax.set_xlim(0.701, 0.756); ax.set_ylim(-0.50, 2.10); ax.set_yticks([])
ax.set_xticks([0.71, 0.72, 0.73, 0.74, 0.75]); ax.tick_params(axis="x", labelsize=10.5)
ax.spines["left"].set_visible(False)
ax.set_title("다음 기간 정답 예측 정확도(AUC)", fontsize=15, loc="left", pad=8)
fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.11); fig.savefig(OUT / "fig4_예측정확도_AUC.png"); plt.close(fig)
check["그림4"] = [round(r[2], 3) for r in rows] + [round(r[4], 3) for r in rows]

# ---------- 그림 5 (03 분석 결과): K-means 학습 유형 — 지표 9개를 '유형이 가장 잘 나뉘는 2개 축'(선형 판별 분석, LDA)으로 요약해
#            학습자마다 점 1개, 유형마다 도형을 다르게. (PCA 2개 축은 정보의 48%만 담고 CE 비율이 거의 안 실려 문법형이 흩어져 보였음)
T = engine.transform(X[engine.FEATURES].astype(float))
Z = engine.standardize(T, X["레이팅 구간"].to_numpy(), S)[engine.FEATURES].astype(float).to_numpy()
y = X["유형번호"].to_numpy(); mu = Z.mean(0)
Sw = sum(np.cov(Z[y == k].T) * (np.sum(y == k) - 1) for k in range(1, 5))
Sb = sum(np.sum(y == k) * np.outer(Z[y == k].mean(0) - mu, Z[y == k].mean(0) - mu) for k in range(1, 5))
ev, V = np.linalg.eig(np.linalg.solve(Sw, Sb)); V = V[:, np.argsort(-ev.real)[:2]].real
P = (Z - mu) @ V
# 가장 가까운 군집 중심 = 저장된 유형 번호인지 확인 (그림이 실제 K-means 결과와 같은지)
C = np.array(S["학습유형"]["군집_중심"]); assert (((Z[:, None] - C[None]) ** 2).sum(-1).argmin(1) + 1 == y).all()
DORANGE, PURPLE = "#D9480F", "#8E44AD"
# 로직 점검형 ● 남색 · 도전적 시행착오형 ▲ 보라 · 효율 개선 필요형 ■ 청록 · 문법 실수·빠른 재제출형 ◆ 주황 (서로 잘 구별되는 4색)
sty = {2: ("o", NAVY), 4: ("^", PURPLE), 1: ("s", "#1F9E89"), 3: ("D", DORANGE)}
rng = np.random.default_rng(42); idx = rng.choice(len(X), size=min(1200, len(X)), replace=False)
fig, ax = plt.subplots(figsize=(5.6, 4.35))
for k in [2, 4, 1, 3]:                                    # 배경: 학습자 점은 작고 옅게
    m = y[idx] == k; mk, col = sty[k]
    ax.scatter(P[idx][m, 0], P[idx][m, 1], s=11, marker=mk, alpha=0.28, color=col, edgecolors="none", zorder=1)
from matplotlib.patches import Ellipse
for k in [2, 4, 1, 3]:                                    # 유형마다 학습자 대부분이 모인 범위(표준편차 1배 타원) + 중심
    mk, col = sty[k]; Q = P[y == k]; c = Q.mean(0); cov = np.cov(Q.T)
    val, vec = np.linalg.eigh(cov); ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    ax.add_patch(Ellipse(c, 2 * np.sqrt(val[1]), 2 * np.sqrt(val[0]), angle=ang, facecolor=col, alpha=0.13, edgecolor="none", zorder=2))
    ax.add_patch(Ellipse(c, 2 * np.sqrt(val[1]), 2 * np.sqrt(val[0]), angle=ang, facecolor="none", edgecolor=col, lw=2.2, zorder=3))
    ax.scatter(*c, s=260, marker=mk, color=col, edgecolors="white", linewidths=2.4, zorder=5)
ax.set_xlim(np.percentile(P[:, 0], 1), np.percentile(P[:, 0], 99)); ax.set_ylim(np.percentile(P[:, 1], 1), np.percentile(P[:, 1], 99))
ax.set_xticks([]); ax.set_yticks([])
for sp in ("left", "bottom"): ax.spines[sp].set_color("#9AA3AE")
ax.set_xlabel("유형을 가장 잘 나누는 축 1 (지표 9개 요약)", fontsize=12); ax.set_ylabel("축 2", fontsize=12)
fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.08); fig.savefig(OUT / "fig5_학습유형_Kmeans.png"); plt.close(fig)
check["그림5 유형 인원"] = {names[str(k - 1)]: int((X["유형번호"] == k).sum()) for k in range(1, 5)}
for k, v in check.items(): print(k, v)
