#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrador.py — Calibracao estatistica (MLE + Cross-Validation + Grid Search)
                dos parametros do simulador.py usando dados historicos reais.

Dados: Copa do Mundo 2018 e 2022 (fase de grupos)
Fonte: github.com/martj42/international_results (dominio publico)
Metrica de forca: Elo Rating calculado de forma rolling sobre toda a historia

Uso: python calibrador.py

Saida:
  - Parametros otimizados para colar no simulador.py
  - calibracao_resultado.png  (graficos de validacao)
"""

import os, sys, math
from collections import Counter, defaultdict

# ── instalacao automatica ────────────────────────────────────────────────────
import subprocess
for pkg, mod in [("numpy","numpy"),("pandas","pandas"),
                 ("scipy","scipy"),("matplotlib","matplotlib")]:
    try:    __import__(mod)
    except ImportError:
        print(f"Instalando {pkg}..."); subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD DOS DADOS HISTORICOS
# ════════════════════════════════════════════════════════════════════════════
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def baixar_historico():
    print("  Baixando resultados historicos (martj42/international_results)...")
    df = pd.read_csv(URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score","away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"]  = df["away_score"].astype(int)
    print(f"  {len(df):,} jogos ({df['date'].dt.year.min()}-{df['date'].dt.year.max()})")
    return df.sort_values("date").reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════
# 2. ELO HISTORICO ROLLING
# ════════════════════════════════════════════════════════════════════════════
def calcular_elo_historico(df, K=20, start=1500):
    """
    Percorre todos os jogos em ordem cronologica e calcula Elo de cada selecao.
    Guarda o Elo ANTES de cada jogo — esse sera o valor usado na calibracao.
    """
    elos   = {}       # elo atual por selecao
    hist   = {}       # (selecao, data) -> elo antes do jogo

    for _, row in df.iterrows():
        h, a, d = row["home_team"], row["away_team"], row["date"].date()
        elo_h = elos.get(h, start)
        elo_a = elos.get(a, start)

        hist[(h, d)] = elo_h
        hist[(a, d)] = elo_a

        exp_h = 1 / (1 + 10**((elo_a - elo_h) / 400))
        gh, ga = row["home_score"], row["away_score"]
        s_h = 1.0 if gh > ga else (0.0 if ga > gh else 0.5)

        elos[h] = elo_h + K * (s_h       - exp_h)
        elos[a] = elo_a + K * ((1 - s_h) - (1 - exp_h))

    print(f"  {len(hist):,} registros de Elo calculados")
    return hist

# ════════════════════════════════════════════════════════════════════════════
# 3. EXTRAIR JOGOS DA FASE DE GRUPOS DAS COPAS
# ════════════════════════════════════════════════════════════════════════════
def extrair_jogos_copa(df, hist_elo, anos=(2018, 2022)):
    """
    Filtra apenas jogos da fase de grupos das Copas indicadas.
    Cada Copa de 32 selecoes tem exatamente 48 jogos de grupos (8 grupos x 6 jogos).
    """
    jogos = []
    wc = df[df["tournament"] == "FIFA World Cup"]

    for ano in anos:
        copa = wc[wc["date"].dt.year == ano].reset_index(drop=True)
        grupos = copa.iloc[:48]   # primeiros 48 = fase de grupos (ordenados por data)

        for _, row in grupos.iterrows():
            h, a, d = row["home_team"], row["away_team"], row["date"].date()
            elo_h = hist_elo.get((h, d))
            elo_a = hist_elo.get((a, d))
            if elo_h is None or elo_a is None:
                continue
            jogos.append({
                "home": h, "away": a,
                "gols_home": int(row["home_score"]),
                "gols_away": int(row["away_score"]),
                "elo_home": elo_h, "elo_away": elo_a,
                "ano": ano,
            })

    print(f"  2018: {sum(1 for j in jogos if j['ano']==2018)} jogos")
    print(f"  2022: {sum(1 for j in jogos if j['ano']==2022)} jogos")
    return jogos

# ════════════════════════════════════════════════════════════════════════════
# 4. MODELO DE LAMBDA (espelho do simulador.py, com parametros livres)
# ════════════════════════════════════════════════════════════════════════════
def calcular_lambda_param(elo_atq, elo_def, min_elo, max_elo,
                           lambda_base, expo, fator_def_amp):
    """
    Lambda_A = lambda_base * norm_A^expo * fator_def_B

    norm em [0.5, 1.5]:
      norm = 0.5 + (elo - min_elo) / (max_elo - min_elo)

    fator_def (baseado no Elo do adversario como proxy de defesa):
      norm_def = 0.5 -> fator = 1 + 0.5*amp  (defesa fraca, facil marcar)
      norm_def = 1.0 -> fator = 1.0           (defesa media)
      norm_def = 1.5 -> fator = 1 - 0.5*amp  (defesa forte, dificil marcar)
    """
    e_range  = max(max_elo - min_elo, 1e-6)
    norm_atq = 0.5 + (elo_atq - min_elo) / e_range
    norm_def = 0.5 + (elo_def - min_elo) / e_range
    fator    = 1.0 + fator_def_amp * (1.0 - norm_def)
    lam      = lambda_base * (norm_atq ** expo) * fator
    return max(0.20, float(lam))

# ════════════════════════════════════════════════════════════════════════════
# 5. CALIBRACAO MLE
# ════════════════════════════════════════════════════════════════════════════
def neg_log_likelihood(params, jogos, min_elo, max_elo):
    """Log-likelihood negativa dos placares observados. Minimizar = melhor ajuste."""
    lb, expo, amp = params
    if lb <= 0.2 or expo <= 0.1 or amp < 0 or amp > 1.5:
        return 1e10
    total = 0.0
    for j in jogos:
        lh = calcular_lambda_param(j["elo_home"], j["elo_away"], min_elo, max_elo, lb, expo, amp)
        la = calcular_lambda_param(j["elo_away"], j["elo_home"], min_elo, max_elo, lb, expo, amp)
        p  = poisson_dist.pmf(j["gols_home"], lh) * poisson_dist.pmf(j["gols_away"], la)
        total -= math.log(max(p, 1e-15))
    return total

def calibrar_mle(jogos, min_elo, max_elo):
    """Otimiza [lambda_base, expo, fator_def_amp] via Nelder-Mead com 4 pontos de partida."""
    print("  Testando 4 pontos de partida para evitar minimos locais...")
    melhor_res, melhor_val = None, 1e10

    pontos_inicio = [
        [1.20, 2.2, 0.40],
        [1.00, 1.5, 0.25],
        [1.40, 3.0, 0.50],
        [0.90, 2.0, 0.30],
    ]
    for i, x0 in enumerate(pontos_inicio, 1):
        res = minimize(
            neg_log_likelihood, x0=x0,
            args=(jogos, min_elo, max_elo),
            method="Nelder-Mead",
            options={"maxiter": 15000, "xatol": 1e-7, "fatol": 1e-7, "adaptive": True},
        )
        print(f"    Ponto {i}: log-loss={res.fun:.4f} | params={[f'{v:.4f}' for v in res.x]}")
        if res.fun < melhor_val:
            melhor_val, melhor_res = res.fun, res

    return melhor_res

# ════════════════════════════════════════════════════════════════════════════
# 6. CALIBRACAO DO LIMIAR DE EMPATE (grid search)
# ════════════════════════════════════════════════════════════════════════════
def calibrar_limiar(jogos, params_ot, min_elo, max_elo, n_sims=300):
    """
    Encontra o LIMIAR que faz o modelo reproduzir a taxa historica de empates.
    Taxa historica WC 2018+2022 fase de grupos: ~24-27%.
    """
    lb, expo, amp = params_ot
    rng = np.random.default_rng(42)

    taxa_hist = sum(1 for j in jogos if j["gols_home"] == j["gols_away"]) / len(jogos)
    print(f"  Taxa historica de empates: {taxa_hist*100:.1f}%")

    melhor_lim, menor_diff = 15, 1.0

    for lim in range(3, 45):
        emp = 0
        for j in jogos:
            lh = calcular_lambda_param(j["elo_home"], j["elo_away"], min_elo, max_elo, lb, expo, amp)
            la = calcular_lambda_param(j["elo_away"], j["elo_home"], min_elo, max_elo, lb, expo, amp)
            sims = [(int(rng.poisson(lh)), int(rng.poisson(la))) for _ in range(n_sims)]
            va = sum(a > b for a, b in sims)
            vb = sum(b > a for a, b in sims)
            if abs(va - vb) < lim:
                emp += 1
        taxa_sim = emp / len(jogos)
        diff = abs(taxa_sim - taxa_hist)
        if diff < menor_diff:
            menor_diff, melhor_lim = diff, lim

    return melhor_lim, taxa_hist

# ════════════════════════════════════════════════════════════════════════════
# 7. CROSS-VALIDATION LEAVE-ONE-OUT
# ════════════════════════════════════════════════════════════════════════════
def cross_validation(jogos_2018, jogos_2022, min_elo, max_elo):
    """
    Treina em um ano, testa no outro.
    Metrica: acuracia do resultado (V/E/D).
    """
    print("  Rodada 1: Treino=2018, Teste=2022")
    print("  Rodada 2: Treino=2022, Teste=2018")

    cv_res = {}
    for treino, teste, label in [
        (jogos_2018, jogos_2022, "Treino_2018_Teste_2022"),
        (jogos_2022, jogos_2018, "Treino_2022_Teste_2018"),
    ]:
        res = minimize(neg_log_likelihood, x0=[1.20, 2.2, 0.40],
                       args=(treino, min_elo, max_elo), method="Nelder-Mead",
                       options={"maxiter": 8000, "adaptive": True})
        lb, expo, amp = res.x
        certos = 0
        for j in teste:
            lh = calcular_lambda_param(j["elo_home"], j["elo_away"], min_elo, max_elo, lb, expo, amp)
            la = calcular_lambda_param(j["elo_away"], j["elo_home"], min_elo, max_elo, lb, expo, amp)
            pred = "H" if lh > la + 0.15 else ("A" if la > lh + 0.15 else "E")
            real = "H" if j["gols_home"] > j["gols_away"] else ("A" if j["gols_away"] > j["gols_home"] else "E")
            if pred == real:
                certos += 1
        acc = certos / len(teste) * 100
        cv_res[label] = {"params": res.x, "accuracy": acc, "n_teste": len(teste)}
        print(f"    {label}: {acc:.1f}% ({certos}/{len(teste)} corretos)")

    return cv_res

# ════════════════════════════════════════════════════════════════════════════
# 8. GRAFICOS DE VALIDACAO
# ════════════════════════════════════════════════════════════════════════════
def gerar_graficos(jogos, params_ot, min_elo, max_elo):
    lb, expo, amp = params_ot
    rng = np.random.default_rng(42)

    gols_reais, lambdas, gols_sim = [], [], []
    for j in jogos:
        for elo_atq, elo_def, gr in [
            (j["elo_home"], j["elo_away"], j["gols_home"]),
            (j["elo_away"], j["elo_home"], j["gols_away"]),
        ]:
            lam = calcular_lambda_param(elo_atq, elo_def, min_elo, max_elo, lb, expo, amp)
            gols_reais.append(gr)
            lambdas.append(lam)
            gols_sim.append(int(rng.poisson(lam)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Calibracao MLE — Copas 2018 + 2022 (fase de grupos)", fontsize=13, fontweight="bold")

    # Grafico 1: Lambda vs Gols Reais
    axes[0].scatter(lambdas, gols_reais, alpha=0.35, s=22, color="steelblue")
    mx = max(lambdas) * 1.1
    axes[0].plot([0, mx], [0, mx], "r--", lw=1.5, label="Ideal")
    axes[0].set_xlabel("Lambda previsto (gols esperados)")
    axes[0].set_ylabel("Gols reais")
    axes[0].set_title("Lambda vs Gols Reais")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Grafico 2: Distribuicao de Gols Real vs Simulado
    max_g = max(max(gols_reais), max(gols_sim)) + 1
    x    = list(range(max_g))
    tot  = len(gols_reais)
    dr   = Counter(gols_reais)
    ds   = Counter(gols_sim)
    axes[1].bar([i - 0.2 for i in x], [dr.get(i,0)/tot for i in x],
                width=0.4, label="Real", color="steelblue", alpha=0.8)
    axes[1].bar([i + 0.2 for i in x], [ds.get(i,0)/tot for i in x],
                width=0.4, label="Simulado", color="coral", alpha=0.8)
    axes[1].set_xlabel("Gols por time por jogo")
    axes[1].set_ylabel("Frequencia relativa")
    axes[1].set_title("Distribuicao de Gols: Real vs Simulado")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Grafico 3: Elo do time vs Lambda previsto
    elos_atq = ([j["elo_home"] for j in jogos] + [j["elo_away"] for j in jogos])
    axes[2].scatter(elos_atq, lambdas, alpha=0.35, s=22, color="green")
    axes[2].set_xlabel("Elo do time atacante")
    axes[2].set_ylabel("Lambda previsto")
    axes[2].set_title("Elo vs Lambda (relacao aprendida)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "calibracao_resultado.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: calibracao_resultado.png")

# ════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  CALIBRADOR — Simulador Copa do Mundo 2026")
    print("  Metodo: MLE (Nelder-Mead) + Cross-Validation + Grid Search")
    print("=" * 62)

    # 1. Dados historicos
    print("\n[1/6] Carregando dados historicos...")
    df = baixar_historico()

    # 2. Elo historico
    print("\n[2/6] Calculando Elo historico (rolling sobre toda a historia)...")
    hist_elo = calcular_elo_historico(df)

    # 3. Extrair jogos WC
    print("\n[3/6] Extraindo jogos das Copas 2018 e 2022 (fase de grupos)...")
    jogos = extrair_jogos_copa(df, hist_elo, anos=(2018, 2022))
    if not jogos:
        print("ERRO: Nenhum jogo extraido. Verifique conexao com a internet.")
        sys.exit(1)

    jogos_2018 = [j for j in jogos if j["ano"] == 2018]
    jogos_2022 = [j for j in jogos if j["ano"] == 2022]
    all_elos   = [j["elo_home"] for j in jogos] + [j["elo_away"] for j in jogos]
    min_elo, max_elo = min(all_elos), max(all_elos)

    # 4. Calibracao MLE
    print("\n[4/6] Calibracao MLE — todos os jogos 2018 + 2022...")
    res_mle = calibrar_mle(jogos, min_elo, max_elo)
    lb_ot, expo_ot, amp_ot = res_mle.x

    # 5. Limiar
    print("\n[5/6] Calibrando LIMIAR de empate (grid search 3-44)...")
    limiar_ot, taxa_hist = calibrar_limiar(jogos, res_mle.x, min_elo, max_elo)
    print(f"  LIMIAR otimo: {limiar_ot} (taxa simulada ~{taxa_hist*100:.1f}%)")

    # 6. Cross-validation
    print("\n[6/6] Cross-Validation Leave-One-Out...")
    cv = cross_validation(jogos_2018, jogos_2022, min_elo, max_elo)

    # Graficos
    print("\n[+] Gerando graficos de validacao...")
    gerar_graficos(jogos, res_mle.x, min_elo, max_elo)

    # ── Relatorio final ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RESULTADO FINAL DA CALIBRACAO")
    print("=" * 62)
    print(f"\n  {'Parametro':<22} {'Valor atual':>12} {'Calibrado':>12}")
    print(f"  {'-'*48}")
    print(f"  {'LAMBDA_BASE':<22} {'1.2000':>12} {lb_ot:>12.4f}")
    print(f"  {'EXPO_FORCA':<22} {'2.2000':>12} {expo_ot:>12.4f}")
    print(f"  {'FATOR_DEF_AMP':<22} {'0.4000':>12} {amp_ot:>12.4f}")
    print(f"  {'LIMIAR':<22} {'15':>12} {limiar_ot:>12d}")
    print(f"\n  Log-Loss total (2018+2022): {res_mle.fun:.4f}")
    print(f"  Convergiu:                  {res_mle.success}")

    print(f"\n  Cross-Validation:")
    for label, r in cv.items():
        print(f"    {label}: {r['accuracy']:.1f}% de acuracia")

    print(f"\n  COPIE PARA O simulador.py:")
    print(f"  {'-'*48}")
    print(f"  LAMBDA_BASE = {lb_ot:.4f}")
    print(f"  EXPO_FORCA  = {expo_ot:.4f}")
    print(f"  # fator_def: 1.0 + {amp_ot:.4f} * (1.0 - norm_def)")
    print(f"  LIMIAR = {limiar_ot}  # dentro de simular_100x")
    print("=" * 62)

if __name__ == "__main__":
    main()
