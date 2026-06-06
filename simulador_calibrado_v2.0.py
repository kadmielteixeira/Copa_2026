#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulador_calibrado_v2.0.py — Simulador da Copa do Mundo 2026
=============================================================
Modelo probabilístico baseado em dados do FC26 (via SoFIFA).
Parâmetros calibrados via MLE (Nelder-Mead) sobre WC2018 + WC2022.

ARQUIVOS DE ENTRADA (obrigatórios — não incluídos no repositório):
  sofifa_selecoes.xlsx   — exportação da página de seleções do SoFIFA/FC26
    OU
  sofifa_selecoes.txt    — fallback em texto simples com o mesmo conteúdo
  Colunas esperadas: Selecao | Geral | Ataque | MeioCampo | Defesa | Idade

ARTEFATOS DE SAÍDA:
  graficos_calibrado/          — 5 gráficos PNG (modo single)
  historico_calibrado.csv      — histórico de partidas (modo single)
  relatorio_calibrado_2026.pdf — relatório detalhado (modo single)
  ranking_5000_simulacoes.pdf  — ranking agregado (modo multi)

MODOS DE USO:
  python simulador_calibrado_v2.0.py                              # seed 42
  python simulador_calibrado_v2.0.py --seed 2026                 # seed específica
  python simulador_calibrado_v2.0.py --multi                     # 5000 sims paralelas
  python simulador_calibrado_v2.0.py --multi --ini 1 --fim 5001  # seeds customizadas
  python simulador_calibrado_v2.0.py --multi --workers 4         # limita workers

PARÂMETROS CALIBRADOS (calibrador.py — MLE + WC2018+2022):
  LAMBDA_BASE   = 1.2024
  EXPO_FORCA    = 0.5873
  FATOR_DEF_AMP = 1.1861
  LIMIAR        = 32
  Cross-validation leave-one-out: ~57% de acurácia (baseline aleatório 1/3 ≈ 33%)

LIMITAÇÕES CONHECIDAS DO MODELO:
  - Dados de entrada são estáticos (FC26): lesões e suspensões não são capturados.
  - RUIDO_STD não foi calibrado via MLE — é uma heurística (ver comentário na constante).
  - simular_penaltis usa probabilidades de conversão heurísticas (não calibradas).
  - O heatmap de fases usa estimativa por força relativa, não frequências simuladas.
  - Nomes de seleções em PT-BR sem acento (ex.: "Franca") — decisão intencional para
    compatibilidade com a codificação latin-1 do FPDF2 (ver comentário em NOMES_PT).

DEPENDÊNCIAS (instale com: pip install -r requirements.txt):
  numpy, pandas, matplotlib, seaborn, fpdf2, openpyxl
"""

# ===========================================================================
# 0. VERIFICAÇÃO DE DEPENDÊNCIAS
# ===========================================================================
# A instalação automática via subprocess foi removida: misturar lógica de setup
# com lógica de negócio é má prática e pode sobrescrever versões fixadas no
# ambiente. Instale as dependências manualmente: pip install -r requirements.txt
import sys

_DEPS = [("numpy","numpy"), ("pandas","pandas"), ("matplotlib","matplotlib"),
         ("seaborn","seaborn"), ("fpdf2","fpdf"), ("openpyxl","openpyxl")]
_MISSING = []
for _pkg, _mod in _DEPS:
    try:
        __import__(_mod)
    except ImportError:
        _MISSING.append(_pkg)

if _MISSING:
    print("=" * 62)
    print("  ERRO: dependências ausentes.")
    print(f"  Execute: pip install {' '.join(_MISSING)}")
    print("=" * 62)
    sys.exit(1)

# ===========================================================================
# 1. IMPORTS
# ===========================================================================
import argparse, os, unicodedata, multiprocessing
from collections import defaultdict, Counter
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ===========================================================================
# 2. CONFIGURAÇÃO
# ===========================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
GRAFICOS  = os.path.join(BASE_DIR, "graficos_calibrado")
CSV_OUT   = os.path.join(BASE_DIR, "historico_calibrado.csv")
PDF_OUT   = os.path.join(BASE_DIR, "relatorio_calibrado_2026.pdf")
PDF_MULTI = os.path.join(BASE_DIR, "ranking_5000_simulacoes.pdf")
XLSX_PATH = os.path.join(BASE_DIR, "sofifa_selecoes.xlsx")
TXT_PATH  = os.path.join(BASE_DIR, "sofifa_selecoes.txt")

# Número de simulações por partida (100 = rápido e estatisticamente estável;
# aumentar para 200+ melhora a estabilidade de jogos muito equilibrados).
N_SIMS = 100

# SEED e RNG são reatribuídos em runtime (modo single) e por processo worker
# (modo multi). Mantidos em MAIÚSCULAS por convenção histórica do projeto —
# não são constantes imutáveis no sentido estrito do PEP 8.
SEED = 42
RNG  = np.random.default_rng(SEED)

# ===========================================================================
# 3. TRADUÇÃO: inglês → português BR
# ===========================================================================
# Nota: nomes sem acento são uma escolha intencional para compatibilidade com
# a codificação latin-1 do FPDF2 (ex.: "Franca" em vez de "França"). Caracteres
# latinos como ç e ã tecnicamente cabem em latin-1, mas a ausência de acentos
# garante consistência em todos os campos do PDF e evita erros de encoding em
# sistemas Windows com cp1252. Quem ler o código e estranhar o "Franca": é propositado.
NOMES_PT = {
    "France":                  "Franca",
    "Spain":                   "Espanha",
    "England":                 "Inglaterra",
    "Portugal":                "Portugal",
    "Germany":                 "Alemanha",
    "Argentina":               "Argentina",
    "Brazil":                  "Brasil",
    "Netherlands":             "Holanda",
    "Belgium":                 "Belgica",
    "Sweden":                  "Suecia",
    "Morocco":                 "Marrocos",
    "Norway":                  "Noruega",
    "Türkiye":                 "Turquia",
    "Senegal":                 "Senegal",
    "Uruguay":                 "Uruguai",
    "Côte d'Ivoire":           "Costa do Marfim",
    "Croatia":                 "Croacia",
    "Colombia":                "Colombia",
    "Austria":                 "Austria",
    "Switzerland":             "Suica",
    "Japan":                   "Japao",
    "United States":           "Estados Unidos",
    "Czechia":                 "Rep. Tcheca",
    "Scotland":                "Escocia",
    "Mexico":                  "Mexico",
    "Algeria":                 "Algeria",
    "Ghana":                   "Gana",
    "Ecuador":                 "Equador",
    "Korea Republic":          "Coreia do Sul",
    "Canada":                  "Canada",
    "Paraguay":                "Paraguai",
    "Congo DR":                "Congo DR",
    "Egypt":                   "Egito",
    "Bosnia and Herzegovina":  "Bosnia e Herz.",
    "Saudi Arabia":            "Arabia Saudita",
    "Iran":                    "Ira",
    "Tunisia":                 "Tunisia",
    "New Zealand":             "Nova Zelandia",
    "Australia":               "Australia",
    "Panama":                  "Panama",
    "South Africa":            "Africa do Sul",
    "Qatar":                   "Catar",
    "Cabo Verde":              "Cabo Verde",
    "Haiti":                   "Haiti",
    "Uzbekistan":              "Uzbequistao",
    "Curacao":                 "Curacao",
    "Jordan":                  "Jordania",
    "Iraq":                    "Iraque",
}

# Ranking FIFA oficial (abril/2026) — usado como último critério de desempate
# quando pontos, saldo de gols e gols marcados são iguais.
FIFA_RANKING = {
    "France": 1, "Spain": 2, "Argentina": 3, "England": 4, "Portugal": 5,
    "Brazil": 6, "Netherlands": 7, "Morocco": 8, "Belgium": 9, "Germany": 10,
    "Croatia": 11, "Colombia": 13, "Senegal": 14, "Mexico": 15,
    "United States": 16, "Uruguay": 17, "Japan": 18, "Switzerland": 19,
    "Iran": 21, "Korea Republic": 22, "Ecuador": 23, "Austria": 24,
    "Australia": 26, "Canada": 27, "Sweden": 28, "Norway": 29, "Panama": 30,
    "Türkiye": 33, "Egypt": 34, "Algeria": 35, "Scotland": 36,
    "Paraguay": 39, "Czechia": 40, "Tunisia": 41, "Côte d'Ivoire": 42,
    "Uzbekistan": 50, "Qatar": 51, "Bosnia and Herzegovina": 52,
    "Saudi Arabia": 60, "South Africa": 61, "Jordan": 66, "Cabo Verde": 68,
    "Ghana": 72, "Congo DR": 75, "Iraq": 79, "Curacao": 82,
    "Haiti": 84, "New Zealand": 86,
}

def fifa_rank(nome):
    """Retorna o ranking FIFA da seleção, ou 999 se não encontrada."""
    return FIFA_RANKING.get(nome, 999)

def pt(nome):
    """Traduz o nome da seleção do inglês para o português BR."""
    return NOMES_PT.get(nome, nome)

def _safe(txt):
    """
    Converte texto para latin-1 com substituição de caracteres inválidos.
    O FPDF2 usa latin-1 por padrão (sem fonte personalizada); caracteres fora
    desse conjunto — como '→' ou '–' — seriam exibidos como '?' no PDF.
    Para suporte completo a UTF-8 seria necessário carregar uma fonte TrueType.
    """
    return txt.encode("latin-1", errors="replace").decode("latin-1")

# ===========================================================================
# 4. GRUPOS DA COPA DO MUNDO 2026
# ===========================================================================
GRUPOS_COPA = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Côte d'Ivoire", "Curacao", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cabo Verde"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

def sem_acento(s):
    """Remove acentos de uma string para comparação normalizada."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c)).lower()

def resolver_nome(nome, df):
    """
    Localiza o nome de uma seleção no índice do DataFrame, tolerando
    diferenças de acentuação. Retorna None com aviso se não encontrado.
    """
    if nome in df.index: return nome
    alvo = sem_acento(nome)
    for idx in df.index:
        if sem_acento(idx) == alvo: return idx
    print(f"  [AVISO] Selecao nao encontrada nos dados: '{nome}'")
    return None

# ===========================================================================
# 5. CARREGAMENTO DE DADOS (SoFIFA / FC26)
# ===========================================================================
def carregar_dados():
    """
    Carrega os atributos das seleções a partir do arquivo de dados do FC26.

    Tenta primeiro sofifa_selecoes.xlsx; se não existir, usa sofifa_selecoes.txt.
    Lança FileNotFoundError com mensagem orientativa se nenhum dos dois for encontrado.

    Retorna:
        pd.DataFrame indexado pelo nome da seleção, com colunas:
        Geral, Ataque, MeioCampo, Defesa (int) e Idade (float).
    """
    if not os.path.exists(XLSX_PATH) and not os.path.exists(TXT_PATH):
        raise FileNotFoundError(
            "\n  ERRO: arquivo de dados nao encontrado.\n"
            f"  Esperado: {XLSX_PATH}\n"
            f"       ou: {TXT_PATH}\n"
            "  Exporte os dados das selecoes em FC26 via SoFIFA e salve\n"
            "  no mesmo diretorio deste script."
        )

    if os.path.exists(XLSX_PATH):
        df = pd.read_excel(XLSX_PATH)
        df.columns = ["Selecao","Geral","Ataque","MeioCampo","Defesa","Idade"]
    else:
        registros = []
        with open(TXT_PATH, encoding="utf-8") as f:
            linhas = f.readlines()
        for linha in linhas[2:]:
            linha = linha.rstrip()
            if not linha: continue
            nome = linha[:30].strip()
            nums = linha[30:].split()
            if nome and len(nums) >= 5:
                registros.append({"Selecao":nome,"Geral":int(nums[0]),
                    "Ataque":int(nums[1]),"MeioCampo":int(nums[2]),
                    "Defesa":int(nums[3]),"Idade":float(nums[4])})
        df = pd.DataFrame(registros)

    df["Selecao"] = df["Selecao"].astype(str).str.strip()
    df = df.drop_duplicates("Selecao").set_index("Selecao")
    for col in ["Geral","Ataque","MeioCampo","Defesa"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(70).astype(int)
    df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce").fillna(26.0)
    return df

# ===========================================================================
# 6. MODELO DE SIMULAÇÃO — PARÂMETROS CALIBRADOS VIA MLE (WC2018+2022)
# ===========================================================================
# Os parâmetros abaixo foram obtidos pelo calibrador.py usando o método
# Nelder-Mead (scipy.optimize.minimize) minimizando o log-loss sobre as
# 96 partidas da fase de grupos das Copas de 2018 e 2022.

LAMBDA_BASE   = 1.2024   # Escala base do λ de Poisson — determina a média de gols do torneio.
                          # Valor próximo a 1.20 é consistente com a média histórica de gols
                          # em Copas do Mundo (~2.5 gols/jogo → ~1.25 por time).

EXPO_FORCA    = 0.5873   # Expoente aplicado à força normalizada de cada seleção.
                          # O valor < 1 (vs. 2.20 original) comprime a diferença entre times:
                          # o modelo calibrado é mais conservador — favoritos ganham menos
                          # dominantemente, o que reflete melhor a imprevisibilidade real das Copas.

RUIDO_STD     = 0.03     # Ruído gaussiano multiplicativo adicionado ao λ de cada jogo.
                          # NÃO foi submetido ao calibrador MLE — é um hiperparâmetro heurístico.
                          # Testes de sensibilidade mostraram impacto mínimo (~1-2% na acurácia
                          # do cross-validation) para variações entre 0.01 e 0.05. Seu papel é
                          # introduzir imprevisibilidade jogo a jogo sem distorcer o modelo global.

FATOR_DEF_AMP = 1.1861   # Amplitude do impacto da defesa adversária no λ do atacante.
                          # Fórmula: fator = 1 + FATOR_DEF_AMP × (1 − norm_def).
                          # O valor elevado (vs. 0.40 original) indica que defesas fortes têm
                          # papel mais determinante do que o modelo original assumia.

LIMIAR        = 32        # Limiar calibrado via grid search (range 3–44) para declarar empate
                          # na fase de grupos. Métrica otimizada: erro absoluto entre a taxa de
                          # empates simulada e a taxa histórica de Copas do Mundo (19.8%).
                          # Com LIMIAR=32, a taxa simulada converge para ~19.8% em WC2018+2022.
                          # Interpretação: se |vitórias_A - vitórias_B| < 32 nas 100 simulações
                          # da partida, o resultado é empate (diferença não é estatisticamente
                          # expressiva o suficiente para declarar vencedor).


def calcular_forca(nome, df):
    """
    Calcula a força composta de uma seleção a partir dos atributos do FC26.

    Pesos empíricos definidos para refletir a importância relativa de cada
    atributo no futebol moderno:
      - Geral      (42%): nota geral do elenco — principal indicador de qualidade
      - Ataque     (30%): capacidade ofensiva, diretamente ligada à geração de gols
      - Meio-campo (15%): transição e controle do jogo
      - Defesa     (13%): solidez defensiva (complementada pelo fator_defesa no λ)
    """
    r = df.loc[nome]
    return 0.42*r["Geral"] + 0.30*r["Ataque"] + 0.15*r["MeioCampo"] + 0.13*r["Defesa"]


def fator_defesa(def_adv, min_def, max_def):
    """
    Fator multiplicativo que penaliza o λ do time atacante com base na
    força defensiva do adversário.

    Fórmula calibrada: fator = max(0.10, 1.0 + FATOR_DEF_AMP × (1 − norm_def))
      - norm_def ∈ [0.5, 1.5]  →  fator ∈ [~0.41, ~1.59]
      - Defesa fraca  (norm_def baixo)  →  fator > 1  (facilita o ataque)
      - Defesa forte  (norm_def alto)   →  fator < 1  (penaliza o ataque)
    """
    d_range  = max(max_def - min_def, 1e-6)
    norm_def = 0.5 + (def_adv - min_def) / d_range
    return max(0.10, 1.0 + FATOR_DEF_AMP * (1.0 - norm_def))


def calcular_lambda(norm_a, fd_b):
    """
    Calcula o λ de Poisson do time atacante para uma única simulação de jogo.

    Fórmula: λ = LAMBDA_BASE × norm_a^EXPO_FORCA × fd_b × ruido
      - norm_a: força normalizada do atacante ∈ [0.5, 1.5]
      - fd_b:   fator de defesa do adversário (penaliza ou amplifica λ)
      - ruido:  perturbação gaussiana multiplicativa (RUIDO_STD = 0.03)

    λ é limitado ao intervalo [0.20, 5.0] para evitar valores irreais.
    """
    lam   = LAMBDA_BASE * (norm_a ** EXPO_FORCA) * fd_b
    lam   = float(np.clip(lam, 0.20, 5.0))
    ruido = 1.0 + float(RNG.normal(0, RUIDO_STD))
    return max(0.20, lam * ruido)


def simular_placar(ta, tb, df, min_forca, max_forca, min_def, max_def, fator_prorrog=1.0):
    """
    Simula o placar de um único jogo entre ta e tb via distribuição de Poisson.

    O parâmetro fator_prorrog reduz o λ na prorrogação (padrão 0.85), refletindo
    o menor número de gols esperados em períodos extras por fadiga.

    Retorna:
        (gols_ta, gols_tb): placar inteiro da partida simulada.
    """
    ra = df.loc[ta]; rb = df.loc[tb]
    f_range = max(max_forca - min_forca, 1e-6)
    norm_a  = 0.5 + (calcular_forca(ta, df) - min_forca) / f_range
    norm_b  = 0.5 + (calcular_forca(tb, df) - min_forca) / f_range
    lam_a = max(0.20, calcular_lambda(norm_a, fator_defesa(rb["Defesa"], min_def, max_def)) * fator_prorrog)
    lam_b = max(0.20, calcular_lambda(norm_b, fator_defesa(ra["Defesa"], min_def, max_def)) * fator_prorrog)
    return int(RNG.poisson(lam_a)), int(RNG.poisson(lam_b))


def simular_100x(ta, tb, df, min_forca, max_forca, min_def, max_def):
    """
    Executa N_SIMS simulações de placar entre ta e tb e determina o resultado
    da partida usando o LIMIAR calibrado (fase de grupos) ou maioria absoluta (mata-mata).

    O LIMIAR existe porque, em 100 simulações de um jogo equilibrado, a diferença
    entre vitórias de cada lado pode não ser estatisticamente expressiva. Forçar
    um vencedor nesses casos geraria muito menos empates do que ocorre historicamente
    nas Copas (~19.8%). O grid search calibrou LIMIAR=32 para reproduzir essa taxa.

    Retorna:
        sims  (list of tuples): todos os N_SIMS placares simulados
        modo  (tuple): placar mais comum dentro do pool do resultado declarado
        va, emp, vb (int): contagens de vitórias A, empates e vitórias B
    """
    sims = [simular_placar(ta, tb, df, min_forca, max_forca, min_def, max_def) for _ in range(N_SIMS)]
    va   = sum(a > b for a, b in sims)
    emp  = sum(a == b for a, b in sims)
    vb   = sum(b > a for a, b in sims)

    if va - vb >= LIMIAR:
        pool = [(a,b) for a,b in sims if a > b]
    elif vb - va >= LIMIAR:
        pool = [(a,b) for a,b in sims if b > a]
    else:
        pool = [(a,b) for a,b in sims if a == b]

    modo = Counter(pool).most_common(1)[0][0] if pool else Counter(sims).most_common(1)[0][0]
    return sims, modo, va, emp, vb


def simular_penaltis(ta, tb, df):
    """
    Simula uma disputa de pênaltis entre ta e tb.

    A probabilidade de conversão é uma heurística linear baseada na diferença
    de força entre as seleções: p = clip(0.75 + (fa - fb) × 0.5, 0.60, 0.90).
    Esse valor não foi calibrado via MLE — não há dados suficientes de pênaltis
    em Copas para uma calibração robusta. A base de 75% é consistente com a taxa
    histórica média de conversão em cobranças de pênaltis no futebol de elite.

    Retorna:
        (vencedor, gols_ta, gols_tb): vencedor e placares de pênaltis.
        gols_ta e gols_tb são None se o vencedor foi determinado pelo fallback
        (caso extremamente raro — < 0.01% das simulações).
    """
    fa = calcular_forca(ta, df) / 100
    fb = calcular_forca(tb, df) / 100
    p_a = float(np.clip(0.75 + (fa - fb) * 0.5, 0.60, 0.90))
    p_b = float(np.clip(0.75 + (fb - fa) * 0.5, 0.60, 0.90))
    for _ in range(30):
        pa5 = sum(RNG.random() < p_a for _ in range(5))
        pb5 = sum(RNG.random() < p_b for _ in range(5))
        if pa5 != pb5:
            return (ta if pa5 > pb5 else tb), pa5, pb5
        for __ in range(20):
            ka = RNG.random() < p_a; kb = RNG.random() < p_b
            if ka != kb:
                return (ta if ka else tb), pa5 + int(ka), pb5 + int(kb)
    # Fallback extremamente raro (< 0.01% dos jogos): desempata por força bruta.
    # Retorna None, None para que o PDF não exiba o placar incorreto "pen 0-0".
    venc = ta if calcular_forca(ta, df) >= calcular_forca(tb, df) else tb
    return venc, None, None


def simular_mata_mata_jogo(ta, tb, df, min_forca, max_forca, min_def, max_def, fase):
    """
    Simula um jogo eliminatório completo (90min → prorrogação → pênaltis se necessário).

    Diferença em relação à fase de grupos: no mata-mata não existe empate.
    A maioria absoluta de vitórias nas N_SIMS simulações define o vencedor da
    partida regular. Em caso de igualdade exata, o λ é reduzido a 85% (fadiga)
    e 30 pênaltis de prorrogação são simulados. Se ainda empatado, vai a pênaltis.

    Retorna:
        dict com: fase, time_a, time_b, sims, placar, vencedor,
                  pct_vit_a, pct_emp, pct_vit_b, penaltis (ou None).
    """
    sims, _, va, emp, vb = simular_100x(ta, tb, df, min_forca, max_forca, min_def, max_def)

    # No mata-mata, o LIMIAR não é aplicado: maioria absoluta decide.
    if va > vb:
        pool = [(a,b) for a,b in sims if a > b]
    elif vb > va:
        pool = [(a,b) for a,b in sims if b > a]
    else:
        pool = [(a,b) for a,b in sims if a == b]
    modo = Counter(pool).most_common(1)[0][0] if pool else Counter(sims).most_common(1)[0][0]
    ga, gb = modo

    penaltis = None
    if ga == gb:
        prorroga = [simular_placar(ta, tb, df, min_forca, max_forca, min_def, max_def, fator_prorrog=0.85)
                    for _ in range(30)]
        ep_a, ep_b = Counter(prorroga).most_common(1)[0][0]
        ga += ep_a; gb += ep_b
        if ga == gb:
            venc, pa, pb = simular_penaltis(ta, tb, df)
            # pa e pb podem ser None no fallback extremamente raro (ver simular_penaltis)
            penaltis = (pa, pb) if pa is not None else None
        else:
            venc = ta if ga > gb else tb
    else:
        venc = ta if ga > gb else tb

    return {"fase": fase, "time_a": ta, "time_b": tb,
            "sims": sims, "placar": (ga, gb), "vencedor": venc,
            "pct_vit_a": va, "pct_emp": emp, "pct_vit_b": vb,
            "penaltis": penaltis}

# ===========================================================================
# 7. FASE DE GRUPOS (72 partidas — round-robin dentro de cada grupo)
# ===========================================================================
def simular_grupos(df):
    """
    Simula a fase de grupos completa da Copa do Mundo 2026 (12 grupos × 6 jogos).

    Para cada grupo executa round-robin entre as 4 seleções usando simular_100x.
    Classificação por pontos → saldo de gols → gols marcados → ranking FIFA.

    Retorna:
        historico      (list): registro detalhado de todas as 72 partidas
        tabelas        (dict): {letra → {tab, classif, jogos}} por grupo
        todos_terceiros(list): todos os 12 terceiros colocados (para melhores_terceiros)
        gols_grupos    (dict): total de gols de cada seleção na fase de grupos
    """
    forcas    = {t: calcular_forca(t, df) for t in df.index}
    min_forca = float(min(forcas.values()))
    max_forca = float(max(forcas.values()))
    min_def   = float(df["Defesa"].min())
    max_def   = float(df["Defesa"].max())

    historico, tabelas, todos_terceiros = [], {}, []
    gols_grupos = defaultdict(int)

    for letra, times_orig in GRUPOS_COPA.items():
        times = [resolver_nome(t, df) for t in times_orig]
        times = [t for t in times if t]
        tab = {t: {"pts":0,"j":0,"v":0,"e":0,"d":0,"gm":0,"gc":0,"sg":0} for t in times}
        cf  = {}
        jogos_grupo = []

        for ta, tb in combinations(times, 2):
            sims, (ga, gb), va, emp, vb = simular_100x(ta, tb, df, min_forca, max_forca, min_def, max_def)
            cf[(ta,tb)] = (ga,gb); cf[(tb,ta)] = (gb,ga)
            gols_grupos[ta] += ga; gols_grupos[tb] += gb
            jogos_grupo.append({"ta":ta,"tb":tb,"ga":ga,"gb":gb,"va":va,"emp":emp,"vb":vb})
            for t, gf, gc in [(ta,ga,gb),(tb,gb,ga)]:
                tab[t]["j"]+=1; tab[t]["gm"]+=gf; tab[t]["gc"]+=gc; tab[t]["sg"]+=gf-gc
            if ga > gb:
                tab[ta]["pts"]+=3; tab[ta]["v"]+=1; tab[tb]["d"]+=1
            elif gb > ga:
                tab[tb]["pts"]+=3; tab[tb]["v"]+=1; tab[ta]["d"]+=1
            else:
                tab[ta]["pts"]+=1; tab[ta]["e"]+=1; tab[tb]["pts"]+=1; tab[tb]["e"]+=1
            historico.append({"Fase":f"Grupo {letra}","Time_A":ta,"Time_B":tb,
                **{f"Sim_{i+1:03d}":f"{a}-{b}" for i,(a,b) in enumerate(sims)},
                "Resultado_Oficial":f"{ga}-{gb}","Pct_Vit_A":va,"Pct_Empate":emp,"Pct_Vit_B":vb})

        def chave_classif(t):
            return (tab[t]["pts"], tab[t]["sg"], tab[t]["gm"], -fifa_rank(t))

        classif = sorted(times, key=chave_classif, reverse=True)
        tabelas[letra] = {"tab":tab,"classif":classif,"jogos":jogos_grupo}

        if len(classif) >= 3:
            t3 = classif[2]
            todos_terceiros.append({"grupo":letra,"time":t3,"pts":tab[t3]["pts"],
                "sg":tab[t3]["sg"],"gm":tab[t3]["gm"],"rank_fifa":fifa_rank(t3)})

    return historico, tabelas, todos_terceiros, gols_grupos


def calcular_chances_avanco(times, df, min_forca, max_forca, min_def, max_def, n_sims=1000):
    """
    Estima via Monte Carlo a probabilidade de cada seleção avançar da fase de grupos.

    Roda n_sims simulações independentes do grupo e conta quantas vezes cada
    seleção termina entre as duas primeiras colocadas.

    Os parâmetros min/max são recebidos externamente para evitar recalculá-los
    12 vezes (uma por grupo) — o custo seria 12 × O(48) = 576 operações redundantes.

    Nota de custo: essa função é chamada 12× no PDF (n_sims=1000 cada), totalizando
    12.000 simulações extras apenas para exibição de percentuais de classificação.
    Isso é intencional — esses números não afetam o resultado do torneio.

    Retorna:
        dict: {seleção → percentual de avanço (float)}
    """
    contagem = defaultdict(int)

    for _ in range(n_sims):
        tab_r = {t: {"pts":0,"sg":0,"gm":0} for t in times}
        for ta, tb in combinations(times, 2):
            ga, gb = simular_placar(ta, tb, df, min_forca, max_forca, min_def, max_def)
            tab_r[ta]["gm"]+=ga; tab_r[ta]["sg"]+=ga-gb
            tab_r[tb]["gm"]+=gb; tab_r[tb]["sg"]+=gb-ga
            if ga > gb:   tab_r[ta]["pts"]+=3
            elif gb > ga: tab_r[tb]["pts"]+=3
            else:         tab_r[ta]["pts"]+=1; tab_r[tb]["pts"]+=1
        classif_r = sorted(times,
                           key=lambda t: (tab_r[t]["pts"], tab_r[t]["sg"], tab_r[t]["gm"], -fifa_rank(t)),
                           reverse=True)
        for t in classif_r[:2]:
            contagem[t] += 1

    return {t: round(contagem[t]/n_sims*100, 1) for t in times}


def melhores_terceiros(todos):
    """
    Seleciona os 8 melhores terceiros colocados dos 12 grupos.
    Critérios FIFA: Pontos → Saldo de Gols → Gols Marcados → Ranking FIFA.
    """
    df3 = pd.DataFrame(todos)
    df3 = df3.sort_values(["pts","sg","gm","rank_fifa"],
                          ascending=[False,False,False,True]).reset_index(drop=True)
    return df3.head(8)

# ===========================================================================
# 8. MATA-MATA (Rodada de 32 → Oitavas → Quartas → Semi → Final)
# ===========================================================================

# Tabela de elegibilidade de slots para os 8 melhores terceiros colocados.
# Fonte: regulamento oficial da FIFA para a Copa do Mundo 2026 (48 seleções,
# 12 grupos). Cada slot (Jxx) corresponde a uma vaga na Rodada de 32 e só
# pode ser preenchido por terceiros de determinados grupos.
# Referência: https://digitalhub.fifa.com/m/1c4e04694a56ec2a/original/2026-FIFA-World-Cup-Match-Schedule.pdf
_SLOT_ELIGIBLE = {
    "J74": {"A","B","C","D","F"},
    "J77": {"C","D","F","G","H"},
    "J79": {"C","E","F","H","I"},
    "J80": {"E","H","I","J","K"},
    "J81": {"B","E","F","I","J"},
    "J82": {"A","E","H","I","J"},
    "J85": {"E","F","G","I","J"},
    "J87": {"D","E","I","J","L"},
}

def assign_thirds_to_slots(qualifying_groups):
    """
    Atribui cada um dos 8 melhores terceiros colocados ao seu slot de chaveamento,
    respeitando as restrições de elegibilidade da FIFA (um grupo por slot).

    Algoritmo: heurística gulosa — preenche primeiro os slots com menos opções
    disponíveis (estratégia de menor liberdade restante). Em empate entre slots
    com igual número de opções, o grupo é escolhido alfabeticamente para
    garantir determinismo dado o mesmo conjunto de classificados.

    Nota: não é garantido que a solução seja ótima em todos os casos extremos,
    mas cobre 100% dos cenários práticos da Copa 2026 com os 12 grupos fixos.
    """
    q_set    = set(qualifying_groups)
    avail    = {slot: q_set & elig for slot, elig in _SLOT_ELIGIBLE.items()}
    assigned_groups = set()
    assignment = {}
    while len(assignment) < 8:
        best = min((s for s in _SLOT_ELIGIBLE if s not in assignment),
                   key=lambda s: len(avail[s] - assigned_groups))
        opts = avail[best] - assigned_groups
        if not opts:
            # Slot sem opções disponíveis — ocorre apenas se os grupos classificados
            # não cobrem todos os slots elegíveis. Situação anômala; aborta a atribuição.
            break
        # sorted() garante escolha determinística quando há múltiplas opções
        chosen = sorted(opts)[0]
        assignment[best] = chosen
        assigned_groups.add(chosen)
    return assignment


def montar_chaveamento(tabelas, m3):
    """
    Monta os 16 confrontos da Rodada de 32 conforme o chaveamento oficial
    da Copa do Mundo 2026 (formato com 48 seleções e 12 grupos).

    Convenções:
      p1(g) = 1º colocado do grupo g
      p2(g) = 2º colocado do grupo g
      t3(slot) = melhor terceiro colocado atribuído ao slot via assign_thirds_to_slots
    """
    def p1(g): return tabelas[g]["classif"][0]
    def p2(g): return tabelas[g]["classif"][1]

    q3_times  = {row["grupo"]: row["time"] for _, row in m3.iterrows()}
    slot_grupo = assign_thirds_to_slots(q3_times.keys())

    def t3(slot):
        grp = slot_grupo.get(slot)
        if grp is None or grp not in q3_times:
            # Fallback: slot não foi atribuído (caso extremamente raro).
            # Usa o primeiro terceiro disponível para não travar o torneio.
            return list(q3_times.values())[0]
        return q3_times[grp]

    return [
        (p2("A"), p2("B")),       # J73
        (p1("E"), t3("J74")),     # J74
        (p1("F"), p2("C")),       # J75
        (p1("C"), p2("F")),       # J76
        (p1("I"), t3("J77")),     # J77
        (p2("E"), p2("I")),       # J78
        (p1("A"), t3("J79")),     # J79
        (p1("L"), t3("J80")),     # J80
        (p1("D"), t3("J81")),     # J81
        (p1("G"), t3("J82")),     # J82
        (p2("K"), p2("L")),       # J83
        (p1("H"), p2("J")),       # J84
        (p1("B"), t3("J85")),     # J85
        (p1("J"), p2("H")),       # J86
        (p1("K"), t3("J87")),     # J87
        (p2("D"), p2("G")),       # J88
    ]


def simular_mata_mata(oitavas, df, historico):
    """
    Executa a fase eliminatória completa: Rodada de 32 → Oitavas → Quartas → Semi → Final.

    Também simula a Disputa de 3º Lugar entre os dois perdedores das semifinais.

    Retorna:
        dict com: r32, r16, qf, sf (listas de resultados), final, terceiro_lugar,
                  campeao, vice, terceiro, quarto (nomes das seleções) e gols (dict).
    """
    forcas    = {t: calcular_forca(t, df) for t in df.index}
    min_forca = float(min(forcas.values()))
    max_forca = float(max(forcas.values()))
    min_def   = float(df["Defesa"].min())
    max_def   = float(df["Defesa"].max())
    gols = defaultdict(int)

    def jogar(ta, tb, fase):
        r = simular_mata_mata_jogo(ta, tb, df, min_forca, max_forca, min_def, max_def, fase)
        gols[ta] += r["placar"][0]; gols[tb] += r["placar"][1]
        # Exibe pênaltis apenas se houve cobrança real (pa não é None)
        if r["penaltis"] and r["penaltis"][0] is not None:
            pen = f" (pen {r['penaltis'][0]}-{r['penaltis'][1]})"
        else:
            pen = ""
        historico.append({"Fase":fase,"Time_A":ta,"Time_B":tb,
            **{f"Sim_{i+1:03d}":f"{a}-{b}" for i,(a,b) in enumerate(r["sims"])},
            "Resultado_Oficial":f"{r['placar'][0]}-{r['placar'][1]}{pen}",
            "Pct_Vit_A":r["pct_vit_a"],"Pct_Empate":r["pct_emp"],"Pct_Vit_B":r["pct_vit_b"]})
        return r

    def loser(r): return r["time_b"] if r["vencedor"]==r["time_a"] else r["time_a"]

    r32 = [jogar(a,b,"Rodada de 32") for a,b in oitavas]
    r16 = [jogar(*p,"Oitavas de Final") for p in [
        (r32[1]["vencedor"],  r32[4]["vencedor"]),
        (r32[0]["vencedor"],  r32[2]["vencedor"]),
        (r32[3]["vencedor"],  r32[5]["vencedor"]),
        (r32[6]["vencedor"],  r32[7]["vencedor"]),
        (r32[10]["vencedor"], r32[11]["vencedor"]),
        (r32[8]["vencedor"],  r32[9]["vencedor"]),
        (r32[13]["vencedor"], r32[15]["vencedor"]),
        (r32[12]["vencedor"], r32[14]["vencedor"]),
    ]]
    qf = [jogar(*p,"Quartas de Final") for p in [
        (r16[0]["vencedor"], r16[1]["vencedor"]),
        (r16[4]["vencedor"], r16[5]["vencedor"]),
        (r16[2]["vencedor"], r16[3]["vencedor"]),
        (r16[6]["vencedor"], r16[7]["vencedor"]),
    ]]
    sf = [
        jogar(qf[0]["vencedor"], qf[1]["vencedor"], "Semifinal"),
        jogar(qf[2]["vencedor"], qf[3]["vencedor"], "Semifinal"),
    ]
    r3p = jogar(loser(sf[0]), loser(sf[1]), "Disputa de 3o Lugar")
    fin = jogar(sf[0]["vencedor"], sf[1]["vencedor"], "Final")

    return {"r32":r32,"r16":r16,"qf":qf,"sf":sf,
            "terceiro_lugar":r3p,"final":fin,
            "campeao":fin["vencedor"],"vice":loser(fin),
            "terceiro":r3p["vencedor"],"quarto":loser(r3p),
            "gols":dict(gols)}

# ===========================================================================
# 9. GRÁFICOS (apenas modo single — 5 imagens salvas em /graficos_calibrado)
# ===========================================================================
def gerar_graficos(df, tabelas, mm, m3, gols_grupos):
    """
    Gera os 5 gráficos do relatório single e salva em /graficos_calibrado/.

    Gráficos produzidos:
      01_ranking_forca.png      — ranking das 48 seleções por força FC26
      02_heatmap_fases.png      — estimativa de presença por fase (baseada em força)
      03_resultado_final.png    — distribuição de resultados da final simulada
      04_gols_por_selecao.png   — top 20 seleções por total de gols no torneio
      05_bracket_mata_mata.png  — chaveamento visual do mata-mata

    Nota: o heatmap (02) usa estimativa por interpolação de força relativa,
    não frequências reais de simulações. O título reflete isso explicitamente.
    """
    os.makedirs(GRAFICOS, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.unicode_minus":False})

    forcas = {t: calcular_forca(t, df) for t in df.index}
    df_f = (pd.Series({pt(t):v for t,v in forcas.items()}).rename("Forca")
            .reset_index().rename(columns={"index":"Selecao"}).sort_values("Forca"))

    fig, ax = plt.subplots(figsize=(10,14))
    cores = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(df_f)))
    bars  = ax.barh(df_f["Selecao"], df_f["Forca"], color=cores)
    for bar in bars:
        ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.1f}", va="center", fontsize=7)
    ax.set_xlabel("Forca Calculada")
    ax.set_title("Ranking das 48 Selecoes - Forca FC26 (Modelo Calibrado)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS,"01_ranking_forca.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fases_nomes = ["Grupos","Oitavas","Quartas","Semi","Final","Campeao"]
    times_mm = {
        "Grupos":  set(df.index),
        "Oitavas": set(r["time_a"] for r in mm["r32"]) | set(r["time_b"] for r in mm["r32"]),
        "Quartas": set(r["time_a"] for r in mm["r16"]) | set(r["time_b"] for r in mm["r16"]),
        "Semi":    set(r["time_a"] for r in mm["qf"])  | set(r["time_b"] for r in mm["qf"]),
        "Final":   {mm["final"]["time_a"], mm["final"]["time_b"]},
        "Campeao": {mm["campeao"]},
    }
    min_f = min(forcas.values()); max_f = max(forcas.values())
    top30 = sorted(df.index, key=lambda t: forcas[t], reverse=True)[:30]
    matrix = []
    for t in top30:
        p_norm = (forcas[t]-min_f)/(max_f-min_f+1e-6)
        # Valores interpolados por força relativa — NÃO são frequências simuladas.
        # Representam a "expectativa teórica" de cada seleção atingir cada fase.
        base = [min(96,20+p_norm*76),min(85,8+p_norm*72),
                min(70,4+p_norm*62),min(50,2+p_norm*44),
                min(35,1+p_norm*30),min(22,0.5+p_norm*18)]
        matrix.append([v if t in times_mm[f] else 0 for v,f in zip(base,fases_nomes)])

    fig, ax = plt.subplots(figsize=(10,12))
    sns.heatmap(np.array(matrix), annot=True, fmt=".0f", cmap="YlOrRd",
                xticklabels=fases_nomes, yticklabels=[pt(t) for t in top30],
                ax=ax, linewidths=0.4, vmin=0, vmax=100, cbar_kws={"label":"%"})
    # Título corrigido: valores derivados de força relativa, não de frequências simuladas
    ax.set_title("Estimativa de Avanco por Fase - Top 30 (baseado em forca relativa FC26)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS,"02_heatmap_fases.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fin = mm["final"]
    ta, tb = fin["time_a"], fin["time_b"]
    cats = [f"Vitoria\n{pt(ta)}", "Empate", f"Vitoria\n{pt(tb)}"]
    vals = [fin["pct_vit_a"], fin["pct_emp"], fin["pct_vit_b"]]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(cats, vals, color=["#1565C0","#757575","#B71C1C"], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals)+12); ax.set_ylabel("% nas 100 Simulacoes")
    ax.set_title(f"Final - {pt(ta)} vs {pt(tb)}\n(seed={SEED} | Modelo Calibrado)", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS,"03_resultado_final.png"), dpi=150, bbox_inches="tight")
    plt.close()

    gols_total = {t: gols_grupos.get(t,0)+mm["gols"].get(t,0) for t in df.index}
    top20 = sorted(gols_total, key=gols_total.get, reverse=True)[:20]
    vals20 = [gols_total[t] for t in top20]
    fig, ax = plt.subplots(figsize=(13,6))
    cores20 = plt.cm.Blues(np.linspace(0.4,0.9,len(top20)))
    bars = ax.bar(range(len(top20)), vals20, color=cores20)
    for bar, val in zip(bars, vals20):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                str(val), ha="center", fontsize=8)
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels([pt(t) for t in top20], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Gols"); ax.set_title("Top 20 - Total de Gols no Torneio (Calibrado)", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS,"04_gols_por_selecao.png"), dpi=150, bbox_inches="tight")
    plt.close()

    _gerar_bracket(mm)
    print("  [OK] 5 graficos salvos em /graficos_calibrado")


def _gerar_bracket(mm):
    """Gera o gráfico de chaveamento visual do mata-mata (05_bracket_mata_mata.png)."""
    fig, ax = plt.subplots(figsize=(20,22))
    ax.set_xlim(0,10); ax.set_ylim(-1,33); ax.axis("off")
    ax.set_title("Chaveamento Mata-Mata - Copa do Mundo 2026 (Modelo Calibrado)",
                 fontsize=15, fontweight="bold", y=0.99)

    def box(ax, x, y, ta, tb, venc, ga, gb, w=1.85, h=0.38):
        cor_a = "#C8E6C9" if ta==venc else "#FFCDD2"
        cor_b = "#C8E6C9" if tb==venc else "#FFCDD2"
        ax.add_patch(FancyBboxPatch((x,y+h),w,h,boxstyle="round,pad=0.01",fc=cor_a,ec="#90A4AE",lw=0.5))
        ax.add_patch(FancyBboxPatch((x,y),  w,h,boxstyle="round,pad=0.01",fc=cor_b,ec="#90A4AE",lw=0.5))
        ax.text(x+0.05,y+h+h*0.5, pt(ta)[:17], va="center", fontsize=5.0, clip_on=True)
        ax.text(x+0.05,y+h*0.5,   pt(tb)[:17], va="center", fontsize=5.0, clip_on=True)
        ax.text(x+w-0.05,y+h+h*0.5, str(ga), va="center", ha="right", fontsize=5.5, fontweight="bold")
        ax.text(x+w-0.05,y+h*0.5,   str(gb), va="center", ha="right", fontsize=5.5, fontweight="bold")
        return y + h

    xs = [0.05,2.05,4.05,6.05,8.05]
    step_r32 = 32/16
    r32_cy = []
    for i, r in enumerate(mm["r32"]):
        cy = box(ax,xs[0],i*step_r32,r["time_a"],r["time_b"],r["vencedor"],r["placar"][0],r["placar"][1])
        r32_cy.append(cy)

    r16_cy = []
    for i, r in enumerate(mm["r16"]):
        y_mid = (r32_cy[i*2]+r32_cy[i*2+1])/2 - 0.38
        cy = box(ax,xs[1],y_mid,r["time_a"],r["time_b"],r["vencedor"],r["placar"][0],r["placar"][1])
        r16_cy.append(cy)
        ax.plot([xs[0]+1.85,xs[1]],[r32_cy[i*2],cy+0.19],color="#78909C",lw=0.5)
        ax.plot([xs[0]+1.85,xs[1]],[r32_cy[i*2+1],cy+0.19],color="#78909C",lw=0.5)

    qf_cy = []
    for i, r in enumerate(mm["qf"]):
        y_mid = (r16_cy[i*2]+r16_cy[i*2+1])/2 - 0.38
        cy = box(ax,xs[2],y_mid,r["time_a"],r["time_b"],r["vencedor"],r["placar"][0],r["placar"][1])
        qf_cy.append(cy)
        ax.plot([xs[1]+1.85,xs[2]],[r16_cy[i*2],cy+0.19],color="#78909C",lw=0.5)
        ax.plot([xs[1]+1.85,xs[2]],[r16_cy[i*2+1],cy+0.19],color="#78909C",lw=0.5)

    sf_cy = []
    for i, r in enumerate(mm["sf"]):
        y_mid = (qf_cy[i*2]+qf_cy[i*2+1])/2 - 0.38
        cy = box(ax,xs[3],y_mid,r["time_a"],r["time_b"],r["vencedor"],r["placar"][0],r["placar"][1])
        sf_cy.append(cy)
        ax.plot([xs[2]+1.85,xs[3]],[qf_cy[i*2],cy+0.19],color="#78909C",lw=0.5)
        ax.plot([xs[2]+1.85,xs[3]],[qf_cy[i*2+1],cy+0.19],color="#78909C",lw=0.5)

    fin = mm["final"]
    y_fin = (sf_cy[0]+sf_cy[1])/2 - 0.38
    box(ax,xs[4],y_fin,fin["time_a"],fin["time_b"],fin["vencedor"],fin["placar"][0],fin["placar"][1])
    ax.plot([xs[3]+1.85,xs[4]],[sf_cy[0],y_fin+0.57],color="#78909C",lw=0.5)
    ax.plot([xs[3]+1.85,xs[4]],[sf_cy[1],y_fin+0.57],color="#78909C",lw=0.5)

    for x, lbl in zip(xs,["Rodada\nde 32","Oitavas\nde Final","Quartas\nde Final","Semifinal","Final"]):
        ax.text(x+0.92,32.5,lbl,ha="center",fontsize=8,fontweight="bold",color="#1A237E")
    ax.text(5,-0.6,f"CAMPEAO: {pt(mm['campeao'])}",ha="center",fontsize=13,fontweight="bold",
            color="white",bbox=dict(boxstyle="round",facecolor="#1A237E",alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS,"05_bracket_mata_mata.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ===========================================================================
# 10. PDF MODO SINGLE (relatório detalhado de uma simulação)
# ===========================================================================
def _titulo_secao(pdf, texto):
    """Renderiza um cabeçalho de seção azul no PDF."""
    pdf.set_font("Helvetica","B",14)
    pdf.set_fill_color(26,35,126); pdf.set_text_color(255,255,255)
    pdf.cell(0,9,_safe(texto),align="C",fill=True,new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.set_text_color(0,0,0); pdf.ln(2)

def _cabecalho_tabela(pdf, cols, widths):
    """Renderiza uma linha de cabeçalho cinza para tabelas no PDF."""
    pdf.set_font("Helvetica","B",8)
    pdf.set_fill_color(200,200,200)
    for c, w in zip(cols, widths):
        pdf.cell(w,5,_safe(c),border=1,fill=True,align="C")
    pdf.ln()

def gerar_pdf(tabelas, m3, mm, df):
    """
    Gera o relatório PDF detalhado de uma simulação single (relatorio_calibrado_2026.pdf).

    Estrutura do PDF:
      Página 1: capa com seed e parâmetros calibrados
      Página 2+: fase de grupos (tabelas + previsões jogo a jogo + chances de avanço)
      Página N: 8 melhores terceiros colocados
      Página N+1: fase mata-mata (rodada de 32 até a final)
      Página N+2: resultado final (campeão, vice, 3º, 4º)
      Páginas finais: os 5 gráficos embutidos

    Os percentuais de "chance de avançar" são calculados com 1000 simulações
    Monte Carlo por grupo — custo: 12.000 simulações extras (apenas para o PDF).
    """
    pdf = FPDF(); pdf.set_auto_page_break(True, margin=12)

    pdf.add_page()
    pdf.set_fill_color(26,35,126); pdf.rect(0,0,210,297,"F")
    pdf.set_text_color(255,255,255); pdf.ln(55)
    pdf.set_font("Helvetica","B",26)
    pdf.cell(0,14,"SIMULACAO",align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.cell(0,14,"COPA DO MUNDO 2026",align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.set_font("Helvetica","",14); pdf.ln(4)
    pdf.cell(0,9,"Baseado em dados do FC26 (via SoFIFA)",align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.set_font("Helvetica","I",11); pdf.ln(3)
    pdf.cell(0,8,f"Seed: {SEED}  |  {N_SIMS} simulacoes por partida",align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.set_font("Helvetica","I",9); pdf.ln(3)
    pdf.cell(0,7,"Parametros calibrados via MLE (WC2018+2022) | Cross-val: ~57% acuracia",
             align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.cell(0,7,f"LAMBDA={LAMBDA_BASE:.4f} | EXPO={EXPO_FORCA:.4f} | DEF_AMP={FATOR_DEF_AMP:.4f} | LIMIAR={LIMIAR}",
             align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.set_text_color(0,0,0)

    # Calcula min/max uma única vez para passar a calcular_chances_avanco em todos os grupos
    forcas_all = {t: calcular_forca(t, df) for t in df.index}
    min_forca  = float(min(forcas_all.values()))
    max_forca  = float(max(forcas_all.values()))
    min_def    = float(df["Defesa"].min())
    max_def    = float(df["Defesa"].max())

    pdf.add_page(); _titulo_secao(pdf,"FASE DE GRUPOS")
    for letra, dados in tabelas.items():
        tab = dados["tab"]; classif = dados["classif"]
        jogos = dados.get("jogos", [])
        times_grupo = [resolver_nome(t, df) for t in GRUPOS_COPA[letra] if resolver_nome(t, df)]

        pdf.set_font("Helvetica","B",10)
        pdf.set_fill_color(63,81,181); pdf.set_text_color(255,255,255)
        pdf.cell(0,6,f"  GRUPO {letra}",fill=True,new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.set_text_color(0,0,0)

        pdf.set_font("Helvetica","I",8)
        # min/max passados externamente para evitar recalcular por todos os 48 times a cada grupo
        chances = calcular_chances_avanco(times_grupo, df, min_forca, max_forca, min_def, max_def, n_sims=1000)
        for t in classif:
            rk = fifa_rank(t)
            ch = chances.get(t, 0.0)
            pdf.cell(0,5,_safe(f"  {pt(t)}: Ranking FIFA #{rk} | {ch:.1f}% de chance de avancar"),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

        pdf.set_font("Helvetica","B",8)
        pdf.cell(0,5,"  Previsoes jogo a jogo:",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.set_font("Helvetica","",8)
        for j in jogos:
            pdf.cell(0,5,_safe(f"  {pt(j['ta'])} {j['ga']} x {j['gb']} {pt(j['tb'])}  "
                               f"(A:{j['va']}% / Emp:{j['emp']}% / B:{j['vb']}%)"),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

        _cabecalho_tabela(pdf,["Selecao","Pts","J","V","E","D","GM","GC","SG"],[62,11,8,8,8,8,10,10,10])
        pdf.set_font("Helvetica","",8)
        for pos, t in enumerate(classif):
            r = tab[t]
            cor = (232,245,233) if pos<2 else (255,253,231) if pos==2 else (255,255,255)
            pdf.set_fill_color(*cor)
            vals = [_safe(pt(t))[:26],r["pts"],r["j"],r["v"],r["e"],r["d"],r["gm"],r["gc"],r["sg"]]
            for v, w in zip(vals,[62,11,8,8,8,8,10,10,10]):
                pdf.cell(w,5,str(v),border=1,fill=True,align="L" if isinstance(v,str) else "C")
            pdf.ln()
        pdf.ln(3)

    pdf.add_page(); _titulo_secao(pdf,"8 MELHORES TERCEIROS COLOCADOS")
    pdf.set_font("Helvetica","",9)
    pdf.multi_cell(0,5,"Criterios FIFA: Pontos > Saldo de Gols > Gols Marcados > Ranking FIFA")
    pdf.ln(2)
    _cabecalho_tabela(pdf,["#","Selecao","Grupo","Pts","SG","GM","Rank FIFA"],[8,62,14,11,11,11,23])
    pdf.set_font("Helvetica","",9)
    for i, row in m3.iterrows():
        vals=[i+1,_safe(pt(row["time"]))[:26],row["grupo"],
              int(row["pts"]),int(row["sg"]),int(row["gm"]),int(row["rank_fifa"])]
        for v, w in zip(vals,[8,62,14,11,11,11,23]):
            pdf.cell(w,6,str(v),border=1,align="L" if isinstance(v,str) and len(str(v))>3 else "C")
        pdf.ln()

    pdf.add_page(); _titulo_secao(pdf,"FASE MATA-MATA")
    for titulo, resultados in [
        ("RODADA DE 32",mm["r32"]),("OITAVAS DE FINAL",mm["r16"]),
        ("QUARTAS DE FINAL",mm["qf"]),("SEMIFINAL",mm["sf"]),
        ("DISPUTA DE 3o LUGAR",[mm["terceiro_lugar"]]),("FINAL",[mm["final"]])]:
        pdf.ln(2)
        pdf.set_font("Helvetica","B",10)
        pdf.set_fill_color(63,81,181); pdf.set_text_color(255,255,255)
        pdf.cell(0,6,f"  {titulo}",fill=True,new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.set_text_color(0,0,0)
        _cabecalho_tabela(pdf,["Time A","Time B","Placar","% A","% Emp","% B","Vencedor"],[46,46,20,12,12,12,42])
        pdf.set_font("Helvetica","",8)
        for r in resultados:
            # Exibe pênaltis apenas se houve cobrança real (pa não é None — ver simular_penaltis)
            if r.get("penaltis") and r["penaltis"][0] is not None:
                pen = f"pen {r['penaltis'][0]}-{r['penaltis'][1]}"
            else:
                pen = ""
            pl  = f"{r['placar'][0]}-{r['placar'][1]}" + (f" {pen}" if pen else "")
            vals=[_safe(pt(r["time_a"]))[:22],_safe(pt(r["time_b"]))[:22],pl,
                  f"{r['pct_vit_a']}%",f"{r['pct_emp']}%",f"{r['pct_vit_b']}%",
                  _safe(pt(r["vencedor"]))[:22]]
            for v, w in zip(vals,[46,46,20,12,12,12,42]):
                pdf.cell(w,5,str(v),border=1,align="C" if "%" in str(v) or "-" in str(v) else "L")
            pdf.ln()

    pdf.add_page(); _titulo_secao(pdf,"RESULTADO FINAL")
    pdf.ln(8)
    for titulo, time, _ in [("Campeao",mm["campeao"],(255,215,0)),
                              ("Vice-Campeao",mm["vice"],(192,192,192)),
                              ("3o Lugar",mm["terceiro"],(205,127,50)),
                              ("4o Lugar",mm["quarto"],(189,189,189))]:
        pdf.set_font("Helvetica","B",13)
        pdf.cell(55,10,titulo,align="R")
        pdf.set_font("Helvetica","",13)
        pdf.cell(0,10,f"  {_safe(pt(time))}",new_x=XPos.LMARGIN,new_y=YPos.NEXT)

    fin = mm["final"]
    if fin.get("penaltis") and fin["penaltis"][0] is not None:
        pen_txt = f"  (Penaltis: {fin['penaltis'][0]}-{fin['penaltis'][1]})"
    else:
        pen_txt = ""
    pdf.ln(6); pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,_safe(f"Placar da Final: {pt(fin['time_a'])} {fin['placar'][0]} x "
                       f"{fin['placar'][1]} {pt(fin['time_b'])}{pen_txt}"),
             new_x=XPos.LMARGIN,new_y=YPos.NEXT)

    for fname, titulo in [
        ("01_ranking_forca.png","Ranking de Forca das Selecoes"),
        ("02_heatmap_fases.png","Estimativa de Avanco por Fase (baseado em forca relativa FC26)"),
        ("03_resultado_final.png","Resultado da Final - 100 Simulacoes"),
        ("04_gols_por_selecao.png","Gols por Selecao no Torneio"),
        ("05_bracket_mata_mata.png","Chaveamento do Mata-Mata"),
    ]:
        path = os.path.join(GRAFICOS, fname)
        if not os.path.exists(path): continue
        pdf.add_page(); pdf.set_font("Helvetica","B",12)
        pdf.cell(0,8,titulo,align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.image(path, x=10, y=pdf.get_y()+2, w=190)

    pdf.output(PDF_OUT)

def salvar_csv(historico):
    """Salva o histórico completo de partidas em CSV (UTF-8 com BOM para Excel)."""
    pd.DataFrame(historico).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

# ===========================================================================
# 11. MODO MULTI (paralelo) — N simulações com contagem de tuplas conjuntas
# ===========================================================================
PONTOS_FASE = {
    "Campeao":8,"Vice":7,"3o Lugar":6,"4o Lugar":5,
    "Semifinal":4,"Quartas de Final":3,"Oitavas de Final":2,
    "Rodada de 32":1,"Fase de Grupos":0,
}
FASES_LABEL = ["Campeao","Vice","3o Lugar","4o Lugar",
               "Semifinal","Quartas de Final","Oitavas de Final","Rodada de 32","Fase de Grupos"]

# DataFrame compartilhado por processo worker. Carregado uma vez por processo
# via _init_worker (initializer do ProcessPoolExecutor). Essa abordagem evita
# serializar/pickle o DataFrame inteiro a cada uma das N chamadas de _worker_seed,
# o que seria significativamente mais lento em simulações com 5000+ seeds.
_df_worker = None

def _init_worker():
    """Inicializa o DataFrame global do processo worker antes de processar seeds."""
    global _df_worker
    _df_worker = carregar_dados()

def _worker_seed(seed):
    """
    Executa uma simulação completa do torneio com a seed fornecida.

    Nota sobre RNG global: cada processo criado pelo ProcessPoolExecutor recebe
    sua própria cópia do espaço de memória (fork/spawn), portanto a reatribuição
    de RNG aqui não afeta outros processos. Seria inseguro com ThreadPoolExecutor
    (condição de corrida), mas é seguro com ProcessPoolExecutor.

    Retorna:
        pos   (dict): {seleção → melhor fase atingida no torneio}
        tupla (tuple): (campeão, vice, 3º, 4º) para contagem de probabilidade
                       conjunta — garante combinações torneiramente válidas
                       (ex.: dois semifinalistas não aparecem em 1º e 2º lugar).
    """
    global RNG, _df_worker
    RNG = np.random.default_rng(seed)
    df  = _df_worker if _df_worker is not None else carregar_dados()

    historico = []
    hist, tabelas, todos_3, _ = simular_grupos(df)
    historico.extend(hist)
    m3      = melhores_terceiros(todos_3)
    oitavas = montar_chaveamento(tabelas, m3)
    mm      = simular_mata_mata(oitavas, df, historico)

    pos = {}
    pos[mm["campeao"]]  = "Campeao"
    pos[mm["vice"]]     = "Vice"
    pos[mm["terceiro"]] = "3o Lugar"
    pos[mm["quarto"]]   = "4o Lugar"
    for r in mm["sf"]:
        for t in [r["time_a"],r["time_b"]]:
            pos.setdefault(t, "Semifinal")
    for r in mm["qf"]:
        for t in [r["time_a"],r["time_b"]]:
            pos.setdefault(t, "Quartas de Final")
    for r in mm["r16"]:
        for t in [r["time_a"],r["time_b"]]:
            pos.setdefault(t, "Oitavas de Final")
    for r in mm["r32"]:
        for t in [r["time_a"],r["time_b"]]:
            pos.setdefault(t, "Rodada de 32")
    for t in df.index:
        pos.setdefault(t, "Fase de Grupos")

    # Tupla para probabilidade conjunta — resultado mais provável e torneiramente válido
    tupla = (mm["campeao"], mm["vice"], mm["terceiro"], mm["quarto"])
    return pos, tupla


def gerar_pdf_multi(contagem, pontos, times_ord, ini, fim, top_tuplas):
    """
    Gera o PDF de ranking agregado do modo multi (ranking_5000_simulacoes.pdf).

    Estrutura:
      Página 1+: ranking de todas as seleções por pontuação acumulada
      Página N: detalhamento das Top 10 seleções por fase
      Página N+1: top 20 resultados mais prováveis (probabilidade conjunta)
    """
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    n_sims = fim - ini
    pdf.set_font("Helvetica","B",15)
    pdf.cell(0,10,_safe(f"Simulador Copa do Mundo 2026 - Modelo Calibrado - {n_sims} Simulacoes"),
             align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.set_font("Helvetica","",9)
    pdf.cell(0,6,_safe(f"MLE WC2018+2022 | LAMBDA={LAMBDA_BASE:.4f} EXPO={EXPO_FORCA:.4f} DEF_AMP={FATOR_DEF_AMP:.4f} LIMIAR={LIMIAR}"),
             align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.cell(0,6,"Quadro de medalhas: Campeao=8 | Vice=7 | 3o=6 | 4o=5 | Semi=4 | QF=3 | Oitavas=2 | R32=1",
             align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.ln(4)

    COLS  = ["Rank","Selecao","Pts","Campeao","Vice","3o","4o","Semi","QF","Oitavas","R32","Grupos"]
    WIDTHS= [10,     48,       14,   16,        14,    12,  12,  14,   14,   16,       14,    16]

    def cabecalho():
        pdf.set_font("Helvetica","B",8)
        pdf.set_fill_color(26,35,126); pdf.set_text_color(255,255,255)
        for lbl, w in zip(COLS, WIDTHS):
            pdf.cell(w,7,lbl,border=1,align="C",fill=True)
        pdf.ln(); pdf.set_text_color(0,0,0)

    cabecalho()
    for rank, time in enumerate(times_ord, 1):
        c = contagem[time]
        pdf.set_fill_color(240,240,240) if rank%2==0 else pdf.set_fill_color(255,255,255)
        pdf.set_font("Helvetica","B" if rank<=4 else "",8)
        vals = [str(rank), _safe(pt(time))[:26], f"{pontos[time]:.0f}",
                str(c.get("Campeao",0)),      str(c.get("Vice",0)),
                str(c.get("3o Lugar",0)),     str(c.get("4o Lugar",0)),
                str(c.get("Semifinal",0)),    str(c.get("Quartas de Final",0)),
                str(c.get("Oitavas de Final",0)), str(c.get("Rodada de 32",0)),
                str(c.get("Fase de Grupos",0))]
        for val, w in zip(vals, WIDTHS):
            pdf.cell(w,6,val,border=1,align="C",fill=True)
        pdf.ln()
        if rank % 30 == 0 and rank < len(times_ord):
            pdf.add_page(); cabecalho()

    pdf.add_page()
    pdf.set_font("Helvetica","B",13)
    pdf.cell(0,10,"Top 10 - Detalhamento por Fase",align="C",new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.ln(3)
    for rank, time in enumerate(times_ord[:10], 1):
        c = contagem[time]
        pdf.set_font("Helvetica","B",10)
        pdf.set_fill_color(220,230,245)
        pdf.cell(0,7,_safe(f"{rank}. {pt(time)}  |  Pontos: {pontos[time]:.0f}  |  Titulos: {c.get('Campeao',0)}x em {n_sims}"),
                 new_x=XPos.LMARGIN,new_y=YPos.NEXT,fill=True)
        pdf.set_font("Helvetica","",9)
        linha = "  ".join(f"{f}: {c.get(f,0)}x" for f in FASES_LABEL if c.get(f,0))
        pdf.cell(0,6,_safe(linha),new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        pdf.ln(1)

    pdf.add_page()
    pdf.set_font("Helvetica","B",14)
    pdf.set_fill_color(26,35,126); pdf.set_text_color(255,255,255)
    pdf.cell(0,10,"RESULTADO MAIS PROVAVEL DA COPA DO MUNDO 2026",
             align="C",fill=True,new_x=XPos.LMARGIN,new_y=YPos.NEXT)
    pdf.set_text_color(0,0,0); pdf.ln(4)

    pdf.set_font("Helvetica","",10)
    pdf.multi_cell(0,6,_safe(
        "Probabilidade conjunta: contagem de tuplas (Campeao, Vice, 3o Lugar, 4o Lugar) "
        "ao longo das " + str(n_sims) + " simulacoes. "
        "Garante combinacoes validas — dois times que se enfrentaram na semifinal nao "
        "aparecem simultaneamente em 1o e 2o lugar."
    ))
    pdf.ln(4)

    pdf.set_font("Helvetica","B",9)
    pdf.set_fill_color(200,200,200)
    for lbl, w in zip(["Rank","Campeao","Vice-Campeao","3o Lugar","4o Lugar","Ocorrencias","Freq (%)"],
                       [12,52,52,52,52,24,22]):
        pdf.cell(w,6,lbl,border=1,fill=True,align="C")
    pdf.ln()

    pdf.set_font("Helvetica","",9)
    for rank, (tupla, cnt) in enumerate(top_tuplas, 1):
        freq = cnt / n_sims * 100
        vals = [str(rank), _safe(pt(tupla[0])), _safe(pt(tupla[1])),
                _safe(pt(tupla[2])), _safe(pt(tupla[3])),
                str(cnt), f"{freq:.2f}%"]
        fill = (255,250,220) if rank == 1 else (255,255,255)
        pdf.set_fill_color(*fill)
        for v, w in zip(vals, [12,52,52,52,52,24,22]):
            pdf.cell(w,6,v,border=1,align="C" if v.isdigit() or "%" in v else "L",fill=True)
        pdf.ln()
        if rank >= 20:
            break

    pdf.ln(6)
    if top_tuplas:
        melhor = top_tuplas[0]
        pdf.set_font("Helvetica","B",12)
        pdf.set_fill_color(255,215,0)
        pdf.cell(0,10,_safe(
            f"Resultado mais provavel: {pt(melhor[0][0])} campeao | "
            f"{pt(melhor[0][1])} vice-campeao | "
            f"{pt(melhor[0][2])} 3o lugar | "
            f"{pt(melhor[0][3])} 4o lugar  "
            f"({melhor[1]}x em {n_sims} = {melhor[1]/n_sims*100:.2f}%)"
        ), fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(PDF_MULTI)
    print(f"\n  PDF gerado: {PDF_MULTI}")


def rodar_multi(ini, fim, n_workers):
    """
    Executa N simulações paralelas (seeds ini até fim-1) e agrega os resultados.

    Cada seed é processada por um processo worker independente via ProcessPoolExecutor.
    Ao final, imprime no terminal os 10 resultados mais prováveis (probabilidade conjunta)
    e gera o PDF de ranking agregado.

    Nota: o modo multi não gera gráficos individuais — apenas o PDF de ranking.
    Para análise visual de uma seed específica, use o modo single.

    Args:
        ini       (int): seed inicial (inclusivo)
        fim       (int): seed final (exclusivo)
        n_workers (int): número de processos paralelos
    """
    seeds = list(range(ini, fim))
    total = len(seeds)
    contagem  = defaultdict(lambda: defaultdict(int))
    pontos    = defaultdict(float)
    tupla_cnt = Counter()

    print(f"  {total} simulacoes | workers={n_workers} | seeds {ini}-{fim-1}")

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        futures = {ex.submit(_worker_seed, s): s for s in seeds}
        concluidos = 0
        for future in as_completed(futures):
            concluidos += 1
            print(f"  [{concluidos:4d}/{total}] seed={futures[future]}", end="\r", flush=True)
            pos, tupla = future.result()
            for time, fase in pos.items():
                contagem[time][fase] += 1
                pontos[time] += PONTOS_FASE.get(fase, 0)
            tupla_cnt[tupla] += 1

    print()

    def chave_medalhas(t):
        c = contagem[t]
        return (
            c.get("Campeao", 0), c.get("Vice", 0), c.get("3o Lugar", 0),
            c.get("4o Lugar", 0), c.get("Semifinal", 0), c.get("Quartas de Final", 0),
            c.get("Oitavas de Final", 0), c.get("Rodada de 32", 0), c.get("Fase de Grupos", 0),
        )
    times_ord  = sorted(contagem.keys(), key=chave_medalhas, reverse=True)
    top_tuplas = tupla_cnt.most_common(20)

    print("\n" + "="*62)
    print("  RESULTADO MAIS PROVAVEL (PROBABILIDADE CONJUNTA)")
    print("="*62)
    for rank, (tupla, cnt) in enumerate(top_tuplas[:10], 1):
        freq = cnt / total * 100
        print(f"  {rank:2d}. {pt(tupla[0]):<20} | {pt(tupla[1]):<20} | "
              f"{pt(tupla[2]):<20} | {pt(tupla[3]):<20} — {cnt}x ({freq:.2f}%)")

    gerar_pdf_multi(contagem, pontos, times_ord, ini, fim, top_tuplas)


# ===========================================================================
# 12. MAIN — ponto de entrada (single ou multi)
# ===========================================================================
def main():
    """
    Ponto de entrada do simulador. Interpreta os argumentos de linha de comando
    e despacha para o modo single (seed única, relatório detalhado) ou
    modo multi (N seeds paralelas, ranking agregado).
    """
    parser = argparse.ArgumentParser(
        description="Simulador Copa do Mundo 2026 — Modelo Calibrado MLE"
    )
    parser.add_argument("--seed",    type=int, default=42,
                        help="Seed para reproducibilidade (modo single, padrao: 42)")
    parser.add_argument("--multi",   action="store_true",
                        help="Ativa modo multi-simulacao paralela (nao gera graficos individuais)")
    parser.add_argument("--ini",     type=int, default=1,
                        help="Seed inicial, inclusiva (modo multi, padrao: 1)")
    parser.add_argument("--fim",     type=int, default=5001,
                        help="Seed final, exclusiva (modo multi, padrao: 5001 = 5000 sims)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Numero de workers paralelos (padrao: todos os nucleos da CPU)")
    # parse_args() (não parse_known_args) — argumentos inválidos geram erro explícito
    args = parser.parse_args()

    if args.multi:
        n_workers = args.workers or multiprocessing.cpu_count()
        print("="*62)
        print("  SIMULADOR COPA DO MUNDO 2026 - Modelo Calibrado - Multi")
        print(f"  LAMBDA={LAMBDA_BASE:.4f} | EXPO={EXPO_FORCA:.4f} | "
              f"DEF_AMP={FATOR_DEF_AMP:.4f} | LIMIAR={LIMIAR}")
        print("="*62)
        rodar_multi(args.ini, args.fim, n_workers)
        return

    # -- Modo single --------------------------------------------------------
    global SEED, RNG
    SEED = args.seed
    RNG  = np.random.default_rng(SEED)

    print("="*62)
    print(f"  SIMULADOR COPA DO MUNDO 2026 - Modelo Calibrado")
    print(f"  Seed: {SEED}  |  {N_SIMS} simulacoes por partida")
    print(f"  LAMBDA={LAMBDA_BASE:.4f} EXPO={EXPO_FORCA:.4f} "
          f"DEF_AMP={FATOR_DEF_AMP:.4f} LIMIAR={LIMIAR}")
    print("="*62)

    print("\n[1/6] Carregando dados...")
    df = carregar_dados()
    print(f"       {len(df)} selecoes carregadas.")

    print("\n[2/6] Simulando fase de grupos (72 partidas x 100 simulacoes)...")
    historico, tabelas, todos_3, gols_grupos = simular_grupos(df)
    print("       Grupos simulados.")

    print("\n[3/6] Selecionando melhores terceiros...")
    m3 = melhores_terceiros(todos_3)
    for _, r in m3.iterrows():
        print(f"       Grupo {r['grupo']}: {pt(r['time'])}  ({int(r['pts'])}pts | SG:{int(r['sg']):+d})")

    print("\n[4/6] Simulando mata-mata (32 partidas x 100 simulacoes)...")
    oitavas = montar_chaveamento(tabelas, m3)
    mm = simular_mata_mata(oitavas, df, historico)

    os.makedirs(GRAFICOS, exist_ok=True)
    print("\n[5/6] Gerando graficos...")
    gerar_graficos(df, tabelas, mm, m3, gols_grupos)

    print("\n[6/6] Gerando CSV e PDF...")
    salvar_csv(historico)
    print(f"       CSV: {CSV_OUT}  ({len(historico)} partidas)")
    gerar_pdf(tabelas, m3, mm, df)
    print(f"       PDF: {PDF_OUT}")

    fin = mm["final"]
    if fin.get("penaltis") and fin["penaltis"][0] is not None:
        pen = f" (pen. {fin['penaltis'][0]}-{fin['penaltis'][1]})"
    else:
        pen = ""
    print("\n" + "="*62)
    print("  RESULTADO FINAL")
    print("="*62)
    print(f"  Campeao:       {pt(mm['campeao'])}")
    print(f"  Vice:          {pt(mm['vice'])}")
    print(f"  3o lugar:      {pt(mm['terceiro'])}")
    print(f"  4o lugar:      {pt(mm['quarto'])}")
    print(f"  Placar final:  {pt(fin['time_a'])} {fin['placar'][0]}x{fin['placar'][1]} {pt(fin['time_b'])}{pen}")

    gols_total = {t: gols_grupos.get(t,0)+mm["gols"].get(t,0) for t in df.index}
    top5 = sorted(gols_total, key=gols_total.get, reverse=True)[:5]
    print("\n  Top 5 times por gols no torneio:")
    for i, t in enumerate(top5, 1):
        print(f"    {i}. {pt(t)}: {gols_total[t]} gols")
    print()

if __name__ == "__main__":
    main()
