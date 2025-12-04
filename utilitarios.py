# ===============================================================================

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Configura backend não-interativo para evitar conflitos com Pygame/Tkinter
import matplotlib.pyplot as plt
import matplotlib.patches as mptch
import sklearn as sk
from collections import Counter
from constants import VS

# ===============================================================================

# Plota os datasets
def plot_graph(df, x, y):

    cores = {0: 'green', 1: 'yellow', 2: 'red', 3: 'black'}
    contagem = Counter(df['tri'])
    print("\nNúmero de vítimas por classificação de Triagem:")
    for k in sorted(contagem.keys()):
        print(f"  {k} ({cores[k]}): {contagem[k]}")
    print("")# Espaçamento

    plt.figure(figsize=(16, 10))
    cores_pontos = df['tri'].map(cores)

    plt.scatter(x, y, color=cores_pontos, edgecolors='black', alpha=0.7)
    plt.title('Distribuição percentual da probabilidade de sobrevivência')
    plt.ylabel('Probabilidade de Sobrevivência')
    plt.xlabel(x.name)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    legendas = [mptch.Patch(color=cores[k], label=f'{k}') for k in sorted(cores.keys())]
    plt.legend(handles=legendas, title='Classificação da Triagem')

    plt.show()

# -------------------------------------------------------------------------------

# Plota confusion matrix
def plot_cmatrix(y_test, y_pred_test, report):
    sk.metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, cmap='viridis')
    plt.title('Matriz de Confusão')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    if report:
        print(sk.metrics.classification_report(y_test, y_pred_test))

# -------------------------------------------------------------------------------

# Plota arvore de decisão
def plot_tree(modelo, x_train, y_train):
    feature_names = x_train.columns.tolist()
    class_names = [str(c) for c in sorted(y_train.unique())]

    plt.figure(figsize=(8, 6))
    sk.tree.plot_tree(modelo, feature_names=feature_names, filled=True, rounded=True, class_names=class_names, fontsize=8)
    plt.show()

# -------------------------------------------------------------------------------

# Plota desempenho das parametrizações RN
def plot_vies_var (num_params, train_scores, vld_scores):

    plt.figure(figsize=(10, 6))
    colors = [
        ["darkblue", "lightblue"],    # param 0
        ["purple", "pink"],           # param 1
        ["darkgreen", "lightgreen"],  # param 2
    ]

    for i in range(num_params):
        plt.plot(range(1, len(train_scores[i]) + 1), train_scores[i], label=f"{i} Train Neg MSE", marker='o',
                 color=colors[i][0])
        plt.axhline(train_scores[i].mean(), color=colors[i][0], linestyle='--',
                    label=f"{i} Train.mean: {train_scores[i].mean():.2f}")
        plt.plot(range(1, len(vld_scores[i]) + 1), vld_scores[i], label=f"{i} Valid Neg MSE", marker='o',
                 color=colors[i][1])
        plt.axhline(vld_scores[i].mean(), color=colors[i][1], linestyle='--',
                    label=f"{i} Valid.mean: {vld_scores[i].mean():.2f}")

    plt.xlabel("Fold")
    plt.ylabel("Neg MSE")
    plt.title("Training and Validation Scores (Bias and Variance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(np.arange(1, 3 + 1, 1))
    plt.grid()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------

# Plota um grafo com pesos da biblioteca networkx
def plot_nxgraph (graph):
    pos = {(x, y): (y, -x) for x, y in graph.nodes()}

    plt.figure(figsize=(6, 6))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=1500,
        node_color="lightblue",
        font_weight="bold",
        arrows=True,
        arrowsize=25,
        connectionstyle="arc3,rad=0.2",  # curva as retas
        edgecolors="black"
    )

    for (u, v), data in nx.get_edge_attributes(graph, 'weight').items():

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels={(u, v): data},
            label_pos=0.5,
            font_size=9,
            rotate=False
        )

    plt.title("Grafo Direcionado", fontsize=14)
    plt.axis("off")
    plt.show()

# -------------------------------------------------------------------------------

# Plota gráfico simples usando contagens diretas por triagem
def plot_saved_vs_found_counts(total_counts, saved_counts, save_path=None, saved_by_rescuer=None):
    labels_map = {0: 'Verde (0)', 1: 'Amarelo (1)', 2: 'Vermelho (2)', 3: 'Preto (3)'}
    categories = [0, 1, 2, 3]

    found_vals = [int(total_counts.get(c, 0)) for c in categories]
    saved_vals = [int(saved_counts.get(c, 0)) for c in categories]

    x = np.arange(len(categories))
    width = 0.24  # grossura das barras

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width/2,
        found_vals,
        width,
        label='Total (env)',
        color='#9e9e9e',
        alpha=0.7,
        edgecolor='#666666',
        linewidth=0.8,
    )

    if saved_by_rescuer and isinstance(saved_by_rescuer, list) and len(saved_by_rescuer) > 0:
        fixed_colors_rgb = [
            (0, 102, 204),   # azul
            (102, 204, 0),   # verde
            (204, 0, 51),    # vermelho
        ]
        bottom = np.zeros(len(categories))
        for ridx, resc_counts in enumerate(saved_by_rescuer):
            rgb = fixed_colors_rgb[ridx % len(fixed_colors_rgb)]
            color = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
            vals = np.array([int(resc_counts.get(c, 0)) for c in categories])
            bars = ax.bar(
                x + width/2,
                vals,
                width,
                bottom=bottom,
                color=color,
                alpha=0.9,
                edgecolor='#444444',
                linewidth=0.6,
                label=f"Salvas - R{ridx+1}"
            )
            for xi, v, btm in zip(x, vals, bottom):
                if v > 0:
                    ax.text(xi + width/2, btm + v/2.0, f"{int(v)}", ha='center', va='center', fontsize=9, color='white')
            bottom += vals
        rects2_vals = saved_vals
    else:
        rects2 = ax.bar(
            x + width/2,
            saved_vals,
            width,
            label='Salvas',
            color='#43a047',
            alpha=0.85,
            edgecolor='#2e7d32',
            linewidth=0.8,
        )

    ax.set_ylabel('Quantidade')
    ax.set_title('Vítimas Totais vs Salvas por Triagem')
    ax.set_xticks(x)
    ax.set_xticklabels([labels_map.get(c, str(c)) for c in categories])
    ax.legend(frameon=False, ncol=2)

    ax.bar_label(rects1, padding=3, fontsize=9)
    if saved_by_rescuer and isinstance(saved_by_rescuer, list) and len(saved_by_rescuer) > 0:
        for xi, total in zip(x, saved_vals):
            ax.text(xi + width/2, total + 0.05, f"{total}", ha='center', va='bottom', fontsize=9)
    else:
        ax.bar_label(rects2, padding=3, fontsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.35)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()

# -------------------------------------------------------------------------------

def draw_rescuer_traces(env, resc_minds, save_path='rescuer_traces.png'):
    """Desenha o grid do ambiente e sobrepõe os trajetos dos socorristas.
    - env: Environment
    - resc_minds: lista de mentes dos socorristas (RescuerMind)
    """
    grid_w = env.dic.get('GRID_WIDTH', 94)
    grid_h = env.dic.get('GRID_HEIGHT', 94)

    # Render usando matplotlib
    fig, ax = plt.subplots(figsize=(10, 9))

    # Desenha obstáculos em escala de cinza
    for x in range(grid_w):
        for y in range(grid_h):
            obst = env.obst[x][y]
            if obst == VS.OBST_WALL:
                color = (0, 0, 0)
            elif obst == VS.OBST_NONE:
                color = (1, 1, 1)
            else:
                # escala de cinza baseada no fator do terreno
                perc = min(max(obst / 3.0, 0.0), 1.0)
                light = (1 - perc) * 1.0 + perc * 0.4
                color = (light, light, light)
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=(0.9, 0.9, 0.9), linewidth=0.3)
            ax.add_patch(rect)

    # Base
    bx, by = env.dic.get('BASE', [grid_w//2, grid_h//2])
    base_rect = plt.Rectangle((bx, by), 1, 1, fill=False, edgecolor=(0, 1, 1), linewidth=2)
    ax.add_patch(base_rect)

    fixed_colors_rgb = [
        (0, 102, 204),   # azul
        (102, 204, 0),   # verde
        (204, 0, 51),    # vermelho
    ]

    # Trajetos e vítimas salvas por agente
    for idx, mind in enumerate(resc_minds):
        rgb = fixed_colors_rgb[idx % len(fixed_colors_rgb)]
        color = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
        path = getattr(mind, 'trace_path', [])
        # desenha segmentos entre pontos consecutivos
        for i in range(1, len(path)):
            x0, y0 = path[i-1]
            x1, y1 = path[i]
            ax.plot([x0 + 0.5, x1 + 0.5], [y0 + 0.5, y1 + 0.5], color=color, linewidth=2)

        # marca vítimas salvas por este agente
        try:
            saved_by_agent = []
            for vid, savers in enumerate(env.saved):
                if any(getattr(phy.mind, 'NAME', None) == mind.NAME for phy in savers):
                    saved_by_agent.append(env.victims[vid])
            for (vx, vy) in saved_by_agent:
                ax.plot(vx + 0.5, vy + 0.5, marker='o', markersize=6, markerfacecolor=color, markeredgecolor='black')
        except Exception:
            pass

    ax.set_xlim(0, grid_w)
    ax.set_ylim(0, grid_h)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # para corresponder ao sistema pygame
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Mapa de trajetos salvo em: {save_path}")

# -------------------------------------------------------------------------------

def plot_found_vs_total_by_explorer(total_counts, found_by_explorer, save_path='found_by_explorer.png'):
    """
    Plota barras: à esquerda Total (env) por triagem; à direita, barras empilhadas
    com vítimas encontradas por explorador (um segmento por explorador) por triagem.

    total_counts: dict {tri: total}
    found_by_explorer: list of dicts, one per explorer, {tri: found}
    """
    labels_map = {0: 'Verde (0)', 1: 'Amarelo (1)', 2: 'Vermelho (2)', 3: 'Preto (3)'}
    categories = [0, 1, 2, 3]

    found_total_vals = [int(total_counts.get(c, 0)) for c in categories]

    x = np.arange(len(categories))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))

    # Barra de totais à esquerda
    rects1 = ax.bar(
        x - width/2,
        found_total_vals,
        width,
        label='Total (env)',
        color='#9e9e9e',
        alpha=0.7,
        edgecolor='#666666',
        linewidth=0.8,
    )

    fixed_colors_rgb = [
        (0, 102, 204),
        (102, 204, 0),
        (204, 0, 51),
    ]

    bottom = np.zeros(len(categories))
    for eidx, exp_counts in enumerate(found_by_explorer):
        rgb = fixed_colors_rgb[eidx % len(fixed_colors_rgb)]
        color = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
        vals = np.array([int(exp_counts.get(c, 0)) for c in categories])
        bars = ax.bar(
            x + width/2,
            vals,
            width,
            bottom=bottom,
            color=color,
            alpha=0.9,
            edgecolor='#444444',
            linewidth=0.6,
            label=f"Encontradas - E{eidx+1}"
        )
        # Label por segmento
        for xi, v, btm in zip(x, vals, bottom):
            if v > 0:
                ax.text(xi + width/2, btm + v/2.0, f"{int(v)}", ha='center', va='center', fontsize=9, color='white')
        bottom += vals

    # Títulos e eixos
    ax.set_ylabel('Quantidade')
    ax.set_title('Vítimas Totais vs Encontradas por Triagem (por Explorador)')
    ax.set_xticks(x)
    ax.set_xticklabels([labels_map.get(c, str(c)) for c in categories])
    ax.legend(frameon=False, ncol=2)

    ax.bar_label(rects1, padding=3, fontsize=9)
    # total de encontradas por categoria
    total_found_per_cat = bottom
    for xi, total in zip(x, total_found_per_cat):
        if total > 0:
            ax.text(xi + width/2, total + 0.05, f"{int(total)}", ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.35)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")