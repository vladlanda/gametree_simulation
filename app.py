import random
from statistics import mean
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from maze_game import LAYOUTS, TREES, GameState, html_player, name_tree, play, tree_trace

st.set_page_config(page_title="Adversarial Search Visualizer", page_icon="🎮", layout="wide")
st.title("Adversarial Search Visualizer")
st.caption("An independent, step-by-step educational implementation inspired by the topics covered in Berkeley CS188.")

def shift_step(key, delta):
    st.session_state[key] += delta

def fmt(v):
    return "-∞" if v == float("-inf") else "+∞" if v == float("inf") else f"{v:g}"

# ---------- game-tree drawing ----------
def tree_layout(tree):
    """Leaves at consecutive x, parents centred above their children."""
    xy, leaf = {}, 0
    def walk(t, path):
        nonlocal leaf
        if isinstance(t, list):
            xs = [walk(c, path + (i,)) for i, c in enumerate(t)]
            x = (xs[0] + xs[-1]) / 2
        else:
            x, leaf = leaf, leaf + 1
        xy[path] = (x, -len(path))
        return x
    walk(tree, ())
    return xy

def subtree(tree, path):
    for i in path:
        tree = tree[i]
    return tree

FONT = 20  # font size of the visualizations


def draw_game_tree(tree, events, step, show_leaves=True):
    xy, names = tree_layout(tree), name_tree(tree)
    values, pruned, entered, bound = {}, set(), set(), {}
    for e in events[:step + 1]:
        if e["kind"] in ("leaf", "return"):
            values[e["path"]] = e["value"]
        if e["kind"] in ("enter", "leaf"):
            entered.add(e["path"])
        if e["kind"] == "prune":
            bound[e["path"]] = "≥" if len(e["path"]) % 2 == 0 else "≤"  # value is only a bound
            for p in e["skipped"]:
                pruned |= {q for q in xy if q[:len(p)] == p}
    cur = events[step]["path"]
    on_path = {cur[:i] for i in range(len(cur) + 1)}
    fig = go.Figure()
    for p, (x, y) in xy.items():
        if not p:
            continue
        px, py = xy[p[:-1]]
        color = "#cbd5e1" if p in pruned else "#f59e0b" if p in on_path else "#64748b"
        fig.add_trace(go.Scatter(x=[px, x], y=[py, y], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=color, width=4 if p in on_path else 2,
                                           dash="dot" if p in pruned else "solid")))
    for p, (x, y) in xy.items():
        leaf = not isinstance(subtree(tree, p), list)
        symbol = "square" if leaf else ("triangle-up" if len(p) % 2 == 0 else "triangle-down")
        fill = "#e2e8f0" if p in pruned else "#fde68a" if p == cur else "#ffffff" if p in entered else "#f8fafc"
        v, color = values.get(p), "#0f172a"
        if p in pruned:   # never looked at
            label, color = (fmt(subtree(tree, p)) + "✂" if leaf and show_leaves else "✂"), "#94a3b8"
        elif v is not None:
            label = bound.get(p, "") + fmt(v)
        elif leaf:        # not evaluated yet
            label, color = (fmt(subtree(tree, p)), "#64748b") if show_leaves else ("?", "#0f172a")
        else:
            label = ""
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"<b>{label}</b>"],
            textposition="middle center", textfont=dict(size=FONT, color=color),
            marker=dict(symbol=symbol, size=62, color=fill,
                        line=dict(color="#f59e0b" if p == cur else "#334155", width=4 if p == cur else 1.5)),
            hovertext=[f"{names[p]}"], hoverinfo="text", showlegend=False, cliponaxis=False))
        if p:
            fig.add_annotation(x=x - 0.34, y=y + 0.32, text=f"<b>{names[p]}</b>", showarrow=False,
                               font=dict(size=FONT, color="#1e293b"), bgcolor="white")
    depth = max(len(p) for p in xy)
    fig.update_layout(height=170 * depth + 140, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      xaxis=dict(visible=False, range=[-0.7, max(x for x, _ in xy.values()) + 0.7]),
                      yaxis=dict(visible=False, range=[-depth - 0.6, 0.6]))
    return fig, pruned

def describe(e, names):
    who = "root" if not e["path"] else names[e["path"]]
    ab = f"(α = {fmt(e['alpha'])}, β = {fmt(e['beta'])})"
    if e["kind"] == "enter":
        return f"Enter **{who}** {ab}"
    if e["kind"] == "leaf":
        return f"Leaf **{who}** = {fmt(e['value'])} {ab}"
    if e["kind"] == "return":
        return f"**{who}** returns {fmt(e['value'])}"
    skipped = ", ".join(names[p] for p in e["skipped"])
    rule = "v ≥ β" if len(e["path"]) % 2 == 0 else "v ≤ α"
    return f"✂ Prune at **{who}**: v = {fmt(e['value'])}, {rule} {ab} → skip {skipped}"

@st.cache_data(show_spinner="Playing…")
def run_game(**kw):
    return play(**kw)

# ---------- UI ----------
tab_tree, tab_maze, tab_about = st.tabs(["Game tree", "Maze game", "About"])

with tab_tree:
    st.subheader("Minimax and alpha-beta on the lecture trees")
    c1, c2, c3 = st.columns([2, 1, 1])
    tree_name = c1.selectbox("Tree", list(TREES), index=1)
    algo = c2.radio("Algorithm", ["Minimax", "Alpha-beta"], horizontal=True)
    show_leaves = c3.checkbox("Show leaf values", True,
                              help="Off: leaves show ? until the search evaluates them (good for asking the class).")
    tree = TREES[tree_name]
    root_value, events = tree_trace(tree, alpha_beta=algo == "Alpha-beta")
    key = f"tree_{tree_name}_{algo}"
    step = st.slider("Execution step", 0, len(events) - 1, key=key)
    left, right = st.columns([2, 1])
    fig, pruned = draw_game_tree(tree, events, step, show_leaves)
    names = name_tree(tree)
    with left:
        st.plotly_chart(fig, width="stretch")
        st.caption("▲ MAX · ▼ MIN · ■ leaf · orange = current node and its path to the root · ✂ dotted grey = pruned "
                   "(never looked at). Letters a, b, c … are the node names used in the lecture quizzes.")
    with right:
        st.markdown(f"### Step {step} / {len(events) - 1}")
        st.markdown(describe(events[step], names))
        leaves_seen = sum(1 for e in events[:step + 1] if e["kind"] == "leaf")
        total_leaves = sum(1 for p in tree_layout(tree) if not isinstance(subtree(tree, p), list))
        st.metric("Leaves evaluated", f"{leaves_seen} / {total_leaves}")
        if pruned:
            st.write("**Pruned:** " + ", ".join(sorted(names[p] for p in pruned)))
        if step == len(events) - 1:
            st.success(f"Root value: {fmt(root_value)}")
        b1, b2 = st.columns(2)
        b1.button("◀ Previous", key="tree_prev", disabled=step == 0, on_click=shift_step, args=(key, -1))
        b2.button("Next ▶", key="tree_next", disabled=step >= len(events) - 1, on_click=shift_step, args=(key, 1))

with tab_maze:
    st.subheader("Maze chase: depth-limited search agent")
    c = st.columns(8)
    layout = c[0].selectbox("Layout", list(LAYOUTS), index=list(LAYOUTS).index("classic"))
    algorithm = c[1].selectbox("Player", ["Alpha-beta", "Minimax", "Expectimax"],
                               help="Expectimax treats ghosts as random (next lecture).")
    depth = c[2].slider("Depth", 1, 10, 3, help="Full rounds: every agent moves once per level.")
    evaluation = c[3].selectbox("Evaluation", ["score", "closer"],
                                help="score = game score only · closer = score − distance to the nearest dot")
    ghosts = c[4].selectbox("Ghosts", ["random", "true_random", "greedy", "smart"],
                            help="random = random at junctions, never turns back (CS188 rule) · true_random = any open "
                                 "direction each move, turning back included (the player's search still assumes no "
                                 "U-turns) · greedy = step toward the player · smart = ghosts run minimax (see ghost depth)")
    ghost_depth = c[5].slider("Ghost depth", 1, 4, 2, disabled=ghosts != "smart",
                              help="Search depth of smart ghosts")
    if "seed" not in st.session_state:
        st.session_state.seed = 0
    seed = c[6].number_input("Seed", 0, 9999, key="seed",
                             help="Same seed = same game (random ghosts and tie-breaks repeat). 🎲 picks a new one.")
    c[6].button("🎲 New random game", on_click=lambda: st.session_state.update(seed=random.randint(0, 9999)))
    max_moves = c[7].number_input("Max moves", 10, 1000, 400 if layout == "classic" else 150)
    kw = dict(layout=layout, depth=depth, evaluate=evaluation, alpha_beta=algorithm == "Alpha-beta",
              ghost_model="expect" if algorithm == "Expectimax" else "min", ghosts=ghosts, ghost_depth=ghost_depth,
              max_moves=int(max_moves), seed=int(seed))
    n_agents = GameState.from_layout(LAYOUTS[layout]).num_agents()
    top = st.columns([1.3, 1, 1, 1, 1, 2])
    tree_plies = top[5].slider("Search-tree levels to record", 1, min(9, depth * n_agents), min(6, depth * n_agents),
                               help=f"One level per move: A, then each ghost ({n_agents} levels = one round). "
                                    "Deeper trees are wider: scroll inside the tree panel.")
    frames, info = run_game(**kw, tree_plies=tree_plies)
    {"win": top[0].success, "lose": top[0].error, "timeout": top[0].warning}[info["result"]](f"Result: {info['result']}")
    top[1].metric("Final score", info["score"])
    top[2].metric("Player moves", info["moves"])
    top[3].metric("Dots left", info["dots_left"])
    k = info["nodes_per_move"]
    top[4].metric("Nodes / move", round(mean(k)) if k else 0)
    s0 = frames[0]
    cell = max(18, min(40, 760 // s0.w))
    components.html(html_player(frames, speed=8, trees=info["trees"], ghost_model=kw["ghost_model"], font=FONT,
                                tree_levels=n_agents),
                    height=cell * s0.h + 2 * FONT + 150)  # the player then resizes its frame to fit the tree
    st.caption("Green A = player (MAX) · numbered squares = ghosts (MIN) · orange = power pellets: ghosts turn "
               "blue (!) for 40 ghost moves and flash before they recover. Scoring: −1 per move, +10 per dot, "
               "+200 per scared ghost, +500 win, −500 caught. Turns: A, then ghost 1, then ghost 2; one agent jumps "
               "one cell per step (orange ring = who just moved). Ghosts never turn back unless stuck (as in CS188), "
               "so a random ghost only chooses at junctions. Tree: blue frame = A (MAX), red frame = ghost (MIN).")
    g1, g2 = st.columns([1, 3])
    n_games = g1.number_input("Games (seeds 0 … n−1)", 5, 100, 20, key="n_games")
    if g1.button("Run many games"):
        results = [play(**{**kw, "seed": i})[1] for i in range(int(n_games))]
        wins = sum(r["result"] == "win" for r in results)
        nodes = [n for r in results for n in r["nodes_per_move"]]
        g2.write(f"**Wins:** {wins} / {len(results)} · **timeouts:** "
                 f"{sum(r['result'] == 'timeout' for r in results)} · "
                 f"**average score:** {mean(r['score'] for r in results):.1f} · "
                 f"**average moves:** {mean(r['moves'] for r in results):.1f}" +
                 (f" · **nodes per move:** {mean(nodes):.0f}" if nodes else ""))
    with st.expander("Suggested lecture demos"):
        st.markdown("""
| Demo | Layout | Player | Depth | Evaluation | Ghosts |
|---|---|---|---|---|---|
| Hopeless: minimax dives into ghost 1 | trapped | Alpha-beta | 3 | score | random |
| Same, but ghosts modelled as random | trapped | Expectimax | 3 | score | random |
| Thrashing (two dots, depth 2) | line / room | Alpha-beta | 2 | score | – |
| Thrashing fixed | room | Alpha-beta | 2 | closer | – |
| Thrashing, open room (lecture video look) | open | Alpha-beta | 2 | score, then closer | – |
| Depth matters (no eval fix) | room | Alpha-beta | 2 vs 6 | score | – |
| Smart ghosts coordinate | arena | Alpha-beta | 2 | closer | smart, ghost depth 1 vs 3 |
| Full board with power pellets | classic | Alpha-beta | 3 | closer | random or greedy |
""")

with tab_about:
    st.subheader("What this prototype demonstrates")
    st.markdown("""
- **Game tree:** minimax values computed bottom-up, and alpha-beta pruning with α / β at every node,
  on the trees from the lecture (including both quizzes).
- **Maze chase:** a depth-limited search agent (minimax, alpha-beta, or expectimax) against ghosts that move
  randomly, greedily, or by running minimax themselves. Compare evaluation functions, depths and node counts.
  Power pellets scare the ghosts (they flee and can be eaten for +200), as in the lecture's Pacman.

The engine is `maze_game.py` (standard library only); the Colab exercise uses the same file.
This is a clean-room educational implementation with its own look, not a copy of Berkeley's Pacman project.
    """)
