"""Maze chase: a tiny multi-agent grid game for teaching adversarial search.

Same rules as the Pacman demos in the CS188 adversarial-search lecture, own code and look:
agent 0 (the player) collects dots, agents 1..n (ghosts) chase it.
Scoring: -1 per player move, +10 per dot, +500 for clearing the board, -500 when caught.
Power pellets ('o') scare every ghost for SCARED_TIME ghost moves; touching a scared ghost
sends it home for +200. Ghosts never reverse direction unless they hit a dead end.

Pure standard library, so the same file runs in Streamlit and in Colab.
Run `python maze_game.py` for the self-check.
"""
import random
from collections import deque

DIRS = {"North": (0, -1), "South": (0, 1), "East": (1, 0), "West": (-1, 0)}
REVERSE = {"North": "South", "South": "North", "East": "West", "West": "East"}
STEP, DOT, WIN, LOSE, EAT_GHOST = -1, 10, 500, -500, 200
SCARED_TIME = 40  # ghost moves, as in CS188

# '%' wall, '.' dot, 'o' power pellet, 'A' player, 'G' ghost (ghosts numbered 1, 2, ... in reading order)
LAYOUTS = {
    # Lecture "thrashing" slide: dot right next to the player (west), another two steps east.
    "line": [
        "%%%%%%",
        "%.A .%",
        "%%%%%%",
    ],
    # Lecture "hopeless" situation: ghost 1 can only come closer, ghost 2 picks up or down.
    "trapped": [
        "%%%%%%%%%%%%",
        "%   A G%%%%%",
        "% %%%%%%%%%%",
        "%G%%%%%%%%%%",
        "%....      %",
        "%%%%%%%%%%%%",
    ],
    # Open room without ghosts: dots far apart, so a depth-2 search sees no points (thrashing).
    "room": [
        "%%%%%%%%%%",
        "%.      .%",
        "% %%%%%% %",
        "%    A   %",
        "% %%%%%% %",
        "%.      .%",
        "%%%%%%%%%%",
    ],
    # Lecture "thrashing" video look: open room, both dots beyond a depth-2 horizon, so the player wanders.
    # (CS188's version has a dot right next to Pacman; without a Stop action ours would just eat it.)
    "open": [
        "%%%%%%%%%%%%",
        "%      A   %",
        "%          %",
        "%          %",
        "%          %",
        "%.        .%",
        "%%%%%%%%%%%%",
    ],
    # Small board with loops and two ghosts.
    "arena": [
        "%%%%%%%%%%%",
        "%....%....%",
        "%.%%...%%.%",
        "%....A....%",
        "%.%%.%.%%.%",
        "%G...%...G%",
        "%%%%%%%%%%%",
    ],
    # Full board with power pellets in the corners and a ghost pen in the middle.
    "classic": [
        "%%%%%%%%%%%%%%%%%%%",
        "%o.......%.......o%",
        "%.%%.%%%.%.%%%.%%.%",
        "%.................%",
        "%.%%.%.%%%%%.%.%%.%",
        "%....%.G   G.%....%",
        "%.%%.%.%%%%%.%.%%.%",
        "%........A........%",
        "%.%%.%%%.%.%%%.%%.%",
        "%o.......%.......o%",
        "%%%%%%%%%%%%%%%%%%%",
    ],
}


class GameState:
    """Immutable game state. Agent 0 is the player (MAX), agents 1.. are ghosts (MIN)."""
    __slots__ = ("walls", "dots", "pos", "dirs", "score", "status", "w", "h", "caps", "scared", "home")

    def __init__(self, walls, dots, pos, dirs, score=0, status="playing", w=0, h=0,
                 caps=frozenset(), scared=None, home=None):
        self.walls, self.dots, self.pos, self.dirs = walls, dots, pos, dirs
        self.score, self.status, self.w, self.h = score, status, w, h
        self.caps = caps                                  # power pellets left
        self.scared = scared or (0,) * len(pos)           # scared timer per agent (index 0 unused)
        self.home = home or pos                           # start positions: eaten ghosts go back here

    @staticmethod
    def from_layout(rows):
        walls, dots, caps, ghosts, player = set(), set(), set(), [], None
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == "%": walls.add((x, y))
                elif ch == ".": dots.add((x, y))
                elif ch == "o": caps.add((x, y))
                elif ch == "A": player = (x, y)
                elif ch == "G": ghosts.append((x, y))
        pos = (player, *ghosts)
        return GameState(frozenset(walls), frozenset(dots), pos, (None,) * len(pos),
                         w=max(map(len, rows)), h=len(rows), caps=frozenset(caps))

    def num_agents(self): return len(self.pos)
    def is_win(self): return self.status == "win"
    def is_lose(self): return self.status == "lose"
    def is_terminal(self): return self.status != "playing"
    def player_pos(self): return self.pos[0]
    def ghost_positions(self): return list(self.pos[1:])
    def is_scared(self, ghost): return self.scared[ghost] > 0

    def legal_actions(self, agent=0):
        if self.is_terminal():
            return []
        x, y = self.pos[agent]
        moves = [a for a, (dx, dy) in DIRS.items() if (x + dx, y + dy) not in self.walls]
        if agent > 0 and self.dirs[agent] and len(moves) > 1:  # ghosts keep going: no U-turn
            moves = [a for a in moves if a != REVERSE[self.dirs[agent]]]
        return moves

    def successor(self, agent, action, u_turn=False):
        """u_turn=True lets a ghost turn back (only the "true_random" ghost policy does that)."""
        ok = action in self.legal_actions(agent) or (
            u_turn and not self.is_terminal() and action in DIRS
            and (self.pos[agent][0] + DIRS[action][0], self.pos[agent][1] + DIRS[action][1]) not in self.walls)
        assert ok, f"illegal {action} for agent {agent}"
        dx, dy = DIRS[action]
        x, y = self.pos[agent]
        pos = list(self.pos); pos[agent] = (x + dx, y + dy)
        dirs = list(self.dirs); dirs[agent] = action
        dots, caps, scared = self.dots, self.caps, list(self.scared)
        score, status = self.score, "playing"
        if agent == 0:
            score += STEP
            if pos[0] in dots:
                dots = dots - {pos[0]}
                score += DOT
                if not dots:
                    score, status = score + WIN, "win"
            if pos[0] in caps:
                caps = caps - {pos[0]}
                scared = [0] + [SCARED_TIME] * (len(pos) - 1)
        elif scared[agent]:
            scared[agent] -= 1
        for g in range(1, len(pos)):
            if status == "playing" and pos[g] == pos[0]:
                if scared[g]:                              # player eats a scared ghost
                    score += EAT_GHOST
                    pos[g], dirs[g], scared[g] = self.home[g], None, 0
                else:
                    score, status = score + LOSE, "lose"
        return GameState(self.walls, dots, tuple(pos), tuple(dirs), score, status, self.w, self.h,
                         caps, tuple(scared), self.home)

    def __repr__(self):
        return "\n".join(self.ascii_rows())

    def ascii_rows(self):
        grid = [[" "] * self.w for _ in range(self.h)]
        for x, y in self.walls: grid[y][x] = "%"
        for x, y in self.dots: grid[y][x] = "."
        for x, y in self.caps: grid[y][x] = "o"
        for i, (x, y) in enumerate(self.pos[1:], 1): grid[y][x] = "s" if self.scared[i] else str(i)
        x, y = self.pos[0]; grid[y][x] = "X" if self.is_lose() else "A"
        return ["".join(r) for r in grid]


# ---------- maze distances (BFS from every cell, cached per wall set) ----------
_DIST = {}

def maze_distance(state, a, b):
    if state.walls not in _DIST:
        cells = [(x, y) for y in range(state.h) for x in range(state.w) if (x, y) not in state.walls]
        table = {}
        for s in cells:
            d, q = {s: 0}, deque([s])
            while q:
                x, y = q.popleft()
                for dx, dy in DIRS.values():
                    n = (x + dx, y + dy)
                    if n not in state.walls and n not in d:
                        d[n] = d[(x, y)] + 1; q.append(n)
            table[s] = d
        _DIST[state.walls] = table
    return _DIST[state.walls][a].get(b, 999)


# ---------- evaluation functions (from the player's point of view: higher = better) ----------
def score_eval(state):
    """Naive: the game score only."""
    return state.score

def closer_eval(state):
    """Lecture fix for thrashing: score plus a bonus for being close to the nearest dot."""
    if state.is_terminal() or not state.dots:
        return state.score
    return state.score - min(maze_distance(state, state.pos[0], d) for d in state.dots)

def chase_eval(state):
    """Ghosts' view (ghosts minimize this): close to the player is good for them, unless they are scared."""
    if state.is_terminal():
        return state.score
    return state.score + sum(maze_distance(state, g, state.pos[0]) * (-1 if state.scared[i] else 1)
                             for i, g in enumerate(state.pos[1:], 1))

EVALS = {"score": score_eval, "closer": closer_eval, "chase": chase_eval}


# ---------- depth-limited minimax / alpha-beta / expectimax ----------
def search(state, depth, evaluate, agent=0, alpha_beta=False, ghost_model="min", rng=None, tree=None,
           tree_plies=3):
    """Choose an action for `agent` by depth-limited search.

    depth counts full rounds (every agent moves once). Player = MAX, ghosts = MIN
    (ghost_model="min") or uniformly random chance nodes (ghost_model="expect").
    Returns (action, value, nodes_expanded). Ties at the root are broken by `rng`.
    tree: pass a dict to also get the top `tree_plies` plies of the search tree. Every node is a dict:
    a = action into it, v = value, b = "≤"/"≥" when v is only a bound (alpha-beta cut there),
    p = 1 when pruned (never searched), c = children, x = 1 on the root child that was chosen,
    q = agent positions in that node (s = scared flags) so the node's board can be drawn.
    """
    n = state.num_agents()
    nodes = 0

    def mark(rec, s):   # board of a recorded node: q = agent positions, s = scared flags
        rec["q"] = s.pos
        if any(s.scared):
            rec["s"] = [int(t > 0) for t in s.scared]
        return rec

    def value(s, who, d, a, b, rec, rp):
        nonlocal nodes
        if rec is not None:
            mark(rec, s)
        if s.is_terminal() or d == 0:
            v = evaluate(s)
            if rec is not None: rec["v"] = v
            return v
        nodes += 1
        nxt, nd = (who + 1) % n, d - (1 if (who + 1) % n == 0 else 0)
        acts = s.legal_actions(who)
        keep = rec is not None and rp > 0
        if keep: rec["c"] = []

        def kid(act):
            if not keep: return None
            c = {"a": act}; rec["c"].append(c); return c

        def done(v, i=None, bound=""):
            if rec is not None:
                rec["v"] = v
                if bound: rec["b"] = bound
                if keep and i is not None:
                    rec["c"] += [mark({"a": x, "p": 1}, s.successor(who, x)) for x in acts[i + 1:]]
            return v
        if who == 0:
            v = float("-inf")
            for i, act in enumerate(acts):
                v = max(v, value(s.successor(who, act), nxt, nd, a, b, kid(act), rp - 1))
                if alpha_beta:
                    if v >= b: return done(v, i, "≥")
                    a = max(a, v)
            return done(v)
        if ghost_model == "expect":
            return done(sum(value(s.successor(who, act), nxt, nd, a, b, kid(act), rp - 1) for act in acts) / len(acts))
        v = float("inf")
        for i, act in enumerate(acts):
            v = min(v, value(s.successor(who, act), nxt, nd, a, b, kid(act), rp - 1))
            if alpha_beta:
                if v <= a: return done(v, i, "≤")
                b = min(b, v)
        return done(v)

    rng = rng or random.Random(0)
    nodes += 1
    nxt = (agent + 1) % n
    nd = depth - (1 if nxt == 0 else 0)
    scored, kids = [], []
    a, b = float("-inf"), float("inf")
    for act in state.legal_actions(agent):
        rec = {"a": act} if tree is not None and tree_plies > 0 else None
        v = value(state.successor(agent, act), nxt, nd, a, b, rec, tree_plies - 1)
        scored.append((v, act)); kids.append(rec)
        # Root bound sits just below (above) the best value so far: children that tie with it
        # still get their exact value, and random tie-breaking stays correct (lecture: root ties).
        if alpha_beta:
            if agent == 0: a = max(a, v - 1e-9)
            else: b = min(b, v + 1e-9)
    best = max(v for v, _ in scored) if agent == 0 else min(v for v, _ in scored)
    action = rng.choice([act for v, act in scored if v == best])
    if tree is not None:
        tree.update(v=best, c=[k for k in kids if k is not None])
        for k in tree["c"]:
            if k["a"] == action: k["x"] = 1
    return action, best, nodes


# ---------- ghost behaviours for actual play ----------
def ghost_action(state, i, policy, rng, depth=2):
    acts = state.legal_actions(i)
    if policy == "true_random":  # any open direction, turning back included (ignores the no-U-turn rule)
        x, y = state.pos[i]
        return rng.choice([a for a, (dx, dy) in DIRS.items() if (x + dx, y + dy) not in state.walls])
    if policy == "random":
        return rng.choice(acts)
    if policy == "greedy":  # step toward the player (away from it while scared)
        sign = -1 if state.scared[i] else 1
        dist = {a: sign * maze_distance(state, state.successor(i, a).pos[i], state.pos[0]) for a in acts}
        m = min(dist.values())
        return rng.choice([a for a in acts if dist[a] == m])
    if policy == "smart":  # ghosts run minimax too, with "close to the player is good"
        return search(state, depth, chase_eval, agent=i, alpha_beta=True, rng=rng)[0]
    raise ValueError(policy)


def play(layout, depth=2, evaluate="score", alpha_beta=True, ghost_model="min",
         ghosts="random", ghost_depth=2, max_moves=150, seed=0, player=None, tree_plies=0):
    """Play one game. Returns (frames, info): frames is the list of states after every move.

    `player(state) -> action` overrides the built-in search player (used by the exercise).
    tree_plies > 0: info["trees"] holds the top of the player's search tree for every player move."""
    rng = random.Random(seed)
    state = GameState.from_layout(LAYOUTS[layout] if isinstance(layout, str) else layout)
    ev = EVALS[evaluate] if isinstance(evaluate, str) else evaluate
    frames, nodes, moves, trees = [state], [], 0, []
    while not state.is_terminal() and moves < max_moves:
        for i in range(state.num_agents()):
            if state.is_terminal():
                break
            if i == 0:
                if player:
                    act = player(state)
                else:
                    t = {} if tree_plies else None
                    act, _, k = search(state, depth, ev, 0, alpha_beta, ghost_model, rng, t, tree_plies)
                    nodes.append(k)
                    if t is not None: trees.append(t)
                moves += 1
            else:
                act = ghost_action(state, i, ghosts, rng, ghost_depth)
            state = state.successor(i, act, u_turn=i > 0 and ghosts == "true_random")
            frames.append(state)
    result = state.status if state.is_terminal() else "timeout"
    return frames, {"result": result, "score": state.score, "moves": moves,
                    "dots_left": len(state.dots), "nodes_per_move": nodes, "trees": trees}


# ---------- live animation: one self-contained HTML/JS player for Jupyter/Colab and Streamlit ----------
_PLAYER_JS = r"""
(function () {
  const D = __DATA__, id = "__ID__", C = D.cell, N = D.F.length - 1, n = D.na, FS = D.fs, S = FS / 16;
  const $ = s => document.getElementById(id + s);
  const cv = $("c"), g = cv.getContext("2d");
  const GC = ["#ef4444", "#a855f7", "#06b6d4", "#ec4899"];
  const AR = {North: "↑", South: "↓", East: "→", West: "←"};
  const fmtv = v => String(Math.round(v * 10) / 10).replace("-", "−");
  const layers = {};
  function mazeLayer(cs, grid) {             // walls (and the grid) pre-rendered once per cell size
    const key = cs + "," + grid;
    if (layers[key]) return layers[key];
    const c = document.createElement("canvas"); c.width = D.w * cs; c.height = D.h * cs;
    const b = c.getContext("2d"), r = cs * 0.3, W = new Set(D.walls.map(([x, y]) => x + "," + y));
    b.fillStyle = "#0b1020"; b.fillRect(0, 0, c.width, c.height); b.fillStyle = "#2b6cb0";
    for (const [x, y] of D.walls) {
      const cx = (x + .5) * cs, cy = (y + .5) * cs;
      b.beginPath(); b.arc(cx, cy, r, 0, 7); b.fill();
      if (W.has((x + 1) + "," + y)) b.fillRect(cx, cy - r, cs, 2 * r);
      if (W.has(x + "," + (y + 1))) b.fillRect(cx - r, cy, 2 * r, cs);
      if (W.has((x + 1) + "," + y) && W.has(x + "," + (y + 1)) && W.has((x + 1) + "," + (y + 1))) b.fillRect(cx, cy, cs, cs);
    }
    if (grid) {                              // semi-transparent grid: agents jump from cell to cell
      b.strokeStyle = "rgba(148,163,184,.22)"; b.lineWidth = 1; b.beginPath();
      for (let x = 0; x <= D.w; x++) { b.moveTo(x * cs + .5, 0); b.lineTo(x * cs + .5, c.height); }
      for (let y = 0; y <= D.h; y++) { b.moveTo(0, y * cs + .5); b.lineTo(c.width, y * cs + .5); }
      b.stroke();
    }
    return layers[key] = c;
  }
  function rr(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.fill(); }
  function text(ctx, s, x, y, col, px, align) { ctx.fillStyle = col; ctx.font = "bold " + px + "px system-ui";
    ctx.textAlign = align || "center"; ctx.textBaseline = "middle"; ctx.fillText(s, x, y + 1); }
  // one board at cell size cs: dots (dotOn(i)), pellets, ghosts, player
  function board(ctx, x0, y0, cs, pos, scared, dotOn, capOn, opts) {
    ctx.save();
    if (opts.crop) { const [cx, cy, cw, ch] = opts.crop;     // show only this window of the board
      ctx.beginPath(); ctx.rect(x0, y0, cw * cs, ch * cs); ctx.clip(); ctx.translate(x0 - cx * cs, y0 - cy * cs); }
    else ctx.translate(x0, y0);
    ctx.drawImage(mazeLayer(cs, opts.grid), 0, 0);
    ctx.fillStyle = "#fde68a";
    D.dots.forEach(([x, y], i) => { if (dotOn(i)) { ctx.beginPath(); ctx.arc((x + .5) * cs, (y + .5) * cs, Math.max(1, cs * .11), 0, 7); ctx.fill(); } });
    ctx.fillStyle = "#fb923c";
    D.caps.forEach(([x, y], i) => { if (capOn(i)) { ctx.beginPath(); ctx.arc((x + .5) * cs, (y + .5) * cs, cs * (opts.pulse || .26), 0, 7); ctx.fill(); } });
    const lbl = cs >= 16, px = Math.round(Math.max(cs * .45, FS * .9));
    for (let j = 1; j < pos.length; j++) {
      const [x, y] = pos[j], h = cs * .8, sc = scared && scared[j] > 0;
      ctx.fillStyle = sc ? (opts.flash && scared[j] <= 8 ? "#f1f5f9" : "#60a5fa") : GC[(j - 1) % 4];
      rr(ctx, (x + .5) * cs - h / 2, (y + .5) * cs - h / 2, h, h, cs * .22);
      if (lbl) text(ctx, sc ? "!" : String(j), (x + .5) * cs, (y + .5) * cs, sc ? "#0b1020" : "#fff", px);
    }
    const [x, y] = pos[0];
    ctx.fillStyle = opts.dead ? "#94a3b8" : "#22c55e";
    ctx.beginPath(); ctx.arc((x + .5) * cs, (y + .5) * cs, cs * .42, 0, 7); ctx.fill();
    if (lbl) text(ctx, "A", (x + .5) * cs, (y + .5) * cs, "#fff", px);
    ctx.restore();
  }
  let k = 0, playing = D.autoplay && N > 0, acc = 0, last = null, treeK = -1, treeKey = "";
  let Z = 1, zNow = 1;                       // tree zoom: a factor, or null = fit the panel's width
  function draw(now) {
    const A = D.F[k];
    board(g, 0, 0, C, A.p, A.s, i => A.d[i] === "1", i => A.c[i] === "1",
          {grid: true, dead: A.st === "lose", flash: Math.floor(now / 180) % 2, pulse: .24 + .05 * Math.sin(now / 160)});
    if (k > 0) {                              // ring around the agent that just moved
      const [x, y] = A.p[(k - 1) % n]; g.strokeStyle = "#f59e0b"; g.lineWidth = 3;
      g.beginPath(); g.arc((x + .5) * C, (y + .5) * C, C * .55, 0, 7); g.stroke();
    }
    if (k === N && D.result) {
      g.fillStyle = "rgba(11,16,32,.65)"; g.fillRect(0, cv.height / 2 - C * .8, cv.width, C * 1.6);
      text(g, D.result, cv.width / 2, cv.height / 2, D.result === "WIN" ? "#4ade80" : D.result === "CAUGHT" ? "#f87171" : "#fbbf24", Math.round(C * .8));
    }
    $("i").textContent = (D.title ? D.title + "  ·  " : "") +
      (N ? "move " + k + "/" + N + "  ·  round " + Math.floor(k / n) + "  ·  " +
           (A.st === "playing" ? "next: " + (k % n ? "ghost " + k % n : "A") + "  ·  " : "") : "") +
      "score " + A.sc + (A.st !== "playing" ? "  ·  " + A.st : "");
    if (N) { $("s").value = k; $("p").textContent = playing ? "⏸" : "▶"; }
    if (D.T) {
      const key = $("l").value + "," + $("bd").checked + "," + Z;
      if ($("t").checked) { if (treeK !== k || treeKey !== key) drawTree(k); }
      else {                                  // frozen: zoom/levels still redraw the frozen tree
        if (treeK >= 0 && treeKey !== key) drawTree(treeK);
        if (treeK !== k) $("h").textContent = treeK < 0 ? "Tick the box to draw the search tree." :
          "⏸ Tree frozen at move " + treeK + ". Tick the box to follow the game again.";
      }
    }
  }
  function drawTree(k) {                     // the player's tree for move k's round; green = moves actually made
    const r = Math.floor(k / n), j = k % n, T = D.T[r], tc = $("T"), tg = tc.getContext("2d");
    const L0 = +$("l").value, boards = $("bd").checked;
    treeK = k; treeKey = L0 + "," + boards + "," + Z;
    if (!T) { tc.width = 400; tc.height = 50; tg.fillStyle = "#0f172a"; tg.fillRect(0, 0, 400, 50);
      text(tg, "Game over: no search here.", 10, 25, "#94a3b8", FS, "left"); $("h").textContent = ""; return; }
    const path = [T]; let off = "";
    for (let f = r * n + 1; f <= k; f++) {
      const mv = D.F[f].m, nd = path[path.length - 1], c = nd.c && nd.c.find(c => c.a === mv);
      if (!c) { off = nd.c && nd.c.length ? " · ⚠ this ghost move is not in the tree: the player's model assumes ghosts never turn back"
                                          : " · (this move is below the stored levels)"; break; }
      path.push(c);
    }
    const on = new Set(path), cur = path[path.length - 1], L = Math.max(L0, path.length - 1);
    let leaves = 0;
    (function lay(nd, d) { const kids = d < L && nd.c && nd.c.length ? nd.c : null;
      nd._d = d; nd._k = kids;
      if (!kids) nd._x = leaves++;
      else { kids.forEach(c => lay(c, d + 1)); nd._x = (kids[0]._x + kids[kids.length - 1]._x) / 2; } })(T, 0);
    // all node boards show the same window: where the agents are in the drawn part of the tree (+1 cell)
    let x1 = 1e9, y1 = 1e9, x2 = -1, y2 = -1;
    (function box(nd, pos) { if (nd.q) pos = nd.q;
      for (const [x, y] of pos) { x1 = Math.min(x1, x); y1 = Math.min(y1, y); x2 = Math.max(x2, x); y2 = Math.max(y2, y); }
      if (nd._k) nd._k.forEach(c => box(c, pos)); })(T, D.F[r * n].p);
    x1 = Math.max(0, x1 - 1); y1 = Math.max(0, y1 - 1); x2 = Math.min(D.w - 1, x2 + 1); y2 = Math.min(D.h - 1, y2 + 1);
    const crop = [x1, y1, x2 - x1 + 1, y2 - y1 + 1];
    const cs = boards ? Math.max(4, Math.min(22, Math.floor(150 / Math.max(crop[2], crop[3] * 1.4)))) : 0;
    const bw = cs * crop[2], bh = cs * crop[3];
    const nw = boards ? bw + 12 : 78 * S, nh = boards ? bh + 12 + FS + 10 : 44 * S;
    const GX = nw + 22 * S, GY = nh + 56 * S, LW = 130 * S;
    const W = LW + leaves * GX + 20, H = 24 + L * GY + nh + 24;
    const sb = tc.parentElement, cw = sb.clientWidth;   // zoom: draw scaled, so text stays sharp
    const z = zNow = Z !== null ? Z : cw > 50 ? Math.min(2, (cw - 2) / W) : 1;
    tc.width = Math.round(W * z); tc.height = Math.round(H * z); tg.setTransform(z, 0, 0, z, 0, 0);
    $("zl").textContent = Math.round(z * 100) + "%";
    const X = nd => LW + (nd._x + .5) * GX, Y = nd => 16 + nd._d * GY + nh / 2;
    tg.fillStyle = "#0f172a"; tg.fillRect(0, 0, W, H);
    for (let d = 0; d <= L; d++) { const who = d % n;
      text(tg, (who === 0 ? "A · MAX" : "ghost " + who + (D.chance ? " · chance" : " · MIN")) + (d === path.length - 1 ? " ◀" : ""),
           6, 16 + d * GY + nh / 2, d === path.length - 1 ? "#4ade80" : "#94a3b8", FS, "left"); }
    (function edges(nd) { if (!nd._k) return; for (const c of nd._k) {
      const x1 = X(nd), y1 = Y(nd) + nh / 2, x2 = X(c), y2 = Y(c) - nh / 2, act = on.has(c);
      tg.setLineDash(c.p ? [6, 5] : []);
      tg.strokeStyle = act ? "#22c55e" : c.x ? "#f59e0b" : c.p ? "#475569" : (nd._d % n === 0 ? "#3b82f6" : "#ef4444");
      tg.lineWidth = act || c.x ? 4.5 : 2.5; tg.beginPath(); tg.moveTo(x1, y1); tg.lineTo(x2, y2); tg.stroke();
      tg.setLineDash([]);
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, q = 14 * S;
      tg.fillStyle = "#0f172a"; tg.fillRect(mx - q, my - q, 2 * q, 2 * q);
      text(tg, c.p && !act ? "✂" : (AR[c.a] || c.a), mx, my, act ? "#4ade80" : c.p ? "#f87171" : c.x ? "#fbbf24" : "#e2e8f0", FS);
      edges(c); } })(T);
    const F0 = D.F[r * n];
    (function nodes(nd, pos, sc, seen) {
      if (nd.q) { pos = nd.q; sc = nd.s || null; seen = seen.concat([pos[0][0] + "," + pos[0][1]]); }
      const x = X(nd), y = Y(nd), who = nd._d % n, now = nd === cur;
      const border = who === 0 ? "#3b82f6" : D.chance ? "#a855f7" : "#ef4444", val = (nd.b || "") + (nd.v === undefined ? "" : fmtv(nd.v));
      if (boards) {
        const x0 = x - nw / 2, y0 = y - nh / 2;
        if (now || nd.x || nd._d === 0) { tg.fillStyle = now ? "#22c55e" : "#f59e0b"; tg.fillRect(x0 - 5, y0 - 5, nw + 10, bh + 22); }
        tg.fillStyle = border; tg.fillRect(x0, y0, nw, bh + 12);
        const eaten = new Set(seen);
        board(tg, x0 + 6, y0 + 6, cs, pos, sc, i => F0.d[i] === "1" && !eaten.has(D.dots[i][0] + "," + D.dots[i][1]),
              i => F0.c[i] === "1" && !eaten.has(D.caps[i][0] + "," + D.caps[i][1]), {grid: cs >= 10, crop});
        if (nd.p) { tg.fillStyle = "rgba(15,23,42,.7)"; tg.fillRect(x0, y0, nw, bh + 12);
          text(tg, "✂", x, y0 + (bh + 12) / 2, "#f87171", Math.round(FS * 1.4)); }
        else text(tg, val, x, y0 + bh + 12 + (FS + 10) / 2, "#f8fafc", FS);
      } else {
        tg.beginPath();
        tg.fillStyle = nd.p ? "#334155" : who === 0 ? "#fde68a" : "#f8fafc";
        tg.strokeStyle = now ? "#22c55e" : nd.p ? "#475569" : nd.x || nd._d === 0 ? "#f59e0b" : "#334155";
        const w = nw, h = nh;
        if (who === 0) { tg.moveTo(x, y - h / 2 - 2); tg.lineTo(x + w / 2, y + h / 2); tg.lineTo(x - w / 2, y + h / 2); }
        else if (D.chance) tg.arc(x, y, h / 2, 0, 7);
        else { tg.moveTo(x - w / 2, y - h / 2); tg.lineTo(x + w / 2, y - h / 2); tg.lineTo(x, y + h / 2 + 2); }
        tg.closePath(); tg.fill(); tg.lineWidth = now ? 4.5 : nd.x || nd._d === 0 ? 3 : 1.25; tg.stroke();
        if (!nd.p) text(tg, val, x, who === 0 ? y + 8 * S : D.chance ? y : y - 7 * S, "#0f172a", FS);
      }
      if (!nd._k && nd.c && nd.c.length) text(tg, "⋯", x, y + nh / 2 + 12 * S, "#94a3b8", FS);
      if (nd._k) nd._k.forEach(c => nodes(c, pos, sc, seen));
    })(T, F0.p, F0.s, [F0.p[0][0] + "," + F0.p[0][1]]);
    sb.scrollLeft = Math.max(0, X(cur) * z - cw / 2);   // keep the current node in view (sideways scroll only)
    fit();
    const ch = T.c.find(c => c.x);
    $("h").textContent = "Round " + r + ": the player searched this tree" +
      (j === 0 ? (ch ? " and picks " + ch.a + " (value " + fmtv(T.v) + ")" : "") :
       " · now ghost " + j + "'s turn · green = moves actually made") +
      (cur.p ? " · ✂ the game went into a branch that alpha-beta pruned (never searched)" : "") + off +
      " · ≤ / ≥ = only a bound · ⋯ = deeper levels";
  }
  function fit() {                           // Streamlit: grow/shrink the frame to the taller of board and tree
    try { const fe = window.frameElement;
      if (!fe) return;
      const b = $("r").getBoundingClientRect().bottom;
      if (b < 50) return;                    // hidden (e.g. an inactive Streamlit tab): measure later
      const h = Math.ceil(b + 12) + "px";
      fe.style.height = h;
      const pc = fe.parentElement;            // Streamlit's wrapper has its own fixed height
      if (pc && pc.classList.contains("stElementContainer")) pc.style.height = h;
    } catch (e) {}
  }
  function loop(now) {                       // discrete steps: one agent jumps one cell per step
    if (playing) {
      acc += (last === null ? 0 : (now - last) / 1000) * +$("v").value;
      while (acc >= 1 && k < N) { k++; acc -= 1; }
      if (k >= N) { playing = false; acc = 0; }
    }
    last = now; draw(now); requestAnimationFrame(loop);
  }
  if (N) {
    $("p").onclick = () => { if (k === N) k = 0; playing = !playing; acc = 0; };
    $("b").onclick = () => { k = 0; playing = false; };
    $("m").onclick = () => { k = Math.max(0, k - 1); playing = false; };
    $("n").onclick = () => { k = Math.min(N, k + 1); playing = false; };
    $("s").oninput = e => { k = +e.target.value; playing = false; };
  }
  requestAnimationFrame(loop);
  setTimeout(fit, 50);
  // when the frame becomes visible or changes size: redraw the tree (re-centres it) and refit the frame
  if (D.T) {
    $("zo").onclick = () => { Z = Math.max(.1, (Z ?? zNow) / 1.25); };
    $("zi").onclick = () => { Z = Math.min(4, (Z ?? zNow) * 1.25); };
    $("zf").onclick = () => { Z = null; };
    $("z1").onclick = () => { Z = 1; };
    const tc = $("tc");   // ⛶: tree in full screen; where that is blocked, tree under the board at full width
    const wide = () => { tc.style.flexBasis = tc.style.flexBasis === "100%" ? "420px" : "100%"; Z = null; };
    $("fs").onclick = () => document.fullscreenElement ? document.exitFullscreen() :
      (document.fullscreenEnabled ? tc.requestFullscreen() : Promise.reject()).then(() => { Z = null; }, wide);
    document.addEventListener("fullscreenchange", () => { treeKey = ""; });
  }
  try { new ResizeObserver(() => { treeKey = ""; fit(); }).observe($("r")); } catch (e) {}
})();
"""


def html_player(frames, title="", speed=8, autoplay=False, trees=None, ghost_model="min", tree_levels=None,
                font=16, boards=True):
    """Animate a game (list of states, e.g. from play()) as self-contained HTML + JavaScript.

    One agent moves per step, cell to cell on a grid, in turn order (A, ghost 1, ghost 2, ...); ▶ starts it.
    trees (info["trees"] from play(..., tree_plies=...)): the player's search tree is drawn beside the board,
    every node showing its board (boards=True) and value; a checkbox freezes the drawing while unticked.
    Jupyter/Colab: IPython.display.HTML(html_player(frames)). Streamlit: components.html(...)."""
    import json, uuid
    s0, n = frames[0], frames[0].num_agents()

    def moved(f):   # the action that led into frame f (None for the start or a ghost sent home)
        if f == 0:
            return None
        j = (f - 1) % n
        (x0, y0), (x1, y1) = frames[f - 1].pos[j], frames[f].pos[j]
        return next((a for a, d in DIRS.items() if d == (x1 - x0, y1 - y0)), None)
    dots, caps = sorted(s0.dots), sorted(s0.caps)
    last = frames[-1]
    data = dict(cell=max(18, min(40, 760 // s0.w)), fs=font, title=title, autoplay=autoplay and len(frames) > 1,
                w=s0.w, h=s0.h, walls=sorted(s0.walls), dots=dots, caps=caps, na=n,
                result={"win": "WIN", "lose": "CAUGHT"}.get(last.status, "TIME UP" if len(frames) > 1 else ""),
                F=[dict(p=s.pos, s=s.scared, sc=s.score, st=s.status, m=moved(i),
                        d="".join("1" if d in s.dots else "0" for d in dots),
                        c="".join("1" if c in s.caps else "0" for c in caps)) for i, s in enumerate(frames)])

    def rounded(t):
        t = dict(t)
        if "v" in t:
            t["v"] = round(t["v"], 1)
        if "c" in t:
            t["c"] = [rounded(c) for c in t["c"]]
        return t
    if trees:
        data.update(T=[rounded(t) for t in trees], chance=ghost_model == "expect")
    uid = "mz" + uuid.uuid4().hex[:10]
    w, h = data["cell"] * s0.w, data["cell"] * s0.h
    btn = f'style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:2px 10px;cursor:pointer;font-size:{font}px;font-weight:600"'
    row = f'style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px;font-size:{font}px;font-weight:600"'
    controls = "" if len(frames) == 1 else f"""
<div {row}>
 <button id="{uid}b" {btn}>⏮</button><button id="{uid}m" {btn}>◀</button>
 <button id="{uid}p" {btn}>▶</button><button id="{uid}n" {btn}>▶|</button>
 <input id="{uid}s" type="range" min="0" max="{len(frames) - 1}" value="0" style="flex:1;min-width:160px">
 <select id="{uid}v" {btn}>{"".join(f'<option{" selected" if v == speed else ""}>{v}</option>' for v in (1, 2, 4, 8, 16, 32))}</select>
 moves/s</div>"""
    plies = max((_plies(t) for t in trees), default=1) if trees else 0
    lv = min(tree_levels or n, plies)
    tree_col = "" if not trees else f"""
<style>#{uid}tc:fullscreen{{overflow:auto;padding:12px;background:#0b1020}}</style>
<div id="{uid}tc" style="flex:1 1 420px;min-width:320px;max-width:100%">
 <div {row} style="margin-top:0">
  <label style="cursor:pointer"><input id="{uid}t" type="checkbox" checked> Show search tree</label>
  <span>levels <select id="{uid}l" {btn}>{"".join(f'<option{" selected" if v == lv else ""}>{v}</option>' for v in range(1, plies + 1))}</select></span>
  <label style="cursor:pointer"><input id="{uid}bd" type="checkbox" {"checked" if boards else ""}> boards in nodes</label>
  <span>zoom <button id="{uid}zo" {btn} title="zoom out">−</button> <span id="{uid}zl">100%</span>
   <button id="{uid}zi" {btn} title="zoom in">+</button> <button id="{uid}zf" {btn} title="whole tree in the panel's width">fit</button>
   <button id="{uid}z1" {btn} title="actual size">1:1</button>
   <button id="{uid}fs" {btn} title="tree in full screen (Esc to leave), or under the board at full width">⛶</button></span>
 </div>
 <div id="{uid}h" style="font-size:{font}px;font-weight:600;color:#cbd5e1;margin:8px 0 6px"></div>
 <div style="overflow-x:auto;overflow-y:hidden;border-radius:8px;background:#0f172a"><canvas id="{uid}T" width="400" height="50"></canvas></div>
</div>"""
    js = _PLAYER_JS.replace("__DATA__", json.dumps(data).replace("</", "<\\/")).replace("__ID__", uid)
    return f"""<div id="{uid}r" style="font-family:system-ui,sans-serif;display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;
background:#0b1020;color:#e2e8f0;padding:10px;border-radius:12px;max-width:100%;box-sizing:border-box">
<div style="flex:0 1 auto;max-width:100%"><div id="{uid}i" style="font-size:{font}px;font-weight:600;margin-bottom:6px"></div>
<canvas id="{uid}c" width="{w}" height="{h}" style="max-width:100%;display:block"></canvas>{controls}</div>{tree_col}</div>
<script>{js}</script>"""


def _plies(t):
    return 0 if not t.get("c") else 1 + max(_plies(c) for c in t["c"])


# ---------- abstract game trees from the lecture (for minimax / alpha-beta by hand) ----------
# Nested lists, leaves are numbers, root is MAX and levels alternate.
# Nodes are named a, b, c, ... in pre-order (root unnamed), as in the lecture quizzes.
TREES = {
    "Minimax example (8 2 | 5 6)": [[8, 2], [5, 6]],
    "Lecture tree (3 12 8 | 2 4 6 | 14 5 2)": [[3, 12, 8], [2, 4, 6], [14, 5, 2]],
    "Properties (10 | 10 0)": [[10], [10, 0]],
    "Quiz 1": [[10, 8], [4, 50]],
    "Quiz 2": [[[10, 6], [100, 8]], [[1, 2], [20, 4]]],
}

NODE_NAMES = list("abcdefghijklmnopqrstuvwxyz") + [f"n{i}" for i in range(2000)]

def name_tree(tree):
    """Return {path: name} with pre-order letters; path = tuple of child indices."""
    names, k = {(): "root"}, 0
    def walk(t, path):
        nonlocal k
        if isinstance(t, list):
            for i, c in enumerate(t):
                names[path + (i,)] = NODE_NAMES[k]; k += 1
                walk(c, path + (i,))
    walk(tree, ())
    return names

def tree_trace(tree, alpha_beta=False):
    """Run minimax (or alpha-beta, pruning on <= / >= as in the lecture) on an abstract tree.

    Returns (root_value, events). Each event: dict(kind, path, value, alpha, beta), kind in
    enter / leaf / return / prune; 'prune' lists the paths of the skipped children."""
    events = []
    def value(t, path, is_max, a, b):
        if not isinstance(t, list):
            events.append(dict(kind="leaf", path=path, value=t, alpha=a, beta=b))
            return t
        events.append(dict(kind="enter", path=path, value=None, alpha=a, beta=b))
        v = float("-inf") if is_max else float("inf")
        for i, c in enumerate(t):
            cv = value(c, path + (i,), not is_max, a, b)
            v = max(v, cv) if is_max else min(v, cv)
            if alpha_beta and ((is_max and v >= b) or (not is_max and v <= a)):
                skipped = [path + (j,) for j in range(i + 1, len(t))]
                if skipped:
                    events.append(dict(kind="prune", path=path, value=v, alpha=a, beta=b, skipped=skipped))
                break
            if alpha_beta:
                if is_max: a = max(a, v)
                else: b = min(b, v)
        events.append(dict(kind="return", path=path, value=v, alpha=a, beta=b))
        return v
    return value(tree, (), True, float("-inf"), float("inf")), events

def pruned_names(tree):
    names = name_tree(tree)
    _, ev = tree_trace(tree, alpha_beta=True)
    return sorted(names[p] for e in ev if e["kind"] == "prune" for p in e["skipped"])


if __name__ == "__main__":
    # game trees: values and pruned nodes from the lecture
    assert tree_trace(TREES["Minimax example (8 2 | 5 6)"])[0] == 5
    t = TREES["Lecture tree (3 12 8 | 2 4 6 | 14 5 2)"]
    assert tree_trace(t)[0] == tree_trace(t, True)[0] == 3
    assert pruned_names(t) == ["g", "h"]                      # the leaves 4 and 6
    assert tree_trace(TREES["Quiz 1"], True)[0] == 8 and pruned_names(TREES["Quiz 1"]) == ["f"]
    assert tree_trace(TREES["Quiz 2"], True)[0] == 10 and pruned_names(TREES["Quiz 2"]) == ["g", "l"]
    assert pruned_names(TREES["Properties (10 | 10 0)"]) == ["e"]  # the 0 is skipped
    # thrashing slide: W then E = 8, E then E = 8, E then W = -2
    s = GameState.from_layout(LAYOUTS["line"])
    assert s.successor(0, "West").successor(0, "East").score == 8
    assert s.successor(0, "East").successor(0, "East").score == 8
    assert s.successor(0, "East").successor(0, "West").score == -2
    assert sorted(search(s, 2, score_eval)[1:2]) == [8]
    # alpha-beta never changes the root value and expands fewer nodes
    s = GameState.from_layout(LAYOUTS["arena"])
    _, v1, n1 = search(s, 2, closer_eval)
    _, v2, n2 = search(s, 2, closer_eval, alpha_beta=True)
    assert v1 == v2 and n2 < n1, (v1, v2, n1, n2)
    # power pellet: ghosts get scared, a scared ghost that is touched goes home for +200
    s = GameState.from_layout(["%%%%%%%", "%Ao G.%", "%%%%%%%"])
    s = s.successor(0, "East")
    assert s.scared == (0, SCARED_TIME) and not s.caps
    s = s.successor(1, "West").successor(0, "East")
    assert s.pos == ((3, 1), (4, 1)) and s.scared == (0, 0) and s.score == -2 + EAT_GHOST and not s.is_terminal()
    frames, _ = play("classic", depth=2, evaluate="closer", max_moves=30)
    assert "<canvas" in html_player(frames) and "<canvas" in html_player(frames[:1])
    frames, info = play("arena", depth=2, evaluate="closer", max_moves=5, tree_plies=3)
    assert len(info["trees"]) == info["moves"] and "Show search tree" in html_player(frames, trees=info["trees"])
    # true_random ghosts may turn back; random ones never do
    def turned_back(g):
        frames, _ = play("arena", depth=1, ghosts=g, max_moves=60, seed=3)
        return any(f.dirs[i] and p.dirs[i] and REVERSE[p.dirs[i]] == f.dirs[i] and len(p.legal_actions(i)) > 1
                   for p, f in zip(frames, frames[1:]) for i in range(1, f.num_agents()))
    assert turned_back("true_random") and not turned_back("random")
    print("all checks passed", n1, n2)
