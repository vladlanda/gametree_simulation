"""Build HM_05.ipynb (student) and HM_05_solution.ipynb from one cell list.
Usage: python build_hm05.py <engine maze_game.py> <out_dir> [answers.json]"""
import ast, json, random, sys
import nbformat as nbf

ENGINE_PATH, OUT = sys.argv[1], sys.argv[2]
ANSWERS = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
sys.path.insert(0, ENGINE_PATH.rsplit("/", 1)[0])
import maze_game as mg

# ---------- engine for the notebook: drop the search code (that is what students write) ----------
src = open(ENGINE_PATH).read()
tree = ast.parse(src)
DROP = {"NODE_NAMES", "closer_eval", "chase_eval", "EVALS", "search", "play", "TREES", "name_tree", "tree_trace",
        "pruned_names"}
parts = []
for node in tree.body:
    name = getattr(node, "name", None) or (node.targets[0].id if isinstance(node, ast.Assign)
                                          and isinstance(node.targets[0], ast.Name) else None)
    if isinstance(node, ast.If):  # __main__ self-check
        continue
    if name in DROP:
        continue
    seg = ast.get_source_segment(src, node)
    if isinstance(node, ast.Expr) and node is tree.body[0]:
        continue  # module docstring
    parts.append(seg)
engine = "\n\n".join(parts)
smart = '''    if policy == "smart":  # ghosts run minimax too, with "close to the player is good"
        return search(state, depth, chase_eval, agent=i, alpha_beta=True, rng=rng)[0]
'''
assert smart in engine
engine = engine.replace(smart, "")
assert "search(" not in engine

# ---------- expected values for the checks (computed with the reference engine) ----------
def rand_tree(rng, b, d):
    return rng.randint(0, 99) if d == 0 else [rand_tree(rng, b, d - 1) for _ in range(b)]
R = random.Random(5)
RANDOM_TREES = [rand_tree(R, b, d) for b, d in [(2, 3), (3, 3), (2, 4), (3, 4), (4, 3)]]
def leaf_order(t):
    _, ev = mg.tree_trace(t, alpha_beta=True)
    names = mg.name_tree(t)
    return [names[e["path"]] for e in ev if e["kind"] == "leaf"]
LECTURE = {k: mg.TREES[k] for k in mg.TREES}
EXPECT_TREES = {k: (mg.tree_trace(t)[0], leaf_order(t)) for k, t in LECTURE.items()}
EXPECT_RANDOM = [(mg.tree_trace(t)[0], len(leaf_order(t))) for t in RANDOM_TREES]

def st(rows): return mg.GameState.from_layout(rows)
line, trapped, arena = (st(mg.LAYOUTS[k]) for k in ("line", "trapped", "arena"))
EXPECT_GAME = {
    "arena_d1_score": mg.search(arena, 1, mg.score_eval)[1],
    "arena_d2_score": mg.search(arena, 2, mg.score_eval)[1],
}
# value of a mid-game arena state, so the check is not only about start positions
mid = arena.successor(0, "West").successor(1, "East").successor(2, "West")
EXPECT_GAME["mid_d2_score"] = mg.search(mid, 2, mg.score_eval)[1]

SETUP = r'''#@title 🔧 תא הגדרות: הריצו אותו ראשון (אין צורך לקרוא את הקוד)
import itertools, math, random, signal, time
from statistics import mean
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import HTML, display

# ============================================================
# מנוע המשחק "מרדף במבוך" (אותו מנוע כמו בסימולטור של הקורס)
# ============================================================
''' + engine + r'''


# ============================================================
# ספירת צמתים: כל קריאה ל-successor היא צומת אחד שנוצר בעץ החיפוש
# ============================================================
NODES = [0]
_plain_successor = GameState.successor
def _counting_successor(self, agent, action, u_turn=False):
    NODES[0] += 1
    return _plain_successor(self, agent, action, u_turn)
GameState.successor = _counting_successor


def run_game(layout, player, ghosts='random', max_moves=150, seed=0):
    """משחק אחד. player(state) מחזיר פעולה לשחקן. מחזיר (frames, info)."""
    rng = random.Random(seed)
    random.seed(seed)                      # שבירת שוויון אקראית אצלכם תהיה זהה בכל הרצה
    state = GameState.from_layout(LAYOUTS[layout] if isinstance(layout, str) else layout)
    frames, nodes, moves = [state], [], 0
    while not state.is_terminal() and moves < max_moves:
        for i in range(state.num_agents()):
            if state.is_terminal():
                break
            if i == 0:
                NODES[0] = 0
                act = player(state)
                nodes.append(NODES[0])
                moves += 1
            else:
                act = ghost_action(state, i, ghosts, rng)
            state = state.successor(i, act, u_turn=i > 0 and ghosts == 'true_random')
            frames.append(state)
    result = state.status if state.is_terminal() else 'timeout'
    return frames, {'result': result, 'score': state.score, 'moves': moves,
                    'nodes_per_move': mean(nodes) if nodes else 0}


def play_many(layout, player, n=10, ghosts='random', max_moves=150, quiet=False):
    infos = [run_game(layout, player, ghosts, max_moves, seed)[1] for seed in range(n)]
    s = {'wins': sum(i['result'] == 'win' for i in infos),
         'timeouts': sum(i['result'] == 'timeout' for i in infos),
         'avg_score': round(mean(i['score'] for i in infos), 1),
         'avg_moves': round(mean(i['moves'] for i in infos), 1),
         'nodes_per_move': round(mean(i['nodes_per_move'] for i in infos))}
    if not quiet:
        print(f"{layout}: wins {s['wins']}/{n} · timeouts {s['timeouts']} · avg score {s['avg_score']} · "
              f"avg moves {s['avg_moves']} · nodes per move {s['nodes_per_move']}")
    return s


# ============================================================
# ציור המבוך ואנימציה "חיה" (אותו נגן כמו בסימולטור של הקורס)
# ============================================================
def show_state(s, title=''):
    display(HTML(html_player([s], title)))

def animate_game(frames, title='', speed=8):
    """אנימציה של משחק: לוחצים ▶ כדי להתחיל. יש גם צעד אחד קדימה או אחורה, פס גלילה ומהירות (מהלכים לשנייה). כל לחיצה מזיזה סוכן אחד, לפי התור."""
    display(HTML(html_player(frames, title, speed)))


# ============================================================
# עצי משחק מופשטים (כמו בהרצאה)
# ============================================================
VISITED = []   # שמות העלים שהחיפוש הסתכל עליהם, לפי הסדר

class Node:
    """צומת בעץ משחק. לצומת פנימי יש player ('MAX' או 'MIN') ו-children. לעלה יש תועלת."""
    def __init__(self, name, player=None, children=(), utility=None):
        self.name, self.player, self.children, self._utility = name, player, list(children), utility
    def is_terminal(self):
        return not self.children
    def utility(self):
        VISITED.append(self.name)          # כך יודעים על אילו עלים החיפוש הסתכל
        return self._utility
    def __repr__(self):
        return f'Node({self.name}, {self.player or self._utility})'

def build_tree(nested, first='MAX'):
    """רשימות מקוננות → Node. השמות: root, ואחריו a, b, c, ... בסדר pre-order (כמו בהרצאה)."""
    letters = itertools.chain('abcdefghijklmnopqrstuvwxyz', (f'n{i}' for i in itertools.count()))
    def build(t, name, player):
        if not isinstance(t, list):
            return Node(name, utility=t)
        other = 'MIN' if player == 'MAX' else 'MAX'
        kids = []
        for c in t:
            kids.append(build(c, next(letters), other))
        return Node(name, player, kids)
    return build(nested, 'root', first)

LECTURE_TREES = ''' + repr(LECTURE) + r'''

def _layout(t, path=(), xy=None, counter=None):
    xy = {} if xy is None else xy; counter = counter or [0]
    if isinstance(t, list):
        xs = [_layout(c, path + (i,), xy, counter)[path + (i,)][0] for i, c in enumerate(t)]
        xy[path] = ((xs[0] + xs[-1]) / 2, -len(path))
    else:
        xy[path] = (counter[0], -len(path)); counter[0] += 1
    return xy

def show_tree(nested, visited=None, title='', root_value=None):
    """מצייר עץ. visited = רשימת העלים שנבדקו; עלים שלא נבדקו (נגזמו) מופיעים באפור ומקווקו."""
    root = build_tree(nested)
    names, nodes = {}, {}
    def walk(n, path):
        names[path], nodes[path] = n.name, n
        for i, c in enumerate(n.children):
            walk(c, path + (i,))
    walk(root, ())
    xy = _layout(nested)
    leaves_under = {p: [q for q in xy if q[:len(p)] == p and nodes[q].is_terminal()] for p in xy}
    pruned = set() if visited is None else {p for p in xy if all(names[q] not in visited for q in leaves_under[p])}
    width = max(x for x, _ in xy.values()) + 1
    depth = max(len(p) for p in xy)
    fig, ax = plt.subplots(figsize=(max(5, 0.95 * width), 1.6 * depth + 1))
    for p, (x, y) in xy.items():
        if p:
            px, py = xy[p[:-1]]
            ax.plot([px, x], [py, y], color='#cbd5e1' if p in pruned else '#64748b',
                    ls=':' if p in pruned else '-', zorder=1)
    for p, (x, y) in xy.items():
        n = nodes[p]
        grey = p in pruned
        if n.is_terminal():
            ax.scatter([x], [y], marker='s', s=1200, color='#e2e8f0' if grey else 'white',
                       edgecolors='#94a3b8' if grey else '#334155', zorder=2)
            ax.text(x, y, str(n._utility), ha='center', va='center', fontsize=16, weight='bold',
                    color='#94a3b8' if grey else 'black', zorder=3)
        else:
            ax.scatter([x], [y], marker='^' if n.player == 'MAX' else 'v', s=1400,
                       color='#e2e8f0' if grey else '#fde68a' if not p else 'white',
                       edgecolors='#94a3b8' if grey else '#334155', zorder=2)
        if p:
            ax.text(x - 0.38, y + 0.34, n.name, fontsize=16, weight='bold', color='#1e293b')
    if root_value is not None:
        ax.text(xy[()][0], 0.45, f'value = {root_value}', ha='center', fontsize=16, weight='bold')
    if pruned:
        top = sorted(names[p] for p in pruned if p[:-1] not in pruned)   # whole pruned subtrees
        ax.set_xlabel('pruned: ' + ', '.join(top), fontsize=16, weight='bold')
    ax.set_title(title, fontsize=16, weight='bold'); ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values(): side.set_visible(False)
    ax.set_ylim(-depth - 0.6, 0.8); plt.show()

def random_tree(b, m, seed=0):
    rng = random.Random(seed)
    def make(d):
        return rng.randint(0, 99) if d == 0 else [make(d - 1) for _ in range(b)]
    return make(m)

def order_tree(nested, value_fn, best_first=True, player='MAX'):
    """ממיין את הילדים בכל צומת לפי ערך ה-Minimax שלהם (best_first: הטוב לשחקן שבוחר קודם)."""
    if not isinstance(nested, list):
        return nested
    other = 'MIN' if player == 'MAX' else 'MAX'
    kids = [order_tree(c, value_fn, best_first, other) for c in nested]
    vals = [value_fn(build_tree(c, other)) for c in kids]
    VISITED.clear()
    reverse = (player == 'MAX') == best_first
    return [k for _, k in sorted(zip(vals, kids), key=lambda vk: vk[0], reverse=reverse)]


# ============================================================
# תשתית בדיקות
# ============================================================
RESULTS = {}
RANDOM_TREES = ''' + repr(RANDOM_TREES) + r'''
EXPECT_TREES = ''' + repr(EXPECT_TREES) + r'''
EXPECT_RANDOM = ''' + repr(EXPECT_RANDOM) + r'''
EXPECT_GAME = ''' + repr(EXPECT_GAME) + r'''

class _Timeout(Exception):
    pass

def _call(fn, args, seconds=10):
    def _handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return 'ok', fn(*args)
    except _Timeout:
        return 'timeout', None
    except Exception as e:
        return 'error', e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

def _report_problem(status, value):
    if status == 'timeout':
        print('❌ הפונקציה לא הסתיימה בזמן: כנראה רקורסיה בלי תנאי עצירה, או עומק שלא יורד.')
    else:
        print(f'❌ הפונקציה זרקה שגיאה: {type(value).__name__}: {value}')
        print('   אם עדיין לא השלמתם את ה-TODO, זה צפוי. אחרת, קראו את הודעת השגיאה ובדקו את השורה הרלוונטית.')

def _finish(name, ok):
    RESULTS[name] = ok
    print(f'✅ {name}: כל הבדיקות עברו!' if ok else f'⛔ {name}: יש בדיקות שנכשלו. תקנו והריצו שוב.')

def check_minimax(value_fn):
    ok = True
    for label, nested in LECTURE_TREES.items():
        VISITED.clear()
        status, got = _call(value_fn, (build_tree(nested),))
        if status != 'ok':
            _report_problem(status, got); ok = False; break
        exp = EXPECT_TREES[label][0]
        if got != exp:
            print(f'❌ [{label}] קיבלנו {got!r}, אבל ערך ה-Minimax הוא {exp}.')
            if got in (math.inf, -math.inf) or got is None:
                print('   רמז: האם max_value / min_value מחזירות את v אחרי הלולאה, ו-value מחזירה את התוצאה שלהן?')
            ok = False
        n_leaves = sum(1 for _ in _leaves(nested))
        if status == 'ok' and len(VISITED) != n_leaves:
            print(f'❌ [{label}] Minimax צריך להסתכל על כל {n_leaves} העלים, אבל הסתכלתם על {len(VISITED)}.'); ok = False
    if ok:
        for t, (exp, _) in zip(RANDOM_TREES, EXPECT_RANDOM):
            status, got = _call(value_fn, (build_tree(t),))
            if status != 'ok' or got != exp:
                print(f'❌ על עץ אקראי קיבלנו {got!r} במקום {exp}. בדקו שצמתי MIN ו-MAX מתחלפים ברקורסיה.'); ok = False; break
        status, got = _call(value_fn, (Node('leaf', utility=7),))
        if ok and got != 7:
            print('❌ על עץ שהוא עלה בודד (ערך 7) צריך להחזיר 7. האם בדקתם is_terminal קודם?'); ok = False
    _finish('minimax (value)', ok)

def _leaves(t):
    if isinstance(t, list):
        for c in t:
            yield from _leaves(c)
    else:
        yield t

def check_alphabeta(ab_fn):
    ok = True
    for label, nested in LECTURE_TREES.items():
        VISITED.clear()
        status, got = _call(ab_fn, (build_tree(nested), -math.inf, math.inf))
        if status != 'ok':
            _report_problem(status, got); ok = False; break
        exp, exp_leaves = EXPECT_TREES[label]
        if got != exp:
            print(f'❌ [{label}] ערך השורש צריך להיות {exp} (אלפא-בטא לא משנה את ערך השורש!), קיבלנו {got!r}.'); ok = False
        elif VISITED != exp_leaves:
            print(f'❌ [{label}] הסתכלתם על העלים {VISITED}, אבל לפי ההרצאה צריך {exp_leaves}.')
            if len(VISITED) > len(exp_leaves):
                print('   רמז: יותר מדי עלים. האם אתם עוצרים (return v) כש-v ≥ β בצומת MAX, וכש-v ≤ α בצומת MIN?')
                print('        והאם אתם מעבירים את α ו-β המעודכנים לקריאה הרקורסיבית?')
            else:
                print('   רמז: פחות מדי עלים. האם עדכנתם את α / β רק אחרי בדיקת הגיזום, ורק בצומת עצמו (לא כלפי מעלה)?')
            ok = False
    if ok:
        for t, (exp, n_exp) in zip(RANDOM_TREES, EXPECT_RANDOM):
            VISITED.clear()
            status, got = _call(ab_fn, (build_tree(t), -math.inf, math.inf))
            if status != 'ok' or got != exp or len(VISITED) != n_exp:
                print(f'❌ על עץ אקראי: ערך {got!r} (צריך {exp}), עלים {len(VISITED)} (צריך {n_exp}).'); ok = False; break
    _finish('alpha-beta (ab_value)', ok)

def check_depth_limited(dl_fn, choose_fn):
    ok = True
    line = GameState.from_layout(LAYOUTS['line'])
    cases = [('line: מערבה ואז מזרחה', (line.successor(0, 'West'), 0, 1, score_eval), 8),
             ('line: מזרחה ואז הכי טוב', (line.successor(0, 'East'), 0, 1, score_eval), 8),
             ('line: עומק 0 מחזיר את ההערכה', (line, 0, 0, score_eval), 0),
             ('trapped: עומק 3', (GameState.from_layout(LAYOUTS['trapped']), 0, 3, score_eval), -501),
             ('arena: עומק 1', (GameState.from_layout(LAYOUTS['arena']), 0, 1, score_eval), EXPECT_GAME['arena_d1_score']),
             ('arena: עומק 2', (GameState.from_layout(LAYOUTS['arena']), 0, 2, score_eval), EXPECT_GAME['arena_d2_score'])]
    a = GameState.from_layout(LAYOUTS['arena'])
    mid = a.successor(0, 'West').successor(1, 'East').successor(2, 'West')
    cases.append(('arena: אמצע משחק, עומק 2', (mid, 0, 2, score_eval), EXPECT_GAME['mid_d2_score']))
    for label, args, exp in cases:
        status, got = _call(dl_fn, args)
        if status != 'ok':
            _report_problem(status, got); ok = False; break
        if got != exp:
            print(f'❌ [{label}] קיבלנו {got!r}, אבל צריך {exp}.')
            if label.startswith('trapped'):
                print('   רמז: הרוחות הן MIN. האם בתור של רוח לוקחים min? והאם העומק יורד רק אחרי שכל הסוכנים זזו?')
            ok = False
    if ok:
        tr = GameState.from_layout(LAYOUTS['trapped'])
        status, got = _call(choose_fn, (tr, 3, score_eval))
        if status != 'ok' or got != 'East':
            print(f"❌ choose_action ב-trapped בעומק 3 צריכה להחזיר 'East' (ראו את ההדגמה בהרצאה), קיבלנו {got!r}."); ok = False
    if ok:
        seen = {choose_fn(line, 2, score_eval) for _ in range(40)}
        if seen != {'West', 'East'}:
            print(f'❌ ב-line בעומק 2 שני המהלכים שווים (8). צריך לבחור ביניהם באקראי, אבל קיבלנו רק {seen}.')
            print('   רמז: שמרו את כל הפעולות עם הערך הטוב ביותר, והחזירו random.choice מתוכן.'); ok = False
    _finish('depth-limited minimax', ok)

def check_evaluation(eval_fn, choose_fn):
    ok = True
    player = lambda s: choose_fn(s, 2, eval_fn)
    status, room = _call(play_many, ('room', player, 10, 'random', 150, True), seconds=120)
    if status != 'ok':
        _report_problem(status, room); _finish('evaluation function', False); return
    print(f"room, עומק 2: {room['wins']}/10 ניצחונות, ממוצע {room['avg_moves']} מהלכים")
    if room['wins'] < 10 or room['avg_moves'] > 40:
        print('❌ צריך 10/10 ניצחונות בממוצע של 40 מהלכים לכל היותר. הסוכן עדיין "משתולל"?'); ok = False
    status, arena = _call(play_many, ('arena', player, 10, 'random', 150, True), seconds=240)
    if status != 'ok':
        _report_problem(status, arena); _finish('evaluation function', False); return
    print(f"arena, עומק 2, רוחות אקראיות: {arena['wins']}/10 ניצחונות")
    if arena['wins'] < 5:
        print('❌ צריך לפחות 5/10 ניצחונות.'); ok = False
    _finish('evaluation function', ok)

def ready(*names):
    missing = [n for n in names if not RESULTS.get(n)]
    if missing:
        print('⚠️ השלימו קודם את המשימות האלה והריצו את תאי הבדיקה שלהן:', ', '.join(missing))
    return not missing

def final_report():
    for n in ['minimax (value)', 'alpha-beta (ab_value)', 'depth-limited minimax', 'evaluation function']:
        mark = {True: '✅', False: '❌'}.get(RESULTS.get(n), '⬜ (לא נבדק)')
        print(f'{mark}  {n}')
'''

def md(text):
    return nbf.v4.new_markdown_cell('<div dir="rtl" align="right">\n\n' + text.strip() + '\n\n</div>')

def answer(key):
    if key in ANSWERS:
        return md('✍️ **תשובה (פתרון):**\n\n' + ANSWERS[key])
    return md('✍️ **תשובה (פתרון):**\n\n*(ימולא אחרי הרצת הפתרון)*')

student_answer = lambda: md('✍️ **תשובה:** *(לחצו פעמיים על התא כדי לכתוב כאן)*')

# (student version, solution version) for each code cell with a TODO
TASK1 = ('''def value(node):
    """ערך Minimax של צומת (dispatch, כמו בשקופית מההרצאה)."""
    # TODO: אם node.is_terminal(): החזירו node.utility()
    #       אם node.player == 'MAX': החזירו max_value(node)
    #       אחרת (MIN): החזירו min_value(node)


def max_value(node):
    v = -math.inf
    # TODO: לכל ילד child ב-node.children:  v = max(v, value(child))

    return v


def min_value(node):
    v = math.inf
    # TODO: לכל ילד child ב-node.children:  v = min(v, value(child))

    return v
''', '''def value(node):
    """ערך Minimax של צומת (dispatch, כמו בשקופית מההרצאה)."""
    if node.is_terminal():
        return node.utility()
    if node.player == 'MAX':
        return max_value(node)
    return min_value(node)


def max_value(node):
    v = -math.inf
    for child in node.children:
        v = max(v, value(child))
    return v


def min_value(node):
    v = math.inf
    for child in node.children:
        v = min(v, value(child))
    return v
''')

TASK2 = ('''def ab_value(node, alpha, beta):
    """ערך השורש עם גיזום אלפא-בטא. α: האפשרות הטובה ביותר של MAX במסלול לשורש, β: של MIN."""
    # TODO: כמו value, אבל קוראים ל-ab_max_value / ab_min_value ומעבירים להן את alpha ו-beta


def ab_max_value(node, alpha, beta):
    v = -math.inf
    for child in node.children:
        # TODO: v = max(v, ab_value(child, alpha, beta))
        #       אם v >= beta: החזירו v מיד (MIN שמעלינו לא ייתן להגיע לכאן)
        #       עדכנו alpha = max(alpha, v)
        pass
    return v


def ab_min_value(node, alpha, beta):
    v = math.inf
    for child in node.children:
        # TODO: v = min(v, ab_value(child, alpha, beta))
        #       אם v <= alpha: החזירו v מיד (MAX שמעלינו לא ייתן להגיע לכאן)
        #       עדכנו beta = min(beta, v)
        pass
    return v
''', '''def ab_value(node, alpha, beta):
    """ערך השורש עם גיזום אלפא-בטא. α: האפשרות הטובה ביותר של MAX במסלול לשורש, β: של MIN."""
    if node.is_terminal():
        return node.utility()
    if node.player == 'MAX':
        return ab_max_value(node, alpha, beta)
    return ab_min_value(node, alpha, beta)


def ab_max_value(node, alpha, beta):
    v = -math.inf
    for child in node.children:
        v = max(v, ab_value(child, alpha, beta))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def ab_min_value(node, alpha, beta):
    v = math.inf
    for child in node.children:
        v = min(v, ab_value(child, alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v
''')

TASK3 = ('''def dl_value(state, agent, depth, evaluate):
    """
    ערך Minimax מוגבל עומק של state, כשתורו של agent לזוז.
    agent 0 = השחקן (MAX), 1, 2, ... = רוחות (MIN).
    depth = כמה סבבים מלאים נשארו (בסבב כל הסוכנים זזים פעם אחת).
    evaluate(state) = פונקציית ההערכה, למצבים שבהם עוצרים.
    """
    # TODO 1: אם state.is_terminal() או depth == 0: החזירו evaluate(state)


    next_agent = (agent + 1) % state.num_agents()
    next_depth = depth - 1 if next_agent == 0 else depth     # סבב נגמר אחרי שהסוכן האחרון זז

    # TODO 2: חשבו את ערכי כל הילדים: לכל פעולה a ב-state.legal_actions(agent):
    #         dl_value(state.successor(agent, a), next_agent, next_depth, evaluate)
    # TODO 3: החזירו max של הערכים אם agent == 0, אחרת min


def choose_action(state, depth, evaluate):
    """בחירת פעולה לשחקן (agent 0): הפעולה שערך ה-Minimax שלה הכי גבוה. בשוויון: בחירה אקראית."""
    next_agent = 1 % state.num_agents()                       # בלי רוחות, התור חוזר לשחקן
    next_depth = depth - 1 if next_agent == 0 else depth
    best_value, best_actions = -math.inf, []
    for action in state.legal_actions(0):
        v = dl_value(state.successor(0, action), next_agent, next_depth, evaluate)
        # TODO 4: אם v > best_value: התחילו רשימה חדשה [action] ועדכנו best_value
        #         אם v == best_value: הוסיפו את action לרשימה


    return random.choice(best_actions)
''', '''def dl_value(state, agent, depth, evaluate):
    """
    ערך Minimax מוגבל עומק של state, כשתורו של agent לזוז.
    agent 0 = השחקן (MAX), 1, 2, ... = רוחות (MIN).
    depth = כמה סבבים מלאים נשארו (בסבב כל הסוכנים זזים פעם אחת).
    evaluate(state) = פונקציית ההערכה, למצבים שבהם עוצרים.
    """
    if state.is_terminal() or depth == 0:
        return evaluate(state)

    next_agent = (agent + 1) % state.num_agents()
    next_depth = depth - 1 if next_agent == 0 else depth     # סבב נגמר אחרי שהסוכן האחרון זז

    values = [dl_value(state.successor(agent, a), next_agent, next_depth, evaluate)
              for a in state.legal_actions(agent)]
    return max(values) if agent == 0 else min(values)


def choose_action(state, depth, evaluate):
    """בחירת פעולה לשחקן (agent 0): הפעולה שערך ה-Minimax שלה הכי גבוה. בשוויון: בחירה אקראית."""
    next_agent = 1 % state.num_agents()                       # בלי רוחות, התור חוזר לשחקן
    next_depth = depth - 1 if next_agent == 0 else depth
    best_value, best_actions = -math.inf, []
    for action in state.legal_actions(0):
        v = dl_value(state.successor(0, action), next_agent, next_depth, evaluate)
        if v > best_value:
            best_value, best_actions = v, [action]
        elif v == best_value:
            best_actions.append(action)
    return random.choice(best_actions)
''')

TASK4 = ('''def my_evaluation(state):
    """
    פונקציית הערכה משלכם: מספר גבוה = מצב טוב לשחקן.
    מותר להשתמש ב: state.score, state.dots, state.player_pos(), state.ghost_positions(),
    maze_distance(state, a, b), state.is_win(), state.is_lose()
    """
    if state.is_terminal():
        return state.score
    # TODO: התחילו מ-state.score, והוסיפו "סיגנל" שאומר שמצב שבו כבר התקרבנו לנקודה טוב יותר.
    #       למשל: החסירו את מרחק המבוך לנקודה הקרובה ביותר.

    return state.score
''', '''def my_evaluation(state):
    """
    פונקציית הערכה משלכם: מספר גבוה = מצב טוב לשחקן.
    מותר להשתמש ב: state.score, state.dots, state.player_pos(), state.ghost_positions(),
    maze_distance(state, a, b), state.is_win(), state.is_lose()
    """
    if state.is_terminal():
        return state.score
    nearest_dot = min(maze_distance(state, state.player_pos(), d) for d in state.dots)
    return state.score - nearest_dot
''')

def guarded(code, *names):
    body = '\n'.join(('    ' + l) if l else l for l in code.split('\n'))
    return nbf.v4.new_code_cell(f"if ready({', '.join(map(repr, names))}):\n" + body)

def build(solution):
    c = lambda pair: nbf.v4.new_code_cell(pair[1] if solution else pair[0])
    ans = (lambda k: answer(k)) if solution else (lambda k: student_answer())
    cells = [
        md('''
# HM_05: חיפוש יריבי: Minimax, אלפא-בטא וחיפוש מוגבל עומק

**קורס: יסודות בינה מלאכותית · תרגול לשיעור 5 (חיפוש יריבי)**

בהרצאה ראינו איך סוכן מחליט מה לעשות כשיש **יריב** שמנסה להפריע לו. בתרגול הזה תממשו בעצמכם את האלגוריתמים, תבדקו אותם על העצים מההרצאה, ואז תריצו אותם על משחק אמיתי: **"מרדף במבוך"**. השחקן (A) אוסף נקודות, והרוחות (ריבועים ממוספרים) רודפות אחריו. זה המשחק מההדגמות בהרצאה, באותם חוקים בדיוק.

1. **Minimax** על עצי משחק.
2. **גיזום אלפא-בטא**, והשפעת סדר הילדים.
3. **Minimax מוגבל עומק** על המשחק: למה הסוכן "מתאבד"? למה הוא "משתולל" הלוך ושוב?
4. **פונקציית הערכה** משלכם.
5. **העומק משנה**: מדידה.

**איך עובדים:** הריצו קודם את תא ההגדרות. בכל ✏️ משימה השלימו את ה-`TODO`, והריצו את תא הבדיקה שאחריה. ב-❓ שאלות כתבו תשובה בתא ✍️ שאחריהן.
'''),
        nbf.v4.new_code_cell(SETUP),
        md('''
## חלק 1: עצי משחק ו-Minimax

עץ משחק בתרגול הוא **רשימות מקוננות**: עלה הוא מספר (התועלת), וצומת פנימי הוא רשימה של ילדים. השורש הוא **MAX**, והרמות מתחלפות: MAX, MIN, MAX, ...

הפונקציה `build_tree` הופכת את הרשימות לאובייקטים של `Node`, עם השמות מההרצאה (a, b, c, ... לפי הסדר). לכל צומת יש:

| מה | משמעות |
|---|---|
| `node.is_terminal()` | האם זה עלה (מצב סופי) |
| `node.utility()` | התועלת של עלה. **כל קריאה נרשמת**, כך שאפשר לדעת על אילו עלים החיפוש הסתכל |
| `node.player` | `'MAX'` או `'MIN'` |
| `node.children` | רשימת הילדים |
'''),
        nbf.v4.new_code_cell('''for name, nested in LECTURE_TREES.items():
    print(f'{name:45s} {nested}')
show_tree(LECTURE_TREES['Lecture tree (3 12 8 | 2 4 6 | 14 5 2)'], title='the tree from the lecture')'''),
        md('''
### ✏️ משימה 1: Minimax

ממשו את שלוש הפונקציות לפי הפסאודו-קוד מההרצאה:

<div dir="ltr">

```
def value(state):
    if the state is a terminal state: return the state's utility
    if the next agent is MAX: return max-value(state)
    if the next agent is MIN: return min-value(state)

def max-value(state):                       def min-value(state):
    initialize v = -∞                           initialize v = +∞
    for each successor of state:                for each successor of state:
        v = max(v, value(successor))                v = min(v, value(successor))
    return v                                    return v
```

</div>
'''),
        c(TASK1),
        nbf.v4.new_code_cell('check_minimax(value)'),
        md('''
### ❓ שאלה 1: Minimax כ-DFS

1. Minimax מסתכל על **כל** העלים. מה סיבוכיות הזמן שלו, ומה סיבוכיות הזיכרון שלו, כשמקדם הסיעוף b והעומק m? למה הזיכרון קטן כל כך?
2. בשחמט b ≈ 35 ו-m ≈ 100. בערך כמה עלים יש בעץ? (סדר גודל מספיק. אפשר להיעזר בתא הבא.)
'''),
        nbf.v4.new_code_cell('# מחשבון: למשל print(f"{35 ** 100:.2e}")'),
        ans('q1'),
        md('''
## חלק 2: גיזום אלפא-בטא

**α:** האפשרות הטובה ביותר של **MAX** לאורך המסלול לשורש. **β:** האפשרות הטובה ביותר של **MIN** לאורך המסלול לשורש. בשורש מתחילים עם α = −∞ ו-β = +∞.

<div dir="ltr">

```
def max-value(state, α, β):                 def min-value(state, α, β):
    initialize v = -∞                           initialize v = +∞
    for each successor of state:                for each successor of state:
        v = max(v, value(successor, α, β))          v = min(v, value(successor, α, β))
        if v ≥ β return v                           if v ≤ α return v
        α = max(α, v)                               β = min(β, v)
    return v                                    return v
```

</div>

**שימו לב:** α ו-β הם **מקומיים**. עדכון שלהם עובר רק **למטה** (לילדים), לא למעלה. למעלה עולה רק v.

### ✏️ משימה 2: אלפא-בטא
'''),
        c(TASK2),
        nbf.v4.new_code_cell('check_alphabeta(ab_value)'),
        md('''
### ❓ שאלה 2: חיזוי (לפני שאתם מריצים!)

אלה שני התרגילים מההרצאה. **לפני** שתריצו את התא הבא, כתבו לכל אחד: מה ערך המשחק, ואילו צמתים נגזמים?

<div dir="ltr">

```
Quiz 1:                 MAX                 Quiz 2:                     MAX
                     a/     \\d                                  a/              \\h
                  MIN        MIN                              MIN                  MIN
                 b/  \\c     e/  \\f                         b/     \\e           i/     \\l
                10    8     4    50                       MAX      MAX         MAX      MAX
                                                         c/ \\d    f/ \\g      j/ \\k     m/ \\n
                                                        10   6  100   8     1   2   20   4
```

</div>

עכשיו הריצו. צדקתם? **אל תתקנו את החיזוי בדיעבד.**
'''),
        ans('q2'),
        nbf.v4.new_code_cell('''for name in ['Lecture tree (3 12 8 | 2 4 6 | 14 5 2)', 'Quiz 1', 'Quiz 2', 'Properties (10 | 10 0)']:
    VISITED.clear()
    v = ab_value(build_tree(LECTURE_TREES[name]), -math.inf, math.inf)
    show_tree(LECTURE_TREES[name], visited=list(VISITED), title=f'alpha-beta: {name}', root_value=v)'''),
        md('''
### 👀 כמה גיזום? תלוי בסדר

הפונקציה `order_tree` ממיינת את הילדים בכל צומת לפי ערך ה-Minimax שלהם: **הטוב ביותר קודם** (סדר "מושלם") או **הגרוע ביותר קודם**. התא הבא סופר על כמה עלים כל אחד מהאלגוריתמים הסתכל, בעצים אקראיים עם b = 3.
'''),
        guarded('''def leaves_visited(fn, nested):
    VISITED.clear()
    fn(build_tree(nested))
    return len(VISITED)

ab = lambda node: ab_value(node, -math.inf, math.inf)
b = 3
print(f"{'m':>3}{'minimax':>10}{'α-β random':>13}{'α-β best first':>17}{'α-β worst first':>18}{'b^(m/2) theory':>17}")
for m in [2, 4, 6, 8]:
    t = random_tree(b, m, seed=m)
    best, worst = order_tree(t, value, True), order_tree(t, value, False)
    theory = b ** math.ceil(m / 2) + b ** (m // 2) - 1
    print(f"{m:>3}{leaves_visited(value, t):>10}{leaves_visited(ab, t):>13}{leaves_visited(ab, best):>17}"
          f"{leaves_visited(ab, worst):>18}{theory:>17}")''', 'minimax (value)', 'alpha-beta (ab_value)'),
        md('''
### ❓ שאלה 3: סדר הילדים

1. העתיקו את הטבלה.
2. בסדר "הטוב ביותר קודם", איך מספר העלים גדל עם m? השוו לעמודה האחרונה, ולטענה מההרצאה ש"אפשר להגיע לעומק כפול".
3. בסדר "הגרוע ביותר קודם" כמעט אין גיזום. למה?
4. במשחק אמיתי לא יודעים מראש מה ערך כל ילד (בשביל זה מחפשים!). איך בכל זאת אפשר לנחש סדר טוב? (רמז: סעיף "סינרגיה" בסוף ההרצאה.)
'''),
        ans('q3'),
        md('''
## חלק 3: המשחק "מרדף במבוך"

**החוקים (כמו בהדגמות בהרצאה):** ‎−1 על כל מהלך של השחקן, ‎+10 על כל נקודה, ‎+500 כשאוכלים את הנקודה האחרונה (ניצחון), ‎−500 כשרוח תופסת את השחקן (הפסד). השחקן והרוחות זזים **לסירוגין**: השחקן, רוח 1, רוח 2, ושוב השחקן. **רוח לא מסתובבת לאחור**, אלא אם היא במבוי סתום.

**כדורי כוח** (העיגולים הכתומים הגדולים, רק בלוח `classic`): כשהשחקן אוכל כדור, כל הרוחות "נבהלות" ל-40 מהלכים של רוח (הן נצבעות כחול עם "!", ומהבהבות לפני שהן חוזרות לעצמן). רוח מבוהלת שהשחקן נוגע בה חוזרת לנקודת ההתחלה שלה, והשחקן מקבל ‎+200.

| מה | משמעות |
|---|---|
| `GameState.from_layout(LAYOUTS['arena'])` | מצב התחלתי. הלוחות: `line`, `trapped`, `room`, `open`, `arena`, `classic` |
| `state.num_agents()` | מספר הסוכנים (השחקן + הרוחות) |
| `state.legal_actions(agent)` | הפעולות החוקיות: `'North'`, `'South'`, `'East'`, `'West'` |
| `state.successor(agent, action)` | המצב **החדש** אחרי הפעולה (המצב הישן לא משתנה) |
| `state.is_terminal()`, `is_win()`, `is_lose()` | האם המשחק נגמר, ואיך |
| `state.score` | הניקוד |
| `maze_distance(state, a, b)` | מרחק במבוך (מספר צעדים) בין שתי משבצות |
| `state.caps`, `state.is_scared(g)` | כדורי הכוח שנשארו, והאם רוח g מבוהלת כרגע |
'''),
        nbf.v4.new_code_cell('''for name in LAYOUTS:
    show_state(GameState.from_layout(LAYOUTS[name]), title=name)

# שחקן אקראי, רק כדי לראות את המשחק זז
frames, info = run_game('arena', player=lambda s: random.choice(s.legal_actions(0)), seed=1)
print(info)
animate_game(frames, 'random player')'''),
        md('''
### ✏️ משימה 3: Minimax מוגבל עומק

במשחק אמיתי אי אפשר לחפש עד הסוף. לכן עוצרים אחרי `depth` **סבבים** (בכל סבב כל הסוכנים זזים פעם אחת), ובמקום תועלת אמיתית מחזירים **הערכה** `evaluate(state)`. השחקן הוא MAX, וכל הרוחות הן MIN.

`choose_action` היא ה"שורש": היא מחשבת את הערך של כל פעולה, ובוחרת את הטובה ביותר. **אם כמה פעולות שוות, בוחרים ביניהן באקראי.**

פונקציית ההערכה הפשוטה ביותר כבר כתובה: `score_eval(state)` מחזירה את הניקוד בלבד.
'''),
        c(TASK3),
        nbf.v4.new_code_cell('check_depth_limited(dl_value, choose_action)'),
        md('''
### 👀 הדגמה 1: מצב חסר סיכוי

רוח 1 יכולה רק להתקרב. רוח 2 תבחר ללכת למעלה או למטה. השחקן מסתכל 3 סבבים קדימה.
'''),
        guarded('''minimax_player = lambda s: choose_action(s, 3, score_eval)
frames, info = run_game('trapped', minimax_player, ghosts='random', seed=0)
print(info)
animate_game(frames, 'trapped, minimax depth 3')
play_many('trapped', minimax_player, n=20)
print('value of the start state:', dl_value(GameState.from_layout(LAYOUTS['trapped']), 0, 3, score_eval))''', 'depth-limited minimax'),
        md('''
### ❓ שאלה 4: למה הוא "מתאבד"?

1. מה השחקן עשה, ומה ערך ה-Minimax שחישב (`dl_value` של המצב ההתחלתי בעומק 3)? הסבירו את המספר.
2. הרוחות בהדגמה זזות **באקראי**. כמה פעמים מתוך 20 רוח 2 בכלל הולכת למטה, ומשאירה דרך פתוחה? (רמז: בכל צומת לרוח שתי אפשרויות.) האם Minimax "צודק" כאן?
3. מה ההנחה של Minimax על היריב, ומתי היא מזיקה? (בהרצאה הבאה נראה אלגוריתם שמניח יריב אקראי.)
'''),
        ans('q4'),
        md('''
### 👀 הדגמה 2: למה הוא "משתולל"?

בלוח `line` יש נקודה צמודה לשחקן ממערב (משמאל), ונקודה נוספת שני צעדים ממזרח (מימין), בדיוק כמו בשקופית מההרצאה. ב-`room` הנקודות רחוקות זו מזו. השחקן מסתכל רק **2** סבבים קדימה, עם `score_eval`.
'''),
        guarded('''line = GameState.from_layout(LAYOUTS['line'])
for a in line.legal_actions(0):
    print(a, '→ value at depth 2 =', dl_value(line.successor(0, a), 0, 1, score_eval))

shallow = lambda s: choose_action(s, 2, score_eval)
frames, info = run_game('room', shallow, seed=0)
print(info)
animate_game(frames, 'room, depth 2, score only')
play_many('room', shallow, n=10)''', 'depth-limited minimax'),
        md('''
### ❓ שאלה 5: Thrashing

1. מה הערך של כל אחד מהמהלכים ב-`line`? למה השחקן לא "ממהר" לאכול?
2. ב-`room`: כמה מהמשחקים נגמרו ב-timeout (150 מהלכים)? מה השחקן "רואה" כשכל הנקודות רחוקות ממנו יותר מ-2 צעדים?
3. בהרצאה נאמר שהסוכן "פותר משחק אחר". איזה משחק?
'''),
        ans('q5'),
        md('''
## חלק 4: פונקציית הערכה

### ✏️ משימה 4: `my_evaluation`

כתבו פונקציית הערכה שנותנת **סיגנל** גם כשאין נקודה בטווח החיפוש. הבדיקה מריצה את השחקן שלכם (עומק 2) על:

- `room`: צריך **10/10** ניצחונות, בממוצע של **40 מהלכים לכל היותר**.
- `arena` עם רוחות אקראיות: לפחות **5/10** ניצחונות.
'''),
        c(TASK4),
        nbf.v4.new_code_cell('''check_evaluation(my_evaluation, choose_action)
if RESULTS.get('evaluation function'):
    frames, info = run_game('room', lambda s: choose_action(s, 2, my_evaluation), seed=0)
    animate_game(frames, 'room, depth 2, my evaluation')'''),
        md('''
### 👀 בונוס: הלוח המלא עם כדורי כוח

אותו שחקן (עומק 2, `my_evaluation`) על הלוח `classic`. שימו לב מה קורה אחרי שהוא אוכל כדור כוח: האם הוא רודף אחרי הרוחות המבוהלות? (הוא לא "יודע" על כדורי הכוח. הוא רק רואה ‎+200 בניקוד, אם זה קורה בתוך עומק החיפוש.)
'''),
        guarded('''frames, info = run_game('classic', lambda s: choose_action(s, 2, my_evaluation), max_moves=400, seed=1)
print(info)
animate_game(frames, 'classic, depth 2, my evaluation')''', 'evaluation function'),
        md('''
## חלק 5: העומק משנה

התא הבא מריץ את השחקן **עם `score_eval` בלבד** (בלי התיקון שלכם) בעומקים שונים, ומודד גם כמה צמתים נוצרו בממוצע בכל מהלך.
'''),
        guarded('''print(f"{'depth':>6}{'wins':>8}{'timeouts':>10}{'avg moves':>11}{'nodes/move':>12}")
for d in [1, 2, 3, 4, 5]:
    s = play_many('room', lambda st, d=d: choose_action(st, d, score_eval), n=10, quiet=True)
    print(f"{d:>6}{s['wins']:>6}/10{s['timeouts']:>10}{s['avg_moves']:>11}{s['nodes_per_move']:>12}")

s = play_many('room', lambda st: choose_action(st, 2, my_evaluation), n=10, quiet=True)
print(f"{'2 + my_evaluation':>17}: wins {s['wins']}/10, avg moves {s['avg_moves']}, nodes/move {s['nodes_per_move']}")''', 'depth-limited minimax', 'evaluation function'),
        md('''
### ❓ שאלה 6: עומק מול הערכה

1. העתיקו את הטבלה.
2. מאיזה עומק `score_eval` לבדה כבר מספיקה? כמה זה עלה בצמתים לכל מהלך, לעומת עומק 2 עם `my_evaluation`?
3. הסבירו בעזרת המשפט מההרצאה: "ככל שפונקציית ההערכה קבורה עמוק יותר בעץ, איכותה משנה פחות". מה ה"מחיר" של כל אחת משתי הדרכים?
'''),
        ans('q6'),
        md('''
## סיכום והגשה

- **Minimax** מחשב את הערך הטוב ביותר נגד יריב **מושלם**. הוא DFS ממצה: זמן O(b^m), זיכרון O(bm).
- **אלפא-בטא** נותן **אותו ערך בשורש**, ומדלג על תתי-עצים שלא יכולים לשנות אותו. עם סדר טוב: O(b^(m/2)).
- במשחק אמיתי מחפשים **לעומק מוגבל** ומחזירים **הערכה**. הערכה חלשה גורמת להתנהגויות מוזרות (thrashing). חיפוש עמוק יותר או הערכה טובה יותר מתקנים אותן, ולכל אחד מהם מחיר.
- **ההנחה על היריב חשובה:** Minimax מניח את הגרוע ביותר, ולכן לפעמים "מתאבד" כשהיריב בכלל לא חכם.

הריצו את התא האחרון. לפני ההגשה ודאו שכל השורות מסומנות ✅, ושכל התשובות ✍️ מלאות.
'''),
        nbf.v4.new_code_cell('final_report()'),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {'colab': {'provenance': [], 'toc_visible': True},
                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                   'language_info': {'name': 'python'}}
    return nb

for sol, name in [(False, 'HM_05.ipynb'), (True, 'HM_05_solution.ipynb')]:
    nbf.write(build(sol), f'{OUT}/{name}')
print('written', OUT)
