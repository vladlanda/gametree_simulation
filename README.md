# Adversarial Search Visualizer

Streamlit app for lesson 5 (adversarial search): minimax and alpha-beta on the lecture trees,
and a maze-chase game (player vs. ghosts, with power pellets) played by a depth-limited search agent.
Games are shown by a live HTML/JS player (`html_player` in the engine), the same one the Colab exercise uses.
The game starts on ▶ and moves one agent per step, cell to cell on a semi-transparent grid, in turn order (A, ghost 1, ghost 2; an orange ring marks who just moved). In the app the player's **search tree** is drawn beside the board: every node shows its board (cropped to where the agents are) and its value, blue frame = A (MAX), red = ghost (MIN), green = moves actually made, ✂ = pruned. Record up to 9 levels; untick *Show search tree* to freeze it.

`maze_game.py` is the engine (standard library only). The Colab exercise HM_05 uses the same code.
Self-check: `python3 maze_game.py`.

## Run

```bash
uv sync
uv run streamlit run app.py
```

## Lecture demos (measured, seeds 0–19)

| Demo | Settings | Result |
|---|---|---|
| Hopeless: minimax dives into ghost 1 | trapped · Minimax d3 · score · random ghosts | 0/20 wins, −501 (caught after 1 move) · 18 nodes/move (alpha-beta: 10) |
| Same, ghosts modelled as random | trapped · Expectimax d3 | 13/20 wins, avg score 169 |
| Thrashing | room · Alpha-beta d2 · score | 14/20 wins, 6/20 timeouts (150 moves), avg 102 moves |
| Thrashing fixed by evaluation | room · d2 · closer | 20/20 wins in 20 moves |
| Thrashing, open room (lecture video look) | open · Alpha-beta d2 · score / closer / d4 score | 14/20 wins, 6/20 timeouts, avg 111 moves / 20/20 wins in 16 moves / 18/20 wins, avg 93 moves |
| Depth matters (no eval fix) | room · d4 · score | 20/20 wins, avg 53 moves |
| Same decisions, fewer nodes | arena · d3 · closer · random ghosts | identical games (9/20 wins); Minimax 179 vs alpha-beta 103 nodes/move |
| Smart ghosts coordinate | arena · d2 · closer · smart ghosts depth 1 / 3 | player wins 9/20 / 0/20 (caught after ~12 moves) |
| Opening: full board with power pellets | classic · Alpha-beta d3 · closer · random ghosts · max 400 | 19/20 wins, avg 163 moves, avg 3 scared ghosts eaten, 95 nodes/move; seed 1: win in 142 moves, score 1708 |

## Regenerating the Colab exercise

`build_hm05.py` rebuilds `moodle/HM_05.ipynb` and `HM_05_solution.ipynb` from this engine (the search code is stripped from the student copy):

```bash
python build_hm05.py maze_game.py ../../moodle answers_hm05.json
```

Needs `nbformat` (and `nbclient` + `ipykernel` to execute the notebooks for a check).
