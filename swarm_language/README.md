# swarm_language

Standalone implementation of the cultural evolution of perceptually grounded compositional lexicons in robot swarms, from:

> R. Sendra-Arranz, Á. Gutiérrez. *Cultural evolution of perceptually grounded compositional lexicons in swarm robotics systems.* Applied Soft Computing 202 (2026) 115879. <https://doi.org/10.1016/j.asoc.2026.115879>

Each agent grounds private meanings on the objects it perceives, learns compositions of meanings through Hebbian co-occurrence, and negotiates shared word forms with its neighbours through a naming game.

The package does not depend on any simulator. It is the code used in the paper's experiments, extracted from the [Mereli](https://github.com/Robolabo/Mereli) simulator, and it only needs `numpy` and `networkx`; `matplotlib` is optional and only used for plotting.

## Quick start

```bash
pip install numpy networkx
python -m swarm_language.demo --agents 20 --steps 3000   # run from the folder containing swarm_language/
```

The demo runs a swarm in a minimal 2D world (`ToyWorld`) with the referents of Experiment 2. Every 250 steps it prints the lexicon sharing score (LSS), the compositionality sharing score (CSS) and the shared lexicon sizes. It takes a few minutes.

## Using it in your own simulator

You only need `LanguageAgent`. Create one per robot, then call `step` once per control cycle on every robot, one after another:

```python
import numpy as np
from swarm_language import LanguageAgent, swarm_metrics

np.random.seed(0)                          # agents draw from the global np.random generator
agents = [LanguageAgent() for _ in range(n_robots)]

for t in range(n_steps):
    for i, agent in enumerate(agents):
        obj = closest_object_in_sensing_range(i)        # dict or None, see below
        messages = [agents[j].message for j in neighbors_within_comm_range(i)]
        agent.step(obj, messages)
    move_robots()

print(swarm_metrics([a.lexicon for a in agents]))
```

- **Observed object:** a dict with at least `'color'` (`'red'`, `'green'` or `'blue'`) and `'shape'` (`'ball'`, `'cube'` or `'pyramid'`). Pass `None` when nothing is perceived. In the paper's setup, a robot perceives the closest object whose surface is within 0.6 m.
- **Messages:** after `step`, `agent.message` holds what the agent broadcasts this cycle, or `None` if it stays silent. It is a plain dict:
  - an elementary word is `{'pattern': prototype_vector, 'word': 'hoxhuna', 'parents': None}`;
  - a composed word is `{'pattern': None, 'word': 'riyifu', 'parents': ('hoxhuna', 'wexewi')}`.

  Agents stepped later in the same cycle already hear the new message, as in the original simulations. In the paper, the communication range is 1 m and omnidirectional.
- **Inspecting an agent:** `agent.lexicon.words`, `.elementary_words`, `.composed_words`, `.assoc_list` (the `(parent1, parent2, word)` tuples), `.meanings` and `.man` (the Meaning Association Network as an adjacency matrix), and `agent.pattern_detector.pattern_mat` (the prototypes). `swarm_language.plotting.plot_man(agent.lexicon)` draws the MAN.

## Code map

| Paper | Module | Main objects |
| --- | --- | --- |
| Sec. 3.2, Eq. 9: sensing (color, shape, quality) | `perception.py` | `perceive_object`, `encode_color`, `encode_shape`, `encode_quality` |
| Sec. 3.3, Eqs. 3–5: iconisation and discrimination | `discrimination.py` | `PatternDetector` |
| Sec. 3.4: semiotic symbols and meaning space | `lexicon.py`, `words.py` | `Lexicon.create_word`, `create_random_word` |
| Sec. 3.5, Eq. 6: MAN and composed meanings | `lexicon.py` | `Lexicon.create_relation`, `propagate_tree` |
| Sec. 3.6, Eqs. 7–8: Hebbian learning of compositions | `lexicon.py` | `Lexicon.step` |
| Sec. 3.7, Fig. 2B–C: naming game (speaker and hearer) | `agent.py` | `LanguageAgent.speak`, `LanguageAgent.hear` |
| Sec. 5.1, Eqs. 10–13: LSS, CSS and lexicon sizes | `metrics.py` | `swarm_metrics` |
| Minimal world for tests and demos | `toy_world.py`, `demo.py` | `ToyWorld` |

## Parameters

All rates are per control step. In the Mereli experiments a robot runs one control step every 0.5 s. The defaults are the values of the implementation used for the published runs.

| Parameter | Where | Default | Meaning |
| --- | --- | --- | --- |
| `threshold` | `PatternDetector` | 0.4 | discrimination threshold θ_ξ |
| `adaptation_rate` | `PatternDetector` | 0.25 | prototype adaptation rate λ_ξ |
| `short_term_memory` | `Lexicon` | 100 | traces decay by `1/short_term_memory` per step (λ_β) |
| `learn_rate` | `Lexicon` | 0.01 | Hebbian increase of α_ij per step of co-activation |
| `forget_rate` | `Lexicon` | 0.001 | decay of α_ij per step |
| `association_threshold` | `Lexicon` | 0.4 | θ_α, creates a composed symbol |
| `meaning_range` | `Lexicon` | 2 | elementary meanings are drawn from U(−2, 2)^dim |
| `speak_prob` | `LanguageAgent` | 0.9 | probability of broadcasting a word each step |
| `NOISE_STD` | `perception.py` | 0.05 | Gaussian noise σ of the sensor readings |

Pass these as `LanguageAgent(speak_prob=..., lexicon_params={...}, discrimination_params={...})`. Two robustness options are also available: `distorted_color=True`, a faulty color sensor (Experiment R2), and `perspective_noise=True`, perspective-dependent shape perception (Experiment R1).

## Implementation details worth knowing

- **Quality comes from color.** The quality feature is derived from the object color: green → +1 (high), red → −1 (low), anything else → 0, plus noise, clipped to [−1, 1].
- **Color encoding.** Colors are encoded with intensity 0.8 (e.g. red = (0.8, 0, 0)), and the shape components are ordered (ball, cube, pyramid).
- **Association update.** α_ij increases with the co-activation of meanings i and j at the current step.
- **Hearer processing stops early.** A hearer stops processing the remaining messages of the cycle when it receives a known composed word whose parents differ from its own composition.
- **Random number usage.** All randomness comes from the global `np.random` generator. Changing the order or the number of random draws changes the results for a given seed.

## Tests

```bash
python -m unittest discover -s swarm_language/tests -t .
```

`test_regression.py` checks that the dynamics are bit-identical to the original implementation. It hashes the complete state of every agent and of the random generator every 100 steps, and compares the hashes with digests generated by the original Mereli controller. Run it after any change. If it fails, the change has altered the published behaviour.

## Reproducing the paper with Mereli

The full experiments run in Mereli: e-puck robots in PyBullet with obstacle avoidance. The robot controller is `build_lexicon_B` (`mereli/controllers/learn_lexicon.py`), which uses this package for the robots' language.

### Installation

```bash
git clone https://github.com/Robolabo/Mereli.git
cd Mereli
git checkout swarm-language

conda create -n mereli -c conda-forge python=3.11 numpy=1.26 scipy networkx matplotlib \
    pybullet click colorama scikit-learn seaborn pygame pymunk pyopengl
conda activate mereli
python -m pip install alphashape
mkdir -p mereli/logs
```

Use Python 3.11 with NumPy 1.26, because Mereli does not run with NumPy 2. On Apple Silicon Macs, the 3D view (`-R`) can be unstable with the native pybullet build; creating the environment in Intel mode fixes this. Prefix the `conda create` command with `CONDA_SUBDIR=osx-64`, then run `conda config --env --set subdir osx-64` inside the environment. Runs without the 3D view are not affected.

### Running

The configurations are in `mereli/config/PaperLanguageEmergence/`:

- `o=5_a=N.json`: the five referents of Experiment 2 with N robots (N = 10, 20, 30, 75, 100);
- `o=3_a=20.json`: three referents with 20 robots;
- `o=3_a=20_robustness.json`: three referents with 20 robots, for the robustness experiments.

```bash
python main.py -f PaperLanguageEmergence/o=5_a=20.json        # no window
python main.py -f PaperLanguageEmergence/o=5_a=20.json -R     # 3D view; starts paused, press r in the window
python main.py -f PaperLanguageEmergence/o=5_a=20.json -l     # also saves the lexicons
```

- **Progress.** The lexicon emergence status of the swarm (LSS, CSS and lexicon sizes, as in the demo) is printed every 1000 steps. Change the period with the controller parameter `"report_every"`, or set it to `0` to disable it.
- **Duration.** One simulation step is 0.05 s. The duration is set by `timesteps` (in `simulation`) and `total_duration` (in `world.task_manager`); keep both equal. For example, 50000 steps are 2500 s.
- **Logs.** With `-l`, the words, elementary and composed words, associations and prototypes of every robot are stored every `logging_period` steps in `mereli/logs/<logging.file>/<logging.file>_<date>.pickle`.
- **Repetitions and seeds.** Each run uses a new random initialisation. The paper's statistics use 50 independent runs per configuration. For reproducible runs, add `"seed": <n>` to the `simulation` block.
- **Parameters.** All the parameters above, plus the robustness options `"perspective_noise"` (R1) and `"faulty_color_robots"` (R2, a list of robot ids), are set in the controller `params` of the config. See the docstring of `BuildLexiconControllerB`.
