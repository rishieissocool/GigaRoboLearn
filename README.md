# gigarobolearn

Reinforcement learning trainer for SSL (Small Size League) robot soccer. Uses PPO with self-play: one policy controls both blue and yellow; observations are mirrored so the same network sees a consistent view. Training runs with an internal physics simulation; optionally you can stream the current game state to [grSim](https://github.com/RoboCup-SSL/grSim) for visualisation.

No Python, no PyTorch—just C++20 and MSVC. Checkpoints are saved under `models/` so you can stop and resume, or run evaluation with a trained policy.

---

## Requirements

- **Windows** (tested on 10/11). The grSim client uses Winsock2; the rest is standard C++.
- **Visual Studio 2022** (or 2019) with **C++20** and **x64** toolset.
- **No vcpkg or external libs** for the core app. Only system dependency: `ws2_32.lib` (Winsock2), which is linked by the project.

---

## Build

1. Open `gigarobolearn.slnx` in Visual Studio (or the `.sln` / `.vcxproj` in the `gigarobolearn` folder).
2. Set configuration to **Release** and platform **x64**.
3. Build. The executable is produced in `x64/Release/gigarobolearn.exe`.

From a developer prompt you can also build with:

```bat
msbuild gigarobolearn.slnx /p:Configuration=Release /p:Platform=x64
```

---

## Usage

Run from the directory that contains the `gigarobolearn` project (or the folder where `models` should be created):

| Flag | Description |
|------|-------------|
| *(none)* | Train using internal physics. Resumes from latest checkpoint in `models/` if present. |
| `--steps N` | Train for at most N more steps (e.g. `--steps 5000000`). |
| `--eval` | Run 20 evaluation episodes with the latest checkpoint (deterministic policy). No training. |
| `--grsim` | Send live game state to grSim for visualisation (see below). |
| `--seed N` | Set random seed for PPO and environment (e.g. `--seed 1234`). |
| `--norand` | Disable random start positions; use fixed kick-off every episode. |
| `--out-rule` | When the ball goes out, respawn ball and robots in random positions instead of bouncing. |
| `--quiet` | Less console output. |

**Examples**

```text
gigarobolearn.exe
gigarobolearn.exe --steps 1000000 --seed 42
gigarobolearn.exe --eval
gigarobolearn.exe --grsim --steps 500000
```

Checkpoints are written every **50k** steps (see `CHECKPOINT_EVERY` in `Config.h`). The last **5** checkpoints are kept; older ones are removed. Checkpoint directories are `models/ckpt_<timestep>/` and contain the actor, critic, and observation stats.

---

## grSim visualisation

With `--grsim`, the trainer sends the current ball and robot positions to grSim so you can watch the game in real time.

- Start grSim first (or your grSim-compatible sim) so it is listening.
- Default host: `127.0.0.1`. Command port: **10300**, sim control port: **10400** (see `Config.h`: `GRSIM_HOST`, `GRSIM_CMD_PORT`, `GRSIM_SIM_CONTROL_PORT`).
- The trainer sends state to both the command and sim-control ports so the sim shows correct orientations (e.g. kicker direction).
- When `--grsim` is used, the process runs at roughly real-time speed (60 Hz step) so the sim view is smooth; training is effectively one game at a time.

If the connection fails, the trainer prints a warning and continues without visualisation.

---

## Configuration and rewards

- **Constants (field, physics, PPO hyperparameters, network size):** `src/Config.h`
- **Reward weights and scales:** `src/RewardWeights.h`

The policy is an MLP: **obs (24) → 768 → 768 → 768 → 384 → 18 actions**, with ReLU and Adam. Observation includes ball and robot positions/velocities, orientations, and kicker-to-ball angle. Actions are discrete (stop, move, rotate, kick, chip, dribble, combos). All of this is fixed in code; change `Config.h` and `RewardWeights.h` to tune.

---

## Project layout

```text
gigarobolearn/
  gigarobolearn.slnx
  gigarobolearn/
    gigarobolearn.cpp    # main, CLI, eval loop
    gigarobolearn.vcxproj
    src/
      Config.h           # Field, PPO, and sim constants
      RewardWeights.h    # Reward scaling
      Environment.h/cpp  # physics, ball/robot state, observations
      Rewards.h/cpp     # Reward computation from step info
      SelfPlay.h/cpp    # Training loop, checkpoints, grSim hook
      PPO.h/cpp         # PPO + GAE, actor/critic, Adam
      NeuralNet.h/cpp   # MLP, ReLU, serialisation
      MLMath.h          # Vec, matvec, softmax, Welford, etc.
      GrSimClient.h/cpp # UDP client for grSim commands/state
      ModelManager.h/cpp# Checkpoint save/load and pruning
```

---

## License

See the repository for license information.
