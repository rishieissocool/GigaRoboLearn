#include "src/SelfPlay.h"
#include "src/Config.h"
#include "src/PPO.h"
#include "src/ModelManager.h"
#include "src/Environment.h"
#include <iostream>
#include <string>
#include <cstdlib>

static void runEval(int numEpisodes = 20) {
    std::cout << "=== Evaluation mode ===\n";

    ModelManager mgr;
    PPOTrainer   ppo;

    if (!mgr.loadLatest(ppo)) {
        std::cout << "No checkpoint found. Train first.\n";
        return;
    }

    Environment env(0);
    auto rewards = makeDefaultRewards();

    int blueWins = 0, yellowWins = 0, draws = 0;
    float totalReward = 0;

    for (int ep = 0; ep < numEpisodes; ep++) {
        env.reset(false);  // standard start
        float epRew = 0;
        GameState prev = env.state();

        for (int step = 0; step < MAX_EP_STEPS; step++) {
            Vec bo = env.getObservation(false);
            Vec yo = env.getObservation(true);

            auto [ba, _b] = ppo.selectAction(bo, true);
            auto [ya, _y] = ppo.selectAction(yo, true);

            bool done = env.step(ba, ya);

            StepInfo info{prev, env.state(),
                          env.blueTouchedBall(), env.yellowTouchedBall(),
                          env.blueKickedBall(),  env.yellowKickedBall(),
                          env.robotsCollidedThisStep(), true,
                          env.ballJustWentOut()};
            epRew += computeReward(rewards, info, true);
            prev = env.state();

            if (done) break;
        }

        totalReward += epRew;
        int bs = env.state().blueScore, ys = env.state().yellowScore;
        std::cout << "  Ep " << ep + 1
                  << ": blue=" << bs << " yellow=" << ys
                  << "  reward=" << epRew << "\n";
        if (bs > ys) blueWins++;
        else if (ys > bs) yellowWins++;
        else draws++;
    }

    std::cout << "\n  Blue wins  : " << blueWins
              << "\n  Yellow wins: " << yellowWins
              << "\n  Draws      : " << draws
              << "\n  Avg reward : " << totalReward / numEpisodes << "\n";
}

int main(int argc, char* argv[]) {
    TrainConfig cfg;
    bool evalMode = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--eval") {
            evalMode = true;
        } else if (arg == "--grsim") {
            cfg.useGrSim = true;
            std::cout << "[main] Visualisation mode: will send game preview to gigarobosim (start the sim first).\n";
        } else if (arg == "--steps" && i + 1 < argc) {
            cfg.totalTimesteps = std::stol(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            cfg.ppoSeed = (unsigned)std::stoul(argv[++i]);
            cfg.envSeed = cfg.ppoSeed;
        } else if (arg == "--norand") {
            cfg.randomStart = false;
        } else if (arg == "--quiet") {
            cfg.verbose = false;
        } else if (arg == "--out-rule") {
            cfg.applyOutRule = true;
        } else if (arg == "--ckpt" && i + 1 < argc) {
        } else {
            std::cout << "Unknown argument: " << arg << "\n";
        }
    }

    std::cout << "===========================================\n"
              << "  SSL Robot Soccer - RL Self-Play Trainer\n"
              << "===========================================\n"
              << "  Algorithm : PPO (Proximal Policy Optimisation)\n"
              << "  Obs size  : " << OBS_SIZE  << "\n"
              << "  Actions   : " << NUM_ACTIONS << "\n"
              << "  Net arch  : " << OBS_SIZE << " -> "
              << HIDDEN1 << " -> " << HIDDEN2 << " -> " << HIDDEN3 << " -> " << HIDDEN4 << " -> actions\n"
              << "  Steps/iter: " << STEPS_PER_ITER << "\n"
              << "  Steps (+) : " << cfg.totalTimesteps << " more\n"
              << "  Seed      : " << cfg.ppoSeed << "\n"
              << "===========================================\n\n";

    if (evalMode) {
        runEval();
        return 0;
    }

    SelfPlayTrainer trainer(cfg);
    trainer.run();

    return 0;
}
