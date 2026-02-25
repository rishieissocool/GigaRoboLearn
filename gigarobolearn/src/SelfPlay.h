#pragma once
#include "PPO.h"
#include "Environment.h"
#include "GrSimClient.h"
#include "Rewards.h"
#include "ModelManager.h"
#include <vector>
#include <string>
#include <chrono>
#include <memory>

struct TrainConfig {
    long  totalTimesteps   = DEFAULT_TOTAL_STEPS;
    int   maxEpSteps       = MAX_EP_STEPS;
    bool  randomStart      = true;
    long  checkpointEvery  = CHECKPOINT_EVERY;
    bool  verbose          = true;
    bool  printIterStats   = true;
    bool  useGrSim         = false;
    bool  applyOutRule     = false;
    unsigned envSeed       = 0;
    unsigned ppoSeed       = 42;
};

struct IterStats {
    long  totalSteps  = 0;
    float meanReward  = 0;
    float meanRewardWhenScored = 0;
    float meanEpLen   = 0;
    int   goals       = 0;
    int   nEpisodes         = 0;
    int   nEpisodesWithGoal = 0;
    int   goalsBlue         = 0;
    int   goalsYellow       = 0;
    int   ownGoals          = 0;
    int   collisionSteps    = 0;
    int   totalEpSteps     = 0;
    PPOStats ppo;
    double iterSeconds = 0;
};

class SelfPlayTrainer {
public:
    explicit SelfPlayTrainer(const TrainConfig& cfg = {});
    void run();

private:
    TrainConfig   cfg_;
    PPOTrainer    ppo_;
    Environment   env_;
    ModelManager  mgr_;
    std::vector<IReward*> rewards_;
    long totalSteps_    = 0;
    long totalEpisodes_ = 0;
    long nextCkpt_      = 0;
    Vec  blueObs_, yellowObs_;
    bool episodeDone_ = true;
    GameState prevState_;
    float epRewardBlue_   = 0;
    float epRewardYellow_ = 0;
    int   epSteps_        = 0;
    std::unique_ptr<GrSimClient> grSim_;
    std::chrono::steady_clock::time_point lastSimSend_;
    std::vector<float> epRewards_;
    std::vector<float> epLens_;
    std::vector<int>   epGoals_;
    std::vector<int>   epGoalBy_;
    std::vector<int>   epLastToucher_;
    std::vector<int>   epCollisions_;
    int   epCollisionsCurr_ = 0;

    void startEpisode();
    bool runStep();
    IterStats collectAndTrain();
    void printStats(const IterStats& s) const;
};
