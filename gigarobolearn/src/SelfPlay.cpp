#include "SelfPlay.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <thread>

SelfPlayTrainer::SelfPlayTrainer(const TrainConfig& cfg)
    : cfg_(cfg)
    , ppo_(cfg.ppoSeed)
    , env_(cfg.envSeed)
    , mgr_(MODELS_DIR, MAX_CHECKPOINTS)
    , rewards_(makeDefaultRewards())
{
    nextCkpt_ = cfg_.checkpointEvery;

    if (cfg_.applyOutRule)
        env_.setApplyOutRule(true);

    if (cfg_.useGrSim) {
        grSim_ = std::make_unique<GrSimClient>();
        if (!grSim_->init()) {
            std::cerr << "[SelfPlay] WARNING: Could not connect to gigarobosim – is the sim running?\n"
                      << "  Start the sim first:  cd gigarobosim && python sim.py\n"
                      << "  Then run: gigarobolearn.exe --grsim\n"
                      << "  Continuing without visualisation.\n";
            grSim_.reset();
        } else {
            std::cout << "[SelfPlay] Sending game preview to gigarobosim at " << GRSIM_HOST
                      << ":" << GRSIM_CMD_PORT << " (you should see the match in the sim window).\n"
                      << "[SelfPlay] Running at real-time speed (60 Hz) for viewing – one game at a time.\n";
            lastSimSend_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(100);
        }
    }

    if (mgr_.hasCheckpoints()) {
        mgr_.loadLatest(ppo_);
        totalSteps_ = mgr_.latestTimestep();
        nextCkpt_   = totalSteps_ + cfg_.checkpointEvery;
        cfg_.totalTimesteps += totalSteps_;
        std::cout << "[SelfPlay] Resumed from step " << totalSteps_ << "\n";
    } else {
        std::cout << "[SelfPlay] No checkpoint found - starting fresh training.\n";
    }
}

void SelfPlayTrainer::startEpisode() {
    env_.reset(cfg_.randomStart);
    blueObs_   = env_.getObservation(false);
    yellowObs_ = env_.getObservation(true);
    prevState_ = env_.state();
    epRewardBlue_ = epRewardYellow_ = 0;
    epSteps_ = 0;
    epCollisionsCurr_ = 0;
    episodeDone_ = false;
}

bool SelfPlayTrainer::runStep() {
    if (grSim_) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastSimSend_).count();
        if (elapsed < (double)SIM_DT) {
            auto sleep_us = (long)(((double)SIM_DT - elapsed) * 1e6);
            if (sleep_us > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
        lastSimSend_ = std::chrono::steady_clock::now();
    }

    auto [blueAction,   blueLogP]   = ppo_.selectAction(blueObs_,   false);
    auto [yellowAction, yellowLogP] = ppo_.selectAction(yellowObs_, false);
    float blueV   = ppo_.getValue(blueObs_);
    float yellowV = ppo_.getValue(yellowObs_);
    bool done = env_.step(blueAction, yellowAction);

    if (grSim_) {
        grSim_->sendState(env_.state());
    }

    StepInfo info{prevState_, env_.state(),
                  env_.blueTouchedBall(), env_.yellowTouchedBall(),
                  env_.blueKickedBall(),  env_.yellowKickedBall(),
                  env_.robotsCollidedThisStep(), true,
                  env_.ballJustWentOut()};

    float blueReward   = computeReward(rewards_, info, true);
    float yellowReward = computeReward(rewards_, info, false);

    ppo_.updateObsStats(blueObs_);
    ppo_.updateObsStats(yellowObs_);

    ppo_.addTransition(false, blueObs_,   blueAction,   blueLogP,   blueReward,   blueV,   done);
    ppo_.addTransition(true,  yellowObs_, yellowAction, yellowLogP, yellowReward, yellowV, done);

    epRewardBlue_   += blueReward;
    epRewardYellow_ += yellowReward;
    if (env_.robotsCollidedThisStep()) epCollisionsCurr_++;
    epSteps_++;
    totalSteps_ += 2;

    Vec nextBlue   = env_.getObservation(false);
    Vec nextYellow = env_.getObservation(true);

    prevState_ = env_.state();
    blueObs_   = nextBlue;
    yellowObs_ = nextYellow;

    if (done) {
        episodeDone_ = true;
        if (env_.state().goalBy == 0)
            epRewards_.push_back(epRewardBlue_);
        else if (env_.state().goalBy == 1)
            epRewards_.push_back(epRewardYellow_);
        else
            epRewards_.push_back((epRewardBlue_ + epRewardYellow_) * 0.5f);
        epLens_.push_back((float)epSteps_);
        epGoals_.push_back(env_.state().blueScore + env_.state().yellowScore);
        epGoalBy_.push_back(env_.state().goalBy);
        epLastToucher_.push_back(env_.state().lastToucherTeam);
        epCollisions_.push_back(epCollisionsCurr_);
    }

    return done;
}

IterStats SelfPlayTrainer::collectAndTrain() {
    auto tStart = std::chrono::steady_clock::now();

    if (episodeDone_) startEpisode();

    while (!ppo_.readyToTrain()) {
        bool done = runStep();
        if (done) startEpisode();
    }

    Vec lastBlue   = blueObs_;
    Vec lastYellow = yellowObs_;
    PPOStats ppoStats = ppo_.train(lastBlue, lastYellow,
                                   episodeDone_, episodeDone_);

    IterStats s;
    s.ppo        = ppoStats;
    s.totalSteps = totalSteps_;

    if (!epRewards_.empty()) {
        s.nEpisodes = (int)epRewards_.size();
        float sum = 0; for (float r : epRewards_) sum += r;
        s.meanReward = sum / epRewards_.size();
        int nScored = 0;
        float sumScored = 0;
        for (size_t i = 0; i < epRewards_.size(); ++i) {
            if (epGoals_[i] >= 1) {
                nScored++;
                sumScored += epRewards_[i];
            }
        }
        s.nEpisodesWithGoal = nScored;
        if (nScored > 0)
            s.meanRewardWhenScored = sumScored / nScored;
    }
    if (!epLens_.empty()) {
        float sum = 0; for (float l : epLens_) sum += l;
        s.meanEpLen = sum / epLens_.size();
        s.totalEpSteps = (int)sum;
    }
    for (int g : epGoals_) s.goals += g;

    for (size_t i = 0; i < epGoalBy_.size(); ++i) {
        int gb = epGoalBy_[i];
        if (gb == 0) s.goalsBlue++;
        if (gb == 1) s.goalsYellow++;
        if (gb >= 0 && epLastToucher_.size() > i) {
            int lt = epLastToucher_[i];
            if ((gb == 0 && lt == 1) || (gb == 1 && lt == 0)) s.ownGoals++;
        }
    }
    for (int c : epCollisions_) s.collisionSteps += c;

    auto tEnd = std::chrono::steady_clock::now();
    s.iterSeconds = std::chrono::duration<double>(tEnd - tStart).count();

    epRewards_.clear();
    epLens_.clear();
    epGoals_.clear();
    epGoalBy_.clear();
    epLastToucher_.clear();
    epCollisions_.clear();

    return s;
}

void SelfPlayTrainer::printStats(const IterStats& s) const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[Step " << std::setw(9) << s.totalSteps << "] "
              << "reward=" << std::setw(8) << s.meanReward;
    if (s.goals >= 1 && s.meanRewardWhenScored != 0)
        std::cout << "  goalEpRew=" << std::setw(8) << s.meanRewardWhenScored;
    std::cout << "  epLen=" << std::setw(6) << s.meanEpLen
              << "  goals=" << std::setw(3) << s.goals
              << "  pLoss=" << std::setw(8) << s.ppo.policyLoss
              << "  vLoss=" << std::setw(8) << s.ppo.valueLoss
              << "  ent="   << std::setw(6) << s.ppo.entropy
              << "  KL="    << std::setw(7) << s.ppo.approxKL
              << "  clip="  << std::setw(5) << s.ppo.clipFrac
              << "  " << std::setprecision(1) << s.iterSeconds << "s"
              << "\n";
    if (s.nEpisodes > 0) {
        float goalRate = 100.0f * s.nEpisodesWithGoal / (float)s.nEpisodes;
        float ownGoalPct = (s.goals > 0) ? (100.0f * s.ownGoals / (float)s.goals) : 0.0f;
        float collisionPct = (s.totalEpSteps > 0) ? (100.0f * s.collisionSteps / (float)s.totalEpSteps) : 0.0f;
        std::cout << "         [1v1] eps=" << s.nEpisodes
                  << "  goalRate=" << std::setprecision(1) << goalRate << "%"
                  << "  ownGoal%=" << ownGoalPct << "%"
                  << "  coll%=" << collisionPct << "%"
                  << "  B=" << s.goalsBlue << " Y=" << s.goalsYellow
                  << "\n";
    }
}

void SelfPlayTrainer::run() {
    std::cout << "[SelfPlay] Starting self-play training.\n"
              << "  Target timesteps : " << cfg_.totalTimesteps << "\n"
              << "  Checkpoint every : " << cfg_.checkpointEvery << " steps\n"
              << "  Max checkpoints  : " << MAX_CHECKPOINTS << "\n"
              << "  Models directory : " << MODELS_DIR << "\n"
              << "  [1v1 targets] goalRate>=40%  ownGoal%<5%  coll%<2%  meanReward>0  (see [1v1] line each iter)\n\n";

    while (totalSteps_ < cfg_.totalTimesteps) {
        IterStats s = collectAndTrain();

        if (cfg_.verbose) printStats(s);

        if (totalSteps_ >= nextCkpt_) {
            mgr_.saveCheckpoint(ppo_, totalSteps_, s.meanReward);
            nextCkpt_ = totalSteps_ + cfg_.checkpointEvery;
        }
    }

    std::cout << "\n[SelfPlay] Training complete at step " << totalSteps_ << ".\n";
    IterStats finalS;
    finalS.totalSteps = totalSteps_;
    finalS.meanReward = epRewards_.empty() ? 0.0f :
        [&]{ float s=0; for(float r:epRewards_) s+=r; return s/(float)epRewards_.size(); }();
    mgr_.saveCheckpoint(ppo_, totalSteps_, finalS.meanReward);
}
