#pragma once

#include "NeuralNet.h"
#include "Config.h"
#include <vector>
#include <random>

struct Transition {
    Vec   obs;
    int   action;
    float logProb;
    float reward;
    float value;
    bool  done;
};

struct Sample {
    Vec   obs;
    int   action;
    float oldLogProb;
    float advantage;
    float targetValue;
};

struct PPOStats {
    float policyLoss  = 0;
    float valueLoss   = 0;
    float entropy     = 0;
    float approxKL    = 0;
    float clipFrac    = 0;
    int   numSamples  = 0;
};

class PPOTrainer {
public:
    PPOTrainer();
    explicit PPOTrainer(unsigned seed);

    std::pair<int, float> selectAction(const Vec& obs, bool deterministic = false);

    float getValue(const Vec& obs);

    void addTransition(bool isOpponent,
                       const Vec& obs, int action, float logProb,
                       float reward, float value, bool done);

    bool readyToTrain() const;

    PPOStats train(const Vec& lastObsBlue, const Vec& lastObsYellow,
                   bool blueDone, bool yellowDone);

    void save(const std::string& dir) const;
    void load(const std::string& dir);

    void updateObsStats(const Vec& obs);
    Vec  normaliseObs(const Vec& obs) const;

    const Network& actor()  const { return actor_;  }
    const Network& critic() const { return critic_; }

private:
    Network actor_;
    Network critic_;

    std::vector<Transition> blueBuffer_;
    std::vector<Transition> yellowBuffer_;

    std::mt19937 rng_;

    WelfordStat obsStat_;
    bool normaliseObs_ = true;

    std::vector<Sample> computeGAE(std::vector<Transition>& buf,
                                   float lastValue,
                                   float gamma, float lambda) const;

    PPOStats runPPO(std::vector<Sample>& samples,
                    int epochs, int miniBatch,
                    float clipEps, float entropyCoef, float valueCoef,
                    float actorLR, float criticLR);

    float computeLogProb(const Vec& logits, int action) const;
};
