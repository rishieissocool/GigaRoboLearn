#include "PPO.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

PPOTrainer::PPOTrainer() : PPOTrainer(42u) {}

PPOTrainer::PPOTrainer(unsigned seed)
    : rng_(seed)
    , obsStat_(OBS_SIZE)
{
    actor_  = Network({OBS_SIZE, HIDDEN1, HIDDEN2, HIDDEN3, HIDDEN4, NUM_ACTIONS}, rng_);
    critic_ = Network({OBS_SIZE, HIDDEN1, HIDDEN2, HIDDEN3, HIDDEN4, 1}, rng_);

    blueBuffer_.reserve(STEPS_PER_ITER + 8);
    yellowBuffer_.reserve(STEPS_PER_ITER + 8);
}

void PPOTrainer::updateObsStats(const Vec& obs) {
    obsStat_.update(obs);
}
Vec PPOTrainer::normaliseObs(const Vec& obs) const {
    if (!normaliseObs_ || obsStat_.count < 100) return obs;
    return obsStat_.normalise(obs);
}

std::pair<int, float> PPOTrainer::selectAction(const Vec& obs, bool deterministic) {
    Vec nObs = normaliseObs(obs);
    Vec logits = actor_.forward(nObs);
    Vec probs  = softmax(logits);

    int action;
    if (deterministic) {
        action = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
    } else {
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        action = dist(rng_);
    }
    float logP = std::log(probs[action] + 1e-8f);
    return {action, logP};
}

float PPOTrainer::getValue(const Vec& obs) {
    Vec nObs = normaliseObs(obs);
    Vec v = critic_.forward(nObs);
    return v[0];
}

float PPOTrainer::computeLogProb(const Vec& logits, int action) const {
    Vec probs = softmax(logits);
    return std::log(probs[action] + 1e-8f);
}

void PPOTrainer::addTransition(bool isOpponent,
                                const Vec& obs, int action, float logProb,
                                float reward, float value, bool done) {
    Transition t{obs, action, logProb, reward, value, done};
    if (!isOpponent)
        blueBuffer_.push_back(std::move(t));
    else
        yellowBuffer_.push_back(std::move(t));
}

bool PPOTrainer::readyToTrain() const {
    return (int)(blueBuffer_.size() + yellowBuffer_.size()) >= STEPS_PER_ITER;
}

std::vector<Sample> PPOTrainer::computeGAE(std::vector<Transition>& buf,
                                            float lastValue,
                                            float gamma, float lambda) const {
    int T = (int)buf.size();
    std::vector<Sample> out(T);
    float lastAdv = 0.0f;

    for (int t = T - 1; t >= 0; t--) {
        float nextV = (t == T - 1) ? lastValue : buf[t + 1].value;
        if (buf[t].done) nextV = 0.0f;

        float delta  = buf[t].reward + gamma * nextV - buf[t].value;
        float gaeC   = buf[t].done ? 0.0f : 1.0f;
        lastAdv = delta + gamma * lambda * gaeC * lastAdv;

        float ret = lastAdv + buf[t].value;
        out[t] = {buf[t].obs, buf[t].action, buf[t].logProb, lastAdv, ret};
    }

    float sumA = 0, sumA2 = 0;
    for (auto& s : out) { sumA += s.advantage; sumA2 += s.advantage * s.advantage; }
    float meanA = sumA / T;
    float stdA  = std::sqrt(sumA2 / T - meanA * meanA + 1e-8f);
    for (auto& s : out) s.advantage = (s.advantage - meanA) / stdA;

    return out;
}

PPOStats PPOTrainer::runPPO(std::vector<Sample>& samples,
                             int epochs, int miniBatch,
                             float clipEps, float entropyCoef, float valueCoef,
                             float actorLR, float criticLR) {
    PPOStats stats;
    int N = (int)samples.size();
    stats.numSamples = N;

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    for (int ep = 0; ep < epochs; ep++) {
        std::shuffle(idx.begin(), idx.end(), rng_);

        for (int bStart = 0; bStart < N; bStart += miniBatch) {
            int bEnd = std::min(bStart + miniBatch, N);
            int bSz  = bEnd - bStart;

            actor_.zeroGrads();
            critic_.zeroGrads();

            float bPolicyLoss = 0, bValueLoss = 0, bEntropy = 0;
            float bKL = 0, bClipFrac = 0;

            for (int bi = bStart; bi < bEnd; bi++) {
                const Sample& s = samples[idx[bi]];
                Vec nObs = normaliseObs(s.obs);

                Vec logits = actor_.forwardTrain(nObs);
                Vec probs  = softmax(logits);
                float newLogP = std::log(probs[s.action] + 1e-8f);
                float H = entropy(probs);

                float ratio   = std::exp(newLogP - s.oldLogProb);
                float clipped = clampf(ratio, 1.0f - clipEps, 1.0f + clipEps);

                float surr1 = ratio   * s.advantage;
                float surr2 = clipped * s.advantage;
                bool  isClipped = (surr1 > surr2);

                float pLoss = -std::min(surr1, surr2);
                bPolicyLoss += pLoss;
                bEntropy    += H;
                bKL         += (s.oldLogProb - newLogP);
                bClipFrac   += isClipped ? 1.0f : 0.0f;

                float grad_ratio = isClipped ? 0.0f : -s.advantage;
                float grad_logP = grad_ratio * ratio;

                Vec grad_logits(NUM_ACTIONS);
                for (int j = 0; j < NUM_ACTIONS; j++) {
                    float ind = (j == s.action) ? 1.0f : 0.0f;
                    grad_logits[j] = grad_logP * (ind - probs[j]);
                }
                for (int j = 0; j < NUM_ACTIONS; j++) {
                    float lp = std::log(probs[j] + 1e-8f);
                    grad_logits[j] += entropyCoef * probs[j] * (lp + H);
                }

                actor_.backward(grad_logits);

                Vec vOut = critic_.forwardTrain(nObs);
                float vPred = vOut[0];
                float vLoss = 0.5f * (vPred - s.targetValue) * (vPred - s.targetValue);
                bValueLoss += vLoss;

                Vec grad_v(1);
                grad_v[0] = valueCoef * (vPred - s.targetValue);
                critic_.backward(grad_v);
            }

            actor_.applyAdam(actorLR,  bSz, ADAM_BETA1, ADAM_BETA2, ADAM_EPS);
            critic_.applyAdam(criticLR, bSz, ADAM_BETA1, ADAM_BETA2, ADAM_EPS);

            if (ep == epochs - 1) {
                stats.policyLoss += bPolicyLoss / bSz;
                stats.valueLoss  += bValueLoss  / bSz;
                stats.entropy    += bEntropy     / bSz;
                stats.approxKL   += bKL          / bSz;
                stats.clipFrac   += bClipFrac    / bSz;
            }
        }
    }

    int nBatches = std::max(1, (N + miniBatch - 1) / miniBatch);
    stats.policyLoss /= nBatches;
    stats.valueLoss  /= nBatches;
    stats.entropy    /= nBatches;
    stats.approxKL   /= nBatches;
    stats.clipFrac   /= nBatches;

    return stats;
}

PPOStats PPOTrainer::train(const Vec& lastObsBlue, const Vec& lastObsYellow,
                            bool blueDone, bool yellowDone) {
    float blueBootstrap   = blueDone   ? 0.0f : getValue(lastObsBlue);
    float yellowBootstrap = yellowDone ? 0.0f : getValue(lastObsYellow);

    auto blueSamples   = computeGAE(blueBuffer_,   blueBootstrap,   GAMMA, GAE_LAMBDA);
    auto yellowSamples = computeGAE(yellowBuffer_, yellowBootstrap, GAMMA, GAE_LAMBDA);

    std::vector<Sample> allSamples;
    allSamples.reserve(blueSamples.size() + yellowSamples.size());
    allSamples.insert(allSamples.end(), blueSamples.begin(),   blueSamples.end());
    allSamples.insert(allSamples.end(), yellowSamples.begin(), yellowSamples.end());

    {
        float sum = 0, sum2 = 0;
        int T = (int)allSamples.size();
        for (auto& s : allSamples) { sum += s.advantage; sum2 += s.advantage * s.advantage; }
        float mean = sum / T;
        float std  = std::sqrt(sum2 / T - mean * mean + 1e-8f);
        for (auto& s : allSamples) s.advantage = (s.advantage - mean) / std;
    }

    PPOStats stats = runPPO(allSamples, PPO_EPOCHS, MINI_BATCH,
                             CLIP_EPS, ENTROPY_COEF, VALUE_COEF,
                             ACTOR_LR, CRITIC_LR);

    blueBuffer_.clear();
    yellowBuffer_.clear();

    return stats;
}

void PPOTrainer::save(const std::string& dir) const {
    fs::create_directories(dir);
    actor_.save(dir + "/actor.bin");
    critic_.save(dir + "/critic.bin");

    std::ofstream os(dir + "/obs_stats.bin", std::ios::binary);
    int32_t n = OBS_SIZE;
    os.write(reinterpret_cast<const char*>(&n),            sizeof(n));
    os.write(reinterpret_cast<const char*>(&obsStat_.count), sizeof(obsStat_.count));
    os.write(reinterpret_cast<const char*>(obsStat_.mean.data()), n * sizeof(float));
    os.write(reinterpret_cast<const char*>(obsStat_.M2.data()),   n * sizeof(float));
}

void PPOTrainer::load(const std::string& dir) {
    actor_.load(dir + "/actor.bin");
    critic_.load(dir + "/critic.bin");

    std::ifstream is(dir + "/obs_stats.bin", std::ios::binary);
    if (is) {
        int32_t n;
        is.read(reinterpret_cast<char*>(&n), sizeof(n));
        is.read(reinterpret_cast<char*>(&obsStat_.count), sizeof(obsStat_.count));
        obsStat_.mean.resize(n);
        obsStat_.M2.resize(n);
        is.read(reinterpret_cast<char*>(obsStat_.mean.data()), n * sizeof(float));
        is.read(reinterpret_cast<char*>(obsStat_.M2.data()),   n * sizeof(float));
    }
}
