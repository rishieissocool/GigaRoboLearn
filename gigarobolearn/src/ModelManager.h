#pragma once

#include "Config.h"
#include <string>
#include <vector>

class PPOTrainer;

struct CheckpointMeta {
    long  timestep;
    float avgReward;
    std::string dir;
};

class ModelManager {
public:
    explicit ModelManager(const std::string& baseDir = MODELS_DIR,
                          int maxKeep = MAX_CHECKPOINTS);

    bool hasCheckpoints() const;

    bool loadLatest(PPOTrainer& trainer) const;

    void saveCheckpoint(const PPOTrainer& trainer, long timestep, float avgReward);

    std::vector<CheckpointMeta> list() const;

    long latestTimestep() const;

private:
    std::string baseDir_;
    int         maxKeep_;

    std::string checkpointDir(long timestep) const;
    void        writeMeta(const std::string& dir, long ts, float avgR) const;
    CheckpointMeta readMeta(const std::string& dir) const;
    void        pruneOld();
};
