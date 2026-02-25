#include "ModelManager.h"
#include "PPO.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

ModelManager::ModelManager(const std::string& baseDir, int maxKeep)
    : baseDir_(baseDir), maxKeep_(maxKeep)
{
    fs::create_directories(baseDir_);
}

std::string ModelManager::checkpointDir(long ts) const {
    std::ostringstream ss;
    ss << baseDir_ << "/ckpt_" << std::setw(10) << std::setfill('0') << ts;
    return ss.str();
}

void ModelManager::writeMeta(const std::string& dir, long ts, float avgR) const {
    std::ofstream f(dir + "/meta.txt");
    f << "timestep=" << ts << "\n";
    f << "avg_reward=" << avgR << "\n";
}

CheckpointMeta ModelManager::readMeta(const std::string& dir) const {
    CheckpointMeta m{};
    m.dir = dir;
    std::ifstream f(dir + "/meta.txt");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("timestep=", 0) == 0)
            m.timestep = std::stol(line.substr(9));
        if (line.rfind("avg_reward=", 0) == 0)
            m.avgReward = std::stof(line.substr(11));
    }
    return m;
}

bool ModelManager::hasCheckpoints() const {
    return !list().empty();
}

std::vector<CheckpointMeta> ModelManager::list() const {
    std::vector<CheckpointMeta> result;
    if (!fs::exists(baseDir_)) return result;

    for (const auto& entry : fs::directory_iterator(baseDir_)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        if (name.rfind("ckpt_", 0) != 0) continue;
        auto meta = readMeta(entry.path().string());
        meta.dir = entry.path().string();
        result.push_back(meta);
    }
    std::sort(result.begin(), result.end(),
              [](const CheckpointMeta& a, const CheckpointMeta& b) {
                  return a.timestep > b.timestep;
              });
    return result;
}

long ModelManager::latestTimestep() const {
    auto ckpts = list();
    return ckpts.empty() ? 0L : ckpts.front().timestep;
}

bool ModelManager::loadLatest(PPOTrainer& trainer) const {
    auto ckpts = list();
    if (ckpts.empty()) return false;
    const auto& best = ckpts.front();
    std::cout << "[ModelManager] Loading checkpoint: " << best.dir
              << "  (ts=" << best.timestep << ", avgR=" << best.avgReward << ")\n";
    trainer.load(best.dir);
    return true;
}

void ModelManager::saveCheckpoint(const PPOTrainer& trainer, long ts, float avgR) {
    std::string dir = checkpointDir(ts);
    fs::create_directories(dir);
    trainer.save(dir);
    writeMeta(dir, ts, avgR);
    std::cout << "[ModelManager] Saved checkpoint: " << dir << "\n";
    pruneOld();
}

void ModelManager::pruneOld() {
    auto ckpts = list();
    while ((int)ckpts.size() > maxKeep_) {
        const auto& old = ckpts.back();
        std::cout << "[ModelManager] Removing old checkpoint: " << old.dir << "\n";
        fs::remove_all(old.dir);
        ckpts.pop_back();
    }
}
