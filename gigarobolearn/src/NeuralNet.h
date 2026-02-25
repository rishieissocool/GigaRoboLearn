#pragma once

#include "MLMath.h"
#include <vector>
#include <string>
#include <random>
#include <iosfwd>

struct Layer {
    int inSize, outSize;

    Vec W;
    Vec b;

    Vec mW, vW;
    Vec mb, vb;
    int adamStep = 0;

    Vec lastInput;
    Vec lastPreact;

    Vec gradW;
    Vec gradb;

    Layer() = default;
    Layer(int inSz, int outSz, std::mt19937& rng);

    Vec forward(const Vec& x);

    void accumGrad(const Vec& delta);

    void applyAdam(float lr, int batchSize,
                   float beta1, float beta2, float eps);

    void zeroGrads();

    void save(std::ostream& os) const;
    void load(std::istream& is);
};

class Network {
public:
    Network() = default;
    Network(const std::vector<int>& sizes, std::mt19937& rng);

    Vec forward(const Vec& obs) const;

    Vec forwardTrain(const Vec& obs);

    void backward(const Vec& lossGrad);

    void applyAdam(float lr, int batchSize,
                   float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    void zeroGrads();

    int outputSize() const;
    int inputSize() const;

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::vector<Layer> layers_;
};
