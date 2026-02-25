#include "NeuralNet.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

Layer::Layer(int inSz, int outSz, std::mt19937& rng)
    : inSize(inSz), outSize(outSz)
{
    W      = heInit(outSz, inSz, rng);
    b      = zeroVec(outSz);
    mW     = zeroVec(outSz * inSz);
    vW     = zeroVec(outSz * inSz);
    mb     = zeroVec(outSz);
    vb     = zeroVec(outSz);
    gradW  = zeroVec(outSz * inSz);
    gradb  = zeroVec(outSz);
    lastInput  = zeroVec(inSz);
    lastPreact = zeroVec(outSz);
}

Vec Layer::forward(const Vec& x) {
    lastInput = x;
    lastPreact = matvec(W, outSize, inSize, x, b);
    return lastPreact;
}

void Layer::accumGrad(const Vec& delta) {
    outerProductAccum(gradW, outSize, inSize, delta, lastInput);
    vecAddInplace(gradb, delta);
}

void Layer::applyAdam(float lr, int batchSize,
                      float beta1, float beta2, float eps) {
    adamStep++;
    float bc1 = 1.0f - std::pow(beta1, (float)adamStep);
    float bc2 = 1.0f - std::pow(beta2, (float)adamStep);
    float invB = 1.0f / (float)batchSize;

    for (int i = 0; i < (int)W.size(); i++) {
        float g  = gradW[i] * invB;
        mW[i] = beta1 * mW[i] + (1.0f - beta1) * g;
        vW[i] = beta2 * vW[i] + (1.0f - beta2) * g * g;
        float mh = mW[i] / bc1;
        float vh = vW[i] / bc2;
        W[i] -= lr * mh / (std::sqrt(vh) + eps);
        gradW[i] = 0.0f;
    }
    for (int i = 0; i < outSize; i++) {
        float g  = gradb[i] * invB;
        mb[i] = beta1 * mb[i] + (1.0f - beta1) * g;
        vb[i] = beta2 * vb[i] + (1.0f - beta2) * g * g;
        float mh = mb[i] / bc1;
        float vh = vb[i] / bc2;
        b[i] -= lr * mh / (std::sqrt(vh) + eps);
        gradb[i] = 0.0f;
    }
}

void Layer::zeroGrads() {
    std::fill(gradW.begin(), gradW.end(), 0.0f);
    std::fill(gradb.begin(), gradb.end(), 0.0f);
}

static void writeVec(std::ostream& os, const Vec& v) {
    int32_t sz = (int32_t)v.size();
    os.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    os.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(float));
}
static void readVec(std::istream& is, Vec& v) {
    int32_t sz;
    is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    v.resize(sz);
    is.read(reinterpret_cast<char*>(v.data()), sz * sizeof(float));
}

void Layer::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&inSize),   sizeof(inSize));
    os.write(reinterpret_cast<const char*>(&outSize),  sizeof(outSize));
    os.write(reinterpret_cast<const char*>(&adamStep), sizeof(adamStep));
    writeVec(os, W); writeVec(os, b);
    writeVec(os, mW); writeVec(os, vW);
    writeVec(os, mb); writeVec(os, vb);
}
void Layer::load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&inSize),   sizeof(inSize));
    is.read(reinterpret_cast<char*>(&outSize),  sizeof(outSize));
    is.read(reinterpret_cast<char*>(&adamStep), sizeof(adamStep));
    readVec(is, W); readVec(is, b);
    readVec(is, mW); readVec(is, vW);
    readVec(is, mb); readVec(is, vb);
    gradW  = zeroVec((int)W.size());
    gradb  = zeroVec(outSize);
    lastInput  = zeroVec(inSize);
    lastPreact = zeroVec(outSize);
}

Network::Network(const std::vector<int>& sizes, std::mt19937& rng) {
    for (int i = 0; i + 1 < (int)sizes.size(); i++) {
        layers_.emplace_back(sizes[i], sizes[i + 1], rng);
    }
}

Vec Network::forward(const Vec& obs) const {
    Vec a = obs;
    for (int l = 0; l < (int)layers_.size(); l++) {
        Vec z = matvec(layers_[l].W, layers_[l].outSize, layers_[l].inSize, a, layers_[l].b);
        if (l < (int)layers_.size() - 1)
            a = relu(z);
        else
            a = z;
    }
    return a;
}

Vec Network::forwardTrain(const Vec& obs) {
    Vec a = obs;
    for (int l = 0; l < (int)layers_.size(); l++) {
        Vec z = layers_[l].forward(a);
        if (l < (int)layers_.size() - 1)
            a = relu(z);
        else
            a = z;
    }
    return a;
}

void Network::backward(const Vec& lossGrad) {
    int L = (int)layers_.size();
    Vec delta = lossGrad;

    for (int l = L - 1; l >= 0; l--) {
        layers_[l].accumGrad(delta);

        if (l > 0) {
            Vec da = matvecT(layers_[l].W, layers_[l].outSize, layers_[l].inSize, delta);
            const Vec& preact = layers_[l - 1].lastPreact;
            for (int j = 0; j < (int)da.size(); j++) {
                if (preact[j] <= 0.0f) da[j] = 0.0f;
            }
            delta = std::move(da);
        }
    }
}

void Network::applyAdam(float lr, int batchSize, float beta1, float beta2, float eps) {
    for (auto& layer : layers_)
        layer.applyAdam(lr, batchSize, beta1, beta2, eps);
}

void Network::zeroGrads() {
    for (auto& layer : layers_) layer.zeroGrads();
}

int Network::outputSize() const {
    return layers_.empty() ? 0 : layers_.back().outSize;
}
int Network::inputSize() const {
    return layers_.empty() ? 0 : layers_.front().inSize;
}

void Network::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("Cannot write: " + path);
    int32_t n = (int32_t)layers_.size();
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& layer : layers_) layer.save(os);
}
void Network::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Cannot read: " + path);
    int32_t n;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    layers_.resize(n);
    for (auto& layer : layers_) layer.load(is);
}
