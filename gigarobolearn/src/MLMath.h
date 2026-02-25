#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <stdlib.h>

#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>

using Vec = std::vector<float>;

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline Vec vecAdd(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] + b[i];
    return r;
}
inline Vec vecSub(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] - b[i];
    return r;
}
inline Vec vecMul(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] * b[i];
    return r;
}
inline Vec vecScale(const Vec& v, float s) {
    Vec r(v.size());
    for (size_t i = 0; i < v.size(); i++) r[i] = v[i] * s;
    return r;
}
inline void vecAddInplace(Vec& a, const Vec& b) {
    for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
}
inline void vecScaleInplace(Vec& v, float s) {
    for (auto& x : v) x *= s;
}
inline float vecDot(const Vec& a, const Vec& b) {
    float s = 0;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}
inline float vecNorm(const Vec& v) {
    return std::sqrt(vecDot(v, v));
}
inline float vecSum(const Vec& v) {
    float s = 0;
    for (float x : v) s += x;
    return s;
}
inline float vecMean(const Vec& v) {
    return vecSum(v) / (float)v.size();
}
inline float vecVar(const Vec& v, float mean) {
    float s = 0;
    for (float x : v) { float d = x - mean; s += d * d; }
    return s / (float)v.size();
}

inline Vec matvec(const Vec& W, int outSize, int inSize, const Vec& x, const Vec& b) {
    Vec y(outSize, 0.0f);
    for (int i = 0; i < outSize; i++) {
        float acc = b[i];
        const float* row = W.data() + i * inSize;
        for (int j = 0; j < inSize; j++) acc += row[j] * x[j];
        y[i] = acc;
    }
    return y;
}

inline Vec matvecT(const Vec& W, int outSize, int inSize, const Vec& x) {
    Vec r(inSize, 0.0f);
    for (int i = 0; i < outSize; i++) {
        const float* row = W.data() + i * inSize;
        for (int j = 0; j < inSize; j++) r[j] += row[j] * x[i];
    }
    return r;
}

inline void outerProductAccum(Vec& G, int outSize, int inSize,
                               const Vec& delta, const Vec& input) {
    for (int i = 0; i < outSize; i++) {
        float* row = G.data() + i * inSize;
        float d = delta[i];
        for (int j = 0; j < inSize; j++) row[j] += d * input[j];
    }
}

inline Vec relu(const Vec& x) {
    Vec r(x.size());
    for (size_t i = 0; i < x.size(); i++) r[i] = x[i] > 0 ? x[i] : 0;
    return r;
}
inline Vec reluGrad(const Vec& preact) {
    Vec g(preact.size());
    for (size_t i = 0; i < preact.size(); i++) g[i] = preact[i] > 0 ? 1.0f : 0.0f;
    return g;
}

inline Vec softmax(const Vec& x) {
    float maxVal = *std::max_element(x.begin(), x.end());
    Vec r(x.size());
    float sum = 0;
    for (size_t i = 0; i < x.size(); i++) { r[i] = std::exp(x[i] - maxVal); sum += r[i]; }
    for (auto& v : r) v /= sum;
    return r;
}

inline Vec logSoftmax(const Vec& x) {
    float maxVal = *std::max_element(x.begin(), x.end());
    float logSum = 0;
    for (float xi : x) logSum += std::exp(xi - maxVal);
    logSum = std::log(logSum) + maxVal;
    Vec r(x.size());
    for (size_t i = 0; i < x.size(); i++) r[i] = x[i] - logSum;
    return r;
}

inline float entropy(const Vec& probs) {
    float H = 0;
    for (float p : probs) {
        if (p > 1e-9f) H -= p * std::log(p);
    }
    return H;
}

inline Vec heInit(int outSize, int inSize, std::mt19937& rng) {
    float std = std::sqrt(2.0f / inSize);
    std::normal_distribution<float> dist(0.0f, std);
    Vec W(outSize * inSize);
    for (auto& w : W) w = dist(rng);
    return W;
}
inline Vec zeroVec(int n) { return Vec(n, 0.0f); }
inline Vec onesVec(int n) { return Vec(n, 1.0f); }

struct WelfordStat {
    Vec mean, M2;
    long count = 0;
    int size;
    explicit WelfordStat(int n) : mean(n, 0.0f), M2(n, 0.0f), size(n) {}

    void update(const Vec& x) {
        count++;
        float c = (float)count;
        for (int i = 0; i < size; i++) {
            float delta = x[i] - mean[i];
            mean[i] += delta / c;
            M2[i] += delta * (x[i] - mean[i]);
        }
    }
    Vec variance() const {
        Vec v(size);
        float c = count > 1 ? (float)(count - 1) : 1.0f;
        for (int i = 0; i < size; i++) v[i] = M2[i] / c;
        return v;
    }
    Vec stddev() const {
        Vec v = variance();
        for (auto& x : v) x = std::sqrt(x + 1e-8f);
        return v;
    }
    Vec normalise(const Vec& x) const {
        Vec r(size);
        Vec sd = stddev();
        for (int i = 0; i < size; i++) r[i] = (x[i] - mean[i]) / sd[i];
        return r;
    }
};
