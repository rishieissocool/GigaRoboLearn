#include "Environment.h"
#include <cmath>
#include <algorithm>

const ActionDef ACTIONS[NUM_ACTIONS] = {
    {  0.0f,   0.0f,  0.0f, 0.0f, false, false },
    {  3.0f,   0.0f,  0.0f, 0.0f, false, false },
    { -3.0f,   0.0f,  0.0f, 0.0f, false, false },
    {  0.0f,   3.0f,  0.0f, 0.0f, false, false },
    {  0.0f,  -3.0f,  0.0f, 0.0f, false, false },
    {  2.12f,  2.12f, 0.0f, 0.0f, false, false },
    {  2.12f, -2.12f, 0.0f, 0.0f, false, false },
    { -2.12f,  2.12f, 0.0f, 0.0f, false, false },
    { -2.12f, -2.12f, 0.0f, 0.0f, false, false },
    {  0.0f,   0.0f,  6.0f, 0.0f, false, false },
    {  0.0f,   0.0f, -6.0f, 0.0f, false, false },
    {  2.0f,   0.0f,  4.0f, 0.0f, false, false },
    {  2.0f,   0.0f, -4.0f, 0.0f, false, false },
    {  0.0f,   0.0f,  0.0f, KICK_SPEED, false, false },
    {  0.0f,   0.0f,  0.0f, CHIP_SPEED, true,  false },
    {  3.0f,   0.0f,  0.0f, KICK_SPEED, false, false },
    {  0.0f,   0.0f,  0.0f, 0.0f, false, true  },
    {  2.0f,   0.0f,  0.0f, 0.0f, false, true  },
};

Environment::Environment(unsigned seed) : rng_(seed) {
    reset(true);
}

float Environment::randInRange(float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng_);
}

void Environment::reset(bool randomStart) {
    state_.blueScore        = 0;
    state_.yellowScore      = 0;
    state_.stepCount        = 0;
    state_.done             = false;
    state_.goalBy           = -1;
    state_.lastToucherTeam  = -1;
    blueTouched_ = yellowTouched_ = blueKicked_ = yellowKicked_ = false;
    robotsCollided_   = false;
    ballJustWentOut_  = false;
    lastToucherTeam_  = -1;
    prevBallOut_      = false;

    if (randomStart) {
        state_.ball.x  = randInRange(-FIELD_HALF_LEN * 0.3f, FIELD_HALF_LEN * 0.3f);
        state_.ball.y  = randInRange(-FIELD_HALF_W   * 0.8f, FIELD_HALF_W   * 0.8f);
        state_.ball.vx = randInRange(-0.5f, 0.5f);
        state_.ball.vy = randInRange(-0.5f, 0.5f);

        state_.blue.x  = randInRange(-FIELD_HALF_LEN * 0.9f, -0.5f);
        state_.blue.y  = randInRange(-FIELD_HALF_W   * 0.8f,  FIELD_HALF_W * 0.8f);
        state_.blue.th = randInRange(-3.14159f, 3.14159f);
        state_.blue.vx = state_.blue.vy = state_.blue.vw = 0;
        state_.blue.hasBall = false;

        state_.yellow.x  = randInRange(0.5f, FIELD_HALF_LEN * 0.9f);
        state_.yellow.y  = randInRange(-FIELD_HALF_W * 0.8f, FIELD_HALF_W * 0.8f);
        state_.yellow.th = randInRange(-3.14159f, 3.14159f);
        state_.yellow.vx = state_.yellow.vy = state_.yellow.vw = 0;
        state_.yellow.hasBall = false;
    } else {
        state_.ball = {};
        state_.blue  = { -1.0f, 0.0f, 0.0f, 0, 0, 0, false };
        state_.yellow = {  1.0f, 0.0f, 3.14159f, 0, 0, 0, false };
    }
}

void Environment::applyAction(RobotState& robot, int actionIdx, float dt) {
    const ActionDef& a = ACTIONS[actionIdx];

    float c = std::cos(robot.th);
    float s = std::sin(robot.th);
    float vxGlobal = a.vxLocal * c - a.vyLocal * s;
    float vyGlobal = a.vxLocal * s + a.vyLocal * c;

    float tau = 0.05f;
    float alpha = dt / (tau + dt);
    robot.vx = robot.vx + alpha * (vxGlobal - robot.vx);
    robot.vy = robot.vy + alpha * (vyGlobal - robot.vy);
    robot.vw = robot.vw + alpha * (a.vw       - robot.vw);

    float spd = std::sqrt(robot.vx * robot.vx + robot.vy * robot.vy);
    if (spd > MAX_ROBOT_SPEED) {
        float sc = MAX_ROBOT_SPEED / spd;
        robot.vx *= sc; robot.vy *= sc;
    }
    if (std::abs(robot.vw) > MAX_ROBOT_ANG_SPEED)
        robot.vw = (robot.vw > 0 ? 1 : -1) * MAX_ROBOT_ANG_SPEED;
}

void Environment::updateBall(float dt) {
    float spd = std::sqrt(state_.ball.vx * state_.ball.vx +
                          state_.ball.vy * state_.ball.vy);
    if (spd > 0.01f) {
        float fric = BALL_FRICTION * dt;
        float newSpd = std::max(0.0f, spd - fric);
        float scale  = newSpd / spd;
        state_.ball.vx *= scale;
        state_.ball.vy *= scale;
    } else {
        state_.ball.vx = 0; state_.ball.vy = 0;
    }

    state_.ball.x += state_.ball.vx * dt;
    state_.ball.y += state_.ball.vy * dt;
}

void Environment::resolveWalls(float dt) {
    if (state_.ball.y + BALL_RADIUS > FIELD_HALF_W_BOUND) {
        state_.ball.y  = FIELD_HALF_W_BOUND - BALL_RADIUS;
        state_.ball.vy = -std::abs(state_.ball.vy) * WALL_RESTITUTION;
    }
    if (state_.ball.y - BALL_RADIUS < -FIELD_HALF_W_BOUND) {
        state_.ball.y  = -FIELD_HALF_W_BOUND + BALL_RADIUS;
        state_.ball.vy =  std::abs(state_.ball.vy) * WALL_RESTITUTION;
    }
    bool inGoalY = std::abs(state_.ball.y) < GOAL_HALF_W;
    if (!inGoalY) {
        if (state_.ball.x + BALL_RADIUS > FIELD_HALF_LEN_BOUND) {
            state_.ball.x  = FIELD_HALF_LEN_BOUND - BALL_RADIUS;
            state_.ball.vx = -std::abs(state_.ball.vx) * WALL_RESTITUTION;
        }
        if (state_.ball.x - BALL_RADIUS < -FIELD_HALF_LEN_BOUND) {
            state_.ball.x  = -FIELD_HALF_LEN_BOUND + BALL_RADIUS;
            state_.ball.vx =  std::abs(state_.ball.vx) * WALL_RESTITUTION;
        }
    }
}

void Environment::resolveBallRobotCollision(RobotState& robot, bool isBlue, float dt) {
    float dx = state_.ball.x - robot.x;
    float dy = state_.ball.y - robot.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    float minDist = ROBOT_RADIUS + BALL_RADIUS;

    if (dist < minDist + 0.001f) {
        if (isBlue) blueTouched_   = true;
        else        yellowTouched_ = true;

        if (dist < 0.0001f) { dx = 1; dy = 0; dist = 1; }

        float nx = dx / dist, ny = dy / dist;

        float overlap = minDist - dist;
        state_.ball.x += nx * overlap * 1.05f;
        state_.ball.y += ny * overlap * 1.05f;

        float relVx = state_.ball.vx - robot.vx;
        float relVy = state_.ball.vy - robot.vy;
        float relVn = relVx * nx + relVy * ny;

        if (relVn < 0) {
            float imp = -(1.0f + WALL_RESTITUTION) * relVn * 0.6f;
            state_.ball.vx += imp * nx;
            state_.ball.vy += imp * ny;
        }
    }

    if (robot.hasBall) {
        float c = std::cos(robot.th), s = std::sin(robot.th);
        float holdDist = ROBOT_RADIUS + BALL_RADIUS + 0.005f;
        state_.ball.x  = robot.x + c * holdDist;
        state_.ball.y  = robot.y + s * holdDist;
        state_.ball.vx = robot.vx;
        state_.ball.vy = robot.vy;
        if (isBlue) blueTouched_ = true;
        else        yellowTouched_ = true;
    }
}

void Environment::resolveRobotBoundary(RobotState& robot) {
    float r = ROBOT_RADIUS;
    if (robot.x < -FIELD_HALF_LEN_BOUND + r) { robot.x = -FIELD_HALF_LEN_BOUND + r; robot.vx = 0; }
    if (robot.x >  FIELD_HALF_LEN_BOUND - r) { robot.x =  FIELD_HALF_LEN_BOUND - r; robot.vx = 0; }
    if (robot.y < -FIELD_HALF_W_BOUND   + r) { robot.y = -FIELD_HALF_W_BOUND   + r; robot.vy = 0; }
    if (robot.y >  FIELD_HALF_W_BOUND   - r) { robot.y =  FIELD_HALF_W_BOUND   - r; robot.vy = 0; }
}

void Environment::resolveRobotRobotCollision() {
    float dx = state_.yellow.x - state_.blue.x;
    float dy = state_.yellow.y - state_.blue.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    float minD = 2.0f * ROBOT_RADIUS;
    if (dist < minD && dist > 0.0001f) {
        robotsCollided_ = true;
        float nx = dx / dist, ny = dy / dist;
        float push = (minD - dist) * 0.5f;
        state_.blue.x   -= nx * push; state_.blue.y   -= ny * push;
        state_.yellow.x += nx * push; state_.yellow.y += ny * push;
        auto killNormal = [&](RobotState& r, float sign) {
            float vn = r.vx * nx + r.vy * ny;
            if (vn * sign < 0) {
                r.vx -= vn * nx * 0.5f;
                r.vy -= vn * ny * 0.5f;
            }
        };
        killNormal(state_.blue,   -1);
        killNormal(state_.yellow, +1);
    }
}

bool Environment::checkGoals() {
    if (state_.ball.x > FIELD_HALF_LEN && std::abs(state_.ball.y) < GOAL_HALF_W) {
        state_.goalBy = 0;
        state_.blueScore++;
        return true;
    }
    if (state_.ball.x < -FIELD_HALF_LEN && std::abs(state_.ball.y) < GOAL_HALF_W) {
        state_.goalBy = 1;
        state_.yellowScore++;
        return true;
    }
    return false;
}

static void tryKick(RobotState& robot, BallState& ball,
                    int actionIdx, bool isBlue,
                    bool& kickedFlag) {
    const ActionDef& act = ACTIONS[actionIdx];
    if (act.kick <= 0.0f) return;

    float dx   = ball.x - robot.x;
    float dy   = ball.y - robot.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > KICKER_RANGE + BALL_RADIUS) return;

    float c = std::cos(robot.th), s = std::sin(robot.th);
    float localX = dx * c + dy * s;
    float localY = -dx * s + dy * c;
    if (localX < 0 || std::abs(localY) > localX * 1.2f) return;

    ball.vx = c * act.kick;
    ball.vy = s * act.kick;
    robot.hasBall = false;
    kickedFlag = true;
}

static void tryDribble(RobotState& robot, const BallState& ball, int actionIdx) {
    const ActionDef& act = ACTIONS[actionIdx];
    if (!act.dribble) { robot.hasBall = false; return; }

    float dx = ball.x - robot.x;
    float dy = ball.y - robot.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > DRIBBLE_RANGE + BALL_RADIUS) { robot.hasBall = false; return; }

    float c = std::cos(robot.th), s = std::sin(robot.th);
    float localX = dx * c + dy * s;
    if (localX < 0) { robot.hasBall = false; return; }

    robot.hasBall = true;
}

bool Environment::step(int blueAction, int yellowAction) {
    blueTouched_ = yellowTouched_ = blueKicked_ = yellowKicked_ = false;
    robotsCollided_ = false;
    state_.goalBy = -1;
    const float dt = SIM_DT;

    applyAction(state_.blue,   blueAction,   dt);
    applyAction(state_.yellow, yellowAction, dt);

    tryDribble(state_.blue,   state_.ball, blueAction);
    tryDribble(state_.yellow, state_.ball, yellowAction);

    tryKick(state_.blue,   state_.ball, blueAction,   true,  blueKicked_);
    tryKick(state_.yellow, state_.ball, yellowAction, false, yellowKicked_);

    state_.blue.x   += state_.blue.vx   * dt;
    state_.blue.y   += state_.blue.vy   * dt;
    state_.blue.th  += state_.blue.vw   * dt;

    state_.yellow.x  += state_.yellow.vx  * dt;
    state_.yellow.y  += state_.yellow.vy  * dt;
    state_.yellow.th += state_.yellow.vw  * dt;

    auto wrapAngle = [](float a) {
        while (a >  3.14159f) a -= 6.28318f;
        while (a < -3.14159f) a += 6.28318f;
        return a;
    };
    state_.blue.th   = wrapAngle(state_.blue.th);
    state_.yellow.th = wrapAngle(state_.yellow.th);

    updateBall(dt);

    resolveBallRobotCollision(state_.blue,   true,  dt);
    resolveBallRobotCollision(state_.yellow, false, dt);
    resolveWalls(dt);
    resolveRobotBoundary(state_.blue);
    resolveRobotBoundary(state_.yellow);
    resolveRobotRobotCollision();

    if (blueTouched_)   lastToucherTeam_ = 0;
    if (yellowTouched_) lastToucherTeam_ = 1;
    state_.lastToucherTeam = lastToucherTeam_;

    bool nowOut = (std::abs(state_.ball.y) > FIELD_HALF_W) ||
                  (std::abs(state_.ball.x) > FIELD_HALF_LEN &&
                   std::abs(state_.ball.y) >= GOAL_HALF_W);
    ballJustWentOut_ = nowOut && !prevBallOut_;
    prevBallOut_     = nowOut;

    if (applyOutRule_ && ballJustWentOut_) {
        state_.ball.x  = randInRange(-FIELD_HALF_LEN * 0.85f, FIELD_HALF_LEN * 0.85f);
        state_.ball.y  = randInRange(-FIELD_HALF_W   * 0.9f,   FIELD_HALF_W   * 0.9f);
        state_.ball.vx = 0.0f;
        state_.ball.vy = 0.0f;

        state_.blue.x   = randInRange(-FIELD_HALF_LEN * 0.9f, -0.4f);
        state_.blue.y   = randInRange(-FIELD_HALF_W   * 0.85f, FIELD_HALF_W   * 0.85f);
        state_.blue.th  = randInRange(-3.14159f, 3.14159f);
        state_.blue.vx  = state_.blue.vy = state_.blue.vw = 0;
        state_.blue.hasBall = false;

        state_.yellow.x   = randInRange(0.4f, FIELD_HALF_LEN * 0.9f);
        state_.yellow.y   = randInRange(-FIELD_HALF_W * 0.85f, FIELD_HALF_W * 0.85f);
        state_.yellow.th  = randInRange(-3.14159f, 3.14159f);
        state_.yellow.vx  = state_.yellow.vy = state_.yellow.vw = 0;
        state_.yellow.hasBall = false;

        lastToucherTeam_       = -1;
        state_.lastToucherTeam = -1;
        prevBallOut_           = false;
    }

    if (checkGoals()) {
        state_.done = true;
        return true;
    }

    state_.stepCount++;
    if (state_.stepCount >= MAX_EP_STEPS) {
        state_.done = true;
        return true;
    }

    return false;
}

Vec Environment::getObservation(bool isYellow) const {
    float xs = isYellow ? -1.0f : 1.0f;

    const RobotState& our  = isYellow ? state_.yellow : state_.blue;
    const RobotState& opp  = isYellow ? state_.blue   : state_.yellow;

    float sinOur  = std::sin(our.th),  cosOur  = std::cos(our.th);
    float sinOpp  = std::sin(opp.th),  cosOpp  = std::cos(opp.th);
    if (isYellow) { cosOur *= -1; cosOpp *= -1; }

    Vec obs(OBS_SIZE);
    obs[0]  = xs * state_.ball.x  / NORM_POS_X;
    obs[1]  =      state_.ball.y  / NORM_POS_Y;
    obs[2]  = xs * state_.ball.vx / NORM_BALL_VEL;
    obs[3]  =      state_.ball.vy / NORM_BALL_VEL;
    obs[4]  = xs * our.x          / NORM_POS_X;
    obs[5]  =      our.y          / NORM_POS_Y;
    obs[6]  =      sinOur;
    obs[7]  =      cosOur;
    obs[8]  = xs * our.vx         / NORM_VEL;
    obs[9]  =      our.vy         / NORM_VEL;
    obs[10] = xs * our.vw         / NORM_ANG;
    obs[11] = xs * (state_.ball.x - our.x) / NORM_POS_X;
    obs[12] =      (state_.ball.y - our.y) / NORM_POS_Y;
    obs[13] = xs * opp.x          / NORM_POS_X;
    obs[14] =      opp.y          / NORM_POS_Y;
    obs[15] =      sinOpp;
    obs[16] =      cosOpp;
    obs[17] = xs * opp.vx         / NORM_VEL;
    obs[18] =      opp.vy         / NORM_VEL;
    obs[19] = xs * opp.vw         / NORM_ANG;
    obs[20] = xs * (state_.ball.x - opp.x) / NORM_POS_X;
    obs[21] =      (state_.ball.y - opp.y) / NORM_POS_Y;

    float dxMir   = xs * (state_.ball.x - our.x);
    float dyMir   =      (state_.ball.y - our.y);
    float distBall = std::sqrt(dxMir * dxMir + dyMir * dyMir);
    if (distBall > 0.01f) {
        obs[22] = (cosOur * dxMir + sinOur * dyMir) / distBall;
        obs[23] = (cosOur * dyMir - sinOur * dxMir) / distBall;
    } else {
        obs[22] = 1.0f;
        obs[23] = 0.0f;
    }

    return obs;
}
