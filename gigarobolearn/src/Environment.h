#pragma once
#include "Config.h"
#include "MLMath.h"
#include <random>
#include <cmath>

struct ActionDef {
    float vxLocal, vyLocal, vw;
    float kick;
    bool  chip;
    bool  dribble;
};

extern const ActionDef ACTIONS[NUM_ACTIONS];

struct BallState {
    float x  = 0, y  = 0;
    float vx = 0, vy = 0;
};

struct RobotState {
    float x   = 0, y   = 0;
    float th  = 0;
    float vx  = 0, vy  = 0;
    float vw  = 0;
    bool  hasBall = false;
};

struct GameState {
    BallState  ball;
    RobotState blue;
    RobotState yellow;
    int  blueScore   = 0;
    int  yellowScore = 0;
    int  stepCount   = 0;
    bool done        = false;
    int  goalBy      = -1;
    int  lastToucherTeam = -1;
};

class Environment {
public:
    explicit Environment(unsigned seed = 0);
    void reset(bool randomStart = true);
    bool step(int blueAction, int yellowAction);
    Vec getObservation(bool isYellow) const;
    const GameState& state() const { return state_; }

    bool blueTouchedBall()   const { return blueTouched_;   }
    bool yellowTouchedBall() const { return yellowTouched_; }
    bool blueKickedBall()    const { return blueKicked_;    }
    bool yellowKickedBall()  const { return yellowKicked_; }
    bool robotsCollidedThisStep() const { return robotsCollided_; }
    bool ballJustWentOut() const { return ballJustWentOut_; }
    void setApplyOutRule(bool v) { applyOutRule_ = v; }

private:
    GameState   state_;
    std::mt19937 rng_;
    bool blueTouched_    = false;
    bool yellowTouched_  = false;
    bool blueKicked_     = false;
    bool yellowKicked_   = false;
    bool robotsCollided_ = false;
    bool ballJustWentOut_ = false;
    int  lastToucherTeam_ = -1;
    bool prevBallOut_     = false;
    bool applyOutRule_    = false;

    void applyAction(RobotState& robot, int actionIdx, float dt);
    void updateBall(float dt);
    void resolveWalls(float dt);
    void resolveBallRobotCollision(RobotState& robot, bool isBlue, float dt);
    void resolveRobotBoundary(RobotState& robot);
    void resolveRobotRobotCollision();
    bool checkGoals();
    float randInRange(float lo, float hi);
};
