#pragma once
#include "Environment.h"
#include "Config.h"

struct StepInfo {
    const GameState& prevState;
    const GameState& currState;
    bool blueTouched;
    bool yellowTouched;
    bool blueKicked;
    bool yellowKicked;
    bool robotsCollided;
    bool ourGoalIsNegX;
    bool ballOut;
};

struct IReward {
    virtual ~IReward() = default;
    virtual float computeBlue(const StepInfo& info) const = 0;
};

struct GoalReward : IReward {
    float scoreReward   = RW_GOAL;
    float concedeReward = RW_CONCEDE;
    float ownGoalPenalty = RW_OWN_GOAL;
    float computeBlue(const StepInfo& info) const override {
        int goalBy = info.currState.goalBy;
        int lastToucher = info.currState.lastToucherTeam;
        if (goalBy == 0) {
            if (lastToucher == 1) return 0.0f;
            return scoreReward;
        }
        if (goalBy == 1) {
            if (lastToucher == 0) return ownGoalPenalty;
            return concedeReward;
        }
        return 0.0f;
    }
};

struct BallToGoalReward : IReward {
    float weight = RW_BALL_TO_GOAL;
    float computeBlue(const StepInfo& info) const override {
        float vx = info.currState.ball.vx;
        float toward = info.ourGoalIsNegX ? vx : -vx;
        float r = weight * toward / NORM_BALL_VEL;
        return r > 0.f ? r : 0.f;
    }
};

struct BallTowardOwnGoalPenalty : IReward {
    float penalty = RW_BALL_TOWARD_OWN_GOAL;
    float computeBlue(const StepInfo& info) const override {
        float vx = info.currState.ball.vx;
        if (info.ourGoalIsNegX && vx >= 0.f) return 0.f;
        if (!info.ourGoalIsNegX && vx <= 0.f) return 0.f;
        float towardOur = info.ourGoalIsNegX ? -vx : vx;
        return penalty * towardOur / NORM_BALL_VEL;
    }
};

struct KickTowardOwnGoalPenalty : IReward {
    float penalty = RW_KICK_TOWARD_OWN_GOAL;
    float computeBlue(const StepInfo& info) const override {
        if (!info.blueKicked) return 0.f;
        float vx = info.currState.ball.vx;
        if (info.ourGoalIsNegX && vx < 0.f) return penalty;
        if (!info.ourGoalIsNegX && vx > 0.f) return penalty;
        return 0.f;
    }
};

struct MoveToBallReward : IReward {
    float weight = RW_MOVE_TO_BALL;
    float computeBlue(const StepInfo& info) const override {
        auto dist = [](const RobotState& r, const BallState& b) {
            float dx = r.x - b.x, dy = r.y - b.y;
            return std::sqrt(dx * dx + dy * dy);
        };
        float dPrev = dist(info.prevState.blue, info.prevState.ball);
        float dCurr = dist(info.currState.blue, info.currState.ball);
        float r = weight * (dPrev - dCurr) / NORM_POS_X;
        return r > 0.f ? r : 0.f;
    }
};

struct FaceBallReward : IReward {
    float weight = RW_FACE_BALL;
    float computeBlue(const StepInfo& info) const override {
        const RobotState& r = info.currState.blue;
        const BallState&  b = info.currState.ball;
        float dx = b.x - r.x, dy = b.y - r.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 0.01f) return 0;
        float fwd_x = std::cos(r.th), fwd_y = std::sin(r.th);
        float dot = (fwd_x * dx + fwd_y * dy) / dist;
        float rw = weight * dot;
        return rw > 0.f ? rw : 0.f;
    }
};

struct KickerTowardBallReward : IReward {
    float weight = RW_KICKER_TOWARD_BALL;
    float computeBlue(const StepInfo& info) const override {
        const RobotState& r = info.currState.blue;
        const BallState&  b = info.currState.ball;
        float dx = b.x - r.x, dy = b.y - r.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 0.01f) return 0;
        float kicker_x = std::cos(r.th), kicker_y = std::sin(r.th);
        float dot = (kicker_x * dx + kicker_y * dy) / dist;
        float rw = weight * dot;
        return rw > 0.f ? rw : 0.f;
    }
};

struct OrientedTouchReward : IReward {
    float weight = RW_ORIENTED_TOUCH;
    float computeBlue(const StepInfo& info) const override {
        if (!info.blueTouched) return 0.0f;
        const RobotState& r = info.currState.blue;
        const BallState&  b = info.currState.ball;
        float dx = b.x - r.x, dy = b.y - r.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 0.01f) return 0.0f;
        float fwd_x = std::cos(r.th), fwd_y = std::sin(r.th);
        float alignment = (fwd_x * dx + fwd_y * dy) / dist;
        if (alignment <= 0.0f) return 0.0f;
        float bspd = std::sqrt(b.vx * b.vx + b.vy * b.vy);
        if (bspd < 0.1f) return 0.0f;
        float towardGoal = info.ourGoalIsNegX ? (b.vx / bspd) : (-b.vx / bspd);
        if (towardGoal <= 0.0f) return 0.0f;
        return weight * alignment * towardGoal;
    }
};

struct BallOutPenalty : IReward {
    float penalty = RW_BALL_OUT_PENALTY;
    float computeBlue(const StepInfo& info) const override {
        if (!info.ballOut) return 0.0f;
        return (info.currState.lastToucherTeam == 0) ? penalty : 0.0f;
    }
};

struct DribbleToGoalReward : IReward {
    float weight = RW_DRIBBLE_TO_GOAL;
    float computeBlue(const StepInfo& info) const override {
        if (!info.currState.blue.hasBall) return 0.0f;
        float vx = info.currState.blue.vx;
        float r = weight * vx / MAX_ROBOT_SPEED;
        return r > 0.0f ? r : 0.0f;
    }
};

struct ClearShotReward : IReward {
    float weight = RW_CLEAR_SHOT;
    float computeBlue(const StepInfo& info) const override {
        if (!info.blueKicked) return 0.0f;
        const BallState&  b = info.currState.ball;
        const RobotState& r = info.currState.blue;
        float bspd = std::sqrt(b.vx * b.vx + b.vy * b.vy);
        if (bspd < 1.0f) return 0.0f;
        float towardGoal = info.ourGoalIsNegX ? (b.vx / bspd) : (-b.vx / bspd);
        if (towardGoal <= 0.0f) return 0.0f;
        float rawPos   = info.ourGoalIsNegX ? (r.x / FIELD_HALF_LEN) : (-r.x / FIELD_HALF_LEN);
        float posBonus = rawPos > 0.0f ? rawPos : 0.0f;
        const RobotState& opp = info.currState.yellow;
        float laneMultiplier = 1.0f;
        if (info.ourGoalIsNegX) {
            if (opp.x > r.x && opp.x < FIELD_HALF_LEN) {
                float yMargin = GOAL_HALF_W + ROBOT_RADIUS * 3.0f;
                if (opp.y > -yMargin && opp.y < yMargin) laneMultiplier = 0.4f;
            }
        } else {
            if (opp.x < r.x && opp.x > -FIELD_HALF_LEN) {
                float yMargin = GOAL_HALF_W + ROBOT_RADIUS * 3.0f;
                if (opp.y > -yMargin && opp.y < yMargin) laneMultiplier = 0.4f;
            }
        }
        float followUp = info.ourGoalIsNegX ? (r.vx > 0.5f ? 1.25f : 1.0f) : (r.vx < -0.5f ? 1.25f : 1.0f);
        return weight * towardGoal * (0.4f + 0.6f * posBonus) * laneMultiplier * followUp;
    }
};

struct KickAccuracyReward : IReward {
    float weight = RW_KICK_ACCURACY;
    float computeBlue(const StepInfo& info) const override {
        if (!info.blueKicked) return 0.0f;
        const BallState&  b = info.currState.ball;
        float bspd = std::sqrt(b.vx * b.vx + b.vy * b.vy);
        if (bspd < 0.5f) return 0.0f;
        float towardGoal = info.ourGoalIsNegX ? (b.vx / bspd) : (-b.vx / bspd);
        if (towardGoal <= 0.0f) return 0.0f;
        float power = (bspd >= NORM_BALL_VEL) ? 1.0f : (bspd / NORM_BALL_VEL);
        const RobotState& r = info.prevState.blue;
        float kickerToGoal = info.ourGoalIsNegX ? std::cos(r.th) : (-std::cos(r.th));
        float aim = (kickerToGoal > 0.0f) ? kickerToGoal : 0.0f;
        return weight * towardGoal * (0.6f + 0.4f * power) * (0.7f + 0.3f * aim);
    }
};

struct CollisionPenalty : IReward {
    float penalty = RW_COLLISION_PENALTY;
    float computeBlue(const StepInfo& info) const override {
        return info.robotsCollided ? penalty : 0.0f;
    }
};

struct SeparationBonus : IReward {
    float weight = RW_SEPARATION_BONUS;
    float computeBlue(const StepInfo& info) const override {
        if (info.robotsCollided) return 0.0f;
        const RobotState& a = info.currState.blue;
        const RobotState& b = info.currState.yellow;
        float dx = a.x - b.x, dy = a.y - b.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        constexpr float safeDist = 2.0f * ROBOT_RADIUS + 0.18f;
        if (dist < safeDist) return 0.0f;
        return weight;
    }
};

struct DefendReward : IReward {
    float weight = RW_DEFEND;
    float computeBlue(const StepInfo& info) const override {
        if (info.currState.blue.hasBall) return 0.0f;
        bool ballInOurHalf = info.ourGoalIsNegX ? (info.currState.ball.x < 0.0f) : (info.currState.ball.x > 0.0f);
        bool justLostPossession = info.prevState.blue.hasBall && !info.currState.blue.hasBall;
        if (!ballInOurHalf && !justLostPossession) return 0.0f;
        const RobotState& r = info.currState.blue;
        const BallState&  b = info.currState.ball;
        float dx = b.x - r.x, dy = b.y - r.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 0.01f) return 0.0f;
        float dirX = dx / dist, dirY = dy / dist;
        float velTowardBall = r.vx * dirX + r.vy * dirY;
        float raw = velTowardBall / MAX_ROBOT_SPEED;
        float rw = weight * raw;
        return rw > 0.0f ? rw : 0.0f;
    }
};

struct ClearShotDribblePenalty : IReward {
    float penalty = RW_CLEAR_SHOT_DRIBBLE_PENALTY;
    float computeBlue(const StepInfo& info) const override {
        if (info.blueKicked) return 0.0f;
        if (!info.currState.blue.hasBall) return 0.0f;
        const RobotState& r = info.currState.blue;
        const BallState&  b = info.currState.ball;
        bool ballInFront = info.ourGoalIsNegX ? (b.x > r.x) : (b.x < r.x);
        if (!ballInFront) return 0.0f;
        float dist = std::sqrt((b.x - r.x) * (b.x - r.x) + (b.y - r.y) * (b.y - r.y));
        if (dist > 0.45f) return 0.0f;
        float fwdX = std::cos(r.th);
        float towardGoal = info.ourGoalIsNegX ? fwdX : (-fwdX);
        if (towardGoal < 0.6f) return 0.0f;
        float ourAttackHalf = info.ourGoalIsNegX ? (r.x > 0.25f * FIELD_HALF_LEN) : (r.x < -0.25f * FIELD_HALF_LEN);
        if (!ourAttackHalf) return 0.0f;
        return penalty;
    }
};

struct WastefulKickPenalty : IReward {
    float penalty = RW_WASTEFUL_KICK_PENALTY;
    float computeBlue(const StepInfo& info) const override {
        if (!info.blueKicked) return 0.0f;
        float vx = info.currState.ball.vx;
        float ourX = info.currState.blue.x;
        bool ballTowardGoal = info.ourGoalIsNegX ? (vx >= 1.0f) : (vx <= -1.0f);
        if (ballTowardGoal) return 0.0f;
        bool inAttackHalf = info.ourGoalIsNegX ? (ourX > 0.2f * FIELD_HALF_LEN) : (ourX < -0.2f * FIELD_HALF_LEN);
        if (!inAttackHalf) return 0.0f;
        float towardGoal = info.ourGoalIsNegX ? vx : -vx;
        if (towardGoal < 0.0f) return penalty;
        return penalty * (1.0f - towardGoal);
    }
};

struct KickWithoutPossessionPenalty : IReward {
    float penalty = RW_KICK_WITHOUT_POSSESSION_PENALTY;
    float computeBlue(const StepInfo& info) const override {
        if (!info.blueKicked) return 0.0f;
        if (info.prevState.blue.hasBall) return 0.0f;
        float vx = info.currState.ball.vx;
        bool towardGoal = info.ourGoalIsNegX ? (vx >= 1.5f) : (vx <= -1.5f);
        if (towardGoal) return 0.0f;
        return penalty;
    }
};

struct VelocityPlayerToBallReward : IReward {
    float weight = RW_VELOCITY_PLAYER_TO_BALL;
    float whenClosestMult = RW_VELOCITY_TO_BALL_WHEN_CLOSEST;
    float computeBlue(const StepInfo& info) const override {
        const RobotState& r = info.currState.blue;
        const BallState&  b = info.currState.ball;
        const RobotState& opp = info.currState.yellow;
        float dx = b.x - r.x, dy = b.y - r.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 0.01f) return 0.0f;
        float dirX = dx / dist, dirY = dy / dist;
        float velTowardBall = r.vx * dirX + r.vy * dirY;
        float raw = velTowardBall / MAX_ROBOT_SPEED;
        if (raw <= 0.0f) return 0.0f;
        float rw = weight * raw;
        float distOpp = std::sqrt((b.x - opp.x) * (b.x - opp.x) + (b.y - opp.y) * (b.y - opp.y));
        if (dist < distOpp) rw *= whenClosestMult;
        return rw;
    }
};

struct StrongTouchReward : IReward {
    float weight = RW_STRONG_TOUCH;
    float minVel = STRONG_TOUCH_MIN_VEL;
    float maxVel = STRONG_TOUCH_MAX_VEL;
    float minAlignment = 0.4f;
    float computeBlue(const StepInfo& info) const override {
        if (!info.blueTouched) return 0.0f;
        const RobotState& r = info.prevState.blue;
        const BallState&  b = info.prevState.ball;
        float dx = b.x - r.x, dy = b.y - r.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 0.01f) return 0.0f;
        float kicker_x = std::cos(r.th), kicker_y = std::sin(r.th);
        float alignment = (kicker_x * dx + kicker_y * dy) / dist;
        if (alignment < minAlignment) return 0.0f;
        float dvx = info.currState.ball.vx - info.prevState.ball.vx;
        float dvy = info.currState.ball.vy - info.prevState.ball.vy;
        float hitForce = std::sqrt(dvx * dvx + dvy * dvy);
        if (hitForce < minVel) return 0.0f;
        float raw = (hitForce >= maxVel) ? 1.f : (hitForce / maxVel);
        return weight * raw * alignment;
    }
};

struct TouchBallReward : IReward {
    float weight = RW_TOUCH_BALL;
    float computeBlue(const StepInfo& info) const override {
        return info.blueTouched ? weight : 0.0f;
    }
};

struct KickTowardGoalReward : IReward {
    float weight = RW_KICK_TOWARD_GOAL;
    float computeBlue(const StepInfo& info) const override {
        if (!info.blueKicked) return 0.0f;
        float vx = info.currState.ball.vx;
        float r = weight * clampf(vx / KICK_SPEED, -1.0f, 1.0f);
        return r > 0.f ? r : 0.f;
    }
};

struct BallProgressReward : IReward {
    float weight = RW_BALL_PROGRESS;
    float computeBlue(const StepInfo& info) const override {
        float dx = info.currState.ball.x - info.prevState.ball.x;
        float r = weight * dx / NORM_POS_X;
        return r > 0.f ? r : 0.f;
    }
};

struct TimePenalty : IReward {
    float penalty = RW_TIME_PENALTY;
    float computeBlue(const StepInfo& /*info*/) const override {
        return penalty;
    }
};

struct KeeperSaveReward : IReward {
    float weight = RW_KEEPER_SAVE;
    float computeBlue(const StepInfo& info) const override {
        const auto& b = info.currState.ball;
        const auto& r = info.currState.blue;
        if (b.x >= 0) return 0;
        if (r.x >= b.x) return 0;
        float dist = std::sqrt((b.x - r.x) * (b.x - r.x) + (b.y - r.y) * (b.y - r.y));
        if (dist > 1.5f) return 0;
        return weight * (1.0f - dist / 1.5f);
    }
};

struct KeeperOOBPenalty : IReward {
    float penalty = RW_KEEPER_OOB;
    float computeBlue(const StepInfo& info) const override {
        const auto& b = info.currState.ball;
        const auto& r = info.currState.blue;
        if (b.x >= 0) return 0;
        bool inBox = (r.x >= -FIELD_HALF_LEN && r.x <= -FIELD_HALF_LEN + DEFENSE_AREA_DEPTH &&
                      std::abs(r.y) <= DEFENSE_AREA_HALF_W);
        return inBox ? 0 : penalty;
    }
};

float computeReward(const std::vector<IReward*>& rewards,
                    const StepInfo& info, bool forBlue);
std::vector<IReward*> makeDefaultRewards();
