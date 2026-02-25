#include "Rewards.h"
#include <vector>
#include <memory>

float computeReward(const std::vector<IReward*>& rewards,
                    const StepInfo& info, bool forBlue) {
    if (forBlue) {
        float total = 0;
        for (auto* r : rewards) total += r->computeBlue(info);
        return total;
    }

    GameState mirroredPrev = info.prevState;
    GameState mirroredCurr = info.currState;
    std::swap(mirroredPrev.blue, mirroredPrev.yellow);
    std::swap(mirroredCurr.blue, mirroredCurr.yellow);

    auto mirrorTeam = [](int t) {
        if (t == 0) return 1;
        if (t == 1) return 0;
        return -1;
    };

    mirroredPrev.goalBy = mirrorTeam(mirroredPrev.goalBy);
    mirroredCurr.goalBy = mirrorTeam(mirroredCurr.goalBy);
    mirroredPrev.lastToucherTeam = mirrorTeam(info.prevState.lastToucherTeam);
    mirroredCurr.lastToucherTeam = mirrorTeam(info.currState.lastToucherTeam);

    StepInfo mirroredInfo{mirroredPrev, mirroredCurr,
                          info.yellowTouched, info.blueTouched,
                          info.yellowKicked,  info.blueKicked,
                          info.robotsCollided, false,
                          info.ballOut};

    float total = 0;
    for (auto* r : rewards) total += r->computeBlue(mirroredInfo);
    return total;
}

std::vector<IReward*> makeDefaultRewards() {
    static GoalReward                    goalR;
    static BallToGoalReward              b2gR;
    static BallTowardOwnGoalPenalty      ballOwnGoalR;
    static KickTowardOwnGoalPenalty      kickOwnGoalR;
    static VelocityPlayerToBallReward   velToBallR;
    static FaceBallReward                faceR;
    static KickerTowardBallReward        kickerTowardBallR;
    static StrongTouchReward             strongTouchR;
    static OrientedTouchReward           orientedTouchR;
    static BallOutPenalty               ballOutR;
    static DribbleToGoalReward          dribbleGoalR;
    static ClearShotReward               clearShotR;
    static KickAccuracyReward            kickAccuracyR;
    static ClearShotDribblePenalty       clearShotDribbleR;
    static WastefulKickPenalty           wastefulKickR;
    static KickWithoutPossessionPenalty  kickNoPossR;
    static CollisionPenalty              collisionR;
    static SeparationBonus              separationR;
    static DefendReward                  defendR;
    static TimePenalty                   timeR;
    static KeeperSaveReward              keeperSaveR;
    static KeeperOOBPenalty              keeperOOBR;

    return { &goalR, &b2gR, &ballOwnGoalR, &kickOwnGoalR, &velToBallR, &faceR, &kickerTowardBallR,
             &strongTouchR, &orientedTouchR, &ballOutR, &dribbleGoalR, &clearShotR, &kickAccuracyR, &clearShotDribbleR,
             &wastefulKickR, &kickNoPossR, &collisionR, &separationR, &defendR,
             &timeR, &keeperSaveR, &keeperOOBR };
}
