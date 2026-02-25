#pragma once
#ifndef REWARD_WEIGHTS_H
#define REWARD_WEIGHTS_H

constexpr float RW_GOAL              =  150.0f;
constexpr float RW_CONCEDE           = -150.0f;
constexpr float RW_OWN_GOAL          = -250.0f;

constexpr float RW_FACE_BALL         =   0.2f;
constexpr float RW_KICKER_TOWARD_BALL=   0.9f;
constexpr float RW_VELOCITY_PLAYER_TO_BALL = 4.5f;
constexpr float RW_VELOCITY_TO_BALL_WHEN_CLOSEST = 1.8f;
constexpr float RW_STRONG_TOUCH     =  35.0f;
constexpr float RW_BALL_TO_GOAL     =   2.0f;

constexpr float RW_COLLISION_PENALTY = -15.0f;
constexpr float RW_SEPARATION_BONUS =   0.5f;

constexpr float RW_BALL_TOWARD_OWN_GOAL = -2.0f;
constexpr float RW_KICK_TOWARD_OWN_GOAL = -40.0f;

constexpr float STRONG_TOUCH_MIN_VEL =  2.0f;
constexpr float STRONG_TOUCH_MAX_VEL = 10.0f;

constexpr float RW_ORIENTED_TOUCH   =  8.0f;
constexpr float RW_BALL_OUT_PENALTY = -50.0f;
constexpr float RW_DRIBBLE_TO_GOAL  =  1.5f;
constexpr float RW_DEFEND           =  2.5f;
constexpr float RW_CLEAR_SHOT       =  22.0f;
constexpr float RW_CLEAR_SHOT_DRIBBLE_PENALTY = -10.0f;
constexpr float RW_KICK_ACCURACY    =  12.0f;
constexpr float RW_WASTEFUL_KICK_PENALTY = -12.0f;
constexpr float RW_KICK_WITHOUT_POSSESSION_PENALTY = -5.0f;

constexpr float RW_MOVE_TO_BALL      =   0.0f;
constexpr float RW_TOUCH_BALL        =   0.0f;
constexpr float RW_KICK_TOWARD_GOAL  =   0.0f;
constexpr float RW_BALL_PROGRESS     =   0.0f;
constexpr float RW_TIME_PENALTY      =   0.0f;
constexpr float RW_OOB_PENALTY       =   0.0f;
constexpr float RW_KEEPER_SAVE       =   0.0f;
constexpr float RW_KEEPER_OOB        =   0.0f;

#endif
