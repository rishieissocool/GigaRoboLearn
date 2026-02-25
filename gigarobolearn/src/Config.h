#pragma once
#include <cstdint>
#include "RewardWeights.h"

constexpr int   LEAGUE_6             = 6;
constexpr int   LEAGUE_11            = 11;
constexpr int   ROBOTS_PER_TEAM      = 1;
constexpr int   KEEPER_INDEX         = 0;

constexpr float DEFENSE_AREA_DEPTH   = 1.80f;
constexpr float DEFENSE_AREA_HALF_W  = 1.80f;

constexpr float FIELD_LEN       = 12.0f;
constexpr float FIELD_WIDTH     =  9.0f;
constexpr float FIELD_HALF_LEN  = FIELD_LEN   / 2.0f;
constexpr float FIELD_HALF_W    = FIELD_WIDTH / 2.0f;
constexpr float FIELD_MARGIN          = 0.30f;
constexpr float FIELD_REFEREE_MARGIN  = 0.0f;
constexpr float WALL_THICKNESS        = 0.050f;
constexpr float _INCREMENT            = FIELD_MARGIN + FIELD_REFEREE_MARGIN + WALL_THICKNESS * 0.5f;
constexpr float FIELD_HALF_LEN_BOUND  = FIELD_HALF_LEN + _INCREMENT - WALL_THICKNESS * 0.5f;
constexpr float FIELD_HALF_W_BOUND    = FIELD_HALF_W   + _INCREMENT - WALL_THICKNESS * 0.5f;
constexpr float GOAL_WIDTH      = 1.8f;
constexpr float GOAL_HALF_W     = GOAL_WIDTH  / 2.0f;
constexpr float GOAL_DEPTH      = 0.18f;
constexpr float BALL_RADIUS     = 0.0215f;
constexpr float ROBOT_RADIUS    = 0.09f;
constexpr float KICKER_RANGE    = 0.13f;
constexpr float DRIBBLE_RANGE   = 0.12f;

constexpr float MAX_ROBOT_SPEED      = 3.0f;
constexpr float MAX_ROBOT_ANG_SPEED  = 8.0f;
constexpr float MAX_BALL_SPEED       = 10.0f;
constexpr float BALL_FRICTION        = 0.5f;
constexpr float WALL_RESTITUTION     = 0.65f;
constexpr float KICK_SPEED           = 6.0f;
constexpr float CHIP_SPEED           = 5.0f;
constexpr float SIM_DT               = 1.0f / 60.0f;

constexpr int   MAX_EP_STEPS    = 600;
constexpr int   NUM_ACTIONS     = 18;
constexpr int   OBS_SIZE        = 24;

constexpr long  DEFAULT_TOTAL_STEPS   = 2'000'000'000;
constexpr long  CHECKPOINT_EVERY      = 50000;
constexpr int   STEPS_PER_ITER        = 10000;
constexpr int   PPO_EPOCHS      = 4;
constexpr int   MINI_BATCH      = 512;
constexpr float GAMMA           = 0.99f;
constexpr float GAE_LAMBDA      = 0.95f;
constexpr float CLIP_EPS        = 0.2f;
constexpr float ENTROPY_COEF    = 0.01f;
constexpr float VALUE_COEF      = 0.5f;
constexpr float ACTOR_LR        = 3e-4f;
constexpr float CRITIC_LR       = 3e-4f;
constexpr float GRAD_CLIP       = 0.5f;
constexpr float ADAM_BETA1      = 0.9f;
constexpr float ADAM_BETA2      = 0.999f;
constexpr float ADAM_EPS        = 1e-8f;

constexpr int   HIDDEN1         = 768;
constexpr int   HIDDEN2         = 768;
constexpr int   HIDDEN3         = 768;
constexpr int   HIDDEN4         = 384;

constexpr int   MAX_CHECKPOINTS = 5;
constexpr char  MODELS_DIR[]    = "models";

constexpr char  GRSIM_HOST[]            = "127.0.0.1";
constexpr int   GRSIM_CMD_PORT          = 10300;
constexpr int   GRSIM_SIM_CONTROL_PORT  = 10400;
constexpr char  VISION_MCAST_ADDR[]     = "224.5.23.2";
constexpr int   VISION_PORT             = 10002;

constexpr float NORM_POS_X      = FIELD_HALF_LEN;
constexpr float NORM_POS_Y      = FIELD_HALF_W;
constexpr float NORM_VEL        = MAX_ROBOT_SPEED;
constexpr float NORM_ANG        = MAX_ROBOT_ANG_SPEED;
constexpr float NORM_BALL_VEL   = MAX_BALL_SPEED;
