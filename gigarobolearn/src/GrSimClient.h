#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include "Config.h"
#include "Environment.h"
#include <string>
#include <vector>
#include <cstdint>

struct RobotCmd {
    uint32_t id          = 0;
    float    kickspeedx  = 0;
    float    kickspeedz  = 0;
    float    veltangent  = 0;
    float    velnormal   = 0;
    float    velangular  = 0;
    bool     spinner     = false;
    bool     wheelsspeed = false;
};

struct VisionRobot {
    uint32_t id  = 0;
    float    x   = 0, y = 0;
    float    ori = 0;
    bool     valid = false;
};
struct VisionBall {
    float x = 0, y = 0;
    bool  valid = false;
};
struct VisionFrame {
    VisionBall  ball;
    VisionRobot blue[16], yellow[16];
    uint32_t    frameNumber = 0;
};

class GrSimClient {
public:
    GrSimClient();
    ~GrSimClient();

    bool init(const std::string& simHost  = GRSIM_HOST,
              int                cmdPort  = GRSIM_CMD_PORT,
              int                simControlPort = GRSIM_SIM_CONTROL_PORT,
              const std::string& mcastAddr = VISION_MCAST_ADDR,
              int                visionPort = VISION_PORT);

    void shutdown();

    void sendCommand(bool isYellow, const RobotCmd& cmd);

    void sendCommands(bool isYellow, const std::vector<RobotCmd>& cmds);

    void resetBall(float x, float y, float vx = 0, float vy = 0);

    void resetRobot(bool isYellow, uint32_t id, float x, float y, float ori);

    void sendState(const GameState& state);

    bool recvVision(VisionFrame& out);

    RobotCmd buildCommand(const GameState& state, bool isYellow,
                          int actionIdx, bool isBotYellow) const;

    bool isReady() const { return ready_; }

private:
    bool    ready_  = false;
    SOCKET  cmdSock_    = INVALID_SOCKET;
    SOCKET  visionSock_ = INVALID_SOCKET;

    sockaddr_in cmdAddr_;
    sockaddr_in simControlAddr_;

    std::vector<uint8_t> encodePacket(bool isYellow, const std::vector<RobotCmd>& cmds) const;
    bool parseVisionFrame(const uint8_t* buf, int len, VisionFrame& out) const;
};
