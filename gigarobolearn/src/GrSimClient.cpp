#include "GrSimClient.h"
#include <cstring>
#include <cstdio>
#include <iostream>
#include <cmath>

namespace {

void encodeVarint(std::vector<uint8_t>& buf, uint64_t v) {
    while (v > 127) {
        buf.push_back((uint8_t)((v & 0x7F) | 0x80));
        v >>= 7;
    }
    buf.push_back((uint8_t)v);
}

void encodeField(std::vector<uint8_t>& buf, int fieldNum, int wireType) {
    encodeVarint(buf, (uint64_t)((fieldNum << 3) | wireType));
}

void encodeUint32(std::vector<uint8_t>& buf, int field, uint32_t v) {
    encodeField(buf, field, 0);
    encodeVarint(buf, v);
}
void encodeBool(std::vector<uint8_t>& buf, int field, bool v) {
    encodeUint32(buf, field, v ? 1 : 0);
}
void encodeFloat(std::vector<uint8_t>& buf, int field, float v) {
    encodeField(buf, field, 5);
    uint8_t bytes[4];
    memcpy(bytes, &v, 4);
    for (auto b : bytes) buf.push_back(b);
}
void encodeDouble(std::vector<uint8_t>& buf, int field, double v) {
    encodeField(buf, field, 1);
    uint8_t bytes[8];
    memcpy(bytes, &v, 8);
    for (auto b : bytes) buf.push_back(b);
}
void encodeMessage(std::vector<uint8_t>& buf, int field, const std::vector<uint8_t>& msg) {
    encodeField(buf, field, 2);
    encodeVarint(buf, msg.size());
    buf.insert(buf.end(), msg.begin(), msg.end());
}

std::vector<uint8_t> encodeRobotCommand(const RobotCmd& cmd) {
    std::vector<uint8_t> buf;
    encodeUint32(buf, 1, cmd.id);
    encodeFloat (buf, 2, cmd.kickspeedx);
    encodeFloat (buf, 3, cmd.kickspeedz);
    encodeFloat (buf, 4, cmd.veltangent);
    encodeFloat (buf, 5, cmd.velnormal);
    encodeFloat (buf, 6, cmd.velangular);
    encodeBool  (buf, 7, cmd.spinner);
    encodeBool  (buf, 8, cmd.wheelsspeed);
    return buf;
}

std::vector<uint8_t> encodeCommands(bool isYellow, const std::vector<RobotCmd>& cmds) {
    std::vector<uint8_t> buf;
    encodeDouble(buf, 1, 0.0);
    encodeBool  (buf, 2, isYellow);
    for (const auto& c : cmds) {
        auto msg = encodeRobotCommand(c);
        encodeMessage(buf, 3, msg);
    }
    return buf;
}

uint64_t decodeVarint(const uint8_t*& p, const uint8_t* end) {
    uint64_t result = 0; int shift = 0;
    while (p < end) {
        uint8_t b = *p++;
        result |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
    }
    return result;
}
float decodeFloat(const uint8_t*& p) {
    float v; memcpy(&v, p, 4); p += 4; return v;
}
double decodeDouble(const uint8_t*& p) {
    double v; memcpy(&v, p, 8); p += 8; return v;
}
uint32_t decodeFixed32(const uint8_t*& p) {
    uint32_t v; memcpy(&v, p, 4); p += 4; return v;
}
uint64_t decodeFixed64(const uint8_t*& p) {
    uint64_t v; memcpy(&v, p, 8); p += 8; return v;
}

void skipField(const uint8_t*& p, const uint8_t* end, int wireType) {
    switch (wireType) {
        case 0: decodeVarint(p, end); break;
        case 1: p += 8; break;
        case 2: { uint64_t len = decodeVarint(p, end); p += len; break; }
        case 5: p += 4; break;
        default: p = end; break;
    }
}

struct ParsedBall  { float x=0,y=0; bool valid=false; };
struct ParsedRobot { uint32_t id=0; float x=0,y=0,ori=0; bool valid=false; };

ParsedBall parseBall(const uint8_t* p, const uint8_t* end) {
    ParsedBall b;
    while (p < end) {
        uint64_t tag  = decodeVarint(p, end);
        int field = (int)(tag >> 3), wire = (int)(tag & 7);
        if      (field == 3) { b.x = decodeFloat(p); b.valid = true; }
        else if (field == 4) { b.y = decodeFloat(p); }
        else                  skipField(p, end, wire);
    }
    return b;
}

ParsedRobot parseRobot(const uint8_t* p, const uint8_t* end) {
    ParsedRobot r;
    while (p < end) {
        uint64_t tag  = decodeVarint(p, end);
        int field = (int)(tag >> 3), wire = (int)(tag & 7);
        if      (field == 2) { r.id  = (uint32_t)decodeVarint(p, end); r.valid = true; }
        else if (field == 3) { r.x   = decodeFloat(p); }
        else if (field == 4) { r.y   = decodeFloat(p); }
        else if (field == 5) { r.ori = decodeFloat(p); }
        else                  skipField(p, end, wire);
    }
    return r;
}

} // anonymous namespace

namespace {
std::vector<uint8_t> encodeReplacement(float bx, float by, float bvx, float bvy) {
    std::vector<uint8_t> ballMsg;
    encodeDouble(ballMsg, 1, bx);
    encodeDouble(ballMsg, 2, by);
    encodeDouble(ballMsg, 3, bvx);
    encodeDouble(ballMsg, 4, bvy);
    std::vector<uint8_t> replMsg;
    encodeMessage(replMsg, 1, ballMsg);
    std::vector<uint8_t> pkt;
    encodeMessage(pkt, 2, replMsg);
    return pkt;
}
std::vector<uint8_t> encodeRobotReplacement(bool isYellow, uint32_t id,
                                             float x, float y, float ori) {
    std::vector<uint8_t> robotMsg;
    encodeDouble(robotMsg, 1, x);
    encodeDouble(robotMsg, 2, y);
    encodeDouble(robotMsg, 3, ori);
    encodeUint32(robotMsg, 4, id);
    encodeBool  (robotMsg, 5, isYellow);
    encodeBool  (robotMsg, 6, true);
    std::vector<uint8_t> replMsg;
    encodeMessage(replMsg, 2, robotMsg);
    std::vector<uint8_t> pkt;
    encodeMessage(pkt, 2, replMsg);
    return pkt;
}

std::vector<uint8_t> encodeStateReplacement(const GameState& st) {
    constexpr double RAD_TO_DEG = 57.29577951308232;  // 180/pi
    auto toDeg360 = [](double rad) {
        double d = std::fmod(rad * RAD_TO_DEG, 360.0);
        if (d < 0) d += 360.0;
        return d;
    };

    std::vector<uint8_t> ballMsg;
    encodeDouble(ballMsg, 1, st.ball.x);
    encodeDouble(ballMsg, 2, st.ball.y);
    encodeDouble(ballMsg, 3, st.ball.vx);
    encodeDouble(ballMsg, 4, st.ball.vy);

    std::vector<uint8_t> blueRobotMsg;
    encodeDouble(blueRobotMsg, 1, st.blue.x);
    encodeDouble(blueRobotMsg, 2, st.blue.y);
    encodeDouble(blueRobotMsg, 3, toDeg360(st.blue.th));
    encodeUint32(blueRobotMsg, 4, 0);
    encodeBool  (blueRobotMsg, 5, false);
    encodeBool  (blueRobotMsg, 6, true);

    std::vector<uint8_t> yellowRobotMsg;
    encodeDouble(yellowRobotMsg, 1, st.yellow.x);
    encodeDouble(yellowRobotMsg, 2, st.yellow.y);
    encodeDouble(yellowRobotMsg, 3, toDeg360(st.yellow.th));
    encodeUint32(yellowRobotMsg, 4, 0);
    encodeBool  (yellowRobotMsg, 5, true);
    encodeBool  (yellowRobotMsg, 6, true);

    std::vector<uint8_t> replMsg;
    encodeMessage(replMsg, 1, ballMsg);
    encodeMessage(replMsg, 2, blueRobotMsg);
    encodeMessage(replMsg, 2, yellowRobotMsg);
    std::vector<uint8_t> pkt;
    encodeMessage(pkt, 2, replMsg);
    return pkt;
}
} // anonymous namespace

GrSimClient::GrSimClient() {}

GrSimClient::~GrSimClient() { shutdown(); }

bool GrSimClient::init(const std::string& simHost, int cmdPort,
                        int simControlPort,
                        const std::string& mcastAddr, int visionPort) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) {
        std::cerr << "[GrSimClient] WSAStartup failed.\n"; return false;
    }
#endif

    cmdSock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (cmdSock_ == INVALID_SOCKET) {
        std::cerr << "[GrSimClient] Cannot create command socket.\n"; return false;
    }
    memset(&cmdAddr_, 0, sizeof(cmdAddr_));
    cmdAddr_.sin_family = AF_INET;
    cmdAddr_.sin_port   = htons((u_short)cmdPort);
    inet_pton(AF_INET, simHost.c_str(), &cmdAddr_.sin_addr);

    memset(&simControlAddr_, 0, sizeof(simControlAddr_));
    simControlAddr_.sin_family = AF_INET;
    simControlAddr_.sin_port   = htons((u_short)simControlPort);
    inet_pton(AF_INET, simHost.c_str(), &simControlAddr_.sin_addr);

    visionSock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (visionSock_ != INVALID_SOCKET) {
        int reuse = 1;
        setsockopt(visionSock_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&reuse), sizeof(reuse));
        sockaddr_in localAddr{};
        localAddr.sin_family      = AF_INET;
        localAddr.sin_port        = htons((u_short)visionPort);
        localAddr.sin_addr.s_addr = INADDR_ANY;
        if (bind(visionSock_, (sockaddr*)&localAddr, sizeof(localAddr)) == 0) {
            ip_mreq mreq{};
            inet_pton(AF_INET, mcastAddr.c_str(), &mreq.imr_multiaddr);
            mreq.imr_interface.s_addr = INADDR_ANY;
            if (setsockopt(visionSock_, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                           reinterpret_cast<const char*>(&mreq), sizeof(mreq)) == 0) {
                u_long nonblock = 1;
                ioctlsocket(visionSock_, FIONBIO, &nonblock);
            }
        } else {
            closesocket(visionSock_);
            visionSock_ = INVALID_SOCKET;
        }
    }

    ready_ = true;
    std::cout << "[GrSimClient] Sending to sim at " << simHost << " cmd:" << cmdPort
              << " simControl:" << simControlPort;
    if (visionSock_ != INVALID_SOCKET)
        std::cout << "  (vision rx " << mcastAddr << ":" << visionPort << ")";
    std::cout << "\n";
    return true;
}

void GrSimClient::shutdown() {
    if (cmdSock_    != INVALID_SOCKET) { closesocket(cmdSock_);    cmdSock_    = INVALID_SOCKET; }
    if (visionSock_ != INVALID_SOCKET) { closesocket(visionSock_); visionSock_ = INVALID_SOCKET; }
#ifdef _WIN32
    WSACleanup();
#endif
    ready_ = false;
}

std::vector<uint8_t> GrSimClient::encodePacket(bool isYellow,
                                                 const std::vector<RobotCmd>& cmds) const {
    auto cmdsBuf = encodeCommands(isYellow, cmds);
    std::vector<uint8_t> pkt;
    encodeMessage(pkt, 1, cmdsBuf);
    return pkt;
}

void GrSimClient::sendCommand(bool isYellow, const RobotCmd& cmd) {
    sendCommands(isYellow, {cmd});
}

void GrSimClient::sendCommands(bool isYellow, const std::vector<RobotCmd>& cmds) {
    if (!ready_) return;
    auto pkt = encodePacket(isYellow, cmds);
    sendto(cmdSock_, reinterpret_cast<const char*>(pkt.data()), (int)pkt.size(), 0,
           (sockaddr*)&cmdAddr_, sizeof(cmdAddr_));
}

void GrSimClient::resetBall(float x, float y, float vx, float vy) {
    if (!ready_) return;
    auto pkt = encodeReplacement(x, y, vx, vy);
    sendto(cmdSock_, reinterpret_cast<const char*>(pkt.data()), (int)pkt.size(), 0,
           (sockaddr*)&cmdAddr_, sizeof(cmdAddr_));
}

void GrSimClient::resetRobot(bool isYellow, uint32_t id, float x, float y, float ori) {
    if (!ready_) return;
    auto pkt = encodeRobotReplacement(isYellow, id, x, y, ori);
    sendto(cmdSock_, reinterpret_cast<const char*>(pkt.data()), (int)pkt.size(), 0,
           (sockaddr*)&cmdAddr_, sizeof(cmdAddr_));
}

void GrSimClient::sendState(const GameState& state) {
    if (!ready_) return;
    auto pkt = encodeStateReplacement(state);
    sendto(cmdSock_, reinterpret_cast<const char*>(pkt.data()), (int)pkt.size(), 0,
           (sockaddr*)&cmdAddr_, sizeof(cmdAddr_));
    sendto(cmdSock_, reinterpret_cast<const char*>(pkt.data()), (int)pkt.size(), 0,
           (sockaddr*)&simControlAddr_, sizeof(simControlAddr_));
}

bool GrSimClient::recvVision(VisionFrame& out) {
    if (!ready_ || visionSock_ == INVALID_SOCKET) return false;
    uint8_t buf[65536];
    int n = recv(visionSock_, reinterpret_cast<char*>(buf), sizeof(buf), 0);
    if (n <= 0) return false;

    const uint8_t* p   = buf;
    const uint8_t* end = buf + n;

    while (p < end) {
        uint64_t tag  = decodeVarint(p, end);
        int field = (int)(tag >> 3), wire = (int)(tag & 7);

        if (field == 1 && wire == 2) {
            uint64_t len = decodeVarint(p, end);
            const uint8_t* detEnd = p + len;

            while (p < detEnd) {
                uint64_t dtag  = decodeVarint(p, detEnd);
                int dfield = (int)(dtag >> 3), dwire = (int)(dtag & 7);

                if (dfield == 1 && dwire == 0) {
                    out.frameNumber = (uint32_t)decodeVarint(p, detEnd);
                } else                 if (dfield == 5 && dwire == 2) {
                    uint64_t blen = decodeVarint(p, detEnd);
                    auto ball = parseBall(p, p + blen);
                    if (ball.valid) { out.ball.x = ball.x; out.ball.y = ball.y; out.ball.valid = true; }
                    p += blen;
                } else if (dfield == 6 && dwire == 2) {
                    uint64_t rlen = decodeVarint(p, detEnd);
                    auto r = parseRobot(p, p + rlen);
                    if (r.valid && r.id < 16) {
                        out.yellow[r.id] = {r.id, r.x, r.y, r.ori, true};
                    }
                    p += rlen;
                } else if (dfield == 7 && dwire == 2) {
                    uint64_t rlen = decodeVarint(p, detEnd);
                    auto r = parseRobot(p, p + rlen);
                    if (r.valid && r.id < 16) {
                        out.blue[r.id] = {r.id, r.x, r.y, r.ori, true};
                    }
                    p += rlen;
                } else {
                    skipField(p, detEnd, dwire);
                }
            }
            p = detEnd;
        } else {
            skipField(p, end, wire);
        }
    }
    return true;
}

RobotCmd GrSimClient::buildCommand(const GameState& state, bool isYellow,
                                    int actionIdx, bool isBotYellow) const {
    const ActionDef& act = ACTIONS[actionIdx];
    RobotCmd cmd;
    cmd.id = 0;
    cmd.veltangent  = act.vxLocal;
    cmd.velnormal   = act.vyLocal;
    cmd.velangular  = act.vw;
    cmd.kickspeedx  = act.chip ? 0.0f : act.kick;
    cmd.kickspeedz  = act.chip ? act.kick * std::sin(30.0f * 3.14159f / 180.0f) : 0.0f;
    cmd.spinner     = act.dribble;
    cmd.wheelsspeed = false;
    return cmd;
}
