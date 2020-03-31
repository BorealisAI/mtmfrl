/*
# Original Work Copyright © 2018-present, MAgents (https://github.com/geek-ai/MAgent). 
# All rights reserved.
*/

#ifndef MATRIX_RENDER_BACKEND_UTILITY_CONFIG_H_
#define MATRIX_RENDER_BACKEND_UTILITY_CONFIG_H_

#include <argp.h>
#include <cerrno>
#include <cstdint>

namespace magent {
namespace render {

struct RenderConfig {
    uint16_t port = 9030;
    bool quiet = false;
};

void parse(int argc, char *argv[], RenderConfig &config);

} // namespace magent
} // namespace ARGPParser

#endif //MATRIX_RENDER_BACKEND_UTILITY_CONFIG_H_