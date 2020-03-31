/*
# Original Work Copyright © 2018-present, MAgents (https://github.com/geek-ai/MAgent). 
# All rights reserved.
*/ 

/**
 * \file utility.cc
 * \brief common utility for the project
 */

#include <cstring>
#include "utility.h"

namespace magent {
namespace utility {

bool strequ(const char *a, const char *b) {
    return strcmp(a, b) == 0;
}

} // namespace utility
} // namespace magent
