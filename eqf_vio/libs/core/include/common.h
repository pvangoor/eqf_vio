#pragma once

#include "yaml-cpp/yaml.h"

template <class T> bool safeConfig(const YAML::Node& cfg, T& var) {
    if (cfg) {
        var = cfg.as<T>();
        return true;
    } else {
        return false;
    }
}
