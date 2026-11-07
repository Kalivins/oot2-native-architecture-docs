// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The stand-in that proves runtime/ui builds and links as C++ (oot2/ui.h, research 104). It uses what a menu library
// leans on: a heap-allocating container, strings and an algorithm from the standard library, so the link needs the
// library and not only its headers. It throws nothing and catches nothing: the build is without exceptions and without
// run-time type information, and an allocation that fails ends the program, as it does in the C code.
#include "oot2/ui.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

extern "C" UiBuild ui_standin_build(void) {
    UiBuild b;
    b.dialect = static_cast<long>(__cplusplus);
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    b.exceptions = 1;
#else
    b.exceptions = 0;
#endif
#if defined(__cpp_rtti) || defined(__GXX_RTTI)
    b.rtti = 1;
#else
    b.rtti = 0;
#endif
    return b;
}

extern "C" size_t ui_standin_list(const char *text, char *out, size_t cap) {
    std::vector<std::string> items;
    const std::string all(text ? text : "");
    for (std::size_t start = 0; start <= all.size();) {
        std::size_t end = all.find(',', start);
        if (end == std::string::npos) end = all.size();
        if (end > start) items.push_back(all.substr(start, end - start));
        start = end + 1;
    }
    std::sort(items.begin(), items.end());
    items.erase(std::unique(items.begin(), items.end()), items.end());

    std::string joined;
    for (std::size_t i = 0; i < items.size(); i++) {
        if (i) joined += " | ";
        joined += items[i];
    }
    if (cap) {
        const std::size_t n = std::min(joined.size(), cap - 1);
        if (out) {
            std::memcpy(out, joined.data(), n);
            out[n] = '\0';
        }
    }
    return joined.size();
}
