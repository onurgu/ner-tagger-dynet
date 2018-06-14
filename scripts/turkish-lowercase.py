#!/usr/bin/env python
# coding=utf8

import sys

CHARMAP = {
    "to_upper": {
        u"ı": u"I",
        u"i": u"İ",
    },
    "to_lower": {
        u"I": u"ı",
        u"İ": u"i",
    }
}


def lower(s):
    for key, value in CHARMAP.get("to_lower").items():
        s = s.replace(key, value)

    return s.lower()

line = sys.stdin.readline()

while line:
    print lower(line.decode("utf8").strip())
    line = sys.stdin.readline()