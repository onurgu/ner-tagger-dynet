
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
    tokens = line.decode("utf8").strip().split(" ")
    for idx in range(1, len(tokens)-1):
        if tokens[idx] == "*UNKNOWN*":
            pass
        else:
            _tags = tokens[idx].split("+")
            _tags[0] = lower(_tags[0])
            tokens[idx] = "+".join(_tags)
    print " ".join(tokens).encode("utf8")
    line = sys.stdin.readline()