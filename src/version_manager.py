VERSION = "3.2.0"

import re

def get_version_info(ver):
    match = re.match("^([0-9]+)\\.([0-9]+)" + "(\\.(.+))?" + "$", ver)
    assert match
    major, minor, _, supplement = match.groups()
    return {"major": major, "minor": minor, "supplement": supplement}

def is_acceptable_version(old_version):
    """
    check the old major version with the current (VERSION).
    """
    current_version_info = get_version_info(VERSION)
    old_version_info = get_version_info(old_version)
    return (current_version_info["major"] == old_version_info["major"])
