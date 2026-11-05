"""Stable H3 media grouping shared by video dispatch and prompt compilation."""


def group_references(items, kind_of):
    groups = {"image": [], "video": [], "audio": []}
    for item in items:
        kind = kind_of(item)
        if kind not in groups:
            raise ValueError("unsupported H3 reference kind")
        groups[kind].append(item)
    return groups
