"""Deterministic ASCII map renderer.

Takes a structured room graph (rooms + edges) and produces a compact ASCII map
with room boxes, connectors, door symbols, and entity markers.
"""

from __future__ import annotations

import re
from typing import Any

# ── Room / connector defaults ──────────────────────────────────────────────
DEFAULT_ROOM_WIDTH = 17
DEFAULT_ROOM_HEIGHT = 5
CONNECTOR_H_LEN = 5  # chars between room right-edge and next room left-edge
CONNECTOR_V_LEN = 1  # lines between room bottom-edge and next room top-edge
MAX_COLS = 72
MAX_ROWS = 30
MAX_ROOMS = 60

# Direction → (dx, dy) on the grid
DIRECTION_OFFSETS: dict[str, tuple[int, int]] = {
    "n": (0, -1),
    "north": (0, -1),
    "s": (0, 1),
    "south": (0, 1),
    "e": (1, 0),
    "east": (1, 0),
    "w": (-1, 0),
    "west": (-1, 0),
    "ne": (1, -1),
    "northeast": (1, -1),
    "nw": (-1, -1),
    "northwest": (-1, -1),
    "se": (1, 1),
    "southeast": (1, 1),
    "sw": (-1, 1),
    "southwest": (-1, 1),
    "up": (0, -1),
    "down": (0, 1),
}

# Door type → connector character (horizontal, vertical)
DOOR_CHARS_H: dict[str, str] = {
    "open": "-",
    "archway": ":",
    "locked": "#",
    "hidden": "~",
    "passage": "=",
}
DOOR_CHARS_V: dict[str, str] = {
    "open": "|",
    "archway": ":",
    "locked": "#",
    "hidden": "~",
    "passage": "=",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:80]


# ── Graph helpers ──────────────────────────────────────────────────────────

def _edge_key(a: str, b: str) -> tuple[str, str]:
    """Normalised undirected edge key."""
    return (min(a, b), max(a, b))


def ensure_room(graph: dict, slug: str, label: str = "", turn: int = 0) -> dict:
    """Add room to graph if missing, return the room dict."""
    rooms = graph.setdefault("rooms", {})
    if slug not in rooms:
        rooms[slug] = {
            "label": label or slug,
            "width": DEFAULT_ROOM_WIDTH,
            "height": DEFAULT_ROOM_HEIGHT,
            "x": None,
            "y": None,
            "positioned": False,
            "first_seen_turn": turn,
        }
    elif label and rooms[slug].get("label") in (slug, "", None):
        rooms[slug]["label"] = label
    return rooms[slug]


def add_edge(
    graph: dict,
    from_slug: str,
    to_slug: str,
    direction: str | None = None,
    door_type: str | None = None,
    bidirectional: bool = True,
) -> bool:
    """Add an edge if a matching one doesn't exist. Returns True if new."""
    edges: list[dict] = graph.setdefault("edges", [])
    key = _edge_key(from_slug, to_slug)
    for e in edges:
        if _edge_key(e["from"], e["to"]) == key:
            # Merge in missing direction/door_type
            if direction and not e.get("direction"):
                e["direction"] = direction
            if door_type and not e.get("door_type"):
                e["door_type"] = door_type
            return False
    edges.append({
        "from": from_slug,
        "to": to_slug,
        "direction": direction or None,
        "door_type": door_type or None,
        "bidirectional": bidirectional,
    })
    return True


def _reverse_direction(d: str | None) -> str | None:
    """Return the opposite compass direction."""
    opposites = {
        "n": "s", "s": "n", "e": "w", "w": "e",
        "ne": "sw", "sw": "ne", "nw": "se", "se": "nw",
        "north": "south", "south": "north", "east": "west", "west": "east",
        "northeast": "southwest", "southwest": "northeast",
        "northwest": "southeast", "southeast": "northwest",
        "up": "down", "down": "up",
    }
    return opposites.get((d or "").lower().strip())


# ── Auto-layout ────────────────────────────────────────────────────────────

def auto_layout(graph: dict) -> None:
    """Assign grid positions to unpositioned rooms.

    Already-positioned rooms are never moved, ensuring map stability.
    """
    rooms: dict[str, dict] = graph.get("rooms", {})
    edges: list[dict] = graph.get("edges", [])
    if not rooms:
        return

    positioned = {k for k, v in rooms.items() if v.get("positioned")}
    unpositioned = {k for k in rooms if k not in positioned}

    if not positioned and unpositioned:
        # Seed: place the first room (by first_seen_turn or alphabetically)
        seed = min(
            unpositioned,
            key=lambda k: (rooms[k].get("first_seen_turn") or 0, k),
        )
        rooms[seed]["x"] = 0
        rooms[seed]["y"] = 0
        rooms[seed]["positioned"] = True
        positioned.add(seed)
        unpositioned.discard(seed)

    occupied: set[tuple[int, int]] = set()
    for slug in positioned:
        r = rooms[slug]
        occupied.add((r["x"], r["y"]))

    # Build adjacency with direction hints
    adjacency: dict[str, list[tuple[str, str | None]]] = {}
    for e in edges:
        a, b = e["from"], e["to"]
        d = (e.get("direction") or "").lower().strip() or None
        adjacency.setdefault(a, []).append((b, d))
        if e.get("bidirectional", True):
            adjacency.setdefault(b, []).append((a, _reverse_direction(d)))

    # Iteratively place rooms that are adjacent to positioned rooms
    changed = True
    while changed and unpositioned:
        changed = False
        for slug in list(unpositioned):
            neighbors = adjacency.get(slug, [])
            placed_neighbor = None
            best_dir = None
            for nb, d in neighbors:
                if nb in positioned:
                    placed_neighbor = nb
                    # d is the direction from slug -> nb.
                    # We need the offset from nb -> slug, so reverse it.
                    best_dir = _reverse_direction(d) if d else None
                    if best_dir:
                        break  # prefer a neighbor with a direction hint
            if placed_neighbor is None:
                continue
            nb_room = rooms[placed_neighbor]
            nx, ny = nb_room["x"], nb_room["y"]
            if best_dir and best_dir in DIRECTION_OFFSETS:
                dx, dy = DIRECTION_OFFSETS[best_dir]
                cx, cy = nx + dx, ny + dy
            else:
                # No direction hint, try E/S/N/W
                cx, cy = nx + 1, ny  # default east
            cx, cy = _find_free_cell(cx, cy, occupied)
            rooms[slug]["x"] = cx
            rooms[slug]["y"] = cy
            rooms[slug]["positioned"] = True
            occupied.add((cx, cy))
            positioned.add(slug)
            unpositioned.discard(slug)
            changed = True

    # Place orphan rooms (no edges to positioned rooms)
    if unpositioned:
        max_y = max((rooms[k]["y"] for k in positioned), default=0)
        col = 0
        for slug in sorted(unpositioned, key=lambda k: (rooms[k].get("first_seen_turn") or 0, k)):
            cx, cy = _find_free_cell(col, max_y + 2, occupied)
            rooms[slug]["x"] = cx
            rooms[slug]["y"] = cy
            rooms[slug]["positioned"] = True
            occupied.add((cx, cy))
            col += 1

    graph.setdefault("layout_version", 0)
    graph["layout_version"] = (graph.get("layout_version") or 0) + 1


def _find_free_cell(
    start_x: int,
    start_y: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    """Find the nearest free cell via spiral search."""
    if (start_x, start_y) not in occupied:
        return start_x, start_y
    # Spiral outward
    for radius in range(1, 30):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) == radius or abs(dy) == radius:
                    if (start_x + dx, start_y + dy) not in occupied:
                        return start_x + dx, start_y + dy
    return start_x, start_y  # fallback (shouldn't happen with < 60 rooms)


# ── Update graph from turn data ───────────────────────────────────────────

def update_room_map_graph(
    campaign_state: dict[str, Any],
    player_state: dict[str, Any],
    *,
    graph_key: str = "_room_map_graph",
    turn_number: int = 0,
    known_locations: dict[str, dict] | None = None,
) -> dict:
    """Extract room connectivity from player_state exits and update the map graph.

    Called after player_state_update is applied each turn. Modifies
    campaign_state[graph_key] in place and returns the graph.
    """
    graph = campaign_state.get(graph_key)
    if not isinstance(graph, dict):
        graph = {"rooms": {}, "edges": [], "layout_version": 0}
        campaign_state[graph_key] = graph

    location = str(player_state.get("location") or "").strip()
    room_title = str(player_state.get("room_title") or "").strip()
    if not location:
        location = slugify(room_title) if room_title else ""
    if not location:
        return graph

    current_slug = slugify(location)
    if not current_slug:
        return graph

    # Ensure current room exists
    ensure_room(graph, current_slug, label=room_title or location, turn=turn_number)

    # Process exits
    exits = player_state.get("exits")
    if not isinstance(exits, list):
        exits = []

    had_new_rooms = False
    for ex in exits:
        if isinstance(ex, dict):
            target_loc = str(ex.get("location") or "").strip()
            target_name = str(ex.get("name") or "").strip()
            direction = str(ex.get("direction") or "").strip() or None
            door = str(ex.get("door") or "").strip() or None
            if not target_loc:
                # Try to fuzzy match against known location cards
                if target_name and known_locations:
                    target_loc = _fuzzy_match_location(target_name, known_locations)
                if not target_loc:
                    continue
            target_slug = slugify(target_loc)
            if not target_slug or target_slug == current_slug:
                continue
            was_new_room = target_slug not in graph.get("rooms", {})
            ensure_room(graph, target_slug, label=target_name or target_loc, turn=turn_number)
            add_edge(graph, current_slug, target_slug, direction=direction, door_type=door)
            if was_new_room:
                had_new_rooms = True
        elif isinstance(ex, str) and ex.strip():
            # Plain string exit — try to match to known locations
            if known_locations:
                target_loc = _fuzzy_match_location(ex.strip(), known_locations)
                if target_loc:
                    target_slug = slugify(target_loc)
                    if target_slug and target_slug != current_slug:
                        was_new_room = target_slug not in graph.get("rooms", {})
                        ensure_room(graph, target_slug, label=ex.strip(), turn=turn_number)
                        add_edge(graph, current_slug, target_slug)
                        if was_new_room:
                            had_new_rooms = True

    # Auto-layout if there are any unpositioned rooms
    has_unpositioned = any(
        not r.get("positioned") for r in graph.get("rooms", {}).values()
    )
    if has_unpositioned:
        auto_layout(graph)

    return graph


def _fuzzy_match_location(
    name: str,
    known_locations: dict[str, dict],
) -> str:
    """Try to match an exit name to a known location card slug."""
    name_lower = name.lower().strip()
    name_slug = slugify(name)
    # Direct slug match
    if name_slug in known_locations:
        return name_slug
    # Match by display name
    for slug, info in known_locations.items():
        card_name = str(info.get("name") or info.get("label") or "").lower().strip()
        if card_name == name_lower:
            return slug
        if name_slug == slugify(card_name):
            return slug
    return ""


# ── ASCII Renderer ─────────────────────────────────────────────────────────

def render_ascii_map(
    graph: dict,
    player_location: str,
    other_players: list[dict] | None = None,
    npcs: list[dict] | None = None,
    viewport_center: str | None = None,
) -> str:
    """Render the room graph as compact ASCII art.

    Args:
        graph: The _room_map_graph dict with rooms and edges.
        player_location: location_key slug of the viewing player.
        other_players: list of {"marker": "A", "character_name": ...,
                        "location_key": ...} dicts.
        npcs: list of {"name": ..., "location_key": ...} dicts.
        viewport_center: slug to center viewport on (defaults to player_location).

    Returns:
        ASCII art string.
    """
    rooms: dict[str, dict] = graph.get("rooms", {})
    edges: list[dict] = graph.get("edges", [])

    if not rooms:
        return ""

    # Ensure all rooms positioned
    has_unpositioned = any(not r.get("positioned") for r in rooms.values())
    if has_unpositioned:
        auto_layout(graph)

    other_players = other_players or []
    npcs = npcs or []

    # Build entity placement: slug -> list of marker strings
    entity_map: dict[str, list[str]] = {}
    player_slug = slugify(player_location)
    entity_map.setdefault(player_slug, []).append("@")
    for op in other_players:
        loc = slugify(str(op.get("location_key") or ""))
        marker = str(op.get("marker") or "?")
        if loc:
            entity_map.setdefault(loc, []).append(marker)

    # Determine grid bounds
    xs = [r["x"] for r in rooms.values() if r.get("x") is not None]
    ys = [r["y"] for r in rooms.values() if r.get("y") is not None]
    if not xs or not ys:
        return ""

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Viewport clipping for large maps
    center = viewport_center or player_location
    center_slug = slugify(center)
    if center_slug in rooms and rooms[center_slug].get("positioned"):
        cx = rooms[center_slug]["x"]
        cy = rooms[center_slug]["y"]
    else:
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2

    # How many grid cells fit in our budget?
    cell_w = DEFAULT_ROOM_WIDTH + CONNECTOR_H_LEN
    cell_h = DEFAULT_ROOM_HEIGHT + CONNECTOR_V_LEN
    max_gx = max(1, MAX_COLS // cell_w)
    max_gy = max(1, MAX_ROWS // cell_h)

    half_gx = max_gx // 2
    half_gy = max_gy // 2
    view_x0 = cx - half_gx
    view_x1 = cx + half_gx
    view_y0 = cy - half_gy
    view_y1 = cy + half_gy

    # Filter rooms to viewport
    visible_rooms = {
        slug: r for slug, r in rooms.items()
        if r.get("positioned")
        and view_x0 <= r["x"] <= view_x1
        and view_y0 <= r["y"] <= view_y1
    }
    if not visible_rooms:
        # fallback: show all
        visible_rooms = {slug: r for slug, r in rooms.items() if r.get("positioned")}

    # Recompute bounds for visible rooms
    xs = [r["x"] for r in visible_rooms.values()]
    ys = [r["y"] for r in visible_rooms.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Canvas dimensions
    grid_w = max_x - min_x + 1
    grid_h = max_y - min_y + 1
    canvas_w = grid_w * DEFAULT_ROOM_WIDTH + (grid_w - 1) * CONNECTOR_H_LEN
    canvas_h = grid_h * DEFAULT_ROOM_HEIGHT + (grid_h - 1) * CONNECTOR_V_LEN

    # Create canvas (list of lists of chars)
    canvas: list[list[str]] = [[" "] * canvas_w for _ in range(canvas_h)]

    def _room_pixel(gx: int, gy: int) -> tuple[int, int]:
        """Grid coord -> top-left pixel on canvas."""
        px = (gx - min_x) * (DEFAULT_ROOM_WIDTH + CONNECTOR_H_LEN)
        py = (gy - min_y) * (DEFAULT_ROOM_HEIGHT + CONNECTOR_V_LEN)
        return px, py

    # Draw rooms
    visible_set = set(visible_rooms.keys())
    for slug, room in visible_rooms.items():
        w = min(room.get("width") or DEFAULT_ROOM_WIDTH, DEFAULT_ROOM_WIDTH)
        h = min(room.get("height") or DEFAULT_ROOM_HEIGHT, DEFAULT_ROOM_HEIGHT)
        px, py = _room_pixel(room["x"], room["y"])
        _draw_room_box(canvas, px, py, w, h, room.get("label") or slug, entity_map.get(slug))

    # Draw connectors
    for edge in edges:
        a, b = edge["from"], edge["to"]
        if a not in visible_set or b not in visible_set:
            continue
        ra, rb = visible_rooms[a], visible_rooms[b]
        door_type = edge.get("door_type") or "open"
        _draw_connector(
            canvas,
            _room_pixel(ra["x"], ra["y"]),
            (ra.get("width") or DEFAULT_ROOM_WIDTH, ra.get("height") or DEFAULT_ROOM_HEIGHT),
            _room_pixel(rb["x"], rb["y"]),
            (rb.get("width") or DEFAULT_ROOM_WIDTH, rb.get("height") or DEFAULT_ROOM_HEIGHT),
            door_type,
        )

    # Convert canvas to string, trim trailing whitespace
    map_lines = []
    for row in canvas:
        line = "".join(row).rstrip()
        map_lines.append(line)
    # Remove trailing empty lines
    while map_lines and not map_lines[-1].strip():
        map_lines.pop()
    # Remove leading empty lines
    while map_lines and not map_lines[0].strip():
        map_lines.pop(0)

    # Append legend
    map_lines.append("")
    map_lines.append("Legend:")
    map_lines.append(f"  @ You")
    for op in other_players:
        marker = str(op.get("marker") or "?")
        name = str(op.get("character_name") or op.get("user_id") or "Unknown")
        loc_display = str(op.get("location_display") or op.get("location_key") or "?")
        map_lines.append(f"  {marker} {name} - {loc_display}")

    return "\n".join(map_lines)


def _draw_room_box(
    canvas: list[list[str]],
    px: int,
    py: int,
    width: int,
    height: int,
    label: str,
    entities: list[str] | None = None,
) -> None:
    """Draw a room box at pixel position (px, py) on the canvas."""
    canvas_h = len(canvas)
    canvas_w = len(canvas[0]) if canvas else 0

    def _put(x: int, y: int, ch: str) -> None:
        if 0 <= y < canvas_h and 0 <= x < canvas_w:
            canvas[y][x] = ch

    # Top border
    for x in range(px, px + width):
        if x == px or x == px + width - 1:
            _put(x, py, "+")
        else:
            _put(x, py, "-")

    # Bottom border
    for x in range(px, px + width):
        if x == px or x == px + width - 1:
            _put(x, py + height - 1, "+")
        else:
            _put(x, py + height - 1, "-")

    # Side borders
    for y in range(py + 1, py + height - 1):
        _put(px, y, "|")
        _put(px + width - 1, y, "|")

    # Interior fill (spaces)
    for y in range(py + 1, py + height - 1):
        for x in range(px + 1, px + width - 1):
            _put(x, y, " ")

    # Label on first interior line, centered
    inner_w = width - 2
    truncated = label[:inner_w]
    pad_left = (inner_w - len(truncated)) // 2
    for i, ch in enumerate(truncated):
        _put(px + 1 + pad_left + i, py + 1, ch)

    # Entity markers on the middle interior line
    if entities:
        mid_y = py + (height // 2)
        if mid_y == py + 1:
            mid_y = py + 2 if height > 3 else py + 1  # avoid overwriting label
        marker_str = " ".join(entities[:6])  # cap at 6 markers
        pad_left = max(0, (inner_w - len(marker_str)) // 2)
        for i, ch in enumerate(marker_str):
            _put(px + 1 + pad_left + i, mid_y, ch)


def _draw_connector(
    canvas: list[list[str]],
    pos_a: tuple[int, int],
    size_a: tuple[int, int],
    pos_b: tuple[int, int],
    size_b: tuple[int, int],
    door_type: str,
) -> None:
    """Draw a connector between two rooms."""
    canvas_h = len(canvas)
    canvas_w = len(canvas[0]) if canvas else 0

    def _put(x: int, y: int, ch: str) -> None:
        if 0 <= y < canvas_h and 0 <= x < canvas_w and canvas[y][x] == " ":
            canvas[y][x] = ch

    ax, ay = pos_a
    aw, ah = size_a
    bx, by = pos_b
    bw, bh = size_b

    # Centre points of each room
    a_cx, a_cy = ax + aw // 2, ay + ah // 2
    b_cx, b_cy = bx + bw // 2, by + bh // 2

    dx = bx - ax
    dy = by - ay

    if dy == 0 and dx != 0:
        # Horizontal connector
        door_ch = DOOR_CHARS_H.get(door_type, "-")
        if dx > 0:
            # A is left, B is right
            start_x = ax + aw
            end_x = bx - 1
        else:
            start_x = bx + bw
            end_x = ax - 1
        mid_y = min(a_cy, b_cy)
        mid_x = (start_x + end_x) // 2
        for x in range(start_x, end_x + 1):
            ch = door_ch if x == mid_x else "-"
            _put(x, mid_y, ch)

    elif dx == 0 and dy != 0:
        # Vertical connector
        door_ch = DOOR_CHARS_V.get(door_type, "|")
        if dy > 0:
            start_y = ay + ah
            end_y = by - 1
        else:
            start_y = by + bh
            end_y = ay - 1
        mid_x = min(a_cx, b_cx)
        mid_y = (start_y + end_y) // 2
        for y in range(start_y, end_y + 1):
            ch = door_ch if y == mid_y else "|"
            _put(mid_x, y, ch)

    else:
        # Diagonal — draw an L-shaped connector (go horizontal then vertical)
        if dx > 0:
            h_start = ax + aw
            h_end = b_cx
        else:
            h_start = b_cx
            h_end = ax - 1

        if dy > 0:
            v_start = a_cy
            v_end = by - 1
        else:
            v_start = by + bh
            v_end = a_cy

        # Horizontal segment
        for x in range(min(h_start, h_end), max(h_start, h_end) + 1):
            _put(x, a_cy, "-")
        # Vertical segment
        for y in range(min(v_start, v_end), max(v_start, v_end) + 1):
            _put(b_cx, y, "|")
        # Corner
        _put(b_cx, a_cy, "+")


def render_single_room_box(
    room_title: str,
    player_name: str,
    other_players: list[dict] | None = None,
) -> str:
    """Fallback: render a single room box when no graph data exists."""
    title = (room_title or "Unknown Room")[:27]
    lines = [
        "+-----------------------------+",
        f"| {title:<27} |",
        "|              @              |",
        "+-----------------------------+",
        "",
        "Legend:",
        f"  @ {player_name}",
    ]
    for op in (other_players or [])[:8]:
        marker = str(op.get("marker") or "?")
        name = str(op.get("character_name") or op.get("user_id") or "Unknown")
        loc = str(op.get("location_display") or op.get("location_key") or "?")
        lines.append(f"  {marker} {name} - {loc}")
    return "\n".join(lines)
