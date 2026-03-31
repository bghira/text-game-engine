"""Tests for the deterministic ASCII map renderer and room graph builder."""

import pytest

from text_game_engine.core.ascii_map import (
    MIN_ROOM_WIDTH,
    _strip_building_name,
    add_edge,
    auto_layout,
    detect_building_clusters,
    ensure_room,
    render_ascii_map,
    render_single_room_box,
    slugify,
    update_room_map_graph,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Fellowship Hall") == "fellowship-hall"

    def test_special_chars(self):
        assert slugify("Penthouse Suite, Escala") == "penthouse-suite-escala"

    def test_max_length(self):
        assert len(slugify("a" * 200)) <= 80


class TestEnsureRoom:
    def test_adds_new_room(self):
        graph = {"rooms": {}, "edges": []}
        r = ensure_room(graph, "lobby", label="Hotel Lobby", turn=1)
        assert "lobby" in graph["rooms"]
        assert r["label"] == "Hotel Lobby"
        assert r["width"] >= MIN_ROOM_WIDTH
        assert r["height"] == 5
        assert r["positioned"] is False

    def test_does_not_overwrite_existing(self):
        graph = {"rooms": {"lobby": {"label": "Hotel Lobby", "width": 20}}, "edges": []}
        r = ensure_room(graph, "lobby", label="New Name")
        assert r["label"] == "Hotel Lobby"  # not overwritten
        assert r["width"] == 20

    def test_updates_slug_label(self):
        graph = {"rooms": {"lobby": {"label": "lobby"}}, "edges": []}
        ensure_room(graph, "lobby", label="Hotel Lobby")
        assert graph["rooms"]["lobby"]["label"] == "Hotel Lobby"


class TestAddEdge:
    def test_adds_new_edge(self):
        graph = {"rooms": {}, "edges": []}
        added = add_edge(graph, "a", "b", direction="east", door_type="open")
        assert added is True
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["from"] == "a"
        assert graph["edges"][0]["direction"] == "east"

    def test_deduplicates(self):
        graph = {"rooms": {}, "edges": []}
        add_edge(graph, "a", "b", direction="east")
        added = add_edge(graph, "b", "a", direction="west")
        assert added is False
        assert len(graph["edges"]) == 1

    def test_merges_missing_direction(self):
        graph = {"rooms": {}, "edges": []}
        add_edge(graph, "a", "b")
        add_edge(graph, "a", "b", direction="north")
        assert graph["edges"][0]["direction"] == "north"


class TestAutoLayout:
    def test_single_room(self):
        graph = {
            "rooms": {"lobby": {"label": "Lobby", "width": 17, "height": 5,
                                "x": None, "y": None, "positioned": False,
                                "first_seen_turn": 0}},
            "edges": [],
        }
        auto_layout(graph)
        assert graph["rooms"]["lobby"]["positioned"] is True
        assert graph["rooms"]["lobby"]["x"] == 0
        assert graph["rooms"]["lobby"]["y"] == 0

    def test_two_rooms_with_direction(self):
        graph = {
            "rooms": {
                "lobby": {"label": "Lobby", "width": 17, "height": 5,
                          "x": None, "y": None, "positioned": False,
                          "first_seen_turn": 0},
                "kitchen": {"label": "Kitchen", "width": 17, "height": 5,
                            "x": None, "y": None, "positioned": False,
                            "first_seen_turn": 1},
            },
            "edges": [{"from": "lobby", "to": "kitchen", "direction": "east",
                       "door_type": None, "bidirectional": True}],
        }
        auto_layout(graph)
        assert graph["rooms"]["lobby"]["x"] == 0
        assert graph["rooms"]["lobby"]["y"] == 0
        assert graph["rooms"]["kitchen"]["x"] == 1
        assert graph["rooms"]["kitchen"]["y"] == 0

    def test_does_not_move_positioned_rooms(self):
        graph = {
            "rooms": {
                "lobby": {"label": "Lobby", "width": 17, "height": 5,
                          "x": 5, "y": 5, "positioned": True,
                          "first_seen_turn": 0},
                "kitchen": {"label": "Kitchen", "width": 17, "height": 5,
                            "x": None, "y": None, "positioned": False,
                            "first_seen_turn": 1},
            },
            "edges": [{"from": "lobby", "to": "kitchen", "direction": "south",
                       "door_type": None, "bidirectional": True}],
        }
        auto_layout(graph)
        assert graph["rooms"]["lobby"]["x"] == 5
        assert graph["rooms"]["lobby"]["y"] == 5
        assert graph["rooms"]["kitchen"]["x"] == 5
        assert graph["rooms"]["kitchen"]["y"] == 6

    def test_three_rooms_chain(self):
        graph = {
            "rooms": {
                "a": {"label": "A", "width": 17, "height": 5,
                      "x": None, "y": None, "positioned": False,
                      "first_seen_turn": 0},
                "b": {"label": "B", "width": 17, "height": 5,
                      "x": None, "y": None, "positioned": False,
                      "first_seen_turn": 1},
                "c": {"label": "C", "width": 17, "height": 5,
                      "x": None, "y": None, "positioned": False,
                      "first_seen_turn": 2},
            },
            "edges": [
                {"from": "a", "to": "b", "direction": "east",
                 "door_type": None, "bidirectional": True},
                {"from": "b", "to": "c", "direction": "south",
                 "door_type": None, "bidirectional": True},
            ],
        }
        auto_layout(graph)
        assert graph["rooms"]["a"]["positioned"] is True
        assert graph["rooms"]["b"]["positioned"] is True
        assert graph["rooms"]["c"]["positioned"] is True
        # a at 0,0; b at 1,0; c at 1,1
        assert graph["rooms"]["a"]["x"] == 0
        assert graph["rooms"]["b"]["x"] == 1
        assert graph["rooms"]["c"]["y"] == 1


class TestUpdateRoomMapGraph:
    def test_creates_graph_from_structured_exits(self):
        campaign_state = {}
        player_state = {
            "location": "lobby",
            "room_title": "Hotel Lobby",
            "exits": [
                {"name": "Kitchen", "location": "kitchen", "direction": "north"},
                {"name": "Garden", "location": "garden", "direction": "east", "door": "archway"},
            ],
        }
        graph = update_room_map_graph(campaign_state, player_state)
        assert "lobby" in graph["rooms"]
        assert "kitchen" in graph["rooms"]
        assert "garden" in graph["rooms"]
        assert len(graph["edges"]) == 2
        # All rooms should be positioned
        for r in graph["rooms"].values():
            assert r["positioned"] is True

    def test_ignores_plain_string_exits_without_known_locations(self):
        campaign_state = {}
        player_state = {
            "location": "lobby",
            "room_title": "Lobby",
            "exits": ["some vague hallway", "a dark passage"],
        }
        graph = update_room_map_graph(campaign_state, player_state)
        assert len(graph["rooms"]) == 1
        assert len(graph["edges"]) == 0

    def test_matches_plain_string_exits_against_known_locations(self):
        campaign_state = {
            "_location_cards": {
                "kitchen": {"name": "Kitchen"},
                "garden": {"name": "Garden Terrace"},
            }
        }
        player_state = {
            "location": "lobby",
            "room_title": "Lobby",
            "exits": ["Kitchen", "unknown exit"],
        }
        graph = update_room_map_graph(
            campaign_state, player_state,
            known_locations=campaign_state["_location_cards"],
        )
        assert "kitchen" in graph["rooms"]
        assert len(graph["edges"]) == 1

    def test_incremental_building(self):
        campaign_state = {}
        # Turn 1: player in lobby
        player_state = {
            "location": "lobby",
            "room_title": "Lobby",
            "exits": [{"name": "Kitchen", "location": "kitchen", "direction": "n"}],
        }
        update_room_map_graph(campaign_state, player_state, turn_number=1)

        # Turn 2: player moves to kitchen
        player_state = {
            "location": "kitchen",
            "room_title": "Kitchen",
            "exits": [
                {"name": "Lobby", "location": "lobby", "direction": "s"},
                {"name": "Pantry", "location": "pantry", "direction": "e"},
            ],
        }
        graph = update_room_map_graph(campaign_state, player_state, turn_number=2)
        assert len(graph["rooms"]) == 3
        assert len(graph["edges"]) == 2  # lobby-kitchen deduplicated

    def test_does_not_create_self_loops(self):
        campaign_state = {}
        player_state = {
            "location": "lobby",
            "room_title": "Lobby",
            "exits": [{"name": "Lobby", "location": "lobby", "direction": "n"}],
        }
        graph = update_room_map_graph(campaign_state, player_state)
        assert len(graph["edges"]) == 0


class TestRenderAsciiMap:
    def _simple_graph(self):
        graph = {
            "rooms": {
                "lobby": {"label": "Hotel Lobby", "width": 17, "height": 5,
                          "x": 0, "y": 0, "positioned": True, "first_seen_turn": 0},
                "kitchen": {"label": "Kitchen", "width": 17, "height": 5,
                            "x": 1, "y": 0, "positioned": True, "first_seen_turn": 1},
            },
            "edges": [
                {"from": "lobby", "to": "kitchen", "direction": "east",
                 "door_type": "open", "bidirectional": True},
            ],
            "layout_version": 1,
        }
        return graph

    def test_renders_rooms(self):
        graph = self._simple_graph()
        result = render_ascii_map(graph, "lobby")
        assert "Hotel Lobby" in result
        assert "Kitchen" in result
        assert "@" in result
        assert "Legend:" in result

    def test_player_marker_in_correct_room(self):
        graph = self._simple_graph()
        result = render_ascii_map(graph, "lobby")
        lines = result.split("\n")
        # Find the line with @ — it should be in the left room (lobby)
        for line in lines:
            if "@" in line and "Legend" not in line:
                # @ should appear before Kitchen's box starts
                at_pos = line.index("@")
                assert at_pos < 17  # within first room box

    def test_other_player_markers(self):
        graph = self._simple_graph()
        others = [
            {"marker": "A", "character_name": "Bob",
             "location_key": "kitchen", "location_display": "Kitchen"},
        ]
        result = render_ascii_map(graph, "lobby", other_players=others)
        assert "A" in result
        assert "A Bob - Kitchen" in result

    def test_connector_between_rooms(self):
        graph = self._simple_graph()
        result = render_ascii_map(graph, "lobby")
        lines = result.split("\n")
        # There should be a horizontal connector between the two rooms
        has_connector = any("-" * 3 in line for line in lines[:5])
        assert has_connector

    def test_vertical_connector(self):
        graph = {
            "rooms": {
                "a": {"label": "Room A", "width": 17, "height": 5,
                      "x": 0, "y": 0, "positioned": True, "first_seen_turn": 0},
                "b": {"label": "Room B", "width": 17, "height": 5,
                      "x": 0, "y": 1, "positioned": True, "first_seen_turn": 1},
            },
            "edges": [
                {"from": "a", "to": "b", "direction": "south",
                 "door_type": "open", "bidirectional": True},
            ],
            "layout_version": 1,
        }
        result = render_ascii_map(graph, "a")
        assert "Room A" in result
        assert "Room B" in result
        lines = result.split("\n")
        # Should have a vertical connector between the two rooms
        map_lines = [l for l in lines if l.strip() and "Legend" not in l and "@" not in l.split("  ")]
        assert len(map_lines) >= 10  # two rooms + connector


class TestBuildingClusters:
    def _vana_kalamaja_graph(self):
        """Build the real-world Vana-Kalamaja apartment graph."""
        rooms = {
            "vana-kalamaja-apartment": {
                "label": "Vana-Kalamaja Apartment", "width": 17, "height": 5,
                "x": 0, "y": 0, "positioned": True, "first_seen_turn": 0,
            },
            "vana-kalamaja-exterior": {
                "label": "Vana-Kalamaja Exterior", "width": 17, "height": 5,
                "x": 1, "y": 0, "positioned": True, "first_seen_turn": 1,
            },
            "rooftop-terrace": {
                "label": "Rooftop Terrace", "width": 17, "height": 5,
                "x": 2, "y": 0, "positioned": True, "first_seen_turn": 2,
            },
            "ground-floor-unit-vana-kalamaja": {
                "label": "Ground Floor Unit, Vana-Kalamaja", "width": 17, "height": 5,
                "x": 0, "y": 1, "positioned": True, "first_seen_turn": 3,
            },
            "upstairs-bathroom-vana-kalamaja": {
                "label": "Upstairs Bathroom, Vana-Kalamaja", "width": 17, "height": 5,
                "x": 1, "y": 1, "positioned": True, "first_seen_turn": 4,
            },
            "bedroom": {
                "label": "Bedroom", "width": 17, "height": 5,
                "x": 2, "y": 1, "positioned": True, "first_seen_turn": 5,
            },
            "sitting-room-vana-kalamaja": {
                "label": "Sitting Room, Vana-Kalamaja", "width": 17, "height": 5,
                "x": 0, "y": 2, "positioned": True, "first_seen_turn": 6,
            },
        }
        edges = [
            {"from": "vana-kalamaja-apartment", "to": "vana-kalamaja-exterior",
             "direction": "east", "door_type": "open", "bidirectional": True},
            {"from": "vana-kalamaja-apartment", "to": "ground-floor-unit-vana-kalamaja",
             "direction": "south", "door_type": "door", "bidirectional": True},
            {"from": "vana-kalamaja-exterior", "to": "upstairs-bathroom-vana-kalamaja",
             "direction": "south", "door_type": "open", "bidirectional": True},
            {"from": "ground-floor-unit-vana-kalamaja", "to": "sitting-room-vana-kalamaja",
             "direction": "south", "door_type": "open", "bidirectional": True},
            {"from": "upstairs-bathroom-vana-kalamaja", "to": "bedroom",
             "direction": "east", "door_type": "door", "bidirectional": True},
            {"from": "rooftop-terrace", "to": "vana-kalamaja-exterior",
             "direction": "west", "door_type": "stair", "bidirectional": True},
            {"from": "bedroom", "to": "sitting-room-vana-kalamaja",
             "direction": "south", "door_type": "open", "bidirectional": True},
        ]
        return {"rooms": rooms, "edges": edges, "layout_version": 1}

    def test_detects_vana_kalamaja_cluster(self):
        graph = self._vana_kalamaja_graph()
        clusters = detect_building_clusters(graph["rooms"], graph["edges"])
        assert len(clusters) == 1
        c = clusters[0]
        assert "Vana-Kalamaja" in c["name"]
        # All 7 rooms should be in the cluster (5 with name + 2 absorbed)
        assert len(c["rooms"]) == 7
        assert "bedroom" in c["rooms"]
        assert "rooftop-terrace" in c["rooms"]

    def test_no_cluster_under_minimum_size(self):
        rooms = {
            "a": {"label": "Hotel Lobby", "width": 17, "height": 5},
            "b": {"label": "Hotel Kitchen", "width": 17, "height": 5},
        }
        edges = [{"from": "a", "to": "b", "direction": "east",
                  "door_type": None, "bidirectional": True}]
        clusters = detect_building_clusters(rooms, edges)
        assert clusters == []

    def test_no_cluster_without_common_term(self):
        rooms = {
            "a": {"label": "Lobby", "width": 17, "height": 5},
            "b": {"label": "Kitchen", "width": 17, "height": 5},
            "c": {"label": "Garden", "width": 17, "height": 5},
        }
        edges = [
            {"from": "a", "to": "b", "direction": "east",
             "door_type": None, "bidirectional": True},
            {"from": "b", "to": "c", "direction": "east",
             "door_type": None, "bidirectional": True},
        ]
        clusters = detect_building_clusters(rooms, edges)
        assert clusters == []

    def test_strip_building_name_from_labels(self):
        assert _strip_building_name("Sitting Room, Vana-Kalamaja", "Vana-Kalamaja") == "Sitting Room"
        assert _strip_building_name("Vana-Kalamaja Exterior", "Vana-Kalamaja") == "Exterior"
        assert _strip_building_name("Ground Floor Unit, Vana-Kalamaja", "Vana-Kalamaja") == "Ground Floor Unit"
        # If stripping leaves nothing, return original
        assert _strip_building_name("Vana-Kalamaja", "Vana-Kalamaja") == "Vana-Kalamaja"

    def test_render_with_building_envelope(self):
        graph = self._vana_kalamaja_graph()
        result = render_ascii_map(graph, "vana-kalamaja-apartment")
        # Building envelope should be present
        assert "╔" in result
        assert "╗" in result
        assert "╚" in result
        assert "╝" in result
        assert "Vana-Kalamaja" in result
        # Stripped labels should appear (not the full "X, Vana-Kalamaja")
        assert "Sitting Room" in result
        assert "Ground Floor Unit" in result
        # Player marker
        assert "@" in result

    def test_render_no_envelope_for_small_graphs(self):
        """Two-room graphs should render without envelopes."""
        graph = {
            "rooms": {
                "lobby": {"label": "Hotel Lobby", "width": 17, "height": 5,
                          "x": 0, "y": 0, "positioned": True, "first_seen_turn": 0},
                "kitchen": {"label": "Hotel Kitchen", "width": 17, "height": 5,
                            "x": 1, "y": 0, "positioned": True, "first_seen_turn": 1},
            },
            "edges": [{"from": "lobby", "to": "kitchen", "direction": "east",
                       "door_type": "open", "bidirectional": True}],
            "layout_version": 1,
        }
        result = render_ascii_map(graph, "lobby")
        assert "╔" not in result  # no envelope
        assert "Hotel Lobby" in result
        assert "Hotel Kitchen" in result

    def test_disconnected_rooms_not_absorbed(self):
        """A room with no edges to the cluster stays outside."""
        rooms = {
            "a": {"label": "Hotel Lobby", "width": 17, "height": 5},
            "b": {"label": "Hotel Kitchen", "width": 17, "height": 5},
            "c": {"label": "Hotel Bar", "width": 17, "height": 5},
            "shed": {"label": "Garden Shed", "width": 17, "height": 5},
        }
        edges = [
            {"from": "a", "to": "b", "direction": "east",
             "door_type": None, "bidirectional": True},
            {"from": "b", "to": "c", "direction": "east",
             "door_type": None, "bidirectional": True},
        ]
        clusters = detect_building_clusters(rooms, edges)
        assert len(clusters) == 1
        assert "shed" not in clusters[0]["rooms"]
        assert len(clusters[0]["rooms"]) == 3


class TestRenderSingleRoomBox:
    def test_basic(self):
        result = render_single_room_box("Lobby", "Player1")
        assert "Lobby" in result
        assert "@" in result
        assert "Player1" in result
        assert "Legend:" in result

    def test_with_others(self):
        others = [{"marker": "A", "character_name": "Bob",
                    "location_display": "Kitchen"}]
        result = render_single_room_box("Lobby", "Player1", other_players=others)
        assert "A Bob - Kitchen" in result
