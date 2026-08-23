from __future__ import annotations

import unittest
from pathlib import Path
import socket
import struct
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dogfight.unreal.client import RemoteClientContext, UnrealAIPilotUDPClient
from dogfight.unreal.protocol import (
    AIType,
    CMD,
    CLIENT_JOIN_INFO_STRUCT,
    ClientJoinInfo,
    PlaneInfo,
    Rotation3D,
    SetPlaneID,
    Vector3D,
    pack_client_join_info,
)
from run_unreal_inference import resolve_team_name


class _RecordingPolicy:
    def __init__(self):
        self.frames: list[tuple[int, int, int]] = []

    def reset(self, context: RemoteClientContext) -> None:
        self.frames.clear()

    def compute_command(self, context: RemoteClientContext) -> CMD:
        own = context.own_plane.frame_index
        enemy = context.enemy_plane.frame_index
        self.frames.append((context.frame_index, own, enemy))
        return CMD(context.plane_id, context.frame_index, 0.0, 0.0, 0.0, 0.5)


def _plane(frame: int, plane_id: int) -> PlaneInfo:
    return PlaneInfo(
        index=frame,
        plane_id=plane_id,
        position=Vector3D(float(plane_id), 0.0, 8000.0),
        rotation=Rotation3D(),
        velocity=Vector3D(250.0, 0.0, 0.0),
    )


class UnrealClientFramePairingTests(unittest.TestCase):
    def setUp(self):
        self.policy = _RecordingPolicy()
        self.client = UnrealAIPilotUDPClient(
            command_policy=self.policy,
            server_ip="127.0.0.1",
            server_port=9999,
            team_name="TEST",
            ai_type=AIType.ReinforcementLearning,
            command_delay_sec=0.0,
        )
        self.commands: list[CMD] = []
        self.client.send_command = self.commands.append
        self.client._handle_set_plane_id(SetPlaneID(plane_id=0))

    def test_sends_once_for_each_matching_frame(self):
        self.client._handle_plane_info(_plane(1, 0))
        self.client._handle_plane_info(_plane(1, 1))
        self.client._handle_plane_info(_plane(1, 0))
        self.client._handle_plane_info(_plane(1, 1))

        self.assertEqual(self.policy.frames, [(1, 1, 1)])
        self.assertEqual([command.index for command in self.commands], [1])

    def test_does_not_pair_different_frames_after_reordering(self):
        self.client._handle_plane_info(_plane(10, 0))
        self.client._handle_plane_info(_plane(9, 1))
        self.assertEqual(self.commands, [])

        self.client._handle_plane_info(_plane(10, 1))
        self.assertEqual(self.policy.frames, [(10, 10, 10)])
        self.assertEqual(self.commands[0].index, 10)

    def test_recovers_on_next_matching_frame_after_packet_loss(self):
        self.client._handle_plane_info(_plane(20, 0))
        self.client._handle_plane_info(_plane(21, 0))
        self.client._handle_plane_info(_plane(20, 1))
        self.assertEqual(self.commands, [])

        self.client._handle_plane_info(_plane(21, 1))
        self.assertEqual(self.policy.frames, [(21, 21, 21)])
        self.assertEqual(self.commands[0].index, 21)


class _FakeSocket:
    def __init__(self, buffers: list[bytes]):
        self._buffers = list(buffers)

    def recv(self, bufsize: int) -> bytes:
        if not self._buffers:
            raise socket.timeout()
        return self._buffers.pop(0)


class UnrealClientPacketErrorRecoveryTests(unittest.TestCase):
    def test_receive_loop_survives_command_policy_exception_and_keeps_processing(self):
        # A single bad frame must not end the match: the server side waits
        # on our CMD to advance simulation, so letting an exception from
        # command handling escape the receive loop freezes both aircraft
        # for the rest of the round with no way to recover.
        client = UnrealAIPilotUDPClient(command_policy=_RecordingPolicy())
        call_count = {"n": 0}

        def _raise_once_then_stop(buffer, remote_endpoint):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("boom")
            client._running = False

        client._process_packet = _raise_once_then_stop
        client._socket = _FakeSocket([b"frame1", b"frame2"])
        client._running = True

        client._receive_loop()

        self.assertEqual(call_count["n"], 2)
        self.assertEqual(client._process_packet_error_count, 1)


class UnrealProtocolTeamNameTests(unittest.TestCase):
    def test_team_name_file_reads_utf8_without_batch_code_page_dependency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "team_name.txt"
            path.write_text("하늘팀\n", encoding="utf-8")
            args = SimpleNamespace(team_name_file=str(path), team_name="ignored")
            self.assertEqual(resolve_team_name(args), "하늘팀")

    def test_team_name_file_rejects_more_than_29_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "team_name.txt"
            path.write_text("가" * 10, encoding="utf-8")
            args = SimpleNamespace(team_name_file=str(path), team_name="ignored")
            with self.assertRaisesRegex(ValueError, "29-byte"):
                resolve_team_name(args)

    def test_client_defaults_to_localhost(self):
        client = UnrealAIPilotUDPClient(command_policy=_RecordingPolicy())
        self.assertEqual(client.server_ip, "127.0.0.1")

    def test_korean_team_name_is_truncated_on_utf8_character_boundary(self):
        packet = pack_client_join_info(
            ClientJoinInfo(
                team_name="가나다라마바사아자차카타파하",
                ai_type=AIType.ReinforcementLearning,
                plane_id=-1,
            )
        )
        _, encoded_name, _, _ = CLIENT_JOIN_INFO_STRUCT.unpack(packet)
        raw_name = encoded_name.split(b"\x00", 1)[0]

        decoded_name = raw_name.decode("utf-8")
        self.assertTrue("가나다라마바사아자차카타파하".startswith(decoded_name))
        self.assertLessEqual(len(raw_name), 29)

    def test_embedded_null_does_not_leak_trailing_team_name_bytes(self):
        packet = pack_client_join_info(
            ClientJoinInfo(
                team_name="TEAM\x00IGNORED",
                ai_type=AIType.ReinforcementLearning,
                plane_id=-1,
            )
        )
        _, encoded_name, _, _ = CLIENT_JOIN_INFO_STRUCT.unpack(packet)
        self.assertEqual(encoded_name.split(b"\x00", 1)[0], b"TEAM")

    def test_unpublished_damage_packet_is_counted_without_being_parsed(self):
        policy = _RecordingPolicy()
        client = UnrealAIPilotUDPClient(command_policy=policy)
        packet = struct.pack("<if", 3, 0.25)

        client._process_packet(packet, "127.0.0.1:9999")

        self.assertEqual(client._rx_counts["MT_Damage"], 1)
        self.assertEqual(client._last_received_packet.message_type, "MT_Damage")
        self.assertEqual(
            client._last_received_packet.fields,
            {"value": {"unparsed_payload_bytes": 4}},
        )

    def test_malformed_packets_are_counted_and_do_not_block_next_valid_packet(self):
        policy = _RecordingPolicy()
        client = UnrealAIPilotUDPClient(command_policy=policy)
        client.send_command = lambda command: None

        client._process_packet(b"\x01\x02", "127.0.0.1:9999")
        client._process_packet(struct.pack("<i", 123456), "127.0.0.1:9999")
        client._process_packet(
            struct.pack("<i", int(2)),
            "127.0.0.1:9999",
        )
        client._process_packet(
            struct.pack("<ib", int(9), 0),
            "127.0.0.1:9999",
        )

        self.assertEqual(client._rx_counts["INVALID"], 3)
        self.assertEqual(client.context.plane_id, 0)
        self.assertEqual(client._rx_counts["MT_SetPlaneID"], 1)


if __name__ == "__main__":
    unittest.main()
