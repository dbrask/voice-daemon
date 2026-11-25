"""Desk satellite: thin voice client (Pi) + relay (on the GPU host).

The relay (`satellite.relay`) is the only component that knows both halves:
it calls the voice daemon over loopback (`:9877` /stt, /tts) and Hermes'
api_server (`/v1/runs`). The client (`satellite.client`) runs on the Pi and
holds nothing but a per-device token.

See satellite/protocol.py for the v1 wire protocol and the README for setup.
"""

__all__ = ["protocol"]
