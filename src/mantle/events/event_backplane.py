"""Optional cross-process signal distribution — the air-gap pattern, applied to signaling.

`TieredContentStore` is the model this module copies deliberately: the local tier is the whole
product, the remote tier is reached only when the local one cannot answer, and `remote=None` is a
**legal, first-class configuration** rather than a degraded one. Substitute "in-process bus" for
"local cache" and "back-plane" for "S3" and the shape is identical:

==================  ==========================================  ==========
tier                role                                        required?
==================  ==========================================  ==========
`event_bus`         the core. Zero dependencies, single-process  always
                    fan-out, complete on its own.
`EventBackplane`    cross-process / cross-node signal            never
                    distribution.
==================  ==========================================  ==========

So `backplane=None` is not "events are broken", it is "this node is one process". A single worker
with no back-plane is a fully supported deployment and every test in the suite runs that way.

What a back-plane is and is not
-------------------------------
It is a **signal** carrier: it moves the already-published `Event` from the process that emitted it
to the other processes serving the same store, so a WebSocket subscriber attached to worker B sees
a write made by worker A. It is not a queue, not a log, and not a durability mechanism — durability
is :class:`event_bus.EventLog`, which is per-store and works with no back-plane at all. A back-plane
that drops a message costs a live subscriber a live update; it never costs a durable subscriber an
event, because the durable subscriber resumes from the log by cursor.

It also carries **no authority**. The wire form is the event's own fields; every receiving process
re-runs its own ACL filter before delivering to any subscriber. A compromised or confused
back-plane can therefore inject or lose signals, but it cannot widen what any subscriber may see —
visibility is decided locally, from local grants, on every delivery.

Two shapes, one protocol
------------------------
* **Redis pub/sub** — the datacenter multi-worker shape: one broker, low latency, fire-and-forget.
* **MQTT** — the edge/mesh shape: offline-tolerant, broadcast, low-bandwidth, and the natural
  companion to `mesh/carrier.py`'s broadcast-medium abstraction.

Both are optional extras. Neither is imported until a node is configured to use one, so the base
install still needs nothing but `cryptography`.

Refusing rather than degrading
------------------------------
With N worker processes and no back-plane, an emit in worker A is invisible to a subscriber on
worker B: the change feed silently loses roughly `(N-1)/N` of its events and the live UI shows a
stale tree while reporting healthy. :func:`require_backplane_for_workers` refuses that
configuration at boot and names the setting, which is the discipline `shard/sqlite_store.py`
already applies when it declines to fall back to a degraded backend.

Loop prevention
---------------
Every process stamps its own :data:`NODE` id on what it publishes and drops what comes back
carrying that id. Without it, one emit re-enters its own bus on the next broker round trip and the
fan-out becomes a fan-out loop.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "NODE",
    "BackplaneUnavailable",
    "EventBackplane",
    "RedisBackplane",
    "MqttBackplane",
    "make_backplane",
    "backplane_from_env",
    "require_backplane_for_workers",
    "KIND_SETTING",
    "URI_SETTING",
    "WORKERS_SETTING",
]

#: This process's identity on the back-plane. Per-process, not per-node: two workers on one host
#: are two distinct publishers and each must ignore only its own echo, not its sibling's.
NODE: str = uuid.uuid4().hex

#: The settings this module reads. Named as constants because the boot refusal quotes them back to
#: the operator, and a refusal that names the wrong knob is worse than no refusal.
KIND_SETTING = "MANTLE_BACKPLANE_KIND"
URI_SETTING = "MANTLE_BACKPLANE_URI"
WORKERS_SETTING = "MANTLE_WORKERS"

#: The channel/topic both implementations default to. One channel for the whole feed: filtering is
#: a subscriber-side concern that already exists (`EventFilter`), and per-container channels would
#: put the container id — which is authorization-relevant vocabulary — into broker metadata.
DEFAULT_CHANNEL = "mantle.events"


class BackplaneUnavailable(ImportError):
    """The configured back-plane's client library is not installed.

    An `ImportError` subclass on purpose: this is the same failure `shard/sqlite_store.py` refuses
    to paper over. A node configured for Redis that silently ran without it would be a node whose
    change feed is quietly wrong, which is the exact outcome the configuration was chosen to avoid.
    """


class EventBackplane:
    """The protocol every back-plane implements. Three methods and a name.

    Structural, not inherited — an operator may pass any object with these methods and
    :func:`event_bus.set_backplane` will use it. Subclassing is available for convenience and is
    what the two shipped adapters do.

    Lifecycle is `start(deliver)` … `publish(wire)` * n … `close()`. `deliver` is called from
    whatever thread the client library uses, so implementations must assume it is not the event
    loop thread; :func:`event_bus.deliver_from_backplane` is thread-safe for exactly that reason.
    """

    #: Short identifier used in logs and in the boot refusal's message.
    name: str = "none"

    def start(self, deliver: Callable[[Dict[str, Any]], None]) -> None:
        """Connect and begin calling *deliver* with each inbound wire dict."""
        raise NotImplementedError

    def publish(self, wire: Dict[str, Any]) -> None:
        """Send one wire dict to every other process. Best-effort by contract."""
        raise NotImplementedError

    def close(self) -> None:
        """Disconnect. Idempotent."""
        raise NotImplementedError

    # -- shared plumbing ---------------------------------------------------------------

    @staticmethod
    def stamp(wire: Dict[str, Any]) -> Dict[str, Any]:
        """Mark a wire dict as originating here, so the echo can be dropped."""
        out = dict(wire)
        out["_node"] = NODE
        return out

    @staticmethod
    def is_echo(wire: Any) -> bool:
        """Did this process publish it? Anything unparseable reads as an echo — dropping an
        unreadable message is the fail-closed direction, since the alternative is feeding a
        malformed payload into the fan-out path."""
        return not isinstance(wire, dict) or wire.get("_node") == NODE


# ---------------------------------------------------------------------------
# Redis pub/sub — the datacenter multi-worker shape
# ---------------------------------------------------------------------------

class RedisBackplane(EventBackplane):
    """Redis pub/sub. Requires the `[redis]` extra.

    Pub/sub rather than a stream: this carries signals, not durability, and a stream would invite
    the mistake of treating the broker as the log. The log is `event_bus.EventLog`, which lives
    beside the store the events describe and survives the broker being absent entirely.
    """

    name = "redis"

    def __init__(self, uri: str, *, channel: str = DEFAULT_CHANNEL):
        self.uri = uri
        self.channel = channel
        self._client = None
        self._pubsub = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self, deliver: Callable[[Dict[str, Any]], None]) -> None:
        try:
            import redis
        except ImportError as exc:
            raise BackplaneUnavailable(
                f"{KIND_SETTING}=redis but the redis client is not installed. Install the extra "
                f"(`pip install agience-mantle[redis]`) or set {KIND_SETTING}=none — running "
                f"without the back-plane this node was configured for would silently drop most of "
                f"the change feed across workers.") from exc
        import json

        self._client = redis.Redis.from_url(self.uri)
        self._pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(self.channel)

        def _pump() -> None:
            while not self._stop.is_set():
                try:
                    msg = self._pubsub.get_message(timeout=1.0)
                except Exception as exc:                   # a broker blip is not a process fault
                    logger.warning("redis back-plane read failed: %s", exc)
                    continue
                if not msg or msg.get("type") != "message":
                    continue
                try:
                    wire = json.loads(msg["data"])
                except Exception:
                    continue
                if self.is_echo(wire):
                    continue
                deliver(wire)

        self._thread = threading.Thread(target=_pump, name="mantle-backplane-redis", daemon=True)
        self._thread.start()

    def publish(self, wire: Dict[str, Any]) -> None:
        import json
        if self._client is None:
            return
        try:
            self._client.publish(self.channel, json.dumps(self.stamp(wire)))
        except Exception as exc:
            # Best-effort by contract: a signal lost here costs a live subscriber a live update.
            # A durable subscriber still resumes by cursor from the log, which is local.
            logger.warning("redis back-plane publish failed: %s", exc)

    def close(self) -> None:
        self._stop.set()
        for obj in (self._pubsub, self._client):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# MQTT — the edge / mesh shape
# ---------------------------------------------------------------------------

class MqttBackplane(EventBackplane):
    """MQTT. Requires the `[mqtt]` extra.

    QoS 0 by default, matching the signal contract: an at-most-once signal in front of an
    at-least-once log is coherent, whereas paying for broker-side retention would duplicate the
    log's job at a second point of truth.
    """

    name = "mqtt"

    def __init__(self, uri: str, *, topic: str = DEFAULT_CHANNEL, qos: int = 0):
        self.uri = uri
        self.topic = topic
        self.qos = int(qos)
        self._client = None

    def start(self, deliver: Callable[[Dict[str, Any]], None]) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise BackplaneUnavailable(
                f"{KIND_SETTING}=mqtt but paho-mqtt is not installed. Install the extra "
                f"(`pip install agience-mantle[mqtt]`) or set {KIND_SETTING}=none — running "
                f"without the back-plane this node was configured for would silently drop most of "
                f"the change feed across workers.") from exc
        import json
        from urllib.parse import urlparse

        parsed = urlparse(self.uri if "://" in self.uri else "mqtt://" + self.uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 1883

        client = mqtt.Client()
        if parsed.username:
            client.username_pw_set(parsed.username, parsed.password or None)

        def _on_connect(cl, _userdata, _flags, _rc, *_a) -> None:
            cl.subscribe(self.topic, qos=self.qos)

        def _on_message(_cl, _userdata, msg) -> None:
            try:
                wire = json.loads(msg.payload)
            except Exception:
                return
            if self.is_echo(wire):
                return
            deliver(wire)

        client.on_connect = _on_connect
        client.on_message = _on_message
        client.connect(host, port)
        client.loop_start()
        self._client = client

    def publish(self, wire: Dict[str, Any]) -> None:
        import json
        if self._client is None:
            return
        try:
            self._client.publish(self.topic, json.dumps(self.stamp(wire)), qos=self.qos)
        except Exception as exc:
            logger.warning("mqtt back-plane publish failed: %s", exc)

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

_KINDS = {"redis": RedisBackplane, "mqtt": MqttBackplane}


def make_backplane(kind: Optional[str], uri: Optional[str] = None,
                   **kwargs: Any) -> Optional[EventBackplane]:
    """Build the configured back-plane, or `None` for the standalone configuration.

    `None`, `""` and `"none"` all mean standalone, and all three are legal answers rather than
    missing configuration — a single-worker node is complete without one. An unrecognized kind is
    refused rather than defaulted to `None`: a typo in the setting must not present as a working
    standalone node when the operator asked for cross-process distribution.
    """
    normalized = (kind or "none").strip().lower()
    if normalized in ("", "none", "off", "disabled"):
        return None
    cls = _KINDS.get(normalized)
    if cls is None:
        raise ValueError(
            f"{KIND_SETTING}={kind!r} is not a back-plane. Known kinds: "
            f"{', '.join(sorted(_KINDS))}, or 'none' for the standalone in-process bus.")
    if not uri:
        raise ValueError(
            f"{KIND_SETTING}={normalized} needs {URI_SETTING}. A broker kind with no address "
            f"cannot connect, and starting anyway would produce a node that looks configured for "
            f"multi-worker but distributes nothing.")
    return cls(uri, **kwargs)


def backplane_from_env(env: Optional[Dict[str, str]] = None) -> Optional[EventBackplane]:
    """`make_backplane` over the two environment settings. Unset means standalone."""
    src = env if env is not None else os.environ
    return make_backplane(src.get(KIND_SETTING), src.get(URI_SETTING))


def require_backplane_for_workers(workers: Any, backplane: Optional[EventBackplane],
                                  *, setting: str = WORKERS_SETTING) -> None:
    """Refuse to start a multi-worker process group with no back-plane configured.

    Refusing over degrading, because the degraded mode is invisible from inside: every worker
    reports healthy, every write succeeds, and only a subscriber on a different worker than the
    writer notices that most of the change feed never arrives. There is no metric a node can emit
    about events it was never told about.

    One worker with no back-plane is silent and correct — the standalone configuration — so this
    raises only when both halves of the hazard are present.
    """
    try:
        count = int(workers)
    except (TypeError, ValueError):
        count = 1
    if count <= 1 or backplane is not None:
        return
    raise RuntimeError(
        f"{setting}={count} with no signalling back-plane configured. With more than one worker "
        f"process an event emitted in one worker never reaches a subscriber attached to another, "
        f"so the change feed silently loses roughly {count - 1}/{count} of its deliveries and the "
        f"live UI goes stale while every worker reports healthy. Set {KIND_SETTING} "
        f"(redis | mqtt) and {URI_SETTING}, or run a single worker — one worker with no "
        f"back-plane is a fully supported standalone configuration.")
