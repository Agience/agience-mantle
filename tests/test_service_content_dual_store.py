import io

import pytest

from mantle.services import content_service as content


@pytest.fixture(autouse=True)
def _clear_edge_presence_memo():
    """The edge-presence memo is process-wide and outlives a test by design.

    Cleared around every test here because these tests assert on what the S3 fakes were asked
    for, and a memo carried in from a neighbour would make a HEAD that did happen look skipped."""
    content.forget_edge_presence()
    yield
    content.forget_edge_presence()


class FakeS3Client:
    def __init__(self, *, presign_base: str = "https://edge.example", objects=None):
        self.presign_base = presign_base
        self.objects = dict(objects or {})
        self.deleted = []

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise Exception("missing")
        item = self.objects[Key]
        return {
            "ContentLength": len(item.get("Body", b"")),
            "ContentType": item.get("ContentType"),
            "CacheControl": item.get("CacheControl"),
        }

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise Exception("missing")
        item = self.objects[Key]
        return {
            "Body": io.BytesIO(item.get("Body", b"")),
            "ContentType": item.get("ContentType"),
            "CacheControl": item.get("CacheControl"),
        }

    def put_object(self, Bucket, Key, Body, ContentType=None, CacheControl=None):
        data = Body.read() if hasattr(Body, "read") else Body
        self.objects[Key] = {
            "Body": data,
            "ContentType": ContentType,
            "CacheControl": CacheControl,
        }
        return {"ETag": "etag"}

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)
        self.objects.pop(Key, None)
        return {}

    def generate_presigned_url(self, operation_name, Params, ExpiresIn):
        return f"{self.presign_base}/{Params['Bucket']}/{Params['Key']}?op={operation_name}"

    def create_bucket(self, **kwargs):
        return {}


def test_persist_object_to_durable_and_evict_edge(monkeypatch):
    key = "tenant/artifact.content"
    edge_internal = FakeS3Client(
        objects={
            key: {
                "Body": b"hello",
                "ContentType": "text/plain",
                "CacheControl": "private, max-age=31536000, immutable",
            }
        }
    )
    durable = FakeS3Client()

    monkeypatch.setattr(content, "_s3_edge_internal", edge_internal)
    monkeypatch.setattr(content, "_s3_durable", durable)
    monkeypatch.setattr(content, "_EDGE_BUCKET", "edge-bucket")
    monkeypatch.setattr(content, "_DURABLE_BUCKET", "durable-bucket")
    monkeypatch.setattr(content, "_EVICT_EDGE_AFTER_DURABLE_SYNC", True)

    assert content.persist_object_to_durable(key) is True
    assert key in durable.objects
    assert key not in edge_internal.objects
    assert edge_internal.deleted == [key]


def test_generate_signed_url_hydrates_edge_from_durable(monkeypatch):
    key = "tenant/artifact.content"
    edge_internal = FakeS3Client()
    edge_public = FakeS3Client(presign_base="https://minio.example")
    durable = FakeS3Client(
        objects={
            key: {
                "Body": b"hello",
                "ContentType": "text/plain",
                "CacheControl": "private, max-age=31536000, immutable",
            }
        }
    )

    monkeypatch.setattr(content, "_s3_edge_internal", edge_internal)
    monkeypatch.setattr(content, "_s3_edge_public", edge_public)
    monkeypatch.setattr(content, "_s3_durable", durable)
    monkeypatch.setattr(content, "_EDGE_BUCKET", "edge-bucket")
    monkeypatch.setattr(content, "_DURABLE_BUCKET", "durable-bucket")
    monkeypatch.setattr(content, "_BUCKET_CHECKED", True)

    url = content.generate_signed_url(key, filename="a.txt", content_type="text/plain")

    assert url.startswith("https://minio.example/edge-bucket/tenant/artifact.content")
    assert key in edge_internal.objects


class CountingS3Client(FakeS3Client):
    """A `FakeS3Client` that records how often it was HEADed."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.heads = []

    def head_object(self, Bucket, Key):
        self.heads.append(Key)
        return super().head_object(Bucket, Key)


def _wire_edge_only(monkeypatch, edge_internal, edge_public):
    monkeypatch.setattr(content, "_s3_edge_internal", edge_internal)
    monkeypatch.setattr(content, "_s3_edge_public", edge_public)
    monkeypatch.setattr(content, "_s3_durable", None)
    monkeypatch.setattr(content, "_DURABLE_BUCKET", "")
    monkeypatch.setattr(content, "_EDGE_BUCKET", "edge-bucket")
    monkeypatch.setattr(content, "_BUCKET_CHECKED", True)


def test_presign_heads_the_edge_object_once_not_once_per_signature(monkeypatch):
    """Every presign HEADed the edge bucket before signing, doubling the round trips on the
    download path. The bucket-existence check beside it is already memoized; this is the same
    memo one level down."""
    key = "tenant/artifact.content"
    edge_internal = CountingS3Client(objects={key: {"Body": b"hello"}})
    _wire_edge_only(monkeypatch, edge_internal, FakeS3Client(presign_base="https://minio.example"))

    for _ in range(4):
        assert content.generate_signed_url(key, filename="a.txt").startswith("https://minio.example")

    assert edge_internal.heads == [key]


def test_an_absent_edge_object_is_never_memoized(monkeypatch):
    """Absence is what triggers hydration from durable storage. Remembering it would pin the
    key as missing for the life of the entry and defeat the hydration it exists to trigger."""
    key = "tenant/gone.content"
    edge_internal = CountingS3Client()
    _wire_edge_only(monkeypatch, edge_internal, FakeS3Client(presign_base="https://minio.example"))

    for _ in range(3):
        content.generate_signed_url(key)

    assert edge_internal.heads == [key, key, key]

    # And once it appears, the next answer is the live one, not a cached absence.
    edge_internal.objects[key] = {"Body": b"back"}
    assert content.ensure_edge_object_present(key) is True


def test_deleting_an_object_forgets_that_it_was_present(monkeypatch):
    """A memo that outlived the object would sign a URL that 404s."""
    key = "tenant/artifact.content"
    edge_internal = CountingS3Client(objects={key: {"Body": b"hello"}})
    _wire_edge_only(monkeypatch, edge_internal, FakeS3Client(presign_base="https://minio.example"))

    assert content.ensure_edge_object_present(key) is True
    content.delete_object(key)

    assert content.ensure_edge_object_present(key) is False
    assert edge_internal.heads == [key, key]        # the second HEAD really was re-issued


def test_the_presence_memo_expires_and_stays_bounded(monkeypatch):
    """Two bounds, because a deletion made outside this process leaves no signal here: a TTL so
    the wrongness is a window rather than a wedged path, and a size cap so a long-running
    process cannot grow the memo without limit."""
    key = "tenant/artifact.content"
    edge_internal = CountingS3Client(objects={key: {"Body": b"hello"}})
    _wire_edge_only(monkeypatch, edge_internal, FakeS3Client(presign_base="https://minio.example"))

    monkeypatch.setattr(content, "_EDGE_PRESENT_TTL_S", -1.0)
    assert content.ensure_edge_object_present(key) is True
    assert content.ensure_edge_object_present(key) is True
    assert edge_internal.heads == [key, key]        # expired immediately, so re-checked

    monkeypatch.setattr(content, "_EDGE_PRESENT_TTL_S", 300.0)
    monkeypatch.setattr(content, "_EDGE_PRESENT_MAX", 8)
    for i in range(64):
        content._remember_edge_presence("k%d" % i)
    assert len(content._edge_present) == 8
    assert content._edge_presence_cached("k63") is True      # newest kept
    assert content._edge_presence_cached("k0") is False      # oldest evicted


def test_get_content_storage_mode_minio_only(monkeypatch):
    monkeypatch.setattr(content, "_s3_durable", None)
    monkeypatch.setattr(content, "_DURABLE_BUCKET", "")
    monkeypatch.delenv("CLOUDFRONT_KEY_ID", raising=False)
    monkeypatch.delenv("CLOUDFRONT_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("CLOUDFRONT_PRIVATE_KEY_PATH", raising=False)

    assert content.get_content_storage_mode() == "minio-only"


def test_get_content_storage_mode_minio_s3_backed(monkeypatch):
    monkeypatch.setattr(content, "_s3_durable", object())
    monkeypatch.setattr(content, "_DURABLE_BUCKET", "durable-bucket")
    monkeypatch.delenv("CLOUDFRONT_KEY_ID", raising=False)
    monkeypatch.delenv("CLOUDFRONT_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("CLOUDFRONT_PRIVATE_KEY_PATH", raising=False)

    assert content.get_content_storage_mode() == "minio-s3-backed"


def test_get_content_storage_mode_cloudfront_s3(monkeypatch):
    monkeypatch.setattr(content, "_s3_durable", object())
    monkeypatch.setattr(content, "_DURABLE_BUCKET", "durable-bucket")
    monkeypatch.setenv("CLOUDFRONT_KEY_ID", "key-id")
    monkeypatch.setenv("CLOUDFRONT_PRIVATE_KEY_PATH", "private.pem")
    monkeypatch.delenv("CLOUDFRONT_PRIVATE_KEY", raising=False)

    assert content.get_content_storage_mode() == "cloudfront-s3"
