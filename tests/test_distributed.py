import sys

from agentic_optimizer import distributed


def test_fallback_path_no_initialized_dist(monkeypatch):
    obj = {"x": [1, 2]}

    monkeypatch.setattr(distributed, "_dist", lambda: None)

    assert distributed.is_available() is False
    assert distributed.rank() == 0
    assert distributed.world_size() == 1
    assert distributed.is_main_process() is True
    assert distributed.backend() is None
    assert distributed.barrier() is None
    assert distributed.broadcast_object(obj) is obj
    assert distributed.all_reduce_mean(3.0) == 3.0
    assert distributed.info() == {
        "enabled": False,
        "rank": 0,
        "world_size": 1,
        "backend": None,
    }


def test_initialized_path_uses_dist(monkeypatch):
    calls = {"barrier": 0, "broadcast": [], "all_reduce": 0}

    class FakeTensor:
        def __init__(self, value):
            self.value = float(value)

        def __truediv__(self, other):
            return FakeTensor(self.value / other)

        def item(self):
            return self.value

    class FakeTorch:
        float64 = object()

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def tensor(value, **kwargs):
            assert kwargs == {"dtype": FakeTorch.float64}
            return FakeTensor(value)

    class FakeReduceOp:
        SUM = object()

    class FakeDist:
        ReduceOp = FakeReduceOp

        @staticmethod
        def get_rank():
            return 2

        @staticmethod
        def get_world_size():
            return 4

        @staticmethod
        def get_backend():
            return "gloo"

        @staticmethod
        def barrier():
            calls["barrier"] += 1

        @staticmethod
        def broadcast_object_list(objects, src=0):
            calls["broadcast"].append(src)
            objects[0] = {"from": src}

        @staticmethod
        def all_reduce(tensor, op=None):
            assert op is FakeReduceOp.SUM
            calls["all_reduce"] += 1
            tensor.value *= 4

    monkeypatch.setattr(distributed, "_dist", lambda: FakeDist)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    assert distributed.is_available() is True
    assert distributed.rank() == 2
    assert distributed.world_size() == 4
    assert distributed.is_main_process() is False
    assert distributed.backend() == "gloo"
    assert distributed.barrier() is None
    assert calls["barrier"] == 1
    assert distributed.broadcast_object({"local": True}, src=1) == {"from": 1}
    assert calls["broadcast"] == [1]
    assert distributed.all_reduce_mean(2.5) == 2.5
    assert calls["all_reduce"] == 1
    assert distributed.info() == {
        "enabled": True,
        "rank": 2,
        "world_size": 4,
        "backend": "gloo",
    }


def test_all_reduce_mean_falls_back_without_torch(monkeypatch):
    class FakeDist:
        pass

    monkeypatch.setattr(distributed, "_dist", lambda: FakeDist())
    monkeypatch.setitem(sys.modules, "torch", None)

    assert distributed.all_reduce_mean(3) == 3.0
