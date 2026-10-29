"""Native model/data adapters, imported only inside each model's environment."""
from contextlib import contextmanager


@contextmanager
def temporary_attribute(target, name, value):
    """Restore a native attribute after a scoped override, including on failure."""
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


def tensor_batches(loader):
    """Collate numeric PI inputs in shared tensor memory before worker transport."""
    from torch.utils.data import default_collate

    original = loader.TorchDataLoader

    def create(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("framework") == "pytorch":
            # Configure before iteration so spawned workers receive the collator.
            result.torch_loader.collate_fn = default_collate
        return result

    return temporary_attribute(loader, "TorchDataLoader", create)
