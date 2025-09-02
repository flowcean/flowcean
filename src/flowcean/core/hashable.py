import hashlib
from typing import Protocol

import cloudpickle


class Hashable(Protocol):
    """A mixin class that provides a method to compute a hash of the object.

    This class uses `cloudpickle` to serialize the object and then computes a
    SHA-256 hash of the serialized bytes. This is useful for caching and
    deduplication purposes.
    """

    def hash(self) -> bytes:
        """Get the hash of the transform.

        Returns:
            The hash of the transform.
        """
        hasher = hashlib.sha256()
        hasher.update(cloudpickle.dumps(self))
        return hasher.digest()
