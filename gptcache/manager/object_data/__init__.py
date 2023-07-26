__all__ = ["ObjectBase"]

from gptcache.utils.lazy_import import LazyImport

object_manager = LazyImport(
    "object_manager", globals(), "gptcache.manager.object_data.manager"
)


def ObjectBase(name: str, **kwargs):
    """Generate specific ObjectStorage with the configuration. For example, setting for
       `ObjectBase` (with `name`) to manage LocalObjectStorage, S3 object storage.
       使用配置生成特定的 ObjectStorage。 例如，设置为
        “ObjectBase”（带有“name”）用于管理 LocalObjectStorage、S3 对象存储。
    """
    return object_manager.ObjectBase.get(name, **kwargs)
