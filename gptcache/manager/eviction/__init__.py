__all__ = ["EvictionBase"]

from gptcache.utils.lazy_import import LazyImport

eviction_manager = LazyImport(
    "eviction_manager", globals(), "gptcache.manager.eviction.manager"
)


def EvictionBase(name: str, **kwargs):
    """Generate specific CacheStorage with the configuration. 使用配置生成特定的 CacheStorage。

    :param name: the name of the eviction, like: memory
    :type name: str

    :param policy: eviction strategy    驱逐策略
    :type policy: str
    :param maxsize: the maxsize of cache data   缓存数据的最大大小
    :type maxsize: int
    :param clean_size: will clean the size of data when the size of cache data reaches the max size
        当缓存数据大小达到最大大小时，将清理数据大小
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
        清理store中数据的功能
    :type  on_evict: Callable[[List[Any]], None]

    Example:
        .. code-block:: python

            from gptcache.manager import EvictionBase

            cache_base = EvictionBase('memory', policy='lru', maxsize=10, clean_size=2, on_evict=lambda x: print(x))
    """
    return eviction_manager.EvictionBase.get(name, **kwargs)
