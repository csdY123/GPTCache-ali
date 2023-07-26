class EvictionManager:
    """
    EvictionManager to manager the eviction policy.
"驱逐策略"通常在计算机科学中用于描述当内存或缓存空间不足时，系统应如何选择删除哪些数据的策略。
    :param scalar_storage: CacheStorage to manager the scalar data. CacheStorage 用于管理标量数据。
    :type scalar_storage: :class:`CacheStorage`
    :param vector_base: VectorBase to manager the vector data. VectorBase 管理矢量数据
    :type vector_base:  :class:`VectorBase`
    """

    MAX_MARK_COUNT = 5000
    MAX_MARK_RATE = 0.1
    BATCH_SIZE = 100000
    REBUILD_CONDITION = 5

    def __init__(self, scalar_storage, vector_base):
        self._scalar_storage = scalar_storage
        self._vector_base = vector_base
        self.delete_count = 0

    def check_evict(self):
        mark_count = self._scalar_storage.count(state=-1)   #设置为假删除的个数
        all_count = self._scalar_storage.count(is_all=True) #所有问题的个数
        if (
            mark_count > self.MAX_MARK_COUNT
            or mark_count / all_count > self.MAX_MARK_RATE
        ):
            return True
        return False

    def delete(self):
        mark_ids = self._scalar_storage.get_ids(deleted=True)
        self._scalar_storage.clear_deleted_data()
        self._vector_base.delete(mark_ids)
        self.delete_count += 1
        if self.delete_count >= self.REBUILD_CONDITION:
            self.rebuild()

    def rebuild(self):
        self._scalar_storage.clear_deleted_data()
        ids = self._scalar_storage.get_ids(deleted=False)
        self._vector_base.rebuild(ids)
        self.delete_count = 0

    def soft_evict(self, marked_keys):
        self._scalar_storage.mark_deleted(marked_keys)  #设置标志位-1
