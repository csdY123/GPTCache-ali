__all__ = ["CacheBase"]

from gptcache.utils.lazy_import import LazyImport
#使用了 LazyImport 函数来延迟导入
scalar_manager = LazyImport(
    "scalar_manager", globals(), "gptcache.manager.scalar_data.manager"
)


def CacheBase(name: str, **kwargs):
    """Generate specific CacheStorage with the configuration. For example, setting for
    #使用配置生成特定的 CacheStorage。 例如，设置为
       `SQLDataBase` (with `name`, `sql_url` and `table_name` params) to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.
        使用配置生成特定的 CacheStorage。 例如，设置为
        “SQLDataBase”（带有“name”、“sql_url”和“table_name”参数）用于管理 SQLite、PostgreSQL、MySQL、MariaDB、SQL Server 和 Oracle。
    """
    return scalar_manager.CacheBase.get(name, **kwargs)
