from abc import ABCMeta
from typing import Any, Dict, Callable, Optional


class BaseCacheLLM(metaclass=ABCMeta):
    """Base LLM, When you have enhanced llm without using the original llm api,
    you can use this class as a proxy to use the ability of the cache.

    NOTE: Please make sure that the custom llm returns the same value as the original llm.

    For example, if you use the openai proxy, you perform delay statistics before sending the openai request,
    and then you package this part of the function, so you may have a separate package, which is different from openai.
    If the api request parameters and return results you wrap are the same as the original ones,
    then you can use this class to obtain cache-related capabilities.
    基础LLM，当您在不使用原始llm api的情况下增强了llm时，
    你可以使用这个类作为代理来使用缓存的能力。
     注意：请确保自定义 llm 返回与原始 llm 相同的值。
     例如，如果使用openai代理，则在发送openai请求之前进行延迟统计，
     然后你把这部分功能封装起来，所以你可能有一个单独的封装，这个和openai不同。
     如果你包装的api请求参数和返回结果与原来的一样，
     那么就可以使用这个类来获取缓存相关的能力。
    Example:
        .. code-block:: python

            import time

            import openai

            from gptcache import Cache
            from gptcache.adapter import openai as cache_openai


            def proxy_openai_chat_complete(*args, **kwargs):
                start_time = time.time()
                res = openai.ChatCompletion.create(*args, **kwargs)
                print("Consume Time Spent =", round((time.time() - start_time), 2))
                return res


            llm_cache = Cache()

            cache_openai.ChatCompletion.llm = proxy_openai_chat_complete
            cache_openai.ChatCompletion.cache_args = {"cache_obj": llm_cache}

            cache_openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "What's GitHub?",
                    }
                ],
            )
    """

    llm: Optional[Callable] = None
    """
    On a cache miss, if that variable is set, it will be called;
    if not, it will call the original llm.
    当缓存未命中时，如果设置了该变量，则会调用该变量；
     如果没有，就会调用原来的llm。
    """

    cache_args: Dict[str, Any] = {}
    """
    It can be used to set some cache-related public parameters.
    If you don't want to set the same parameters every time when using cache, say cache_obj, you can use it.
    可以用来设置一些缓存相关的公共参数。
     如果你不想每次使用缓存时都设置相同的参数，比如说cache_obj，你可以使用它。
    """

    @classmethod
    def fill_base_args(cls, **kwargs):
        """ Fill the base args to the cache args 将基本参数填充到缓存参数中
        """
        for key, value in cls.cache_args.items():
            if key not in kwargs:
                kwargs[key] = value

        return kwargs
