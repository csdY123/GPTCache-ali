import time

import numpy as np

from gptcache import cache
from gptcache.processor.post import temperature_softmax
from gptcache.utils.error import NotInitError
from gptcache.utils.log import gptcache_log
from gptcache.utils.time import time_cal


def adapt(llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs):
    """Adapt to different llm

    :param llm_handler: LLM calling method, when the cache misses, this function will be called LLM调用方法，当缓存未命中时，会调用该函数
    :param cache_data_convert: When the cache hits, convert the answer in the cache to the format of the result returned by llm 当缓存命中时，将缓存中的答案转换为llm返回结果的格式
    :param update_cache_callback: If the cache misses, after getting the result returned by llm, save the result to the cache 如果缓存未命中，获取llm返回的结果后，将结果保存到缓存中
    :param args: llm args
    :param kwargs: llm kwargs
    :return: llm result
    """
    start_time = time.time()
    search_only_flag = kwargs.pop("search_only", False)
    #kwargs.pop("search_only", False) 的作用是从 kwargs 字典中获取键为 "search_only" 的值，并在获取后将其从字典中删除。如果字典中不存在键为 "search_only" 的值，则返回默认值 False。
    user_temperature = "temperature" in kwargs
    #判断 kwargs 字典中是否包含键 "temperature"。如果存在该键，则 user_temperature 的值为 True，否则为 False。
    user_top_k = "top_k" in kwargs
    #user_top_k 是一个用于指定用户（或程序）的结果返回数量的参数或配置选项。
    #在某些应用中，可能需要限制返回给用户的结果数量，以控制结果的数量和展示方式。user_top_k 参数或配置选项用于指定需要返回的结果的数量，通常以整数值表示。
    #具体来说，当 user_top_k 被设置为一个正整数时，表示只返回最高排名的前 k 个结果给用户。这个参数用于限制结果的数量，确保只返回最相关或最重要的结果，从而避免结果过多造成信息过载或性能问题。
    temperature = kwargs.pop("temperature", 0.0)
    chat_cache = kwargs.pop("cache_obj", cache)
    session = kwargs.pop("session", None)   #跟踪用户状态和维持持久连接的概念???
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."   #适配器需要对象存储
    if not chat_cache.has_init:
        raise NotInitError()
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)    #直接返回true
    context = kwargs.pop("cache_context", {})
    embedding_data = None
    # you want to retry to send the request to chatgpt when the cache is negative
    # 当缓存为负值时，您想重试向 chatgpt 发送请求

    if 0 < temperature < 2:
        cache_skip_options = [True, False]
        prob_cache_skip = [0, 1]
        #cache_skip 是一个变量或标志，用于控制在缓存机制中是否跳过缓存的使用。
        cache_skip = kwargs.pop(
            "cache_skip",
            temperature_softmax(
                messages=cache_skip_options,
                scores=prob_cache_skip,
                temperature=temperature,
            ),
        )
    elif temperature >= 2:
        cache_skip = kwargs.pop("cache_skip", True)
    else:  # temperature <= 0
        cache_skip = kwargs.pop("cache_skip", False)
    cache_factor = kwargs.pop("cache_factor", 1.0)  #缓存因子，用于在缓存机制中影响缓存的行为。
    # 通过将 time_cal 装饰器应用到函数上，可以实现对函数执行时间的自动计时和记录。这可以用于分析函数的性能、进行调试或优化。
    #这段代码的目的是使用装饰器函数对 chat_cache.pre_embedding_func 进行计时和记录，并调用该函数进行预处理（pre-processing）的操作。预处理的具体逻辑和功能由 chat_cache.pre_embedding_func 函数定义。同时，通过传递额外的参数，可以对预处理函数进行更灵活的配置和操作。
    pre_embedding_res = time_cal(       #得到结果是“what‘s github”
        chat_cache.pre_embedding_func,
        func_name="pre_process",
        report_func=chat_cache.report.pre,
    )(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
        cache_config=chat_cache.config,
    )
    #返回对象是否是类或其子类的实例。
    if isinstance(pre_embedding_res, tuple):
        pre_store_data = pre_embedding_res[0]
        pre_embedding_data = pre_embedding_res[1]
    else:
        pre_store_data = pre_embedding_res
        pre_embedding_data = pre_embedding_res

    if chat_cache.config.input_summary_len is not None:
        pre_embedding_data = _summarize_input(
            pre_embedding_data, chat_cache.config.input_summary_len
        )

    if cache_enable:
        embedding_data = time_cal(  #没事做，返回原始数据 在相似查询时不是这样
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    if cache_enable and not cache_skip:
        search_data_list = time_cal(    #得到了cache中的数据[("what's github",Answer(answer=''))] [（0.0,1）]
            chat_cache.data_manager.search,     #寻找可能的值，接近的向量
            func_name="search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", 5)
            if (user_temperature and not user_top_k)
            else kwargs.pop("top_k", -1),
        )
        if search_data_list is None:
            search_data_list = []
        cache_answers = []
        #用于定义相似性的阈值，用于筛选在相似性评估中的结果。具体来说，它是一个表示相似性的度量值，在一些应用中用于判断两个对象或数据的相似程度。
        similarity_threshold = chat_cache.config.similarity_threshold   #similarity_threshold: float = 0.8,
        min_rank, max_rank = chat_cache.similarity_evaluation.range()   #进行相似性评估 相似度分数的范围。0,1
        rank_threshold = (max_rank - min_rank) * similarity_threshold * cache_factor    #cache_factor缓存因子默认1.0，结果是0.8
        rank_threshold = (          #代码确保 rank_threshold 的值不会超出指定的最小值和最大值范围
            max_rank
            if rank_threshold > max_rank
            else min_rank
            if rank_threshold < min_rank
            else rank_threshold
        )
        for search_data in search_data_list:
            cache_data = time_cal(  #是CacheData类型的对象
                chat_cache.data_manager.get_scalar_data,    #d得到索引对应的数据 #根据给定的数据 ID，获取对应的缓存数据，并进行会话命中检查和答案转换，最后返回获取到的缓存数据对象。
                func_name="get_data",
                report_func=chat_cache.report.data,
            )(
                search_data,
                extra_param=context.get("get_scalar_data", None),   #context中一直什么都没啊
                session=session,
            )
            if cache_data is None:
                continue

            # cache consistency check   缓存一致性检查
            if chat_cache.config.data_check:
                is_healthy = cache_health_check(
                    chat_cache.data_manager.v,
                    {
                        "embedding": cache_data.embedding_data,
                        "search_result": search_data,
                    },
                )
                if not is_healthy:
                    continue

            if "deps" in context and hasattr(cache_data.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None,
                }
                eval_cache_data = {
                    "question": cache_data.question.deps[0].data,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_store_data, #都是“”what's github"
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": cache_data.question,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,#tuple
                    "cache_data": cache_data,
                    "embedding": cache_data.embedding_data,
                }
            rank = time_cal(    #极其简单的处理，原问题与cache中的问题一致则返回1
                chat_cache.similarity_evaluation.evaluation,    #相似性评估
                func_name="evaluation",
                report_func=chat_cache.report.evaluation,
            )(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )
            gptcache_log.debug(
                "similarity: [user question] %s, [cache question] %s, [value] %f",
                pre_store_data,
                cache_data.question,
                rank,
            )
            if rank_threshold <= rank:  #rank高于门槛值，则加入到cache_answers
                cache_answers.append(
                    (float(rank), cache_data.answers[0].answer, search_data, cache_data)
                )
                chat_cache.data_manager.hit_cache_callback(search_data)
        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        answers_dict = dict((d[1], d) for d in cache_answers)   #回答字典 由元组组成的列表 cache_answers 转换为一个字典 answers_dict
        if len(cache_answers) != 0:

            def post_process():
                if chat_cache.post_process_messages_func is temperature_softmax:
                    return_message = chat_cache.post_process_messages_func(
                        messages=[t[1] for t in cache_answers],
                        scores=[t[0] for t in cache_answers],
                        temperature=temperature,
                    )
                else:
                    return_message = chat_cache.post_process_messages_func(
                        [t[1] for t in cache_answers]
                    )
                return return_message
            #返回提供的信息“github是。。。”
            return_message = time_cal(
                post_process,
                func_name="post_process",
                report_func=chat_cache.report.post,
            )()
            chat_cache.report.hint_cache()
            cache_whole_data = answers_dict.get(str(return_message))
            if session and cache_whole_data:
                chat_cache.data_manager.add_session(
                    cache_whole_data[2], session.name, pre_embedding_data
                )
            if cache_whole_data:
                # user_question / cache_question / cache_question_id / cache_answer / similarity / consume time/ time
                report_cache_data = cache_whole_data[3] #CacheData对象
                report_search_data = cache_whole_data[2]    #cache中查找到的对象
                chat_cache.data_manager.report_cache(   #目前pass
                    pre_store_data if isinstance(pre_store_data, str) else "",
                    report_cache_data.question
                    if isinstance(report_cache_data.question, str)
                    else "",
                    report_search_data[1],
                    report_cache_data.answers[0].answer
                    if isinstance(report_cache_data.answers[0].answer, str)
                    else "",
                    cache_whole_data[0],
                    round(time.time() - start_time, 6),
                )
            return cache_data_convert(return_message)   #return_message：非常纯粹的问题的回答 string类型 函数返回一个字典
    #没有命中的事了
    next_cache = chat_cache.next_cache
    if next_cache:
        kwargs["cache_obj"] = next_cache
        kwargs["cache_context"] = context
        kwargs["cache_skip"] = cache_skip
        kwargs["cache_factor"] = cache_factor
        kwargs["search_only"] = search_only_flag
        llm_data = adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )
    else:
        if search_only_flag:
            # cache miss
            return None
        llm_data = time_cal(
            llm_handler, func_name="llm_request", report_func=chat_cache.report.llm
        )(*args, **kwargs)

    if not llm_data:
        return None

    if cache_enable:
        try:

            def update_cache_func(handled_llm_data, question=None):
                if question is None:
                    question = pre_store_data
                else:
                    question.content = pre_store_data
                time_cal(
                    chat_cache.data_manager.save,
                    func_name="save",
                    report_func=chat_cache.report.save,
                )(
                    question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func", None),
                    session=session,
                )
                if (
                    chat_cache.report.op_save.count > 0
                    and chat_cache.report.op_save.count % chat_cache.config.auto_flush
                    == 0
                ):
                    chat_cache.flush()

            llm_data = update_cache_callback(
                llm_data, update_cache_func, *args, **kwargs
            )
        except Exception as e:  # pylint: disable=W0703
            gptcache_log.warning("failed to save the data to cache, error: %s", e)
    return llm_data


async def aadapt(
    llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
):
    """Simple copy of the 'adapt' method to different llm for 'async llm function'
    将“适应”方法简单复制到“异步 llm 函数”的不同 llm 大型语言模型

    :param llm_handler: Async LLM calling method, when the cache misses, this function will be called
    :param cache_data_convert: When the cache hits, convert the answer in the cache to the format of the result returned by llm
    :param update_cache_callback: If the cache misses, after getting the result returned by llm, save the result to the cache
    :param args: llm args
    :param kwargs: llm kwargs
    :return: llm result
    """
    start_time = time.time()
    user_temperature = "temperature" in kwargs
    user_top_k = "top_k" in kwargs
    temperature = kwargs.pop("temperature", 0.0)
    chat_cache = kwargs.pop("cache_obj", cache)
    session = kwargs.pop("session", None)
    require_object_store = kwargs.pop("require_object_store", False)
    if require_object_store:
        assert chat_cache.data_manager.o, "Object store is required for adapter."
    if not chat_cache.has_init:
        raise NotInitError()
    cache_enable = chat_cache.cache_enable_func(*args, **kwargs)
    context = kwargs.pop("cache_context", {})
    embedding_data = None
    # you want to retry to send the request to chatgpt when the cache is negative

    if 0 < temperature < 2:
        cache_skip_options = [True, False]
        prob_cache_skip = [0, 1]
        cache_skip = kwargs.pop(
            "cache_skip",
            temperature_softmax(
                messages=cache_skip_options,
                scores=prob_cache_skip,
                temperature=temperature,
            ),
        )
    elif temperature >= 2:
        cache_skip = kwargs.pop("cache_skip", True)
    else:  # temperature <= 0
        cache_skip = kwargs.pop("cache_skip", False)
    cache_factor = kwargs.pop("cache_factor", 1.0)
    pre_embedding_res = time_cal(
        chat_cache.pre_embedding_func,
        func_name="pre_process",
        report_func=chat_cache.report.pre,
    )(
        kwargs,
        extra_param=context.get("pre_embedding_func", None),
        prompts=chat_cache.config.prompts,
        cache_config=chat_cache.config,
    )
    if isinstance(pre_embedding_res, tuple):
        pre_store_data = pre_embedding_res[0]
        pre_embedding_data = pre_embedding_res[1]
    else:
        pre_store_data = pre_embedding_res
        pre_embedding_data = pre_embedding_res

    if chat_cache.config.input_summary_len is not None:
        pre_embedding_data = _summarize_input(
            pre_embedding_data, chat_cache.config.input_summary_len
        )

    if cache_enable:
        embedding_data = time_cal(
            chat_cache.embedding_func,
            func_name="embedding",
            report_func=chat_cache.report.embedding,
        )(pre_embedding_data, extra_param=context.get("embedding_func", None))
    if cache_enable and not cache_skip:
        search_data_list = time_cal(
            chat_cache.data_manager.search,
            func_name="search",
            report_func=chat_cache.report.search,
        )(
            embedding_data,
            extra_param=context.get("search_func", None),
            top_k=kwargs.pop("top_k", 5)
            if (user_temperature and not user_top_k)
            else kwargs.pop("top_k", -1),
        )
        if search_data_list is None:
            search_data_list = []
        cache_answers = []
        similarity_threshold = chat_cache.config.similarity_threshold
        min_rank, max_rank = chat_cache.similarity_evaluation.range()
        rank_threshold = (max_rank - min_rank) * similarity_threshold * cache_factor
        rank_threshold = (
            max_rank
            if rank_threshold > max_rank
            else min_rank
            if rank_threshold < min_rank
            else rank_threshold
        )
        for search_data in search_data_list:
            cache_data = time_cal(
                chat_cache.data_manager.get_scalar_data,
                func_name="get_data",
                report_func=chat_cache.report.data,
            )(
                search_data,
                extra_param=context.get("get_scalar_data", None),
                session=session,
            )
            if cache_data is None:
                continue

            if "deps" in context and hasattr(cache_data.question, "deps"):
                eval_query_data = {
                    "question": context["deps"][0]["data"],
                    "embedding": None,
                }
                eval_cache_data = {
                    "question": cache_data.question.deps[0].data,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": None,
                }
            else:
                eval_query_data = {
                    "question": pre_store_data,
                    "embedding": embedding_data,
                }

                eval_cache_data = {
                    "question": cache_data.question,
                    "answer": cache_data.answers[0].answer,
                    "search_result": search_data,
                    "cache_data": cache_data,
                    "embedding": cache_data.embedding_data,
                }
            rank = time_cal(
                chat_cache.similarity_evaluation.evaluation,
                func_name="evaluation",
                report_func=chat_cache.report.evaluation,
            )(
                eval_query_data,
                eval_cache_data,
                extra_param=context.get("evaluation_func", None),
            )
            gptcache_log.debug(
                "similarity: [user question] %s, [cache question] %s, [value] %f",
                pre_store_data,
                cache_data.question,
                rank,
            )
            if rank_threshold <= rank:
                cache_answers.append(
                    (float(rank), cache_data.answers[0].answer, search_data, cache_data)
                )
                chat_cache.data_manager.hit_cache_callback(search_data)
        cache_answers = sorted(cache_answers, key=lambda x: x[0], reverse=True)
        answers_dict = dict((d[1], d) for d in cache_answers)
        if len(cache_answers) != 0:
            def post_process():
                if chat_cache.post_process_messages_func is temperature_softmax:
                    return_message = chat_cache.post_process_messages_func(
                        messages=[t[1] for t in cache_answers],
                        scores=[t[0] for t in cache_answers],
                        temperature=temperature,
                    )
                else:
                    return_message = chat_cache.post_process_messages_func(
                        [t[1] for t in cache_answers]
                    )
                return return_message

            return_message = time_cal(
                post_process,
                func_name="post_process",
                report_func=chat_cache.report.post,
            )()
            chat_cache.report.hint_cache()
            cache_whole_data = answers_dict.get(str(return_message))
            if session and cache_whole_data:
                chat_cache.data_manager.add_session(
                    cache_whole_data[2], session.name, pre_embedding_data
                )
            if cache_whole_data:
                # user_question / cache_question / cache_question_id / cache_answer / similarity / consume time/ time
                report_cache_data = cache_whole_data[3]
                report_search_data = cache_whole_data[2]
                chat_cache.data_manager.report_cache(
                    pre_store_data if isinstance(pre_store_data, str) else "",
                    report_cache_data.question
                    if isinstance(report_cache_data.question, str)
                    else "",
                    report_search_data[1],
                    report_cache_data.answers[0].answer
                    if isinstance(report_cache_data.answers[0].answer, str)
                    else "",
                    cache_whole_data[0],
                    round(time.time() - start_time, 6),
                )
            return cache_data_convert(return_message)

    next_cache = chat_cache.next_cache
    if next_cache:
        kwargs["cache_obj"] = next_cache
        kwargs["cache_context"] = context
        kwargs["cache_skip"] = cache_skip
        kwargs["cache_factor"] = cache_factor
        llm_data = adapt(
            llm_handler, cache_data_convert, update_cache_callback, *args, **kwargs
        )
    else:
        llm_data = await llm_handler(*args, **kwargs)

    if cache_enable:
        try:

            def update_cache_func(handled_llm_data, question=None):
                if question is None:
                    question = pre_store_data
                else:
                    question.content = pre_store_data
                time_cal(
                    chat_cache.data_manager.save,
                    func_name="save",
                    report_func=chat_cache.report.save,
                )(
                    question,
                    handled_llm_data,
                    embedding_data,
                    extra_param=context.get("save_func", None),
                    session=session,
                )
                if (
                    chat_cache.report.op_save.count > 0
                    and chat_cache.report.op_save.count % chat_cache.config.auto_flush
                    == 0
                ):
                    chat_cache.flush()

            llm_data = update_cache_callback(
                llm_data, update_cache_func, *args, **kwargs
            )
        except Exception:  # pylint: disable=W0703
            gptcache_log.error("failed to save the data to cache", exc_info=True)
    return llm_data


_input_summarizer = None


def _summarize_input(text, text_length):
    if len(text) <= text_length:
        return text

    # pylint: disable=import-outside-toplevel
    from gptcache.processor.context.summarization_context import (
        SummarizationContextProcess,
    )

    global _input_summarizer
    if _input_summarizer is None:
        _input_summarizer = SummarizationContextProcess()
    summarization = _input_summarizer.summarize_to_sentence([text], text_length)
    return summarization


def cache_health_check(vectordb, cache_dict):
    """This function checks if the embedding
    from vector store matches one in cache store.
    If cache store and vector store are out of
    sync with each other, cache retrieval can
    be incorrect.
    If this happens, force the similary score
    to the lowerest possible value.
    """
    emb_in_cache = cache_dict["embedding"]
    _, data_id = cache_dict["search_result"]
    emb_in_vec = vectordb.get_embeddings(data_id)
    flag = np.all(emb_in_cache == emb_in_vec)
    if not flag:
        gptcache_log.critical("Cache Store and Vector Store are out of sync!!!")
        # 0: identical, inf: different
        cache_dict["search_result"] = (
            np.inf,
            data_id,
        )
        # self-healing by replacing entry
        # in the vec store with the one
        # from cache store by the same
        # entry_id.
        vectordb.update_embeddings(
            data_id,
            emb=cache_dict["embedding"],
        )
    return flag
