import random
from typing import List, Any

import numpy

from gptcache.utils import softmax


def random_one(messages: List[Any]) -> Any:
    """Randomly select one result after evaluation. 评估后随机选取一项结果。

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import random_one

            messages = ["message 1", "message 2", "message 3"]
            answer = random_one(messages)
    """
    return random.choice(messages)


def first(messages: List[Any]) -> Any:
    """Get the first result after evaluation. 评估后得到第一个结果。

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import first

            messages = ["message 1", "message 2", "message 3"]
            answer = first(messages)
            assert answer = messages[0]
    """
    return messages[0]


def nop(messages: List[Any]) -> Any:
    """No change after evaluation. 评估后无变化。

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import nop

            messages = ["message 1", "message 2", "message 3"]
            answer = nop(messages)
            assert answer = messages
    """
    return messages


def temperature_softmax(messages: List[Any], scores: List[float], temperature: float = 0.0) -> Any:
    """Post processing with temperature softmax after evaluation.评估后使用温度softmax进行后处理。

    :param messages: A list of candidate outputs.
    :type messages: List[Any]
    :param scores: A list of evaluation scores corresponding to `messages` `messages` 对应的评估分数列表
    :type scores: List[float]
    :param temperature: A non-negative number of sampling temperature, defaults to 0.   采样温度的非负数，默认为0。
                        A higher temperature makes the output more random.  较高的温度使输出更加随机。
                        A lower temperature means a more deterministic and confident output.    较低的温度意味着更加确定和自信的输出。
    :type temperature: float

    Example:
        .. code-block:: python

            from gptcache.processor.post import temperature_softmax

            messages = ["message 1", "message 2", "message 3"]
            scores = [0.9, 0.5, 0.1]
            answer = temperature_softmax(messages, scores, temperature=0.5)
    """

    if temperature > 0:
        scores = softmax([x / temperature for x in scores])
        return numpy.random.choice(messages, size=1, p=scores)[0]
    else:
        m_s = list(zip(messages, scores))
        return sorted(m_s, key=lambda x: x[1], reverse=True)[0][0]
    '''
    首先，list(zip(messages, scores)) 将 messages 和 scores 列表进行压缩，生成一个由消息和对应分数组成的元组列表 m_s。每个元组 (message, score) 表示一个消息和其对应的分数。

    然后，使用 sorted() 函数对 m_s 进行排序。sorted() 函数接受一个可迭代对象作为输入，并根据指定的排序键进行排序。在这里，使用了一个 lambda 函数作为排序键，即 lambda x: x[1]，它指定按照每个元组的第二个元素（即分数）进行排序。

    通过将 reverse=True 传递给 sorted() 函数，可以实现按照分数降序排序，即分数最高的消息排在最前面。

    最后，通过索引 [0][0] 取出排序后的列表的第一个元组的第一个元素，即具有最高分数的消息。这个消息将作为函数的返回值返回。

    因此，这段代码的目的是根据分数对消息进行排序，并返回具有最高分数的消息作为结果。如果多个消息具有相同的最高分数，将返回其中的一个。
    '''
