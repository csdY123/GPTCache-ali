# -*- coding: utf-8 -*-


from gptcache.utils import softmax


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
