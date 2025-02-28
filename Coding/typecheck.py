import time
import multiprocessing
from random import random
import functools
import inspect
import warnings
import unittest


def chunk_sum(chunk):
    return [sum(sublist) for sublist in chunk]


def parallel_sum(data, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    chunk_size = len(data) // num_processes
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    if len(data) % num_processes:
        chunks[-1].extend(data[num_processes * chunk_size:])
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(chunk_sum, chunks)
    return [item for sublist in results for item in sublist]


def type_check(func=None, *, at_mismatch='warning'):
    if func is None:
        return lambda f: type_check(f, at_mismatch=at_mismatch)

    signature = inspect.signature(func)
    expected_types = {}
    for name, param in signature.parameters.items():
        if param.default is not inspect.Parameter.empty:
            expected_types[name] = type(param.default)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for name, value in bound_args.arguments.items():
            if name in expected_types and not isinstance(value, expected_types[name]):
                message = (f" '{name}' type mismatch: expected {expected_types[name].__name__}, "
                           f"but {type(value).__name__} instead.")
                if at_mismatch == 'exception':
                    raise TypeError(message)
                else:
                    warnings.warn(message, RuntimeWarning)
        return func(*args, **kwargs)

    return wrapper


class Test_7_1(unittest.TestCase):
    def test_parallel_sum_small(self):
        data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        expected = list(map(sum, data))
        result = parallel_sum(data, num_processes=2)
        self.assertEqual(expected, result)

    def test_parallel_sum_random(self):
        data = [[random(), random()] for _ in range(10_000_000)]
        start = time.time()
        expected = list(map(sum, data))
        map_time = time.time() - start
        print(f"Map time: {map_time:.3f} seconds")
        start = time.time()
        result = parallel_sum(data)
        parallel_time = time.time() - start
        print(f"Multiprocessing time: {parallel_time:.3f} seconds")

        self.assertEqual(expected, result)


class Test_7_2(unittest.TestCase):
    def test_type_check_warning(self):
        @type_check
        def func(*, a=12, b='quant'):
            return a, b

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertEqual(func(a=2, b='pt'), (2, 'pt'))
            self.assertEqual(len(w), 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(a='wrong', b=sum)
            for warn in w:
                print(warn.message)
            # 'a' type mismatch: expected int, but str instead.
            # 'b' type mismatch: expected str, but builtin_function_or_method instead.
            self.assertEqual(result, ('wrong', sum))
            self.assertTrue(len(w) > 0)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))

    def test_type_check_exception(self):
        @type_check(at_mismatch='exception')
        def func(*, a='apple', b=(2, 3)):
            return a, b

        self.assertEqual(func(a='abandon', b=(666,)), ('abandon', (666,)))
        with self.assertRaises(TypeError):
            func(a=2025, b=['p', 'a', 'n', 'd', 't', 'o', 'n', 'g'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
