def benchmark(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result
    return wrapper

@benchmark
def add(a, b):
    import time
    time.sleep(1)
    return a + b


if __name__ == '__main__':
    add(1, 2)
