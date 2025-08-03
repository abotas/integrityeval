import time
from functools import wraps

def retry(retries=1, delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            model_id = None
            for arg in args:
                if hasattr(arg, 'model_id'):
                    model_id = arg.model_id
                    break
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries:
                        if model_id:
                            print(f"Attempt {attempt + 1} failed for {model_id}: {e}. Retrying in {delay}s...")
                        else:
                            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator