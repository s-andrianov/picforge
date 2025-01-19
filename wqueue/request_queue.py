import asyncio
from typing import Callable, Awaitable, Any
import time

class RequestQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.current_task = None
        self.start_time = None

    async def add_task(self, task: Callable[..., Awaitable[Any]], *args, **kwargs):
        await self.queue.put((task, args, kwargs))

    async def process_queue(self):
        while True:
            task, args, kwargs = await self.queue.get()
            self.current_task = task.__name__
            self.start_time = time.time()
            try:
                await task(*args, **kwargs)
            finally:
                self.queue.task_done()
                self.current_task = None
                self.start_time = None

    @property
    def queue_size(self):
        return self.queue.qsize()

    @property
    def current_task_name(self):
        return self.current_task

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

