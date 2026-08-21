import threading
class LatestWinsWorker:
    """Single owner thread; submissions replace only the pending request."""
    def __init__(self,fn,done):
        self.fn=fn; self.done=done; self.cv=threading.Condition(); self.pending=None; self.stopping=False; self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
    def submit(self,request):
        with self.cv:
            if self.stopping:return False
            self.pending=request; self.cv.notify(); return True
    def _run(self):
        while True:
            with self.cv:
                while self.pending is None and not self.stopping:self.cv.wait()
                if self.stopping:return
                request,self.pending=self.pending,None
            try: result=self.fn(request)
            except Exception as exc: result=exc
            with self.cv: stale=self.pending is not None
            self.done(request,result,stale)
    def shutdown(self,timeout=5):
        with self.cv:self.stopping=True; self.pending=None; self.cv.notify_all()
        self.thread.join(timeout); return not self.thread.is_alive()
