import threading
class LatestWinsWorker:
    def __init__(self,fn,done): self.fn=fn; self.done=done; self.lock=threading.Lock(); self.pending=None; self.running=False
    def submit(self,request):
        with self.lock:
            self.pending=request
            if self.running:return
            self.running=True
        threading.Thread(target=self._run,daemon=True).start()
    def _run(self):
        while True:
            with self.lock: req,self.pending=self.pending,None
            if req is None:
                with self.lock:
                    if self.pending is None: self.running=False; return
                continue
            try: result=self.fn(req)
            except Exception as exc: result=exc
            with self.lock: stale=self.pending is not None
            self.done(req,result,stale)
