import logging
import webdataset as wds
from webdataset.utils import pytorch_worker_info
from .utils import get_ds_name

class StatefulUrls(wds.PipelineStage):
    def __init__(self, url='', last_seen=0, replacement: bool = False):
        self.url = url
        self.last_seen = last_seen
        self.first_call = True
        self.replacement = replacement
    
    def run(self, src):
        logger = logging.getLogger()
        # start from beginning or resume from url
        if self.first_call:
            self.first_call = False
            # at first, we avoid resuming if we have a null URL 
            self.found = (self.url == '')
            self.seen = 0
            self.logged = False
        
        # - with replacement, we have an infinite stream
        # - w/o replacement, we offset the last_seen index to 0 every epoch
        if not self.replacement:
            self.seen = 0
        
        # loop through samples (should be only dicts with URLs)
        for sample in src:
            # skip until correct url
            if not self.found:
                if (sample['url'] == self.url) and (self.seen == self.last_seen):
                    self.found = True
            
            # once found, yield from there (self.url included)
            if self.found:
                # log the resuming state (only once)
                if not self.logged:
                    self.logged = True
                    rank, _, worker_id, _ = pytorch_worker_info(group=None)
                    if self.url:
                        msg = f"[{rank=}, {worker_id=}] Resuming from {sample['url']} after {self.seen} URLs."
                    else:
                        msg = f"[{rank=}, {worker_id=}] Starting from scratch."
                    logger.error(msg=msg)
                
                # save the state
                self.url = sample['url']
                self.last_seen = self.seen
                yield sample
            
            # update the sample count
            self.seen += 1


class StatefulSamples(wds.PipelineStage):
    def __init__(self, sample_key=''):
        self.sample_key = sample_key
        self.first_call = True
    
    def run(self, src):
        # start from beginning or resume from sample_key
        # see comments above in StatefulUrls
        if self.first_call:
            self.first_call = False
            self.found = (self.sample_key == '')
        for sample in src:
            # skip until correct key
            if not self.found:
                if sample['__key__'] == self.sample_key:
                    self.found = True
                continue
            # once found, save state and yield
            if self.found:
                self.sample_key = sample['__key__']
                yield sample
