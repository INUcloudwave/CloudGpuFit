import importlib.util
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import prediction_scheduler as scheduler


class SchedulerTests(unittest.TestCase):
    def test_threshold_inclusive_and_reserve(self):
        self.assertTrue(scheduler.admission_allowed({'used_bytes':20,'available_bytes':30},20,6))
        self.assertFalse(scheduler.admission_allowed({'used_bytes':21,'available_bytes':30},20,6))
        self.assertFalse(scheduler.admission_allowed({'used_bytes':10,'available_bytes':5},20,6))

    def test_memory_accounting_uses_available_not_free(self):
        with patch.object(scheduler.Path,'read_text',return_value='MemTotal: 100 kB\nMemFree: 10 kB\nMemAvailable: 60 kB\n'):
            self.assertEqual(scheduler.host_memory(),{'total_bytes':102400,'available_bytes':61440,'used_bytes':40960})

    def test_high_memory_waits_then_resumes_without_failed_case(self):
        admissions=[];events=[];reads=0
        def memory():
            nonlocal reads
            reads+=1
            return {'used_bytes':21 if reads<3 else 20,'available_bytes':30}
        def execute(case,cancel):
            admissions.append(reads);return {'status':'ok'}
        recorded=[]
        flags=scheduler.run_cases([{'case_id':'a'}],execute,lambda c,r:recorded.append((c,r)),
            jobs=2,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
            launch_interval_seconds=0,memory_reader=memory,notify=lambda event,**kw:events.append((event,kw)))
        self.assertEqual(admissions,[3]);self.assertEqual(len(recorded),1)
        self.assertEqual(recorded[0][1]['status'],'ok')
        self.assertEqual([e[1]['reason'] for e in events if e[0]=='wait'],['memory'])
        self.assertFalse(flags['interrupted'])

    def test_jobs_overlap_and_maximum_is_enforced(self):
        lock=threading.Lock();active=peak=0
        def execute(case,cancel):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            time.sleep(.02)
            with lock:active-=1
            return {'status':'ok'}
        completed=[]
        scheduler.run_cases([{'case_id':str(i)} for i in range(6)],execute,lambda c,r:completed.append(c['case_id']),
            jobs=2,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
            launch_interval_seconds=0,memory_reader=lambda:{'used_bytes':2,'available_bytes':40})
        self.assertEqual(peak,2);self.assertEqual(sorted(completed),list('012345'))

    def test_memory_rise_blocks_third_start_and_does_not_cancel_running_jobs(self):
        active=0;reads=[];lock=threading.Lock();release=threading.Event();done=[]
        def memory():
            with lock:count=active
            reads.append(count)
            if count==2:release.set()
            return {'used_bytes':21 if count>=2 else 2,'available_bytes':30}
        def execute(case,cancel):
            nonlocal active
            with lock:active+=1
            release.wait(timeout=2)
            self.assertFalse(cancel.is_set())
            time.sleep(.01)
            with lock:active-=1
            return {'status':'ok'}
        scheduler.run_cases([{'case_id':str(i)} for i in range(3)],execute,lambda c,r:done.append(c['case_id']),
            jobs=3,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
            launch_interval_seconds=.003,memory_reader=memory)
        self.assertIn(2,reads);self.assertEqual(len(done),3)

    def test_reference_stop_cancels_and_drains_peer_without_launching_pending(self):
        started=[];results=[];peer_started=threading.Event()
        def execute(case,cancel):
            started.append(case['case_id'])
            if case['case_id']=='a':
                peer_started.wait(timeout=2);return {'status':'ok'}
            peer_started.set()
            self.assertTrue(cancel.wait(timeout=2))
            return {'status':'error','error_kind':'cancelled'}
        def record(case,result):
            results.append((case['case_id'],result));return case['case_id']!='a'
        flags=scheduler.run_cases([{'case_id':c} for c in 'abc'],execute,record,
            jobs=2,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
            launch_interval_seconds=0,memory_reader=lambda:{'used_bytes':2,'available_bytes':40})
        self.assertEqual(started,['a','b']);self.assertEqual(len(results),2)
        self.assertTrue(flags['validation_failed']);self.assertFalse(flags['interrupted'])

    def test_interrupt_reaches_all_active_workers_and_records_results(self):
        started=[];results=[];both=threading.Event()
        def execute(case,cancel):
            started.append(case['case_id'])
            if len(started)==2:both.set()
            self.assertTrue(cancel.wait(timeout=2))
            return {'status':'error','error_kind':'cancelled'}
        original_wait=scheduler.wait;raised=False
        def wait(*args,**kwargs):
            nonlocal raised
            if both.is_set() and not raised:
                raised=True;raise KeyboardInterrupt
            return original_wait(*args,**kwargs)
        with patch.object(scheduler,'wait',side_effect=wait):
            flags=scheduler.run_cases([{'case_id':c} for c in 'abc'],execute,lambda c,r:results.append(r),
                jobs=2,threshold_bytes=20,reserve_bytes=6,poll_seconds=.001,
                launch_interval_seconds=0,memory_reader=lambda:{'used_bytes':2,'available_bytes':40})
        self.assertEqual(started,['a','b']);self.assertEqual(len(results),2)
        self.assertTrue(flags['interrupted'])


if __name__=='__main__':unittest.main()
