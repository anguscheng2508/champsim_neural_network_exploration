#!/bin/env python
#https://docs.python.org/2/library/queue.html

import os
from subprocess import call
import commands
import operator
import sys
import random


import time
start_time=time.time()

binary = ""

resultsBuffer = []

DesignSpace = []

#trace_dir = "/home/opt/Benchmarks/speccpu/"
trace_dir = "/[path]/[to]/[top 30 sensitive tracers]/"
#bins="champsim champsim_ampmlite champsim_ipstride champsim_nextline champsim_spp".split()
#bins="champsim champsim_nextline champsim_d10 champsim_d10mf champsim_ip_stride".split() # champsim_hp champsim_gshare".split()
bins="champsim_nn".split()

benchmarks= commands.getoutput("ls -1 "+trace_dir+" | grep simtrace").split()

index=0
for bi in bins:
        for ben in benchmarks:
                DesignSpace+=[(bi,ben)]
                index+=1

random.shuffle(DesignSpace)
#DesignSpace=DesignSpace[0:5]

def par_func(value):
        command = "time bin/%s --warmup_instructions 2000000 --simulation_instructions 5000000 %s" %(value[0], trace_dir+value[1])
        
        name = "%s,%s"%(value[0],value[1].replace(".champsimtrace.xz",""))
        suffix = "&> results/"+name+".txt"
        #compute = commands.getstatusoutput(command)
        #resultsBuffer.append((value[0],compute))
        os.system(command+suffix)

        i = value[2]
        left = (time.time()-start_time)/(i+1.0)*index/60.0*((index-i-1.0)/index)
        bars =("="*(30*(i+1)/index))+(">"*(30*(index-i-1)/index))
        sys.stderr.write("\r%d/%d Finished! (%s) Estimated time left: %.2f minutes [%s]" %(i+1,index, name, left,bars)+(" ")*5)

from threading import Thread
from Queue import Queue
num_worker_threads = 5

def worker():
    while True:
        item = q.get()
        par_func(item)#str(item))
        q.task_done()

q = Queue()
for i in range(num_worker_threads):
     t = Thread(target=worker)
     t.daemon = True
     t.start()

ind=0
for var in DesignSpace:
        q.put((var[0],var[1],ind))
        ind+=1

q.join()

resultsBuffer.sort(key=operator.itemgetter(0))
for r in resultsBuffer:
        print r.replace("(","").replace(")","").replace(",","").replace("'","")
