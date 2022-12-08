#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import os, sys
import os.path
import tensorflow as tf
from pathlib import Path

# pyxir
import pyxir
import pyxir.contrib.target.DPUCADF8H
import pyxir.contrib.target.DPUCAHX8H
import pyxir.contrib.target.DPUCAHX8L
import pyxir.contrib.target.DPUCVDX8H
import pyxir.contrib.target.DPUCZDX8G

# tvm, relay
import tvm
from tvm import te
from tvm import contrib
import tvm.relay as relay


# BYOC
from tvm.relay import transform
from tvm.contrib import utils, graph_executor as graph_runtime
from tvm.contrib.target import vitis_ai
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import annotation
# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai
import onnx
from tvm.contrib.download import download_testdata
import cv2
from typing import List

import time
from typing import List
import logging
from pathlib import Path


# In[15]:


FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
TVM_VAI_HOME   = os.getenv('TVM_VAI_HOME')


# In[16]:


onnx_model = onnx.load("enc_dec.onnx")


# In[17]:


shape_dict = {"inputa" : (1, 3, 1024, 768)}


# In[18]:


target = "llvm"
dpu_target = 'DPUCZDX8G-zcu104'


# In[19]:


mod, params = relay.frontend.from_onnx(onnx_model,{"inputa":(1,3,1024,768)})


# with tvm.transform.PassContext(opt_level=1):
#     executor = relay.build_module.create_executor(
#         "graph", mod, tvm.cpu(0), target, params
#     ).evaluate()

# In[20]:


input_name  = "inputa"
input_shape = (1,3,1024,768)
shape_dict  = {input_name:input_shape}
tvm_target  = 'llvm'
lib_kwargs  = {}


# In[21]:


print("Vitis Target:  DPUCZDX8G-zcu104")
postprocessing = []
vitis_target   = "DPUCZDX8G-zcu104"
tvm_target     = 'llvm'
lib_kwargs     = {}


# In[22]:


export_rt_mod_file = os.path.join(os.getcwd(), 'vitis_ai.rtmod')

build_options = {
    'dpu': vitis_target,
    'export_runtime_module': export_rt_mod_file
}

with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):   
    lib = relay.build(mod, tvm_target, params=params)


# In[23]:


# Export runtime module
temp = utils.tempdir()
lib.export_library(temp.relpath("tvm_lib.so"))

# Build and export lib for aarch64 target
tvm_target = tvm.target.arm_cpu('ultra96')
lib_kwargs = {
    'fcompile': contrib.cc.create_shared,
    'cc': "/usr/aarch64-linux-gnu/bin/ld"
}
build_options = {
    'load_runtime_module': export_rt_mod_file
}

with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}): 
    lib_dpuczdx8g = relay.build(mod, tvm_target, params=params)

lib_dpuczdx8g.export_library('tvm_dpu_cpu.so', **lib_kwargs)

    
print("Finished storing the compiled model as tvm_dpu_cpu.so")


# In[26]:


#utils.tempdir().relpath("tvm_lib.so")


# In[28]:


#lib_path = "deploy_lib.so"
#lib.export_library(lib_path)

