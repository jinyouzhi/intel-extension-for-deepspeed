"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
from .builder import SYCLOpBuilder


class CPUAdagradBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        return []

    def include_paths(self):
        return []
