"""Celery 应用实例定义。

本模块只负责创建并配置 `layout_celery`：
- broker/result_backend
- 默认队列
- 任务超时与预取参数

任务注册与编排由 `tasks.py` 完成。
"""

from __future__ import annotations

import importlib

from celery import Celery

from .config import get_settings

settings = get_settings()

layout_celery = Celery(settings.service_name)
layout_celery.conf.update(
    broker_url=settings.celery.broker_url,
    result_backend=settings.celery.result_backend,
    task_default_queue=settings.celery.default_queue,
    task_routes={
        "layout.dataset.process.start": {"queue": "layout_train_dataset_process"},
        "layout.dataset.process": {"queue": "layout_train_dataset_process"},
        "layout.dataset.process.chunk": {"queue": "layout_train_dataset_process"},
        "layout.dataset.process.finalize": {"queue": "layout_train_dataset_process"},
    },
    task_time_limit=settings.celery.task_time_limit_sec,
    worker_prefetch_multiplier=settings.celery.prefetch_multiplier,
)

# 保障 worker 启动时可注册到 `layout_training_kit.tasks` 中的任务。
layout_celery.autodiscover_tasks(["layout_training_kit"])
importlib.import_module("layout_training_kit.tasks")
