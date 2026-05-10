"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # optimizer 초기화
        self.optimizer = model.configure_optimizers(config)

        # DataLoader 초기화
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        # 일반적으로 하는대로 epoch단위로 학습하는게 아니라 iter단위로 학습길이 정함.
        while True:

            # iter단위로 학습하기때문에 Dataloader 무한반복시켜야됨
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            # 데이터 텐서 device로 이동
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # 순전파, 손실함수 계산. 
            # minGPT구현에서는 forward가 prediction과 loss를 모두 출력함.
            logits, self.loss = model(x, y)

            # 그라디언트 초기화
            model.zero_grad(set_to_none=True)
            # 역전파
            self.loss.backward()
            # 그라디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            # 가중치 업데이트
            self.optimizer.step()

            # on_batch_end 이벤트에 반응하는 콜백함수 실행
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # 설정한 iter 초과시 종료
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
