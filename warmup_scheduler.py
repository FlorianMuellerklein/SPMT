import numpy as np

class AddWarmup():
    def __init__(self, scheduler, starting_lr, ending_lr, warmup_dur, verbose=False):
        self.scheduler = scheduler
        self.warmup_dur = warmup_dur
        self.starting_lr = starting_lr
        self.ending_lr = ending_lr
        self.steps = 0
        self.verbose = verbose

        self.lr_schedule = np.linspace(starting_lr, ending_lr, warmup_dur+1)

        if warmup_dur > 0:
            self._set_lr()

    def step(self):
        if self.steps < self.warmup_dur:
            self.steps += 1
            self._set_lr()
        else:
            self.scheduler.step()

    def _set_lr(self):
        if self.verbose:
            print()
            print('Setting LR: ', self.lr_schedule[self.steps])
        for g in self.scheduler.optimizer.param_groups:
            g['lr'] = self.lr_schedule[self.steps]

