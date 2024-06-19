from collections import deque
from typing import Deque, Tuple

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> Tuple[float, ...]:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))

    def compare_priority(self, now: float, seq1: SequenceGroup,
                         seq2: SequenceGroup) -> bool:
        return self.get_priority(now, seq1) < self.get_priority(now, seq2)

    def forces_preemption(self) -> bool:
        raise NotImplementedError

    def sort_waiting(self) -> bool:
        raise NotImplementedError


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> Tuple[float, ...]:
        return (now - seq_group.metrics.arrival_time)

    def forces_preemption(self) -> bool:
        return False

    def sort_waiting(self) -> bool:
        return False


class SP(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> Tuple[float, ...]:
        return (-seq_group.sched_metadata["priority"],
                now - seq_group.metrics.arrival_time)

    def forces_preemption(self) -> bool:
        return True

    def sort_waiting(self) -> bool:
        return True


class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS, 'sp': SP}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
