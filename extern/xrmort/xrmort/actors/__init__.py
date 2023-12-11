#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from enum import Enum
from pathlib import Path
from typing import Optional, Type, Union

from .base_actor import BaseActor
from .dynamic import DynamicActor
from .smpl import SMPL
from .ue import Mannequin
from .ybot import YBot

__all__ = ['ActorNameEnum', 'DynamicActor']


class ActorNameEnum(str, Enum):
    dynamic = "dynamic"
    smpl = "smpl"
    smplx = "smplx"
    ybot = "ybot"
    mannequin = "mannequin"

    @classmethod
    def get_actor_class(cls, actor_name: str) -> Type[BaseActor]:
        _mapping_ = dict(
            dynamic=DynamicActor,
            smpl=SMPL,
            smplx=SMPL,
            ybot=YBot,
            mannequin=Mannequin,
        )
        _mapping_ = {k.lower(): v for k, v in _mapping_.items()}
        actor_name = actor_name.lower()
        try:
            return _mapping_[actor_name]
        except KeyError as e:
            raise NotImplementedError(
                f"{actor_name} not in {list(_mapping_.keys())}"
            ) from e


def get_actor(
    actor_name: str, actor_conf: Optional[Union[str, Path]] = ''
) -> BaseActor:
    """actor name to SkeletonInfo instance.

    Args:
        actor_name (str): actor's name
        actor_conf (Optional[Union[str, Path]], optional):
            path to `actor.json` file. Required when actor_name == "dynamic".
            Ignored in other cases. Defaults to "".

    Returns:
        BaseActor:
    """
    actor_cls_ = ActorNameEnum.get_actor_class(actor_name)
    if actor_cls_ is not DynamicActor:
        actor = actor_cls_()
    else:
        if not actor_conf:
            raise ValueError(
                f'actor_conf if required when `actor_name={actor_name}`.'
            )
        actor_conf = Path(actor_conf)
        if actor_conf.is_dir() and (actor_conf / 'actor.json').exists():
            actor_conf = actor_conf / 'actor.json'
        if not actor_conf.exists():
            raise ValueError(f'Config not found: {actor_conf}')
        actor = DynamicActor.from_json(actor_conf)
    return actor
