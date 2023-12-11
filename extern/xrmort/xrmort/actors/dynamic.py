#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Optional, Union

from .base_actor import BaseActor


class DynamicActor(BaseActor):
    actor_path: Optional[Path] = None           # FBX
    actor_meshed_path: Optional[Path] = None    # FBX
    skeleton_path: Optional[Path] = None        # json

    name_to_bone_mapping = {}
    bone_to_name_mapping = {
        v: k for k, v in name_to_bone_mapping.items() if k and v
    }

    def __init__(
        self,
        actor_path: Optional[Union[str, Path]] = None,
        actor_meshed_path: Optional[Union[str, Path]] = None,
        skeleton_path: Optional[Union[str, Path]] = None,
        name_to_bone_mapping: Optional[Dict[str, str]] = None
    ):
        if actor_path:
            actor_path = actor_path.absolute()
        if actor_meshed_path:
            actor_meshed_path = actor_meshed_path.absolute()
        if skeleton_path:
            skeleton_path = skeleton_path.absolute()

        if not skeleton_path:
            raise ValueError("No skeleton_path provided!")
        if not name_to_bone_mapping:
            raise ValueError("No name_to_bone_mapping provided!")

        super().__init__(
            actor_path,
            actor_meshed_path,
            skeleton_path,
            name_to_bone_mapping
        )

    @classmethod
    def from_json(cls, json_path: Union[Path, str]) -> 'DynamicActor':
        json_path = Path(json_path).absolute()
        if not json_path.exists():
            raise ValueError(f"File not found: {json_path}")
        with json_path.open("r") as f:
            actor_data = json.load(f)
        actor_dir = json_path.parent

        actor_path = actor_data['actor_path']
        actor_path = actor_dir / actor_path if actor_path else None
        actor_meshed_path = actor_data['actor_meshed_path']
        actor_meshed_path = \
            actor_dir / actor_meshed_path if actor_meshed_path else None
        skeleton_path = actor_data['skeleton_path']
        skeleton_path = actor_dir / skeleton_path if skeleton_path else None
        name_to_bone_mapping = actor_data['name_to_bone_mapping'] or {}

        bone_prefix = actor_data.get('bone_prefix') or ''
        if bone_prefix:
            name_to_bone_mapping = {
                k: f'{bone_prefix}{v}' if v else ''
                for k, v in name_to_bone_mapping.items()
            }

        return DynamicActor(
            actor_path,
            actor_meshed_path,
            skeleton_path,
            name_to_bone_mapping
        )
