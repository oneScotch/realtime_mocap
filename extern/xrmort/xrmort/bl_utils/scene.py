#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import bpy


class SceneWrapper:
    def __init__(self, scene_name: str = "Scene"):
        self.scene_name = scene_name

    @property
    def scene(self):
        return bpy.data.scenes[self.scene_name]

    @classmethod
    def clear_all(cls):
        # Will collect meshes from delete objects
        meshes = set()

        collection = bpy.context.collection
        # Get objects in the collection if they are meshes
        for obj in (o for o in collection.objects if o.type == 'MESH'):
            # Store the internal mesh
            meshes.add(obj.data)
            # Delete the object
            bpy.data.objects.remove(obj)

        # Look at meshes that are orphaned after objects removal
        # for mesh in [m for m in meshes if m.users == 0]:
        for mesh in meshes:
            # Delete the meshes
            bpy.data.meshes.remove(mesh)

        scene = bpy.context.scene
        for obj in scene.objects:
            bpy.data.objects.remove(obj)

    @property
    def frame_current(self) -> int:
        return self.scene.frame_current

    def set_frame_current(self, frame: int):
        self.scene.frame_set(frame)

    @property
    def frame_start(self) -> int:
        return self.scene.frame_start

    @frame_start.setter
    def frame_start(self, frame: int):
        self.scene.frame_start = frame

    @property
    def frame_end(self) -> int:
        return self.scene.frame_end

    @frame_end.setter
    def frame_end(self, frame: int):
        self.scene.frame_end = frame

    def get_fps(self) -> float:
        fps = self.scene.render.fps / self.scene.render.fps_base
        return fps

    def set_fps(self, fps: int = 30):
        self.scene.render.fps = fps
