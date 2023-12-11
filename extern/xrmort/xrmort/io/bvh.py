#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import ply.lex as lex


class BVHLexer:
    states = (
        ('hierarchy', 'exclusive'),
        ('joint', 'exclusive'),

        ('addjoint', 'exclusive'),
        ('addchannel', 'exclusive'),
        ('addoffset', 'exclusive'),

        ('motion', 'exclusive'),
        ('setframes', 'exclusive'),
        ('setframetime', 'exclusive'),
        ('addframedata', 'exclusive'),
    )

    tokens = (
        # == key words ==
        # "HIERARCHY",
        "ROOT",
        "OFFSET",
        "CHANNELS",
        "JOINT",
        "ENDSITE",
        "MOTION",
        "FRAMES",
        "FRAMETIME",

        # == syntax ==
        # "LCURL",
        # "RCURL",
        "EOL",
        # "BLANK",
        "BLANKLINE"

        # variables
        "CNAME",
        "NUMBER",
        "WORD",
    )

    # >>> begin `hierarchy` state <<<
    def t_begin_hierarchy(self, t):
        "HIERARCHY"
        t.lexer.begin('hierarchy')
        self.lexer.level = 0

    # >>> enter `addjoint` state
    def t_hierarchy_ROOT(self, t):
        "ROOT"
        self.num_joints += 1
        self.lexer.push_state('addjoint')

    # >>> enter `addjoint` state
    def t_joint_JOINT(self, t):
        "JOINT"
        self.num_joints += 1
        self.lexer.push_state('addjoint')

    def t_addjoint_WORD(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        self.joint_names.append(t.value)
        if self._names_queue:
            parent_name = self._names_queue[-1]
        else:
            parent_name = ""
        self.parent_names.append(parent_name)
        self.lexer.pop_state()

    # >>> enter `joint` state
    def t_hierarchy_joint_begin_joint(self, t):
        r'\{'
        self._names_queue.append(self.joint_names[-1])
        t.lexer.level += 1
        t.lexer.push_state('joint')

    # >>> enter `addoffset` state
    def t_joint_OFFSET(self, t):
        "OFFSET"
        self._offsets_remained = 3
        self.lexer.push_state('addoffset')

    def t_addoffset_NUMBER(self, t):
        r'(([-+]?\d*\.?\d+)|([-+]?\d+\.?\d*))([eE][+-]?\d+)?'
        self.offsets.append(float(t.value))
        self._offsets_remained -= 1
        if self._offsets_remained <= 0:
            self.lexer.pop_state()

    # >>> enter `addchannel` state
    def t_joint_CHANNELS(self, t):
        "CHANNELS"
        self.lexer.push_state('addchannel')

    def t_addchannel_NUMBER(self, t):
        r'\d+'
        self._current_n_channel = int(t.value)
        if self._current_n_channel <= 0:
            self.lexer.pop_state()
        self.num_channels += self._current_n_channel

    def t_addchannel_CNAME(self, t):
        r'[XYZ]((position)|(rotation))'
        cname = f"{self.joint_names[-1]}.{t.value}"
        self.channel_names.append(cname)

        self._current_n_channel -= 1
        if self._current_n_channel <= 0:
            self.lexer.pop_state()

    def t_joint_ENDSITE(self, t):
        r"End\s+Site"
        self.num_joints += 1
        self.joint_names.append(f"{self.joint_names[-1]}_end")

    # leave `joint` state <<<
    def t_hierarchy_joint_end_joint(self, t):
        r'\}'
        t.lexer.level -= 1
        self._names_queue.pop()
        t.lexer.pop_state()    # Back to the previous state
        if t.lexer.level == 0:
            t.lexer.begin("INITIAL")

    # >>> begin `motion` state <<<
    def t_begin_motion(self, t):
        "MOTION"
        t.lexer.begin('motion')

    # >>> enter `setframes` state
    def t_motion_FRAMES(self, t):
        r"Frames\s*:"
        t.lexer.push_state("setframes")

    def t_setframes_NUMBER(self, t):
        r'(([-+]?\d*\.?\d+)|([-+]?\d+\.?\d*))([eE][+-]?\d+)?'
        self.num_frames = float(t.value)
        t.lexer.pop_state()

    def t_motion_FRAMETIME(self, t):
        r"Frame\s+Time\s*:"
        t.lexer.push_state("setframetime")

    # >>> enter `setframetime` state
    def t_setframetime_NUMBER(self, t):
        r'(([-+]?\d*\.?\d+)|([-+]?\d+\.?\d*))([eE][+-]?\d+)?'
        self.frame_time = float(t.value)
        t.lexer.pop_state()

    # >>> enter `setframedata` state
    def t_motion_EOL(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
        if self.num_frames is not None and self.frame_time is not None:
            self._current_frame = 0
            self._current_frame_data = []
            t.lexer.push_state("addframedata")

    def t_addframedata_EOL(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
        n_channels_found = len(self._current_frame_data)
        if n_channels_found:
            if n_channels_found < self.num_channels:
                msg = (
                    f"Not enough channel data of frame:{self._current_frame}."
                    f" Requires {len(self._current_frame_data)} channels,"
                    f" but found {self.num_channels} at ln:{t.lexer.lineno}."
                )
                raise ValueError(msg)
            elif n_channels_found > self.num_channels:
                msg = (
                    f"Too many channel data of frame:{self._current_frame}."
                    f" Requires {len(self._current_frame_data)} channels,"
                    f" but found {self.num_channels} at ln:{t.lexer.lineno}."
                )
            else:
                # next frame
                self.frames_data.append(self._current_frame_data)
                self._current_frame += 1
                self._current_frame_data = []
        else:
            # skip redundant EOL
            pass

    def t_addframedata_NUMBER(self, t):
        r'(([-+]?\d*\.?\d+)|([-+]?\d+\.?\d*))([eE][+-]?\d+)?'
        self._current_frame_data.append(float(t.value))

    def t_ANY_ignore_BLANK(self, t):
        r"[ \t]+"

    # Define a rule so we can track line numbers
    def t_ANY_ignore_EOL(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    # Error handling rules
    def t_ANY_error(self, t):
        msg = f"Illegal character '{t.value[0]}' [ln:{t.lexer.lineno}]"
        raise ValueError(msg)
        # t.lexer.skip(1)

    def __init__(self, **kwargs):
        self.__reset()
        self.lexer = self._build(**kwargs)

    def __reset(self):
        self.num_joints = 0
        self.num_channels = 0
        self.num_frames = None
        self.frame_time = None

        self.joint_names = []
        self.parent_names = []
        self.offsets = []
        self.channel_names = []
        self.frames_data = []

        self._current_n_channel = 0
        self._offsets_remained = 0

        self._current_frame = -1
        self._current_frame_data = []
        self._names_queue = []

    # Build the lexer
    def _build(self, **kwargs) -> lex.Lexer:
        return lex.lex(object=self, **kwargs)

    def input(self, data: str, **kwargs):
        self.__reset()
        self.lexer.input(data, **kwargs)

    def __call__(self, data: str, **kwargs):
        self.input(data, **kwargs)
        while True:
            try:
                tok = self.lexer.token()
            except Exception as e:
                raise ValueError(
                    f"Error while lexing ln:{self.lexer.lineno}"
                ) from e
            if not tok:
                break

    # Test its output
    def test(self, data: str, debug: bool = True, **kwargs):
        self.input(data, debug=debug, **kwargs)
        while True:
            try:
                tok = self.lexer.token()
            except Exception as e:
                raise ValueError(
                    f"Error while lexing ln:{self.lexer.lineno}"
                ) from e
            if not tok:
                break
            print(tok)


class BVHImporter:
    def __init__(self) -> None:
        self._lexer = BVHLexer()

        # self.joint_names = []
        # self.joint_parent_idx = []
        # self.translations = []
        # self.rotations = []

    def parse(self, data: str):
        self._lexer(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the lexer")
    parser.add_argument("--file", type=str, help="The file to be lexed")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        content = f.read()
    lexer = BVHLexer(content)
