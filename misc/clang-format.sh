#!/bin/zsh
clang-format -i ../apps/**/*.cc \
	../apps/**/*.h \
	../kaminpar/**/*.cc \
	../kaminpar/**/*.h \
	../tests/**/*.cc \
	../tests/**/*.h \
	../dkaminpar/**/*.cc \
	../dkaminpar/**/*.h \
	../dtests/**/*.cc \
	../dtests/**/*.h \
	../library/**/*.cc \
	../library/**/*.h

