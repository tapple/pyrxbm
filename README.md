# pyrxbm

Ported to Python from https://github.com/MaximumADHD/Roblox-File-Format

Very incomplete. Currently only supports the bare minimum to read/write KeyframeSequence assets. Contributions welcome

Progress
- XML support: not started/planned
- Chunk types: 4/6 (Missing SSTR and SIGN)
- PropertyTypes: 6/31
- DataTypes: 1/40 (just CFrame)
- Classes: 5/hundreds (Roblox-File-Format has a Roblox Studio Plugin to periodically sync this with the Lua library. Porting that has not started)
- Enums: 0/hundreds (also synced by the Classes plugin)
- Lua -> python code generator for Classes/Enums: not started/planned

The code style is also a mess, with a mix of C#/python conventions. The goal is to match Lua code for things that are part of the Roblox library (UpperCaseMethodNames), and PEP8 otherwise
(underscore_method_names)
