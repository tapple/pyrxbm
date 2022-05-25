from enum import Enum


# fmt: off
class PropertyType(Enum):
    Unknown            =  0
    String             =  1
    Bool               =  2
    Int                =  3
    Float              =  4
    Double             =  5
    UDim               =  6
    UDim2              =  7
    Ray                =  8
    Faces              =  9
    Axes               = 10
    BrickColor         = 11
    Color3             = 12
    Vector2            = 13
    Vector3            = 14

    CFrame             = 16
    Quaternion         = 17
    Enum               = 18
    Ref                = 19
    Vector3int16       = 20
    NumberSequence     = 21
    ColorSequence      = 22
    NumberRange        = 23
    Rect               = 24
    PhysicalProperties = 25
    Color3uint8        = 26
    Int64              = 27
    SharedString       = 28
    ProtectedString    = 29
    OptionalCFrame     = 30
    UniqueId           = 31
    FontFace           = 32
# fmt: on
