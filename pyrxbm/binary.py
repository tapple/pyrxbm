from __future__ import annotations

from typing import Callable, Any, BinaryIO
from enum import Enum

import lz4.block
import struct
from io import BytesIO
from dataclasses import dataclass, fields
import numpy as np
import numpy.typing as npt

from . import classes
from .datatypes import (
    orient_id_to_rotation_matrix,
    CFrame,
    rotation_matrices_to_orient_ids,
)
from .tree import Instance

"""
using RobloxFiles.Enums;
using RobloxFiles.DataTypes;
using RobloxFiles.Utility;
"""


# https://blog.roblox.com/2013/05/condense-and-compress-our-custom-binary-file-format/
# these functions all return an array with native endian, regardless of the endiannes of the input
def decode_int(i: np.ndarray) -> np.ndarray:
    return (i >> 1) ^ (-(i & 1))


def decode_float(i: np.ndarray) -> np.ndarray:
    return (i >> 1) | (i << 31)


def encode_int(i: np.ndarray) -> np.ndarray:
    return (i << 1) ^ (i >> 31)


def encode_int64(i: np.ndarray) -> np.ndarray:
    return (i << 1) ^ (i >> 63)


# http://stackoverflow.com/questions/442188/readint-readbyte-readstring-etc-in-python
class BinaryStream:
    UINT32 = np.dtype(">u4")
    INT32 = np.dtype(">i4")
    INT64 = np.dtype(">i8")
    F32 = np.dtype(">f4")
    F32LE = np.dtype("<f4")

    def __init__(self, base_stream):
        self.base_stream = base_stream

    def read_bytes(self, length) -> bytes:
        return self.base_stream.read(length)

    def write_bytes(self, value):
        self.base_stream.write(value)

    def unpack(self, fmt):
        return struct.unpack(fmt, self.read_bytes(struct.calcsize(fmt)))

    def pack(self, fmt, *data):
        return self.write_bytes(struct.pack(fmt, *data))

    def read_string_undecoded(self) -> bytes:
        (length,) = self.unpack("<I")
        return self.read_bytes(length)

    def write_string_unencoded(self, s: bytes) -> None:
        self.pack("<I", len(s))
        self.write_bytes(s)

    def read_string(self, encoding="utf8") -> str:
        return self.read_string_undecoded().decode(encoding)

    def write_string(self, s: str, encoding="utf8") -> None:
        self.write_string_unencoded(s.encode(encoding))

    # https://blog.roblox.com/2013/05/condense-and-compress-our-custom-binary-file-format/
    def read_interleaved(self, rows: int, dtype: np.dtype, cols=1) -> np.ndarray:
        a = np.frombuffer(self.read_bytes(rows * dtype.itemsize * cols), np.uint8)
        b = a.reshape(cols, dtype.itemsize, rows).transpose(2, 0, 1)
        c = np.ascontiguousarray(b).view(dtype)
        d = c.reshape((rows,) if cols == 1 else (rows, cols))
        return d

    def write_interleaved(self, values, dtype: np.dtype = None):
        # accepts a dtype because values almost always need to be byteswapped
        a = np.asarray(values, dtype)
        self.write_bytes(a.reshape(a.shape[0], -1).view(np.uint8).T.tobytes())

    def read_uints(self, rows, cols=1):
        # no negative encoding
        return self.read_interleaved(rows, self.UINT32, cols)

    def write_uints(self, values):
        # no negative encoding
        self.write_interleaved(values, self.UINT32)

    def read_ints(self, rows, cols=1):
        return decode_int(self.read_interleaved(rows, self.INT32, cols))

    def write_ints(self, values):
        self.write_interleaved(encode_int(np.asarray(values, np.int32)), self.INT32)

    def read_longs(self, rows, cols=1):
        return decode_int(self.read_interleaved(rows, self.INT64, cols))

    def write_longs(self, values):
        self.write_interleaved(encode_int64(np.asarray(values, np.int64)), self.INT64)

    def read_floats(self, rows, cols=1):
        return decode_float(self.read_uints(rows, cols)).view(np.float32)

    def write_floats(self, values):
        self.write_uints(encode_int(np.asarray(values, np.float32).view(np.uint32)))

    def read_instance_ids(self, count):
        """Reads and accumulates an interleaved buffer of integers."""
        return self.read_ints(count).cumsum()

    def write_instance_ids(self, values):
        """Accumulatively writes an interleaved array of integers."""
        self.write_ints(np.ediff1d(np.asarray(values), to_begin=values[0]))

    # http://stackoverflow.com/questions/32774910/clean-way-to-read-a-null-terminated-c-style-string-from-a-file
    def readCString(self):
        buf = bytearray()
        while True:
            b = self.base_stream.read(1)
            if b is None or b == b"\0":
                return buf
            else:
                buf.extend(b)

    def writeCString(self, string):
        self.write_bytes(string)
        self.write_bytes(b"\0")


class META:
    def __init__(self):
        self.Data = {}

    def deserialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        (numEntries,) = stream.unpack("<i")
        for i in range(numEntries):
            key = stream.read_string()
            value = stream.read_string()
            self.Data[key] = value
        file.META = self

    def serialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        stream.pack("<i", len(self.Data))
        for key, value in self.Data.items():
            stream.write_string(key)
            stream.write_string(value)

    def dump(self):
        print(f"- NumEntries: {len(self.Data)}")
        for key, value in self.Data.items():
            print(f"  - {key}: {value}")


class INST:
    def __init__(self):
        self.ClassIndex = 0
        self.ClassName = ""
        self.is_service = False
        self.RootedServices = []
        self.NumInstances = 0
        self.InstanceIds = []

    @property
    def Class(self):
        try:
            return getattr(classes, self.ClassName)
        except AttributeError:
            raise ValueError(
                f"INST - Unknown class: {self.ClassName} while reading INST chunk."
            )

    def __str__(self):
        return f"{self.ClassIndex}: {self.ClassName}x{self.NumInstances}"

    def deserialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        (self.ClassIndex,) = stream.unpack("<i")
        self.ClassName = stream.read_string()
        self.is_service, self.NumInstances = stream.unpack("<?i")
        self.InstanceIds = stream.read_instance_ids(self.NumInstances)
        file.Classes[self.ClassIndex] = self
        instType = self.Class

        if self.is_service:
            self.RootedServices = stream.unpack(f"{self.NumInstances}?")

        for i in range(self.NumInstances):
            instId = self.InstanceIds[i]
            inst = instType()
            inst.referent = str(instId)
            inst.is_service = self.is_service
            if self.is_service:
                isRooted = self.RootedServices[i]
                inst.Parent = file if isRooted else None
            file.Instances[instId] = inst

    def serialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        stream.pack("<i", self.ClassIndex)
        stream.write_string(self.ClassName)
        stream.pack("<?i", self.is_service, self.NumInstances)
        stream.write_instance_ids(self.InstanceIds)
        if self.is_service:
            # stream.pack(f"{self.NumInstances}?", *([False] * self.NumInstances))
            stream.write_bytes(b"\0" * self.NumInstances)

    def dump(self):
        print(f"- ClassIndex:   {self.ClassIndex}")
        print(f"- ClassName:    {self.ClassName}")
        print(f"- is_service:    {self.is_service}")

        if self.is_service and self.RootedServices is not None:
            print(f"- RootedServices: `{', '.join(self.RootedServices)}`")

        print(f"- NumInstances: {self.NumInstances}")
        print(f"- InstanceIds: `{', '.join(self.InstanceIds)}`")


# fmt: off
class PropertyType(Enum):
    Unknown              =  0
    String               =  1
    Bool                 =  2
    Int                  =  3
    Float                =  4
    Double               =  5
    UDim                 =  6
    UDim2                =  7
    Ray                  =  8
    Faces                =  9
    Axes                 = 10
    BrickColor           = 11
    Color3               = 12
    Vector2              = 13
    Vector3              = 14

    CFrame               = 16
    Quaternion           = 17
    Enum                 = 18
    Ref                  = 19
    Vector3int16         = 20
    NumberSequence       = 21
    ColorSequence        = 22
    NumberRange          = 23
    Rect                 = 24
    PhysicalProperties   = 25
    Color3uint8          = 26
    Int64                = 27
    SharedString         = 28
    ProtectedString      = 29
    OptionalCFrame       = 30
    UniqueId             = 31
    FontFace             = 32
    SecurityCapabilities = 33

PROPERTY_TYPE_MAP = {
    "str": PropertyType.String,  "bytes": PropertyType.String,
    "bool"                : PropertyType.Bool                ,
    "int"                 : PropertyType.Int                 ,
    "float"               : PropertyType.Float               ,
    "Double"              : PropertyType.Double              ,
    "UDim"                : PropertyType.UDim                ,
    "UDim2"               : PropertyType.UDim2               ,
    "Ray"                 : PropertyType.Ray                 ,
    "Faces"               : PropertyType.Faces               ,
    "Axes"                : PropertyType.Axes                ,
    "BrickColor"          : PropertyType.BrickColor          ,
    "Color3"              : PropertyType.Color3              ,
    "Vector2"             : PropertyType.Vector2             ,
    "Vector3"             : PropertyType.Vector3             ,

    "CFrame"              : PropertyType.CFrame              ,
    "Quaternion"          : PropertyType.Quaternion          ,
    "Enum"                : PropertyType.Enum                ,
    "Ref"                 : PropertyType.Ref                 ,
    "Vector3int16"        : PropertyType.Vector3int16        ,
    "NumberSequence"      : PropertyType.NumberSequence      ,
    "ColorSequence"       : PropertyType.ColorSequence       ,
    "NumberRange"         : PropertyType.NumberRange         ,
    "Rect"                : PropertyType.Rect                ,
    "PhysicalProperties"  : PropertyType.PhysicalProperties  ,
    "Color3uint8"         : PropertyType.Color3uint8         ,
    "Int64"               : PropertyType.Int64               ,
    "SharedString"        : PropertyType.SharedString        ,
    "ProtectedString"     : PropertyType.ProtectedString     ,
    "OptionalCFrame"      : PropertyType.OptionalCFrame      ,
    "UniqueId"            : PropertyType.UniqueId            ,
    "FontFace"            : PropertyType.FontFace            ,
    "SecurityCapabilities": PropertyType.SecurityCapabilities,
}
# fmt: on


@dataclass
class PROP:
    File: BinaryRobloxFile = None
    Name: str = ""
    ClassIndex: int = -1
    Type: PropertyType = PropertyType.Unknown
    python_type: str = ""

    @property
    def Class(self) -> INST:
        return self.File.Classes[self.ClassIndex]

    @property
    def ClassName(self):
        return self.Class.ClassName if self.Class else "UnknownClass"

    def __str__(self):
        return f"{self.Type} {self.ClassName}.{self.Name}"

    def deserialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        self.File = file
        (self.ClassIndex,) = stream.unpack("<i")
        self.Name = stream.read_string()

        (propType,) = stream.unpack("<b")
        self.Type = PropertyType(propType)

        assert (
            self.Class is not None
        ), f"Unknown class index {self.ClassIndex} (@ {self})!"
        ids = self.Class.InstanceIds
        instCount = self.Class.NumInstances
        props = [None] * instCount
        instances = [self.File.Instances[id] for id in ids]

        """
        for i in range(instCount):
            id = ids[i]
            instance = self.File.Instances[id]

            if instance is None:
                RobloxFile.LogError(
                    f"PROP: No instance @{id} for property {self.ClassName}.{self.Name}"
                )
                continue

            prop = Property(instance, self)
            props[i] = prop

            instance.AddProperty(prop)
        """

        # Setup some short-hand functions for actions used during the read procedure.
        def read_ints():
            return stream.read_ints(instCount)

        def read_floats():
            return stream.read_floats(instCount)

        def read_properties(read: Callable[[int], Any]):
            for i, instance in enumerate(instances):
                if instance is not None:
                    setattr(instance, self.Name, read(i))

        if self.Type == PropertyType.String:
            if self.Name in ("Tags", "AttributesSerialize"):
                read_properties(lambda i: stream.read_string_undecoded())
            else:
                read_properties(lambda i: stream.read_string())
        elif self.Type == PropertyType.Bool:
            read_properties(lambda i: stream.unpack("<?")[0])
            """
        elif self.Type == PropertyType.Int:
            {
                int[] ints = read_ints();
                read_properties(i => ints[i]);
                break;
            }
            """
        elif self.Type == PropertyType.Float:
            floats = read_floats()
            read_properties(lambda i: floats[i])
            """
        elif self.Type == PropertyType.Double:
            {
                read_properties(i => reader.ReadDouble());
                break;
            }
        elif self.Type == PropertyType.UDim:
            {
                float[] UDim_Scales = read_floats();
                int[] UDim_Offsets = read_ints();

                read_properties(i =>
                {
                    float scale = UDim_Scales[i];
                    int offset = UDim_Offsets[i];
                    return new UDim(scale, offset);
                });

                break;
            }
        elif self.Type == PropertyType.UDim2:
            {
                float[] UDim2_Scales_X = read_floats(),
                        UDim2_Scales_Y = read_floats();

                int[] UDim2_Offsets_X = read_ints(),
                      UDim2_Offsets_Y = read_ints();

                read_properties(i =>
                {
                    float scaleX = UDim2_Scales_X[i],
                          scaleY = UDim2_Scales_Y[i];

                    int offsetX = UDim2_Offsets_X[i],
                        offsetY = UDim2_Offsets_Y[i];

                    return new UDim2(scaleX, offsetX, scaleY, offsetY);
                });

                break;
            }
        elif self.Type == PropertyType.Ray:
            {
                read_properties(i =>
                {
                    float posX = reader.ReadFloat(),
                          posY = reader.ReadFloat(),
                          posZ = reader.ReadFloat();

                    float dirX = reader.ReadFloat(),
                          dirY = reader.ReadFloat(),
                          dirZ = reader.ReadFloat();

                    var origin = new Vector3(posX, posY, posZ);
                    var direction = new Vector3(dirX, dirY, dirZ);

                    return new Ray(origin, direction);
                });

                break;
            }
        elif self.Type == PropertyType.Faces:
            {
                read_properties(i =>
                {
                    byte faces = reader.ReadByte();
                    return (Faces)faces;
                });

                break;
            }
        elif self.Type == PropertyType.Axes:
            {
                read_properties(i =>
                {
                    byte axes = reader.ReadByte();
                    return (Axes)axes;
                });

                break;
            }
        elif self.Type == PropertyType.BrickColor:
            {
                int[] BrickColorIds = read_ints();

                read_properties(i =>
                {
                    BrickColor color = BrickColorIds[i];
                    return color;
                });

                break;
            }
        elif self.Type == PropertyType.Color3:
            {
                float[] Color3_R = read_floats(),
                        Color3_G = read_floats(),
                        Color3_B = read_floats();

                read_properties(i =>
                {
                    float r = Color3_R[i],
                          g = Color3_G[i],
                          b = Color3_B[i];

                    return new Color3(r, g, b);
                });

                break;
            }
        elif self.Type == PropertyType.Vector2:
            {
                float[] Vector2_X = read_floats(),
                        Vector2_Y = read_floats();

                read_properties(i =>
                {
                    float x = Vector2_X[i],
                          y = Vector2_Y[i];

                    return new Vector2(x, y);
                });

                break;
            }
        elif self.Type == PropertyType.Vector3:
            {
                float[] Vector3_X = read_floats(),
                        Vector3_Y = read_floats(),
                        Vector3_Z = read_floats();

                read_properties(i =>
                {
                    float x = Vector3_X[i],
                          y = Vector3_Y[i],
                          z = Vector3_Z[i];

                    return new Vector3(x, y, z);
                });

                break;
            }
            """
        elif self.Type == PropertyType.CFrame:
            rots = np.zeros((instCount, 9), dtype=np.float32)
            for i in range(instCount):
                raw_orient_id = stream.read_bytes(1)[0]
                if raw_orient_id > 0:
                    rots[i] = orient_id_to_rotation_matrix(raw_orient_id - 1)
                else:
                    rots[i] = np.frombuffer(stream.read_bytes(36), stream.F32LE)
            poss = stream.read_floats(instCount, 3)
            read_properties(lambda i: CFrame(*poss[i], *rots[i]))
            """
        elif self.Type in (
            PropertyType.CFrame,
            PropertyType.Quaternion,
            PropertyType.OptionalCFrame,
        ):
            {
                float[][] matrices = new float[instCount][];

                if (Type == PropertyType.OptionalCFrame)
                {
                    byte cframeType = (byte)PropertyType.CFrame;
                    byte readType = reader.ReadByte();

                    if (readType != cframeType)
                    {
                        RobloxFile.LogError($"Unexpected property type in OptionalCFrame (expected {cframeType}, got {readType})");
                        read_properties(i => null);
                        break;
                    }
                }

                for (int i = 0; i < instCount; i++)
                {
                    byte rawOrientId = reader.ReadByte();

                    if (rawOrientId > 0)
                    {
                        // Make sure this value is in a safe range.
                        int orientId = (rawOrientId - 1) % 36;

                        NormalId xColumn = (NormalId)(orientId / 6);
                        Vector3 R0 = Vector3.FromNormalId(xColumn);

                        NormalId yColumn = (NormalId)(orientId % 6);
                        Vector3 R1 = Vector3.FromNormalId(yColumn);

                        // Compute R2 using the cross product of R0 and R1.
                        Vector3 R2 = R0.Cross(R1);

                        // Generate the rotation matrix.
                        matrices[i] = new float[9]
                        {
                            R0.X, R0.Y, R0.Z,
                            R1.X, R1.Y, R1.Z,
                            R2.X, R2.Y, R2.Z,
                        };
                    }
                    else if (Type == PropertyType.Quaternion)
                    {
                        float qx = reader.ReadFloat(),
                              qy = reader.ReadFloat(),
                              qz = reader.ReadFloat(),
                              qw = reader.ReadFloat();

                        var quaternion = new Quaternion(qx, qy, qz, qw);
                        var rotation = quaternion.ToCFrame();

                        matrices[i] = rotation.GetComponents();
                    }
                    else
                    {
                        float[] matrix = new float[9];

                        for (int m = 0; m < 9; m++)
                        {
                            float value = reader.ReadFloat();
                            matrix[m] = value;
                        }

                        matrices[i] = matrix;
                    }
                }

                float[] CFrame_X = read_floats(),
                        CFrame_Y = read_floats(),
                        CFrame_Z = read_floats();

                var CFrames = new CFrame[instCount];

                for (int i = 0; i < instCount; i++)
                {
                    float[] matrix = matrices[i];

                    float x = CFrame_X[i],
                          y = CFrame_Y[i],
                          z = CFrame_Z[i];

                    float[] components;

                    if (matrix.Length == 12)
                    {
                        matrix[0] = x;
                        matrix[1] = y;
                        matrix[2] = z;

                        components = matrix;
                    }
                    else
                    {
                        float[] position = new float[3] { x, y, z };
                        components = position.Concat(matrix).ToArray();
                    }

                    CFrames[i] = new CFrame(components);
                }

                if (Type == PropertyType.OptionalCFrame)
                {
                    byte boolType = (byte)PropertyType.Bool;
                    byte readType = reader.ReadByte();

                    if (readType != boolType)
                    {
                        RobloxFile.LogError($"Unexpected property type in OptionalCFrame (expected {boolType}, got {readType})");
                        read_properties(i => null);
                        break;
                    }

                    for (int i = 0; i < instCount; i++)
                    {
                        CFrame cf = CFrames[i];
                        bool archivable = reader.ReadBoolean();

                        if (!archivable)
                            cf = null;

                        CFrames[i] = new Optional<CFrame>(cf);
                    }
                }

                read_properties(i => CFrames[i]);
                break;
            }
            """
        elif self.Type == PropertyType.Enum:
            enums = stream.read_uints(instCount)
            read_properties(lambda i: enums[i])
            """ TODO: Typecast by instance field type
            {
                uint[] enums = reader.ReadUInts(instCount);

                read_properties(i =>
                {
                    Property prop = props[i];
                    Instance instance = prop.Instance;

                    Type instType = instance.GetType();
                    uint value = enums[i];

                    try
                    {
                        var info = ImplicitMember.Get(instType, Name);

                        if (info == null)
                        {
                            RobloxFile.LogError($"Enum cast failed for {ClassName}.{Name} using value {value}!");
                            return value;
                        }

                        return Enum.Parse(info.MemberType, value.ToInvariantString());
                    }
                    catch
                    {
                        RobloxFile.LogError($"Enum cast failed for {ClassName}.{Name} using value {value}!");
                        return value;
                    }
                });

                break;
            }
            """
            """
        elif self.Type == PropertyType.Ref:
            {
                var instIds = reader.ReadInstanceIds(instCount);

                read_properties(i =>
                {
                    int instId = instIds[i];

                    if (instId >= File.NumInstances)
                    {
                        RobloxFile.LogError($"Got out of bounds referent index in {ClassName}.{Name}!");
                        return null;
                    }

                    return instId >= 0 ? File.Instances[instId] : null;
                });

                break;
            }
        elif self.Type == PropertyType.Vector3int16:
            {
                read_properties(i =>
                {
                    short x = reader.ReadInt16(),
                          y = reader.ReadInt16(),
                          z = reader.ReadInt16();

                    return new Vector3int16(x, y, z);
                });

                break;
            }
        elif self.Type == PropertyType.NumberSequence:
            {
                read_properties(i =>
                {
                    int numKeys = reader.ReadInt32();
                    var keypoints = new NumberSequenceKeypoint[numKeys];

                    for (int key = 0; key < numKeys; key++)
                    {
                        float Time = reader.ReadFloat(),
                              Value = reader.ReadFloat(),
                              Envelope = reader.ReadFloat();

                        keypoints[key] = new NumberSequenceKeypoint(Time, Value, Envelope);
                    }

                    return new NumberSequence(keypoints);
                });

                break;
            }
        elif self.Type == PropertyType.ColorSequence:
            {
                read_properties(i =>
                {
                    int numKeys = reader.ReadInt32();
                    var keypoints = new ColorSequenceKeypoint[numKeys];

                    for (int key = 0; key < numKeys; key++)
                    {
                        float Time = reader.ReadFloat(),
                                 R = reader.ReadFloat(),
                                 G = reader.ReadFloat(),
                                 B = reader.ReadFloat();

                        Color3 Value = new Color3(R, G, B);
                        int Envelope = reader.ReadInt32();

                        keypoints[key] = new ColorSequenceKeypoint(Time, Value, Envelope);
                    }

                    return new ColorSequence(keypoints);
                });

                break;
            }
        elif self.Type == PropertyType.NumberRange:
            {
                read_properties(i =>
                {
                    float min = reader.ReadFloat();
                    float max = reader.ReadFloat();

                    return new NumberRange(min, max);
                });

                break;
            }
        elif self.Type == PropertyType.Rect:
            {
                float[] Rect_X0 = read_floats(), Rect_Y0 = read_floats(),
                        Rect_X1 = read_floats(), Rect_Y1 = read_floats();

                read_properties(i =>
                {
                    float x0 = Rect_X0[i], y0 = Rect_Y0[i],
                          x1 = Rect_X1[i], y1 = Rect_Y1[i];

                    return new Rect(x0, y0, x1, y1);
                });

                break;
            }
        elif self.Type == PropertyType.PhysicalProperties:
            {
                read_properties(i =>
                {
                    bool custom = reader.ReadBoolean();

                    if (custom)
                    {
                        float Density = reader.ReadFloat(),
                              Friction = reader.ReadFloat(),
                              Elasticity = reader.ReadFloat(),
                              FrictionWeight = reader.ReadFloat(),
                              ElasticityWeight = reader.ReadFloat();

                        return new PhysicalProperties
                        (
                            Density,
                            Friction,
                            Elasticity,
                            FrictionWeight,
                            ElasticityWeight
                        );
                    }

                    return null;
                });

                break;
            }
        elif self.Type == PropertyType.Color3uint8:
            {
                byte[] Color3uint8_R = reader.ReadBytes(instCount),
                       Color3uint8_G = reader.ReadBytes(instCount),
                       Color3uint8_B = reader.ReadBytes(instCount);

                read_properties(i =>
                {
                    byte r = Color3uint8_R[i],
                         g = Color3uint8_G[i],
                         b = Color3uint8_B[i];

                    Color3uint8 result = Color3.FromRGB(r, g, b);
                    return result;
                });

                break;
            }
            """
        elif self.Type == PropertyType.Int64:
            longs = stream.read_longs(instCount)
            read_properties(lambda i: longs[i])
            """
        elif self.Type == PropertyType.SharedString:
            {
                uint[] SharedKeys = reader.ReadUInts(instCount);

                read_properties(i =>
                {
                    uint key = SharedKeys[i];
                    return File.SharedStrings[key];
                });

                break;
            }
        elif self.Type == PropertyType.ProtectedString:
            {
                read_properties(i =>
                {
                    int length = reader.ReadInt32();
                    byte[] buffer = reader.ReadBytes(length);

                    return new ProtectedString(buffer);
                });

                break;
            }
        elif self.Type == PropertyType.UniqueId:
            {
                read_properties(i =>
                {
                    var index = reader.ReadUInt32();
                    var time = reader.ReadUInt32();
                    var random = reader.ReadUInt64();
                    return new UniqueId(index, time, random);
                });

                break;
            }
        elif self.Type == PropertyType.FontFace:
            {
                read_properties(i =>
                {
                    string family = reader.ReadString();

                    if (family.EndsWith(".otf") || family.EndsWith(".ttf"))
                        return new FontFace(family);

                    var weight = (FontWeight)reader.ReadUInt16();
                    var style = (FontStyle)reader.ReadByte();

                    return new FontFace(family, weight, style);
                });

                break;
            }
            """
        elif self.Type == PropertyType.SecurityCapabilities:
            """
            {
                var capabilities = reader.ReadInterleaved(instCount, BitConverter.ToUInt64);
                readProperties(i => capabilities[i]);
                break;
            }
            """
        else:
            raise NotImplementedError(
                f"Unhandled property type: {self.Type} in {self}!"
            )

    @classmethod
    def _collect_properties(self, inst: INST) -> list[PROP]:
        return [
            PROP(
                Name=field.name,
                ClassIndex=inst.ClassIndex,
                Type=PROPERTY_TYPE_MAP[field.type],
                python_type=field.type,
            )
            for field in fields(inst.Class)
        ]

    def serialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        stream.pack("<i", self.ClassIndex)
        stream.write_string(self.Name)
        stream.pack("<b", self.Type.value)

        self.File = file
        ids = self.Class.InstanceIds
        inst_count = len(ids)
        instances = [self.File.Instances[id] for id in ids]
        props = [getattr(instance, self.Name) for instance in instances]

        def write_properties(write: Callable[[Any], None]):
            for prop in props:
                write(prop)

        if self.Type == PropertyType.String:
            if self.python_type == "bytes":
                write_properties(lambda prop: stream.write_string_unencoded(prop))
            else:
                write_properties(lambda prop: stream.write_string(prop))
        elif self.Type == PropertyType.Bool:
            stream.pack(f"{inst_count}?", *props)
            """
        elif self.Type == PropertyType.Int:
                {
                    var ints = props
                        .Select(prop => prop.CastValue<int>())
                        .ToList();

                    writer.WriteInts(ints);
                    break;
                }
            """
        elif self.Type == PropertyType.Float:
            stream.write_floats(props)
            """
        elif self.Type == PropertyType.Double:
                {
                    props.ForEach(prop =>
                    {
                        double value = prop.CastValue<double>();
                        writer.Write(BinaryRobloxFileWriter.GetBytes(value));
                    });

                    break;
                }
        elif self.Type == PropertyType.UDim:
                {
                    var UDim_Scales = new List<float>();
                    var UDim_Offsets = new List<int>();

                    props.ForEach(prop =>
                    {
                        UDim value = prop.CastValue<UDim>();
                        UDim_Scales.Add(value.Scale);
                        UDim_Offsets.Add(value.Offset);
                    });

                    writer.WriteFloats(UDim_Scales);
                    writer.WriteInts(UDim_Offsets);

                    break;
                }
        elif self.Type == PropertyType.UDim2:
                {
                    var UDim2_Scales_X = new List<float>();
                    var UDim2_Scales_Y = new List<float>();

                    var UDim2_Offsets_X = new List<int>();
                    var UDim2_Offsets_Y = new List<int>();

                    props.ForEach(prop =>
                    {
                        UDim2 value = prop.CastValue<UDim2>();

                        UDim2_Scales_X.Add(value.X.Scale);
                        UDim2_Scales_Y.Add(value.Y.Scale);

                        UDim2_Offsets_X.Add(value.X.Offset);
                        UDim2_Offsets_Y.Add(value.Y.Offset);
                    });

                    writer.WriteFloats(UDim2_Scales_X);
                    writer.WriteFloats(UDim2_Scales_Y);

                    writer.WriteInts(UDim2_Offsets_X);
                    writer.WriteInts(UDim2_Offsets_Y);

                    break;
                }
        elif self.Type == PropertyType.Ray:
                {
                    props.ForEach(prop =>
                    {
                        Ray ray = prop.CastValue<Ray>();

                        Vector3 pos = ray.Origin;
                        writer.Write(pos.X);
                        writer.Write(pos.Y);
                        writer.Write(pos.Z);

                        Vector3 dir = ray.Direction;
                        writer.Write(dir.X);
                        writer.Write(dir.Y);
                        writer.Write(dir.Z);
                    });

                    break;
                }
        elif self.Type == PropertyType.Faces:
        elif self.Type == PropertyType.Axes:
                {
                    props.ForEach(prop =>
                    {
                        byte value = prop.CastValue<byte>();
                        writer.Write(value);
                    });

                    break;
                }
        elif self.Type == PropertyType.BrickColor:
                {
                    var brickColorIds = props
                        .Select(prop => prop.CastValue<BrickColor>())
                        .Select(value => value.Number)
                        .ToList();

                    writer.WriteInts(brickColorIds);
                    break;
                }
        elif self.Type == PropertyType.Color3:
                {
                    var Color3_R = new List<float>();
                    var Color3_G = new List<float>();
                    var Color3_B = new List<float>();

                    props.ForEach(prop =>
                    {
                        Color3 value = prop.CastValue<Color3>();
                        Color3_R.Add(value.R);
                        Color3_G.Add(value.G);
                        Color3_B.Add(value.B);
                    });

                    writer.WriteFloats(Color3_R);
                    writer.WriteFloats(Color3_G);
                    writer.WriteFloats(Color3_B);

                    break;
                }
        elif self.Type == PropertyType.Vector2:
                {
                    var Vector2_X = new List<float>();
                    var Vector2_Y = new List<float>();

                    props.ForEach(prop =>
                    {
                        Vector2 value = prop.CastValue<Vector2>();
                        Vector2_X.Add(value.X);
                        Vector2_Y.Add(value.Y);
                    });

                    writer.WriteFloats(Vector2_X);
                    writer.WriteFloats(Vector2_Y);

                    break;
                }
        elif self.Type == PropertyType.Vector3:
                {
                    var Vector3_X = new List<float>();
                    var Vector3_Y = new List<float>();
                    var Vector3_Z = new List<float>();

                    props.ForEach(prop =>
                    {
                        Vector3 value = prop.CastValue<Vector3>();
                        Vector3_X.Add(value.X);
                        Vector3_Y.Add(value.Y);
                        Vector3_Z.Add(value.Z);
                    });

                    writer.WriteFloats(Vector3_X);
                    writer.WriteFloats(Vector3_Y);
                    writer.WriteFloats(Vector3_Z);

                    break;
                }
            """
        elif self.Type == PropertyType.CFrame:
            components = np.row_stack([p.GetComponents() for p in props])
            poss = components[:, :3]
            rots = components[:, 3:]
            orient_ids = rotation_matrices_to_orient_ids(rots)
            for orient_id, rot in zip(orient_ids, rots):
                if orient_id is None:
                    stream.write_bytes(b"\0" + np.asarray(rot, stream.F32LE).tobytes())
                else:
                    stream.write_bytes(bytes([orient_id + 1]))
            stream.write_floats(poss)
            """
        elif self.Type == PropertyType.CFrame:
        elif self.Type == PropertyType.Quaternion:
        elif self.Type == PropertyType.OptionalCFrame:
                {
                    var CFrame_X = new List<float>();
                    var CFrame_Y = new List<float>();
                    var CFrame_Z = new List<float>();

                    if (Type == PropertyType.OptionalCFrame)
                        writer.Write((byte)PropertyType.CFrame);

                    props.ForEach(prop =>
                    {
                        CFrame value = null;

                        if (prop.Value is Quaternion q)
                            value = q.ToCFrame();
                        else
                            value = prop.CastValue<CFrame>();

                        if (value == null)
                            value = new CFrame();

                        Vector3 pos = value.Position;
                        CFrame_X.Add(pos.X);
                        CFrame_Y.Add(pos.Y);
                        CFrame_Z.Add(pos.Z);

                        int orientId = value.GetOrientId();
                        writer.Write((byte)(orientId + 1));

                        if (orientId == -1)
                        {
                            if (Type == PropertyType.Quaternion)
                            {
                                Quaternion quat = new Quaternion(value);
                                writer.Write(quat.X);
                                writer.Write(quat.Y);
                                writer.Write(quat.Z);
                                writer.Write(quat.W);
                            }
                            else
                            {
                                float[] components = value.GetComponents();

                                for (int i = 3; i < 12; i++)
                                {
                                    float component = components[i];
                                    writer.Write(component);
                                }
                            }
                        }
                    });

                    writer.WriteFloats(CFrame_X);
                    writer.WriteFloats(CFrame_Y);
                    writer.WriteFloats(CFrame_Z);

                    if (Type == PropertyType.OptionalCFrame)
                    {
                        writer.Write((byte)PropertyType.Bool);

                        props.ForEach(prop =>
                        {
                            if (prop.Value is null)
                            {
                                writer.Write(false);
                                return;
                            }

                            if (prop.Value is Optional<CFrame> optional)
                            {
                                writer.Write(optional.HasValue);
                                return;
                            }

                            var cf = prop.Value as CFrame;
                            writer.Write(cf != null);
                        });
                    }

                    break;
                }
            """
        elif self.Type == PropertyType.Enum:
            stream.write_uints(props)
            """
        elif self.Type == PropertyType.Ref:
                {
                    var InstanceIds = new List<int>();

                    props.ForEach(prop =>
                    {
                        int referent = -1;

                        if (prop.Value != null)
                        {
                            Instance value = prop.CastValue<Instance>();

                            if (value.IsDescendantOf(File))
                            {
                                string refValue = value.referent;
                                int.TryParse(refValue, out referent);
                            }
                        }

                        InstanceIds.Add(referent);
                    });

                    writer.WriteInstanceIds(InstanceIds);
                    break;
                }
        elif self.Type == PropertyType.Vector3int16:
                {
                    props.ForEach(prop =>
                    {
                        Vector3int16 value = prop.CastValue<Vector3int16>();
                        writer.Write(value.X);
                        writer.Write(value.Y);
                        writer.Write(value.Z);
                    });

                    break;
                }
        elif self.Type == PropertyType.NumberSequence:
                {
                    props.ForEach(prop =>
                    {
                        NumberSequence value = prop.CastValue<NumberSequence>();

                        var keyPoints = value.Keypoints;
                        writer.Write(keyPoints.Length);

                        foreach (var keyPoint in keyPoints)
                        {
                            writer.Write(keyPoint.Time);
                            writer.Write(keyPoint.Value);
                            writer.Write(keyPoint.Envelope);
                        }
                    });

                    break;
                }
        elif self.Type == PropertyType.ColorSequence:
                {
                    props.ForEach(prop =>
                    {
                        ColorSequence value = prop.CastValue<ColorSequence>();

                        var keyPoints = value.Keypoints;
                        writer.Write(keyPoints.Length);

                        foreach (var keyPoint in keyPoints)
                        {
                            Color3 color = keyPoint.Value;
                            writer.Write(keyPoint.Time);

                            writer.Write(color.R);
                            writer.Write(color.G);
                            writer.Write(color.B);

                            writer.Write(keyPoint.Envelope);
                        }
                    });

                    break;
                }
        elif self.Type == PropertyType.NumberRange:
                {
                    props.ForEach(prop =>
                    {
                        NumberRange value = prop.CastValue<NumberRange>();
                        writer.Write(value.Min);
                        writer.Write(value.Max);
                    });

                    break;
                }
        elif self.Type == PropertyType.Rect:
                {
                    var Rect_X0 = new List<float>();
                    var Rect_Y0 = new List<float>();

                    var Rect_X1 = new List<float>();
                    var Rect_Y1 = new List<float>();

                    props.ForEach(prop =>
                    {
                        Rect value = prop.CastValue<Rect>();

                        Vector2 min = value.Min;
                        Rect_X0.Add(min.X);
                        Rect_Y0.Add(min.Y);

                        Vector2 max = value.Max;
                        Rect_X1.Add(max.X);
                        Rect_Y1.Add(max.Y);
                    });

                    writer.WriteFloats(Rect_X0);
                    writer.WriteFloats(Rect_Y0);

                    writer.WriteFloats(Rect_X1);
                    writer.WriteFloats(Rect_Y1);

                    break;
                }
        elif self.Type == PropertyType.PhysicalProperties:
                {
                    props.ForEach(prop =>
                    {
                        bool custom = (prop.Value != null);
                        writer.Write(custom);

                        if (custom)
                        {
                            PhysicalProperties value = prop.CastValue<PhysicalProperties>();

                            writer.Write(value.Density);
                            writer.Write(value.Friction);
                            writer.Write(value.Elasticity);

                            writer.Write(value.FrictionWeight);
                            writer.Write(value.ElasticityWeight);
                        }
                    });

                    break;
                }
        elif self.Type == PropertyType.Color3uint8:
                {
                    var Color3uint8_R = new List<byte>();
                    var Color3uint8_G = new List<byte>();
                    var Color3uint8_B = new List<byte>();

                    props.ForEach(prop =>
                    {
                        Color3uint8 value = prop.CastValue<Color3uint8>();
                        Color3uint8_R.Add(value.R);
                        Color3uint8_G.Add(value.G);
                        Color3uint8_B.Add(value.B);
                    });

                    byte[] rBuffer = Color3uint8_R.ToArray();
                    writer.Write(rBuffer);

                    byte[] gBuffer = Color3uint8_G.ToArray();
                    writer.Write(gBuffer);

                    byte[] bBuffer = Color3uint8_B.ToArray();
                    writer.Write(bBuffer);

                    break;
                }
            """
        elif self.Type == PropertyType.Int64:
            stream.write_longs(props)
            """
        elif self.Type == PropertyType.SharedString:
                {
                    var sharedKeys = new List<uint>();
                    SSTR sstr = file.SSTR;

                    if (sstr == null)
                    {
                        sstr = new SSTR();
                        file.SSTR = sstr;
                    }

                    props.ForEach(prop =>
                    {
                        var shared = prop.CastValue<SharedString>();

                        if (shared == null)
                        {
                            byte[] empty = Array.Empty<byte>();
                            shared = SharedString.FromBuffer(empty);
                        }

                        string key = shared.Key;

                        if (!sstr.Lookup.ContainsKey(key))
                        {
                            uint id = (uint)sstr.Lookup.Count;
                            sstr.Strings.Add(id, shared);
                            sstr.Lookup.Add(key, id);
                        }

                        uint hashId = sstr.Lookup[key];
                        sharedKeys.Add(hashId);
                    });

                    writer.WriteInterleaved(sharedKeys);
                    break;
                }
        elif self.Type == PropertyType.ProtectedString:
                {
                    props.ForEach(prop =>
                    {
                        var protect = prop.CastValue<ProtectedString>();
                        byte[] buffer = protect.RawBuffer;

                        writer.Write(buffer.Length);
                        writer.Write(buffer);
                    });

                    break;
                }
        elif self.Type == PropertyType.UniqueId:
                {
                    props.ForEach(prop =>
                    {
                        var uniqueId = prop.CastValue<UniqueId>();
                        writer.Write(uniqueId.Index);
                        writer.Write(uniqueId.Time);
                        writer.Write(uniqueId.Random);
                    });

                    break;
                }
        elif self.Type == PropertyType.FontFace:
                {
                    props.ForEach(prop =>
                    {
                        var font = prop.CastValue<FontFace>();

                        string family = font.Family;
                        writer.WriteString(font.Family);

                        if (family.EndsWith(".otf") || family.EndsWith(".ttf"))
                            return;

                        var weight = (ushort)font.Weight;
                        writer.Write(weight);

                        var style = (byte)font.Style;
                        writer.Write(style);
                    });

                    break;
                }
            """
        elif self.Type == PropertyType.SecurityCapabilities:
            """
            {
                // FIXME: Verify this is correct once we know what SecurityCapabilities actually does.
                var capabilities = new List<ulong>();

                props.ForEach(prop =>
                {
                    var value = prop.CastValue<ulong>();
                    capabilities.Add(value);
                });

                writer.WriteInterleaved(capabilities);
                break;
            }
            """
        else:
            raise NotImplementedError(
                f"Unhandled property type: {self.Type} in {self}!"
            )

    def write_info(self):
        """
                public void WriteInfo(StringBuilder builder)
                {
                    builder.AppendLine($"- Name:       {Name}");
                    builder.AppendLine($"- Type:       {Type}");
                    builder.AppendLine($"- TypeId:     {TypeId}");
                    builder.AppendLine($"- ClassName:  {ClassName}");
                    builder.AppendLine($"- ClassIndex: {ClassIndex}");

                    builder.AppendLine($"| InstanceId |           Value           |");
                    builder.AppendLine($"|-----------:|---------------------------|");

                    INST inst = File.Classes[ClassIndex];

                    foreach (var instId in inst.InstanceIds)
                    {
                        Instance instance = File.Instances[instId];
                        Property prop = instance?.GetProperty(Name);

                        object value = prop?.Value;
                        string str = value?.ToInvariantString() ?? "null";

                        if (value is byte[] buffer)
                            str = Convert.ToBase64String(buffer);

                        if (str.Length > 25)
                            str = str.Substring(0, 22) + "...";

                        str = str.Replace('\r', ' ');
                        str = str.Replace('\n', ' ');

                        string row = string.Format("| {0, 10} | {1, -25} |", instId, str);
                        builder.AppendLine(row);
                    }
                }
            }
        }"""


@dataclass
class PRNT:
    FORMAT: int = 0
    File: BinaryRobloxFile = None

    def deserialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        format, id_count = stream.unpack("<bi")
        assert (
            format == self.FORMAT
        ), f"Unexpected PRNT format: {format} (expected {self.FORMAT}!)"
        child_ids = stream.read_instance_ids(id_count)
        parent_ids = stream.read_instance_ids(id_count)
        for child_id, parent_id in zip(child_ids, parent_ids):
            child = file.Instances[child_id]
            parent = file.Instances[parent_id] if parent_id >= 0 else file
            """ will raise a KeyError on the lines above
                if (child == null)
                {
                    RobloxFile.LogError($"PRNT: could not parent {childId} to {parentId} because child {childId} was null.");
                    continue;
                }

                if (parentId >= 0 && parent == null)
                {
                    RobloxFile.LogError($"PRNT: could not parent {childId} to {parentId} because parent {parentId} was null.");
                    continue;
                }
            """
            child.Parent = parent

    def serialize(self, stream: BinaryStream, file: BinaryRobloxFile):
        child_ids = [int(inst.referent) for inst in file.PostInstances]
        parent_ids = [
            int(inst.Parent.referent) if inst.Parent is not None else -1
            for inst in file.PostInstances
        ]

        stream.pack("<bi", self.FORMAT, len(child_ids))
        stream.write_instance_ids(child_ids)
        stream.write_instance_ids(parent_ids)

    def write_info(self):
        """
        public void WriteInfo(StringBuilder builder)
        {
            var childIds = new List<int>();
            var parentIds = new List<int>();

            foreach (Instance inst in File.GetDescendants())
            {
                Instance parent = inst.Parent;

                int childId = int.Parse(inst.referent);
                int parentId = -1;

                if (parent != null)
                    parentId = int.Parse(parent.referent);

                childIds.Add(childId);
                parentIds.Add(parentId);
            }

            builder.AppendLine($"- Format:    {FORMAT}");
            builder.AppendLine($"- ChildIds:  {string.Join(", ", childIds)}");
            builder.AppendLine($"- ParentIds: {string.Join(", ", parentIds)}");
        }
        """


class BinaryRobloxFileChunk:
    """
    BinaryRobloxFileChunk represents a generic LZ4 - compressed chunk
    of data in Roblox's Binary File Format.
    """

    def __init__(self):
        self.ChunkType = b""
        self.Reserved = -1
        self.CompressedSize = -1
        self.Size = -1
        self.CompressedData = b""
        self.Data = b""
        self.HasWriteBuffer = False
        self.WriteBuffer = bytearray()
        self.Handler = None

    @classmethod
    def with_data(cls, chunk_type: bytes, data: bytes, compress: bool = True):
        if len(chunk_type) != 4:
            raise ValueError(
                f"chunk_type {chunk_type!r} has length {len(chunk_type)}, but should have length 4"
            )
        chunk = cls()
        chunk.ChunkType = chunk_type
        chunk.Data = data
        chunk.Size = len(data)
        if compress:
            chunk.CompressedData = lz4.block.compress(chunk.Data, store_size=False)
            chunk.CompressedSize = len(chunk.CompressedData)
        return chunk

    @property
    def HasCompressedData(self):
        return self.CompressedSize > 0

    def __str__(self):
        chunkType = self.ChunkType.replace(b"\0", b" ")
        return f"'{chunkType}' Chunk ({self.Size} bytes) [{self.Handler}]"

    def deserialize(self, stream: BinaryStream):
        (
            self.ChunkType,
            self.CompressedSize,
            self.Size,
            self.Reserved,
        ) = stream.unpack("<4siii")

        if self.HasCompressedData:
            self.CompressedData = stream.read_bytes(self.CompressedSize)
            self.Data = lz4.block.decompress(self.CompressedData, self.Size)
        else:
            self.Data = stream.read_bytes(self.Size)

    def serialize(self, stream: BinaryStream):
        stream.pack(
            "<4siii",
            self.ChunkType,
            self.CompressedSize,
            self.Size,
            self.Reserved,
        )

        if self.HasCompressedData:
            stream.write_bytes(self.CompressedData)
        else:
            stream.write_bytes(self.Data)


class BinaryRobloxFile(Instance):  # (RobloxFile):
    # Header Specific
    MAGIC_HEADER = b"<roblox!\x89\xff\x0d\x0a\x1a\x0a"

    def __init__(self):
        super().__init__(self.__class__.__name__)
        # Header Specific
        self.Version = 0
        self.NumClasses = 0
        self.NumInstances = 0
        self.Reserved = 0

        # Runtime Specific
        self.ChunksImpl: list[BinaryRobloxFileChunk] = []

        self.Instances: list[Instance] = []  # reading/writing. parent -> child order
        self.Classes: list[INST | None] = []  # reading, writing
        self.ClassMap: dict[str, INST] = {}  # writing
        self.PostInstances: list[Instance] = []  # writing. child -> parent order

        self.META = None
        self.SSTR = None
        self.SIGN = None

        self.Name = "Bin:"
        self.referent = "-1"
        self.ParentLocked = True

    @property
    def Chunks(self):
        return self.ChunksImpl

    @property
    def HasMetadata(self):
        return self.META is not None

    @property
    def Metadata(self):
        return self.META.Data if self.META else {}

    @property
    def HasSharedStrings(self):
        return self.SSTR is not None

    @property
    def SharedStrings(self):
        return self.SSTR.Strings if self.SSTR else {}

    @property
    def HasSignatures(self):
        return self.SIGN is not None

    @property
    def Signatures(self):
        return self.SIGN.Signatures if self.SIGN else []

    def deserialize(self, file):
        stream = BinaryStream(file)
        # Verify the signature of the file.
        signature = stream.read_bytes(len(self.MAGIC_HEADER))
        if signature != self.MAGIC_HEADER:
            raise ValueError(
                "Provided file's signature does not match BinaryRobloxFile.MAGIC_HEADER!"
            )

        # Read header data.
        (
            self.Version,
            self.NumClasses,
            self.NumInstances,
            self.Reserved,
        ) = stream.unpack("<HIIq")

        # Begin reading the file chunks.
        reading = True

        self.Classes = [None] * self.NumClasses
        self.Instances = [None] * self.NumInstances

        while reading:
            chunk = BinaryRobloxFileChunk()
            chunk.deserialize(stream)
            handler = None
            if chunk.ChunkType == b"INST":
                handler = INST()
            elif chunk.ChunkType == b"PROP":
                handler = PROP()
            elif chunk.ChunkType == b"PRNT":
                handler = PRNT()
            elif chunk.ChunkType == b"META":
                handler = META()
            elif chunk.ChunkType == b"SSTR":
                handler = None  # SSTR();
            elif chunk.ChunkType == b"SIGN":
                handler = None  # SIGN();
            elif chunk.ChunkType == b"END\0":
                reading = False
            else:
                self.LogError(
                    f"BinaryRobloxFile - Unhandled chunk-type: {chunk.ChunkType}!"
                )
            if handler:
                chunk_stream = BinaryStream(BytesIO(chunk.Data))
                chunk.Handler = handler
                handler.deserialize(chunk_stream, self)
            self.ChunksImpl.append(chunk)

    def _record_instances(self, instances: list[Instance]) -> None:
        for instance in instances:
            if not instance.Archivable:
                continue

            inst_id = self.NumInstances
            self.NumInstances += 1
            class_name = instance.ClassName

            instance.referent = str(inst_id)
            self.Instances.append(instance)

            cls = self.ClassMap.get(class_name)
            if cls is None:
                cls = INST()
                cls.ClassName = class_name
                cls.IsService = instance.is_service
                self.ClassMap[class_name] = cls

            cls.NumInstances += 1
            cls.InstanceIds.append(inst_id)

            self._record_instances(instance.Children)
            self.PostInstances.append(instance)

    def _build_chunk(self, handler, compress: bool = True) -> BinaryRobloxFileChunk:
        stream = BinaryStream(BytesIO())
        handler.serialize(stream, self)
        return BinaryRobloxFileChunk.with_data(
            handler.__class__.__name__.encode(),
            stream.base_stream.getvalue(),
            compress,
        )

    def serialize(self, file):
        """Generate the chunk data."""
        # Clear the existing data.
        self.referent = "-1"
        self.ChunksImpl.clear()
        self.NumInstances = 0
        self.NumClasses = 0
        self.SSTR = None
        self.Instances.clear()
        self.PostInstances.clear()
        self.ClassMap.clear()

        # Recursively capture all instances and classes.
        self._record_instances(self.Children)
        self.Classes = sorted(self.ClassMap.values(), key=lambda c: c.ClassName)
        for i, cls in enumerate(self.Classes):
            cls.ClassIndex = i
        self.NumClasses = len(self.Classes)

        # Write the INST chunks.
        for inst in self.Classes:
            self.ChunksImpl.append(self._build_chunk(inst))

        # Write the PROP chunks.
        for inst in self.Classes:
            for prop in PROP._collect_properties(inst):
                self.ChunksImpl.append(self._build_chunk(prop))

        # Write the PRNT chunk.
        self.ChunksImpl.append(self._build_chunk(PRNT()))

        # Write the SSTR chunk.
        # if self.HasSharedStrings:
        #     self.ChunksImpl.insert(0, self._build_chunk(self.SSTR))

        # Write the META chunk.
        if self.HasMetadata:
            self.ChunksImpl.insert(0, self._build_chunk(self.META))

        # Write the SIGN chunk.
        # if self.HasSignatures:
        #     self.ChunksImpl.append(self._build_chunk(self.SIGN))

        # Write the END chunk.
        self.ChunksImpl.append(
            BinaryRobloxFileChunk.with_data(b"END\0", b"</roblox>", compress=False)
        )

        # Write the chunk buffers with the header data
        stream = BinaryStream(file)
        stream.write_bytes(self.MAGIC_HEADER)
        stream.pack(
            "<HIIq",
            self.Version,
            self.NumClasses,
            self.NumInstances,
            self.Reserved,
        )
        for chunk in self.Chunks:
            chunk.serialize(stream)
